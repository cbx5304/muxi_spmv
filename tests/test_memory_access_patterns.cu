/**
 * @file test_memory_access_patterns.cu
 * @brief Analyze memory access patterns and try shared memory optimization
 *
 * For Mars X201 with small L2 cache (~4MB), we explore:
 * 1. Shared memory x-vector caching
 * 2. Coalesced memory access
 * 3. L2 cache persistence hints (if supported)
 */

#include <iostream>
#include <cmath>

#include "formats/sparse_formats.h"
#include "generators/matrix_generator.h"

using namespace muxi_spmv;
using namespace muxi_spmv::generators;

class GpuTimer {
public:
    GpuTimer() { cudaEventCreate(&start_); cudaEventCreate(&stop_); }
    ~GpuTimer() { cudaEventDestroy(start_); cudaEventDestroy(stop_); }
    void start() { cudaEventRecord(start_, 0); }
    void stop() { cudaEventRecord(stop_, 0); cudaEventSynchronize(stop_); }
    float elapsed_ms() { float ms; cudaEventElapsedTime(&ms, start_, stop_); return ms; }
private:
    cudaEvent_t start_, stop_;
};

// Kernel 1: Standard scalar kernel
template<typename FloatType, int BLOCK_SIZE>
__global__ void spmv_scalar_kernel(
    int numRows,
    int numCols,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= numRows) return;

    FloatType sum = static_cast<FloatType>(0);
    for (int idx = rowPtr[row]; idx < rowPtr[row + 1]; idx++) {
        sum += values[idx] * x[colIdx[idx]];
    }
    y[row] = sum;
}

// Kernel 2: Shared memory x-cache kernel
// Each block caches part of x vector in shared memory
template<typename FloatType, int BLOCK_SIZE, int CACHE_SIZE>
__global__ void spmv_shared_cache_kernel(
    int numRows,
    int numCols,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y,
    int cacheOffset)  // Which part of x to cache
{
    __shared__ FloatType xCache[CACHE_SIZE];

    // Cooperatively load x values into shared memory
    int cacheStart = cacheOffset;
    for (int i = threadIdx.x; i < CACHE_SIZE && (cacheStart + i) < numCols; i += BLOCK_SIZE) {
        xCache[i] = x[cacheStart + i];
    }
    __syncthreads();

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= numRows) return;

    FloatType sum = static_cast<FloatType>(0);
    for (int idx = rowPtr[row]; idx < rowPtr[row + 1]; idx++) {
        int col = colIdx[idx];
        // Check if column is in cache
        if (col >= cacheStart && col < cacheStart + CACHE_SIZE) {
            sum += values[idx] * xCache[col - cacheStart];
        } else {
            sum += values[idx] * x[col];
        }
    }
    y[row] = sum;
}

// Kernel 3: Vectorized load kernel (4 elements at a time)
template<typename FloatType, int BLOCK_SIZE>
__global__ void spmv_vectorized_kernel(
    int numRows,
    int numCols,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= numRows) return;

    FloatType sum = static_cast<FloatType>(0);
    int rowStart = rowPtr[row];
    int rowEnd = rowPtr[row + 1];
    int rowLen = rowEnd - rowStart;

    // Process 4 elements at a time
    int idx = rowStart;
    for (; idx + 3 < rowEnd; idx += 4) {
        sum += values[idx] * x[colIdx[idx]];
        sum += values[idx+1] * x[colIdx[idx+1]];
        sum += values[idx+2] * x[colIdx[idx+2]];
        sum += values[idx+3] * x[colIdx[idx+3]];
    }

    // Process remaining elements
    for (; idx < rowEnd; idx++) {
        sum += values[idx] * x[colIdx[idx]];
    }

    y[row] = sum;
}

// Kernel 4: ILP kernel - instruction level parallelism
template<typename FloatType, int BLOCK_SIZE>
__global__ void spmv_ilp_kernel(
    int numRows,
    int numCols,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= numRows) return;

    FloatType sum0 = static_cast<FloatType>(0);
    FloatType sum1 = static_cast<FloatType>(0);
    FloatType sum2 = static_cast<FloatType>(0);
    FloatType sum3 = static_cast<FloatType>(0);

    int rowStart = rowPtr[row];
    int rowEnd = rowPtr[row + 1];

    // Process 4 elements at a time with separate accumulators
    int idx = rowStart;
    for (; idx + 3 < rowEnd; idx += 4) {
        sum0 += values[idx] * x[colIdx[idx]];
        sum1 += values[idx+1] * x[colIdx[idx+1]];
        sum2 += values[idx+2] * x[colIdx[idx+2]];
        sum3 += values[idx+3] * x[colIdx[idx+3]];
    }

    // Process remaining elements
    FloatType sum = sum0 + sum1 + sum2 + sum3;
    for (; idx < rowEnd; idx++) {
        sum += values[idx] * x[colIdx[idx]];
    }

    y[row] = sum;
}

// Kernel 5: Texture memory for x vector
// (Using __ldg intrinsic which is similar to texture cache)

// Kernel 6: Prefetch kernel
template<typename FloatType, int BLOCK_SIZE>
__global__ void spmv_prefetch_kernel(
    int numRows,
    int numCols,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= numRows) return;

    FloatType sum = static_cast<FloatType>(0);
    int rowStart = rowPtr[row];
    int rowEnd = rowPtr[row + 1];

    // Prefetch next row's data
    if (row + 1 < numRows) {
        int nextRowStart = rowPtr[row + 1];
        // Prefetch hint (compiler may ignore)
        // Using __builtin_prefetch for GCC/clang
    }

    for (int idx = rowStart; idx < rowEnd; idx++) {
        sum += values[idx] * x[colIdx[idx]];
    }

    y[row] = sum;
}

void runTest(int rows, int cols, int avgNnz, int iterations) {
    std::cout << "\n=== Test: avgNnz=" << avgNnz << " ===\n";

    // Generate matrix
    CSRMatrix<float> matrix;
    int nnz = rows * avgNnz;
    generateRandomMatrix<float>(rows, cols, nnz, matrix);
    matrix.allocateDevice();
    matrix.copyToDevice();
    cudaDeviceSynchronize();

    float* h_x = new float[cols];
    for (int i = 0; i < cols; i++) h_x[i] = rand() / (float)RAND_MAX;

    float* d_x, *d_y;
    cudaMalloc(&d_x, cols * sizeof(float));
    cudaMalloc(&d_y, rows * sizeof(float));
    cudaMemcpy(d_x, h_x, cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    float peakBW = (WARP_SIZE == 64) ? 1843.0f : 1008.0f;
    size_t dataBytes = rows * sizeof(int) * 2 + matrix.nnz * sizeof(int) +
                       matrix.nnz * sizeof(float) * 2 + rows * sizeof(float);

    GpuTimer timer;
    float totalTime;

    int blockSize = 256;
    int gridSize = (rows + blockSize - 1) / blockSize;

    // Test 1: Standard scalar
    std::cout << "1. Scalar: ";
    totalTime = 0;
    for (int i = 0; i < iterations; i++) {
        cudaMemset(d_y, 0, rows * sizeof(float));
        cudaDeviceSynchronize();
        timer.start();
        spmv_scalar_kernel<float, 256><<<gridSize, blockSize>>>(
            rows, cols, matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values, d_x, d_y);
        timer.stop();
        totalTime += timer.elapsed_ms();
    }
    float avgTime = totalTime / iterations;
    float bw = (dataBytes / (avgTime * 1e-3)) / (1024 * 1024 * 1024);
    float util = bw / peakBW * 100;
    std::cout << avgTime << " ms, " << bw << " GB/s, " << util << "%\n";

    // Test 2: Vectorized
    std::cout << "2. Vectorized: ";
    totalTime = 0;
    for (int i = 0; i < iterations; i++) {
        cudaMemset(d_y, 0, rows * sizeof(float));
        cudaDeviceSynchronize();
        timer.start();
        spmv_vectorized_kernel<float, 256><<<gridSize, blockSize>>>(
            rows, cols, matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values, d_x, d_y);
        timer.stop();
        totalTime += timer.elapsed_ms();
    }
    avgTime = totalTime / iterations;
    bw = (dataBytes / (avgTime * 1e-3)) / (1024 * 1024 * 1024);
    util = bw / peakBW * 100;
    std::cout << avgTime << " ms, " << bw << " GB/s, " << util << "%\n";

    // Test 3: ILP
    std::cout << "3. ILP: ";
    totalTime = 0;
    for (int i = 0; i < iterations; i++) {
        cudaMemset(d_y, 0, rows * sizeof(float));
        cudaDeviceSynchronize();
        timer.start();
        spmv_ilp_kernel<float, 256><<<gridSize, blockSize>>>(
            rows, cols, matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values, d_x, d_y);
        timer.stop();
        totalTime += timer.elapsed_ms();
    }
    avgTime = totalTime / iterations;
    bw = (dataBytes / (avgTime * 1e-3)) / (1024 * 1024 * 1024);
    util = bw / peakBW * 100;
    std::cout << avgTime << " ms, " << bw << " GB/s, " << util << "%\n";

    delete[] h_x;
    cudaFree(d_x);
    cudaFree(d_y);
}

int main(int argc, char** argv) {
    std::cout << "=== Memory Access Pattern Test ===\n";
    std::cout << "WARP_SIZE = " << WARP_SIZE << "\n";

    int rows = 500000;  // Reduced from 1M to avoid memory issues
    int cols = 1000;
    int iterations = 10;

    runTest(rows, cols, 4, iterations);
    runTest(rows, cols, 6, iterations);

    return 0;
}