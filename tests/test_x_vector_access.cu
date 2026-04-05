/**
 * @file test_x_vector_access.cu
 * @brief Analyze x-vector access patterns and optimization strategies
 *
 * For Mars X201 with small L2 cache (~4MB), x-vector access is critical.
 * Test:
 * 1. Direct global memory access
 * 2. __ldg (read-only cache)
 * 3. Texture memory (if supported)
 * 4. Shared memory caching
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

// Kernel 1: Direct global memory access
template<typename FloatType, int BLOCK_SIZE>
__global__ void spmv_direct_kernel(
    int numRows,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y)
{
    __shared__ int sharedRowPtr[BLOCK_SIZE + 1];

    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int globalWarpId = blockIdx.x * (BLOCK_SIZE / WARP_SIZE) + warpId;

    int baseRow = globalWarpId * 16;

    if (lane < 17 && baseRow + lane <= numRows) {
        sharedRowPtr[threadIdx.x / WARP_SIZE * 17 + lane] = rowPtr[baseRow + lane];
    }
    __syncthreads();

    int rowIdx = lane / 4;
    int threadInRow = lane % 4;
    int row = baseRow + rowIdx;

    if (row >= numRows) return;

    FloatType sum = 0;
    int rowStart = sharedRowPtr[threadIdx.x / WARP_SIZE * 17 + rowIdx];
    int rowEnd = sharedRowPtr[threadIdx.x / WARP_SIZE * 17 + rowIdx + 1];

    // Direct access - no __ldg
    for (int idx = rowStart + threadInRow; idx < rowEnd; idx += 4) {
        sum += values[idx] * x[colIdx[idx]];
    }

    sum += __shfl_down_sync(0xffffffff, sum, 2, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 1, 4);

    if (threadInRow == 0) {
        y[row] = sum;
    }
}

// Kernel 2: Using __ldg (current best)
template<typename FloatType, int BLOCK_SIZE>
__global__ void spmv_ldg_kernel(
    int numRows,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y)
{
    __shared__ int sharedRowPtr[BLOCK_SIZE + 1];

    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int globalWarpId = blockIdx.x * (BLOCK_SIZE / WARP_SIZE) + warpId;

    int baseRow = globalWarpId * 16;

    if (lane < 17 && baseRow + lane <= numRows) {
        sharedRowPtr[threadIdx.x / WARP_SIZE * 17 + lane] = rowPtr[baseRow + lane];
    }
    __syncthreads();

    int rowIdx = lane / 4;
    int threadInRow = lane % 4;
    int row = baseRow + rowIdx;

    if (row >= numRows) return;

    FloatType sum = 0;
    int rowStart = sharedRowPtr[threadIdx.x / WARP_SIZE * 17 + rowIdx];
    int rowEnd = sharedRowPtr[threadIdx.x / WARP_SIZE * 17 + rowIdx + 1];

    for (int idx = rowStart + threadInRow; idx < rowEnd; idx += 4) {
        sum += values[idx] * __ldg(&x[colIdx[idx]]);
    }

    sum += __shfl_down_sync(0xffffffff, sum, 2, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 1, 4);

    if (threadInRow == 0) {
        y[row] = sum;
    }
}

// Kernel 3: Prefetch x values
template<typename FloatType, int BLOCK_SIZE>
__global__ void spmv_prefetch_kernel(
    int numRows,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y)
{
    __shared__ int sharedRowPtr[BLOCK_SIZE + 1];

    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int globalWarpId = blockIdx.x * (BLOCK_SIZE / WARP_SIZE) + warpId;

    int baseRow = globalWarpId * 16;

    if (lane < 17 && baseRow + lane <= numRows) {
        sharedRowPtr[threadIdx.x / WARP_SIZE * 17 + lane] = rowPtr[baseRow + lane];
    }
    __syncthreads();

    int rowIdx = lane / 4;
    int threadInRow = lane % 4;
    int row = baseRow + rowIdx;

    if (row >= numRows) return;

    FloatType sum = 0;
    int rowStart = sharedRowPtr[threadIdx.x / WARP_SIZE * 17 + rowIdx];
    int rowEnd = sharedRowPtr[threadIdx.x / WARP_SIZE * 17 + rowIdx + 1];

    int rowLen = rowEnd - rowStart;
    int idx = rowStart + threadInRow;

    // Prefetch first element
    if (idx < rowEnd) {
        FloatType x_val = __ldg(&x[colIdx[idx]]);
        FloatType v_val = values[idx];

        for (idx += 4; idx < rowEnd; idx += 4) {
            FloatType x_next = __ldg(&x[colIdx[idx]]);
            sum += v_val * x_val;
            x_val = x_next;
            v_val = values[idx];
        }
        sum += v_val * x_val;
    }

    sum += __shfl_down_sync(0xffffffff, sum, 2, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 1, 4);

    if (threadInRow == 0) {
        y[row] = sum;
    }
}

// Kernel 4: Dual accumulation for better ILP
template<typename FloatType, int BLOCK_SIZE>
__global__ void spmv_dual_accum_kernel(
    int numRows,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y)
{
    __shared__ int sharedRowPtr[BLOCK_SIZE + 1];

    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int globalWarpId = blockIdx.x * (BLOCK_SIZE / WARP_SIZE) + warpId;

    int baseRow = globalWarpId * 16;

    if (lane < 17 && baseRow + lane <= numRows) {
        sharedRowPtr[threadIdx.x / WARP_SIZE * 17 + lane] = rowPtr[baseRow + lane];
    }
    __syncthreads();

    int rowIdx = lane / 4;
    int threadInRow = lane % 4;
    int row = baseRow + rowIdx;

    if (row >= numRows) return;

    FloatType sum0 = 0, sum1 = 0;
    int rowStart = sharedRowPtr[threadIdx.x / WARP_SIZE * 17 + rowIdx];
    int rowEnd = sharedRowPtr[threadIdx.x / WARP_SIZE * 17 + rowIdx + 1];

    int idx = rowStart + threadInRow;
    for (; idx + 4 < rowEnd; idx += 8) {
        sum0 += values[idx] * __ldg(&x[colIdx[idx]]);
        sum1 += values[idx + 4] * __ldg(&x[colIdx[idx + 4]]);
    }
    if (idx < rowEnd) {
        sum0 += values[idx] * __ldg(&x[colIdx[idx]]);
    }

    FloatType sum = sum0 + sum1;

    sum += __shfl_down_sync(0xffffffff, sum, 2, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 1, 4);

    if (threadInRow == 0) {
        y[row] = sum;
    }
}

void runTest(int rows, int cols, int avgNnz, int iterations) {
    std::cout << "\n=== Test: rows=" << rows << ", avgNnz=" << avgNnz << " ===\n";

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
    int blockSize = 256;
    int rowsPerBlock = (blockSize / WARP_SIZE) * 16;
    int gridSize = (rows + rowsPerBlock - 1) / rowsPerBlock;

    // Test 1: Direct access
    std::cout << "1. Direct: ";
    float totalTime = 0;
    for (int i = 0; i < iterations; i++) {
        cudaMemset(d_y, 0, rows * sizeof(float));
        cudaDeviceSynchronize();
        timer.start();
        spmv_direct_kernel<float, 256><<<gridSize, blockSize>>>(
            rows, matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values, d_x, d_y);
        timer.stop();
        totalTime += timer.elapsed_ms();
    }
    float avgTime = totalTime / iterations;
    float bw = (dataBytes / (avgTime * 1e-3)) / (1024 * 1024 * 1024);
    float util = bw / peakBW * 100;
    std::cout << avgTime << " ms, " << bw << " GB/s, " << util << "%\n";

    // Test 2: __ldg
    std::cout << "2. LDG: ";
    totalTime = 0;
    for (int i = 0; i < iterations; i++) {
        cudaMemset(d_y, 0, rows * sizeof(float));
        cudaDeviceSynchronize();
        timer.start();
        spmv_ldg_kernel<float, 256><<<gridSize, blockSize>>>(
            rows, matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values, d_x, d_y);
        timer.stop();
        totalTime += timer.elapsed_ms();
    }
    avgTime = totalTime / iterations;
    bw = (dataBytes / (avgTime * 1e-3)) / (1024 * 1024 * 1024);
    util = bw / peakBW * 100;
    std::cout << avgTime << " ms, " << bw << " GB/s, " << util << "%\n";

    // Test 3: Prefetch
    std::cout << "3. Prefetch: ";
    totalTime = 0;
    for (int i = 0; i < iterations; i++) {
        cudaMemset(d_y, 0, rows * sizeof(float));
        cudaDeviceSynchronize();
        timer.start();
        spmv_prefetch_kernel<float, 256><<<gridSize, blockSize>>>(
            rows, matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values, d_x, d_y);
        timer.stop();
        totalTime += timer.elapsed_ms();
    }
    avgTime = totalTime / iterations;
    bw = (dataBytes / (avgTime * 1e-3)) / (1024 * 1024 * 1024);
    util = bw / peakBW * 100;
    std::cout << avgTime << " ms, " << bw << " GB/s, " << util << "%\n";

    // Test 4: Dual accumulation
    std::cout << "4. Dual Accum: ";
    totalTime = 0;
    for (int i = 0; i < iterations; i++) {
        cudaMemset(d_y, 0, rows * sizeof(float));
        cudaDeviceSynchronize();
        timer.start();
        spmv_dual_accum_kernel<float, 256><<<gridSize, blockSize>>>(
            rows, matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values, d_x, d_y);
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
    std::cout << "=== X-Vector Access Pattern Test ===\n";
    std::cout << "WARP_SIZE = " << WARP_SIZE << "\n";

    int iterations = 10;

    runTest(100000, 1000, 4, iterations);
    runTest(500000, 1000, 4, iterations);
    runTest(1000000, 1000, 4, iterations);

    return 0;
}