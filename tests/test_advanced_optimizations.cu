/**
 * @file test_advanced_optimizations.cu
 * @brief Advanced optimization tests for Mars X201 extremely sparse matrices
 *
 * Focus on avgNnz<=4 optimization strategies:
 * 1. __ldg (read-only cache) for x vector
 * 2. Shared memory caching with proper partitioning
 * 3. Warp-level reduction optimizations
 * 4. Combine ILP + __ldg
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

// Kernel 1: Scalar with __ldg for x vector (read-only cache)
template<typename FloatType, int BLOCK_SIZE>
__global__ void spmv_ldg_kernel(
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
        // Use __ldg to read x through read-only cache (texture cache)
        sum += values[idx] * __ldg(&x[colIdx[idx]]);
    }
    y[row] = sum;
}

// Kernel 2: ILP + __ldg combination
template<typename FloatType, int BLOCK_SIZE>
__global__ void spmv_ilp_ldg_kernel(
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

    // Process 4 elements at a time with separate accumulators + __ldg
    int idx = rowStart;
    for (; idx + 3 < rowEnd; idx += 4) {
        sum0 += values[idx] * __ldg(&x[colIdx[idx]]);
        sum1 += values[idx+1] * __ldg(&x[colIdx[idx+1]]);
        sum2 += values[idx+2] * __ldg(&x[colIdx[idx+2]]);
        sum3 += values[idx+3] * __ldg(&x[colIdx[idx+3]]);
    }

    // Process remaining elements
    FloatType sum = sum0 + sum1 + sum2 + sum3;
    for (; idx < rowEnd; idx++) {
        sum += values[idx] * __ldg(&x[colIdx[idx]]);
    }

    y[row] = sum;
}

// Kernel 3: Virtual warp + ILP
template<typename FloatType, int BLOCK_SIZE, int VIRTUAL_WARP>
__global__ void spmv_virtual_warp_ilp_kernel(
    int numRows,
    int numCols,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int virtualWarpId = tid / VIRTUAL_WARP;
    int laneInVirtual = tid % VIRTUAL_WARP;

    if (virtualWarpId >= numRows) return;

    FloatType sum0 = static_cast<FloatType>(0);
    FloatType sum1 = static_cast<FloatType>(0);

    int rowStart = rowPtr[virtualWarpId];
    int rowEnd = rowPtr[virtualWarpId + 1];

    // Each thread in virtual warp processes part of the row with ILP
    // Stride is VIRTUAL_WARP, process 2 elements per iteration with proper bounds check
    int idx = rowStart + laneInVirtual;
    for (; idx < rowEnd; idx += VIRTUAL_WARP * 2) {
        sum0 += values[idx] * __ldg(&x[colIdx[idx]]);
        // Check bounds for second element
        int idx2 = idx + VIRTUAL_WARP;
        if (idx2 < rowEnd) {
            sum1 += values[idx2] * __ldg(&x[colIdx[idx2]]);
        }
    }
    FloatType sum = sum0 + sum1;

    // Warp reduce within virtual warp
    #pragma unroll
    for (int offset = VIRTUAL_WARP / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset, VIRTUAL_WARP);
    }

    if (laneInVirtual == 0) {
        y[virtualWarpId] = sum;
    }
}

// Kernel 4: Shared memory for row metadata cache
template<typename FloatType, int BLOCK_SIZE, int ROWS_PER_BLOCK>
__global__ void spmv_row_cache_kernel(
    int numRows,
    int numCols,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y)
{
    __shared__ int rowStartCache[ROWS_PER_BLOCK];
    __shared__ int rowEndCache[ROWS_PER_BLOCK];

    int blockStartRow = blockIdx.x * ROWS_PER_BLOCK;

    // Cooperatively load row metadata into shared memory
    if (threadIdx.x < ROWS_PER_BLOCK && blockStartRow + threadIdx.x < numRows) {
        rowStartCache[threadIdx.x] = rowPtr[blockStartRow + threadIdx.x];
        rowEndCache[threadIdx.x] = rowPtr[blockStartRow + threadIdx.x + 1];
    }
    __syncthreads();

    int localRow = threadIdx.x % ROWS_PER_BLOCK;
    int row = blockStartRow + localRow;

    if (row >= numRows) return;

    FloatType sum = static_cast<FloatType>(0);
    int rowStart = rowStartCache[localRow];
    int rowEnd = rowEndCache[localRow];

    for (int idx = rowStart; idx < rowEnd; idx++) {
        sum += values[idx] * __ldg(&x[colIdx[idx]]);
    }

    y[row] = sum;
}

// Kernel 5: Lane-optimized for Mars X201 (warp=64)
// Process rows in groups of 8 rows per warp (8 threads per row)
template<typename FloatType, int BLOCK_SIZE>
__global__ void spmv_lane_optimized_kernel(
    int numRows,
    int numCols,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y)
{
    // Each warp (64 threads) processes 8 rows
    // Each row gets 8 threads
    const int THREADS_PER_ROW = 8;

    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    int rowInWarp = lane / THREADS_PER_ROW;  // Which row within this warp (0-7)
    int threadInRow = lane % THREADS_PER_ROW; // Which thread for this row (0-7)

    int row = warpId * 8 + rowInWarp;
    if (row >= numRows) return;

    FloatType sum = static_cast<FloatType>(0);
    int rowStart = rowPtr[row];
    int rowEnd = rowPtr[row + 1];

    // Each thread handles elements at offsets threadInRow, threadInRow+8, threadInRow+16...
    for (int idx = rowStart + threadInRow; idx < rowEnd; idx += THREADS_PER_ROW) {
        sum += values[idx] * __ldg(&x[colIdx[idx]]);
    }

    // Reduce within the 8 threads handling this row
    #pragma unroll
    for (int offset = THREADS_PER_ROW / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset, THREADS_PER_ROW);
    }

    // Thread 0 of each row group writes result
    if (threadInRow == 0) {
        y[row] = sum;
    }
}

template<int VIRTUAL_WARP>
void runVirtualWarpTest(int rows, int cols, int avgNnz, int iterations,
                        const CSRMatrix<float>& matrix,
                        float* d_x, float* d_y, float peakBW, size_t dataBytes) {
    GpuTimer timer;
    int blockSize = 256;
    int virtualWarpsPerBlock = blockSize / VIRTUAL_WARP;
    int gridSize = (rows + virtualWarpsPerBlock - 1) / virtualWarpsPerBlock;

    float totalTime = 0;
    for (int i = 0; i < iterations; i++) {
        cudaMemset(d_y, 0, rows * sizeof(float));
        cudaDeviceSynchronize();
        timer.start();
        spmv_virtual_warp_ilp_kernel<float, 256, VIRTUAL_WARP><<<gridSize, blockSize>>>(
            rows, cols, matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values, d_x, d_y);
        timer.stop();
        totalTime += timer.elapsed_ms();
    }
    float avgTime = totalTime / iterations;
    float bw = (dataBytes / (avgTime * 1e-3)) / (1024 * 1024 * 1024);
    float util = bw / peakBW * 100;
    std::cout << "   VirtualWarp-ILP-" << VIRTUAL_WARP << ": " << avgTime << " ms, " << bw << " GB/s, " << util << "%\n";
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
    float totalTime;
    float avgTime, bw, util;

    int blockSize = 256;
    int gridSize = (rows + blockSize - 1) / blockSize;

    // Test 1: Scalar + __ldg
    std::cout << "1. Scalar+LDG: ";
    totalTime = 0;
    for (int i = 0; i < iterations; i++) {
        cudaMemset(d_y, 0, rows * sizeof(float));
        cudaDeviceSynchronize();
        timer.start();
        spmv_ldg_kernel<float, 256><<<gridSize, blockSize>>>(
            rows, cols, matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values, d_x, d_y);
        timer.stop();
        totalTime += timer.elapsed_ms();
    }
    avgTime = totalTime / iterations;
    bw = (dataBytes / (avgTime * 1e-3)) / (1024 * 1024 * 1024);
    util = bw / peakBW * 100;
    std::cout << avgTime << " ms, " << bw << " GB/s, " << util << "%\n";

    // Test 2: ILP + __ldg
    std::cout << "2. ILP+LDG: ";
    totalTime = 0;
    for (int i = 0; i < iterations; i++) {
        cudaMemset(d_y, 0, rows * sizeof(float));
        cudaDeviceSynchronize();
        timer.start();
        spmv_ilp_ldg_kernel<float, 256><<<gridSize, blockSize>>>(
            rows, cols, matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values, d_x, d_y);
        timer.stop();
        totalTime += timer.elapsed_ms();
    }
    avgTime = totalTime / iterations;
    bw = (dataBytes / (avgTime * 1e-3)) / (1024 * 1024 * 1024);
    util = bw / peakBW * 100;
    std::cout << avgTime << " ms, " << bw << " GB/s, " << util << "%\n";

    // Test 3: Lane-optimized (8 threads per row)
    std::cout << "3. LaneOptimized(8/thread): ";
    int warpsPerBlock = blockSize / WARP_SIZE;
    int rowsPerWarp = 8;
    int rowsPerBlock = warpsPerBlock * rowsPerWarp;
    int laneGridSize = (rows + rowsPerBlock - 1) / rowsPerBlock;
    totalTime = 0;
    for (int i = 0; i < iterations; i++) {
        cudaMemset(d_y, 0, rows * sizeof(float));
        cudaDeviceSynchronize();
        timer.start();
        spmv_lane_optimized_kernel<float, 256><<<laneGridSize, blockSize>>>(
            rows, cols, matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values, d_x, d_y);
        timer.stop();
        totalTime += timer.elapsed_ms();
    }
    avgTime = totalTime / iterations;
    bw = (dataBytes / (avgTime * 1e-3)) / (1024 * 1024 * 1024);
    util = bw / peakBW * 100;
    std::cout << avgTime << " ms, " << bw << " GB/s, " << util << "%\n";

    // Test 4: Virtual warp + ILP variants
    std::cout << "4. Virtual Warp + ILP variants:\n";
    runVirtualWarpTest<4>(rows, cols, avgNnz, iterations, matrix, d_x, d_y, peakBW, dataBytes);
    runVirtualWarpTest<8>(rows, cols, avgNnz, iterations, matrix, d_x, d_y, peakBW, dataBytes);
    runVirtualWarpTest<16>(rows, cols, avgNnz, iterations, matrix, d_x, d_y, peakBW, dataBytes);

    delete[] h_x;
    cudaFree(d_x);
    cudaFree(d_y);
}

int main(int argc, char** argv) {
    std::cout << "=== Advanced Optimization Test ===\n";
    std::cout << "WARP_SIZE = " << WARP_SIZE << "\n";

    // Test with different matrix sizes
    int iterations = 10;

    // Small matrix (100K rows) - L2 cache can cover most
    runTest(100000, 1000, 4, iterations);

    // Medium matrix (500K rows)
    runTest(500000, 1000, 4, iterations);

    // Larger matrix (1M rows) - exceeds L2 cache
    runTest(1000000, 1000, 4, iterations);

    return 0;
}