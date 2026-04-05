/**
 * @file test_kernel_comparison_fair.cu
 * @brief Fair comparison of Adaptive Warp kernels
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

// Original successful kernel from test_lane_allocation.cu
template<typename FloatType, int BLOCK_SIZE>
__global__ void spmv_adaptive_original_kernel(
    int numRows,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y)
{
    __shared__ int sharedRowPtr[BLOCK_SIZE + 1];  // Larger shared memory

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

// Optimized version from test_adaptive_warp_optimized.cu
template<typename FloatType, int BLOCK_SIZE>
__global__ void spmv_adaptive_optimized_kernel(
    int numRows,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y)
{
    __shared__ int sharedRowPtr[BLOCK_SIZE / WARP_SIZE * 17];  // Smaller shared memory

    int warpIdInBlock = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int globalWarpId = blockIdx.x * (BLOCK_SIZE / WARP_SIZE) + warpIdInBlock;

    int baseRow = globalWarpId * 16;

    if (lane < 17 && baseRow + lane <= numRows) {
        sharedRowPtr[warpIdInBlock * 17 + lane] = rowPtr[baseRow + lane];
    }
    __syncthreads();

    int rowIdx = lane / 4;
    int threadInRow = lane % 4;
    int row = baseRow + rowIdx;

    if (row >= numRows) return;

    FloatType sum = 0;
    int rowStart = sharedRowPtr[warpIdInBlock * 17 + rowIdx];
    int rowEnd = sharedRowPtr[warpIdInBlock * 17 + rowIdx + 1];

    for (int idx = rowStart + threadInRow; idx < rowEnd; idx += 4) {
        sum += values[idx] * __ldg(&x[colIdx[idx]]);
    }

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

    // Original kernel (larger shared memory)
    std::cout << "Original (large SMEM): ";
    float totalTime = 0;
    for (int i = 0; i < iterations; i++) {
        cudaMemset(d_y, 0, rows * sizeof(float));
        cudaDeviceSynchronize();
        timer.start();
        spmv_adaptive_original_kernel<float, 256><<<gridSize, blockSize>>>(
            rows, matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values, d_x, d_y);
        timer.stop();
        totalTime += timer.elapsed_ms();
    }
    float avgTime = totalTime / iterations;
    float bw = (dataBytes / (avgTime * 1e-3)) / (1024 * 1024 * 1024);
    float util = bw / peakBW * 100;
    std::cout << avgTime << " ms, " << bw << " GB/s, " << util << "%\n";

    // Optimized kernel (smaller shared memory)
    std::cout << "Optimized (small SMEM): ";
    totalTime = 0;
    for (int i = 0; i < iterations; i++) {
        cudaMemset(d_y, 0, rows * sizeof(float));
        cudaDeviceSynchronize();
        timer.start();
        spmv_adaptive_optimized_kernel<float, 256><<<gridSize, blockSize>>>(
            rows, matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values, d_x, d_y);
        timer.stop();
        totalTime += timer.elapsed_ms();
    }
    avgTime = totalTime / iterations;
    bw = (dataBytes / (avgTime * 1e-3)) / (1024 * 1024 * 1024);
    util = bw / peakBW * 100;
    std::cout << avgTime << " ms, " << bw << " GB/s, " << util << "%\n";

    // Compare shared memory usage
    std::cout << "Shared memory sizes:\n";
    std::cout << "  Original: " << (256 + 1) * sizeof(int) << " bytes\n";
    std::cout << "  Optimized: " << (256 / WARP_SIZE * 17) * sizeof(int) << " bytes\n";

    delete[] h_x;
    cudaFree(d_x);
    cudaFree(d_y);
}

int main(int argc, char** argv) {
    std::cout << "=== Fair Kernel Comparison ===\n";
    std::cout << "WARP_SIZE = " << WARP_SIZE << "\n";

    int iterations = 10;

    runTest(100000, 1000, 4, iterations);
    runTest(500000, 1000, 4, iterations);
    runTest(1000000, 1000, 4, iterations);

    return 0;
}