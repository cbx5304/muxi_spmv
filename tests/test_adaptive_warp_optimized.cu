/**
 * @file test_adaptive_warp_optimized.cu
 * @brief Optimized Adaptive Warp kernel for Mars X201
 *
 * Key insight: Shared memory caching of row pointers + efficient lane allocation
 * achieves 25%+ utilization for avgNnz=4 matrices
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

// Optimized Adaptive Warp kernel - Variant 1: 16 rows per warp (4 threads each)
template<typename FloatType, int BLOCK_SIZE>
__global__ void spmv_adaptive_v1_kernel(
    int numRows,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y)
{
    __shared__ int sharedRowPtr[BLOCK_SIZE / WARP_SIZE * 17];  // Cache row pointers for each warp

    int warpIdInBlock = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int globalWarpId = blockIdx.x * (BLOCK_SIZE / WARP_SIZE) + warpIdInBlock;

    int baseRow = globalWarpId * 16;  // Each warp handles 16 rows

    // Load row pointers cooperatively within warp
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

    // Reduce within 4 threads
    sum += __shfl_down_sync(0xffffffff, sum, 2, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 1, 4);

    if (threadInRow == 0) {
        y[row] = sum;
    }
}

// Optimized Adaptive Warp kernel - Variant 2: 8 rows per warp (8 threads each)
template<typename FloatType, int BLOCK_SIZE>
__global__ void spmv_adaptive_v2_kernel(
    int numRows,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y)
{
    __shared__ int sharedRowPtr[BLOCK_SIZE / WARP_SIZE * 9];

    int warpIdInBlock = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int globalWarpId = blockIdx.x * (BLOCK_SIZE / WARP_SIZE) + warpIdInBlock;

    int baseRow = globalWarpId * 8;  // Each warp handles 8 rows

    if (lane < 9 && baseRow + lane <= numRows) {
        sharedRowPtr[warpIdInBlock * 9 + lane] = rowPtr[baseRow + lane];
    }
    __syncthreads();

    int rowIdx = lane / 8;
    int threadInRow = lane % 8;
    int row = baseRow + rowIdx;

    if (row >= numRows) return;

    FloatType sum = 0;
    int rowStart = sharedRowPtr[warpIdInBlock * 9 + rowIdx];
    int rowEnd = sharedRowPtr[warpIdInBlock * 9 + rowIdx + 1];

    for (int idx = rowStart + threadInRow; idx < rowEnd; idx += 8) {
        sum += values[idx] * __ldg(&x[colIdx[idx]]);
    }

    #pragma unroll
    for (int offset = 4; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset, 8);
    }

    if (threadInRow == 0) {
        y[row] = sum;
    }
}

// Optimized Adaptive Warp kernel - Variant 3: With ILP
template<typename FloatType, int BLOCK_SIZE>
__global__ void spmv_adaptive_ilp_kernel(
    int numRows,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y)
{
    __shared__ int sharedRowPtr[BLOCK_SIZE / WARP_SIZE * 17];

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

    FloatType sum0 = 0, sum1 = 0;
    int rowStart = sharedRowPtr[warpIdInBlock * 17 + rowIdx];
    int rowEnd = sharedRowPtr[warpIdInBlock * 17 + rowIdx + 1];

    // ILP: process 2 elements per iteration
    int idx = rowStart + threadInRow;
    for (; idx + 4 < rowEnd; idx += 8) {
        sum0 += values[idx] * __ldg(&x[colIdx[idx]]);
        int idx2 = idx + 4;
        if (idx2 < rowEnd) {
            sum1 += values[idx2] * __ldg(&x[colIdx[idx2]]);
        }
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

// Adaptive Warp kernel - Variant 4: Precompute row lengths
template<typename FloatType, int BLOCK_SIZE>
__global__ void spmv_adaptive_precompute_kernel(
    int numRows,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y)
{
    __shared__ int sharedRowPtr[BLOCK_SIZE / WARP_SIZE * 17];
    __shared__ int sharedRowLen[BLOCK_SIZE / WARP_SIZE * 16];  // Row lengths

    int warpIdInBlock = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int globalWarpId = blockIdx.x * (BLOCK_SIZE / WARP_SIZE) + warpIdInBlock;

    int baseRow = globalWarpId * 16;

    // Load row pointers
    if (lane < 17 && baseRow + lane <= numRows) {
        sharedRowPtr[warpIdInBlock * 17 + lane] = rowPtr[baseRow + lane];
    }
    // Compute row lengths
    if (lane < 16 && baseRow + lane < numRows) {
        sharedRowLen[warpIdInBlock * 16 + lane] =
            sharedRowPtr[warpIdInBlock * 17 + lane + 1] - sharedRowPtr[warpIdInBlock * 17 + lane];
    }
    __syncthreads();

    int rowIdx = lane / 4;
    int threadInRow = lane % 4;
    int row = baseRow + rowIdx;

    if (row >= numRows) return;

    FloatType sum = 0;
    int rowStart = sharedRowPtr[warpIdInBlock * 17 + rowIdx];
    int rowLen = sharedRowLen[warpIdInBlock * 16 + rowIdx];

    // Process with known length
    for (int idx = rowStart + threadInRow; idx < rowStart + rowLen; idx += 4) {
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
    int warpsPerBlock = blockSize / WARP_SIZE;

    // V1: 16 rows/warp, 4 threads/row
    std::cout << "V1 (16 rows/warp, 4 t/row): ";
    int rowsPerBlockV1 = warpsPerBlock * 16;
    int gridSizeV1 = (rows + rowsPerBlockV1 - 1) / rowsPerBlockV1;
    float totalTime = 0;
    for (int i = 0; i < iterations; i++) {
        cudaMemset(d_y, 0, rows * sizeof(float));
        cudaDeviceSynchronize();
        timer.start();
        spmv_adaptive_v1_kernel<float, 256><<<gridSizeV1, blockSize>>>(
            rows, matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values, d_x, d_y);
        timer.stop();
        totalTime += timer.elapsed_ms();
    }
    float avgTime = totalTime / iterations;
    float bw = (dataBytes / (avgTime * 1e-3)) / (1024 * 1024 * 1024);
    float util = bw / peakBW * 100;
    std::cout << avgTime << " ms, " << bw << " GB/s, " << util << "%\n";

    // V2: 8 rows/warp, 8 threads/row
    std::cout << "V2 (8 rows/warp, 8 t/row): ";
    int rowsPerBlockV2 = warpsPerBlock * 8;
    int gridSizeV2 = (rows + rowsPerBlockV2 - 1) / rowsPerBlockV2;
    totalTime = 0;
    for (int i = 0; i < iterations; i++) {
        cudaMemset(d_y, 0, rows * sizeof(float));
        cudaDeviceSynchronize();
        timer.start();
        spmv_adaptive_v2_kernel<float, 256><<<gridSizeV2, blockSize>>>(
            rows, matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values, d_x, d_y);
        timer.stop();
        totalTime += timer.elapsed_ms();
    }
    avgTime = totalTime / iterations;
    bw = (dataBytes / (avgTime * 1e-3)) / (1024 * 1024 * 1024);
    util = bw / peakBW * 100;
    std::cout << avgTime << " ms, " << bw << " GB/s, " << util << "%\n";

    // V3: ILP variant
    std::cout << "V3 (ILP): ";
    totalTime = 0;
    for (int i = 0; i < iterations; i++) {
        cudaMemset(d_y, 0, rows * sizeof(float));
        cudaDeviceSynchronize();
        timer.start();
        spmv_adaptive_ilp_kernel<float, 256><<<gridSizeV1, blockSize>>>(
            rows, matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values, d_x, d_y);
        timer.stop();
        totalTime += timer.elapsed_ms();
    }
    avgTime = totalTime / iterations;
    bw = (dataBytes / (avgTime * 1e-3)) / (1024 * 1024 * 1024);
    util = bw / peakBW * 100;
    std::cout << avgTime << " ms, " << bw << " GB/s, " << util << "%\n";

    // V4: Precompute row lengths
    std::cout << "V4 (Precompute): ";
    totalTime = 0;
    for (int i = 0; i < iterations; i++) {
        cudaMemset(d_y, 0, rows * sizeof(float));
        cudaDeviceSynchronize();
        timer.start();
        spmv_adaptive_precompute_kernel<float, 256><<<gridSizeV1, blockSize>>>(
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
    std::cout << "=== Optimized Adaptive Warp Kernel Test ===\n";
    std::cout << "WARP_SIZE = " << WARP_SIZE << "\n";

    int iterations = 10;

    // Test different matrix sizes
    runTest(100000, 1000, 4, iterations);
    runTest(500000, 1000, 4, iterations);
    runTest(1000000, 1000, 4, iterations);

    // Test different avgNnz
    runTest(500000, 1000, 6, iterations);
    runTest(500000, 1000, 8, iterations);
    runTest(500000, 1000, 10, iterations);

    return 0;
}