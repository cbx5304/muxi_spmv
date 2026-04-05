/**
 * @file test_combined_optimization.cu
 * @brief Combined optimization test for Mars X201
 *
 * Optimizations found:
 * 1. Block size: 512 threads
 * 2. Shared memory: Large allocation
 * 3. __ldg for x-vector access
 * 4. Adaptive Warp: 16 rows per warp, 4 threads per row
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

// Optimized kernel combining all findings
template<typename FloatType, int BLOCK_SIZE, int SMEM_INTS>
__global__ void spmv_optimized_kernel(
    int numRows,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y)
{
    __shared__ int sharedRowPtr[SMEM_INTS];

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

// Prefetch variant
template<typename FloatType, int BLOCK_SIZE, int SMEM_INTS>
__global__ void spmv_optimized_prefetch_kernel(
    int numRows,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y)
{
    __shared__ int sharedRowPtr[SMEM_INTS];

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

    int idx = rowStart + threadInRow;
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

template<int BLOCK_SIZE, int SMEM_INTS>
void runOptimizedTest(int rows, int cols, int avgNnz, int iterations,
                      const CSRMatrix<float>& matrix,
                      float* d_x, float* d_y, float peakBW, size_t dataBytes,
                      bool usePrefetch) {
    GpuTimer timer;
    int rowsPerBlock = (BLOCK_SIZE / WARP_SIZE) * 16;
    int gridSize = (rows + rowsPerBlock - 1) / rowsPerBlock;

    float totalTime = 0;
    for (int i = 0; i < iterations; i++) {
        cudaMemset(d_y, 0, rows * sizeof(float));
        cudaDeviceSynchronize();
        timer.start();
        if (usePrefetch) {
            spmv_optimized_prefetch_kernel<float, BLOCK_SIZE, SMEM_INTS><<<gridSize, BLOCK_SIZE>>>(
                rows, matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values, d_x, d_y);
        } else {
            spmv_optimized_kernel<float, BLOCK_SIZE, SMEM_INTS><<<gridSize, BLOCK_SIZE>>>(
                rows, matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values, d_x, d_y);
        }
        timer.stop();
        totalTime += timer.elapsed_ms();
    }
    float avgTime = totalTime / iterations;
    float bw = (dataBytes / (avgTime * 1e-3)) / (1024 * 1024 * 1024);
    float util = bw / peakBW * 100;

    std::string config = "B" + std::to_string(BLOCK_SIZE) + "_S" + std::to_string(SMEM_INTS);
    if (usePrefetch) config += "_PF";

    std::cout << "   " << config << ": " << avgTime << " ms, " << bw << " GB/s, " << util << "%\n";
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

    std::cout << "Configuration comparison:\n";

    // Test various combinations
    runOptimizedTest<256, 256>(rows, cols, avgNnz, iterations, matrix, d_x, d_y, peakBW, dataBytes, false);
    runOptimizedTest<256, 512>(rows, cols, avgNnz, iterations, matrix, d_x, d_y, peakBW, dataBytes, false);
    runOptimizedTest<512, 512>(rows, cols, avgNnz, iterations, matrix, d_x, d_y, peakBW, dataBytes, false);
    runOptimizedTest<512, 1024>(rows, cols, avgNnz, iterations, matrix, d_x, d_y, peakBW, dataBytes, false);
    runOptimizedTest<512, 1024>(rows, cols, avgNnz, iterations, matrix, d_x, d_y, peakBW, dataBytes, true);

    delete[] h_x;
    cudaFree(d_x);
    cudaFree(d_y);
}

int main(int argc, char** argv) {
    std::cout << "=== Combined Optimization Test ===\n";
    std::cout << "WARP_SIZE = " << WARP_SIZE << "\n";

    int iterations = 10;

    runTest(100000, 1000, 4, iterations);
    runTest(500000, 1000, 4, iterations);
    runTest(1000000, 1000, 4, iterations);

    // Test other avgNnz values
    runTest(500000, 1000, 6, iterations);
    runTest(500000, 1000, 8, iterations);

    return 0;
}