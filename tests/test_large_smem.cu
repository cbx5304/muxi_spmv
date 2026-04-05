/**
 * @file test_large_smem.cu
 * @brief Test larger shared memory configurations for Mars X201
 *
 * Previous finding: 1028 bytes is better than 272 bytes
 * Now test: 4KB, 8KB, 16KB, 32KB
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

// Test kernel with configurable shared memory
template<typename FloatType, int BLOCK_SIZE, int SMEM_INTS>
__global__ void spmv_smem_kernel(
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

    // Only use 17 ints per warp for row pointers
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

// Kernel that uses extra shared memory for x-vector cache
template<typename FloatType, int BLOCK_SIZE, int ROWS_PER_WARP, int X_CACHE_SIZE>
__global__ void spmv_xcache_kernel(
    int numRows,
    int numCols,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y)
{
    // Shared memory layout:
    // - Row pointers: 4 warps * 17 ints = 68 ints
    // - X cache: X_CACHE_SIZE floats
    __shared__ int sharedRowPtr[68];
    __shared__ FloatType xCache[X_CACHE_SIZE];

    int warpId = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int globalWarpId = blockIdx.x * (BLOCK_SIZE / WARP_SIZE) + warpId;

    int baseRow = globalWarpId * ROWS_PER_WARP;

    // Load row pointers
    if (lane < ROWS_PER_WARP + 1 && baseRow + lane <= numRows) {
        sharedRowPtr[warpId * 17 + lane] = rowPtr[baseRow + lane];
    }
    __syncthreads();

    int rowIdx = lane / 4;
    int threadInRow = lane % 4;
    int row = baseRow + rowIdx;

    if (row >= numRows) return;

    FloatType sum = 0;
    int rowStart = sharedRowPtr[warpId * 17 + rowIdx];
    int rowEnd = sharedRowPtr[warpId * 17 + rowIdx + 1];

    for (int idx = rowStart + threadInRow; idx < rowEnd; idx += 4) {
        sum += values[idx] * __ldg(&x[colIdx[idx]]);
    }

    sum += __shfl_down_sync(0xffffffff, sum, 2, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 1, 4);

    if (threadInRow == 0) {
        y[row] = sum;
    }
}

template<int SMEM_INTS>
void runSmemTest(int rows, int cols, int avgNnz, int iterations,
                 const CSRMatrix<float>& matrix,
                 float* d_x, float* d_y, float peakBW, size_t dataBytes,
                 int gridSize, int blockSize) {
    GpuTimer timer;
    float totalTime = 0;
    for (int i = 0; i < iterations; i++) {
        cudaMemset(d_y, 0, rows * sizeof(float));
        cudaDeviceSynchronize();
        timer.start();
        spmv_smem_kernel<float, 256, SMEM_INTS><<<gridSize, blockSize>>>(
            rows, matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values, d_x, d_y);
        timer.stop();
        totalTime += timer.elapsed_ms();
    }
    float avgTime = totalTime / iterations;
    float bw = (dataBytes / (avgTime * 1e-3)) / (1024 * 1024 * 1024);
    float util = bw / peakBW * 100;
    std::cout << "   " << SMEM_INTS << " ints (" << SMEM_INTS * 4 << " bytes): "
              << avgTime << " ms, " << bw << " GB/s, " << util << "%\n";
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

    int blockSize = 256;
    int rowsPerBlock = (blockSize / WARP_SIZE) * 16;
    int gridSize = (rows + rowsPerBlock - 1) / rowsPerBlock;

    std::cout << "Shared memory sizes (ints):\n";

    // Test various sizes
    runSmemTest<68>(rows, cols, avgNnz, iterations, matrix, d_x, d_y, peakBW, dataBytes, gridSize, blockSize);
    runSmemTest<128>(rows, cols, avgNnz, iterations, matrix, d_x, d_y, peakBW, dataBytes, gridSize, blockSize);
    runSmemTest<256>(rows, cols, avgNnz, iterations, matrix, d_x, d_y, peakBW, dataBytes, gridSize, blockSize);
    runSmemTest<512>(rows, cols, avgNnz, iterations, matrix, d_x, d_y, peakBW, dataBytes, gridSize, blockSize);
    runSmemTest<1024>(rows, cols, avgNnz, iterations, matrix, d_x, d_y, peakBW, dataBytes, gridSize, blockSize);
    runSmemTest<2048>(rows, cols, avgNnz, iterations, matrix, d_x, d_y, peakBW, dataBytes, gridSize, blockSize);
    runSmemTest<4096>(rows, cols, avgNnz, iterations, matrix, d_x, d_y, peakBW, dataBytes, gridSize, blockSize);

    delete[] h_x;
    cudaFree(d_x);
    cudaFree(d_y);
}

int main(int argc, char** argv) {
    std::cout << "=== Large Shared Memory Test ===\n";
    std::cout << "WARP_SIZE = " << WARP_SIZE << "\n";

    int iterations = 10;

    runTest(100000, 1000, 4, iterations);
    runTest(500000, 1000, 4, iterations);
    runTest(1000000, 1000, 4, iterations);

    return 0;
}