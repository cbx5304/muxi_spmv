/**
 * @file test_shared_memory_impact.cu
 * @brief Test different shared memory sizes and patterns for Mars X201
 *
 * Key finding: Larger shared memory allocation improves performance!
 * Need to understand why and optimize further.
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

// Kernel with configurable shared memory allocation
template<typename FloatType, int BLOCK_SIZE, int SMEM_SIZE>
__global__ void spmv_adaptive_smem_kernel(
    int numRows,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y)
{
    __shared__ int sharedRowPtr[SMEM_SIZE];

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

// Kernel with padding to avoid bank conflicts
template<typename FloatType, int BLOCK_SIZE>
__global__ void spmv_adaptive_padded_kernel(
    int numRows,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y)
{
    // Pad to avoid bank conflicts (assuming 32 banks)
    // Each warp gets 17 padded entries (17 = 16 rows + 1 for end pointer)
    __shared__ int sharedRowPtr[BLOCK_SIZE / WARP_SIZE * 18];  // 18 instead of 17 for padding

    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int globalWarpId = blockIdx.x * (BLOCK_SIZE / WARP_SIZE) + warpId;

    int baseRow = globalWarpId * 16;

    // Load with padding offset
    int warpOffset = warpId * 18;
    if (lane < 17 && baseRow + lane <= numRows) {
        sharedRowPtr[warpOffset + lane] = rowPtr[baseRow + lane];
    }
    __syncthreads();

    int rowIdx = lane / 4;
    int threadInRow = lane % 4;
    int row = baseRow + rowIdx;

    if (row >= numRows) return;

    FloatType sum = 0;
    int rowStart = sharedRowPtr[warpOffset + rowIdx];
    int rowEnd = sharedRowPtr[warpOffset + rowIdx + 1];

    for (int idx = rowStart + threadInRow; idx < rowEnd; idx += 4) {
        sum += values[idx] * __ldg(&x[colIdx[idx]]);
    }

    sum += __shfl_down_sync(0xffffffff, sum, 2, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 1, 4);

    if (threadInRow == 0) {
        y[row] = sum;
    }
}

// Kernel without shared memory (direct global memory access)
template<typename FloatType, int BLOCK_SIZE>
__global__ void spmv_no_smem_kernel(
    int numRows,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y)
{
    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int globalWarpId = blockIdx.x * (BLOCK_SIZE / WARP_SIZE) + warpId;

    int baseRow = globalWarpId * 16;

    int rowIdx = lane / 4;
    int threadInRow = lane % 4;
    int row = baseRow + rowIdx;

    if (row >= numRows) return;

    FloatType sum = 0;
    int rowStart = rowPtr[row];
    int rowEnd = rowPtr[row + 1];

    for (int idx = rowStart + threadInRow; idx < rowEnd; idx += 4) {
        sum += values[idx] * __ldg(&x[colIdx[idx]]);
    }

    sum += __shfl_down_sync(0xffffffff, sum, 2, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 1, 4);

    if (threadInRow == 0) {
        y[row] = sum;
    }
}

template<int SMEM_SIZE>
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
        spmv_adaptive_smem_kernel<float, 256, SMEM_SIZE><<<gridSize, blockSize>>>(
            rows, matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values, d_x, d_y);
        timer.stop();
        totalTime += timer.elapsed_ms();
    }
    float avgTime = totalTime / iterations;
    float bw = (dataBytes / (avgTime * 1e-3)) / (1024 * 1024 * 1024);
    float util = bw / peakBW * 100;
    std::cout << "   SMEM " << SMEM_SIZE << " (" << SMEM_SIZE * sizeof(int) << " bytes): "
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

    GpuTimer timer;
    int blockSize = 256;
    int rowsPerBlock = (blockSize / WARP_SIZE) * 16;
    int gridSize = (rows + rowsPerBlock - 1) / rowsPerBlock;

    // Test different shared memory sizes
    std::cout << "Shared memory size comparison:\n";
    runSmemTest<68>(rows, cols, avgNnz, iterations, matrix, d_x, d_y, peakBW, dataBytes, gridSize, blockSize);
    runSmemTest<72>(rows, cols, avgNnz, iterations, matrix, d_x, d_y, peakBW, dataBytes, gridSize, blockSize);
    runSmemTest<128>(rows, cols, avgNnz, iterations, matrix, d_x, d_y, peakBW, dataBytes, gridSize, blockSize);
    runSmemTest<256>(rows, cols, avgNnz, iterations, matrix, d_x, d_y, peakBW, dataBytes, gridSize, blockSize);
    runSmemTest<257>(rows, cols, avgNnz, iterations, matrix, d_x, d_y, peakBW, dataBytes, gridSize, blockSize);
    runSmemTest<512>(rows, cols, avgNnz, iterations, matrix, d_x, d_y, peakBW, dataBytes, gridSize, blockSize);
    runSmemTest<1024>(rows, cols, avgNnz, iterations, matrix, d_x, d_y, peakBW, dataBytes, gridSize, blockSize);
    runSmemTest<1028>(rows, cols, avgNnz, iterations, matrix, d_x, d_y, peakBW, dataBytes, gridSize, blockSize);

    // Padded version
    std::cout << "Padded (72 elements): ";
    float totalTime = 0;
    for (int i = 0; i < iterations; i++) {
        cudaMemset(d_y, 0, rows * sizeof(float));
        cudaDeviceSynchronize();
        timer.start();
        spmv_adaptive_padded_kernel<float, 256><<<gridSize, blockSize>>>(
            rows, matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values, d_x, d_y);
        timer.stop();
        totalTime += timer.elapsed_ms();
    }
    float avgTime = totalTime / iterations;
    float bw = (dataBytes / (avgTime * 1e-3)) / (1024 * 1024 * 1024);
    float util = bw / peakBW * 100;
    std::cout << avgTime << " ms, " << bw << " GB/s, " << util << "%\n";

    // No shared memory version
    std::cout << "No shared memory: ";
    totalTime = 0;
    for (int i = 0; i < iterations; i++) {
        cudaMemset(d_y, 0, rows * sizeof(float));
        cudaDeviceSynchronize();
        timer.start();
        spmv_no_smem_kernel<float, 256><<<gridSize, blockSize>>>(
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
    std::cout << "=== Shared Memory Impact Test ===\n";
    std::cout << "WARP_SIZE = " << WARP_SIZE << "\n";

    int iterations = 10;

    runTest(100000, 1000, 4, iterations);
    runTest(500000, 1000, 4, iterations);
    runTest(1000000, 1000, 4, iterations);

    return 0;
}