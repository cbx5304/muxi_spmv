/**
 * @file test_lane_allocation.cu
 * @brief Test different lane allocation strategies for Mars X201 (warp=64)
 *
 * For avgNnz<=4 extremely sparse matrices, explore:
 * - 4 threads per row (16 rows per warp)
 * - 8 threads per row (8 rows per warp)
 * - 16 threads per row (4 rows per warp)
 * - Hybrid strategies
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

// Lane-optimized kernel with configurable threads per row
template<typename FloatType, int BLOCK_SIZE, int THREADS_PER_ROW>
__global__ void spmv_lane_kernel(
    int numRows,
    int numCols,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y)
{
    const int ROWS_PER_WARP = WARP_SIZE / THREADS_PER_ROW;

    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    int rowInWarp = lane / THREADS_PER_ROW;  // Which row within this warp
    int threadInRow = lane % THREADS_PER_ROW; // Which thread for this row

    int row = warpId * ROWS_PER_WARP + rowInWarp;
    if (row >= numRows) return;

    FloatType sum = static_cast<FloatType>(0);
    int rowStart = rowPtr[row];
    int rowEnd = rowPtr[row + 1];

    // Each thread handles elements at offsets threadInRow, threadInRow+THREADS_PER_ROW, etc.
    for (int idx = rowStart + threadInRow; idx < rowEnd; idx += THREADS_PER_ROW) {
        sum += values[idx] * __ldg(&x[colIdx[idx]]);
    }

    // Reduce within the threads handling this row
    #pragma unroll
    for (int offset = THREADS_PER_ROW / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset, THREADS_PER_ROW);
    }

    // Thread 0 of each row group writes result
    if (threadInRow == 0) {
        y[row] = sum;
    }
}

// Hybrid kernel: dynamically choose threads per row based on row length
template<typename FloatType, int BLOCK_SIZE>
__global__ void spmv_hybrid_kernel(
    int numRows,
    int numCols,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y,
    int avgNnzThreshold)  // Threshold for switching strategies
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= numRows) return;

    int rowLen = rowPtr[row + 1] - rowPtr[row];

    FloatType sum = static_cast<FloatType>(0);
    int rowStart = rowPtr[row];

    // Use different strategies based on row length
    if (rowLen <= avgNnzThreshold) {
        // Short row: simple scalar approach
        for (int idx = rowStart; idx < rowStart + rowLen; idx++) {
            sum += values[idx] * __ldg(&x[colIdx[idx]]);
        }
        y[row] = sum;
    } else {
        // Longer row: process with loop unrolling
        int idx = rowStart;
        FloatType sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
        for (; idx + 3 < rowStart + rowLen; idx += 4) {
            sum0 += values[idx] * __ldg(&x[colIdx[idx]]);
            sum1 += values[idx+1] * __ldg(&x[colIdx[idx+1]]);
            sum2 += values[idx+2] * __ldg(&x[colIdx[idx+2]]);
            sum3 += values[idx+3] * __ldg(&x[colIdx[idx+3]]);
        }
        sum = sum0 + sum1 + sum2 + sum3;
        for (; idx < rowStart + rowLen; idx++) {
            sum += values[idx] * __ldg(&x[colIdx[idx]]);
        }
        y[row] = sum;
    }
}

// Adaptive warp scheduling kernel
// Each warp processes multiple rows adaptively
template<typename FloatType, int BLOCK_SIZE>
__global__ void spmv_adaptive_warp_kernel(
    int numRows,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y)
{
    __shared__ int sharedRowPtr[BLOCK_SIZE + 1];  // Cache row pointers

    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int globalWarpId = blockIdx.x * (BLOCK_SIZE / WARP_SIZE) + warpId;

    // Base row for this warp
    int baseRow = globalWarpId * 16;  // Each warp handles up to 16 rows (4 threads each)

    // Load row pointers cooperatively
    if (lane < 17 && baseRow + lane <= numRows) {
        sharedRowPtr[threadIdx.x / WARP_SIZE * 17 + lane] = rowPtr[baseRow + lane];
    }
    __syncthreads();

    // Each group of 4 threads handles one row
    int rowIdx = lane / 4;  // Which row (0-15)
    int threadInRow = lane % 4;  // Thread position (0-3)
    int row = baseRow + rowIdx;

    if (row >= numRows) return;

    FloatType sum = 0;
    int rowStart = sharedRowPtr[threadIdx.x / WARP_SIZE * 17 + rowIdx];
    int rowEnd = sharedRowPtr[threadIdx.x / WARP_SIZE * 17 + rowIdx + 1];

    // Process with stride of 4
    for (int idx = rowStart + threadInRow; idx < rowEnd; idx += 4) {
        sum += values[idx] * __ldg(&x[colIdx[idx]]);
    }

    // Reduce within the 4 threads
    sum += __shfl_down_sync(0xffffffff, sum, 2, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 1, 4);

    if (threadInRow == 0) {
        y[row] = sum;
    }
}

template<int THREADS_PER_ROW>
void runLaneTest(int rows, int cols, int avgNnz, int iterations,
                 const CSRMatrix<float>& matrix,
                 float* d_x, float* d_y, float peakBW, size_t dataBytes) {
    GpuTimer timer;
    int blockSize = 256;
    const int ROWS_PER_WARP = WARP_SIZE / THREADS_PER_ROW;
    int rowsPerBlock = (blockSize / WARP_SIZE) * ROWS_PER_WARP;
    int gridSize = (rows + rowsPerBlock - 1) / rowsPerBlock;

    float totalTime = 0;
    for (int i = 0; i < iterations; i++) {
        cudaMemset(d_y, 0, rows * sizeof(float));
        cudaDeviceSynchronize();
        timer.start();
        spmv_lane_kernel<float, 256, THREADS_PER_ROW><<<gridSize, blockSize>>>(
            rows, cols, matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values, d_x, d_y);
        timer.stop();
        totalTime += timer.elapsed_ms();
    }
    float avgTime = totalTime / iterations;
    float bw = (dataBytes / (avgTime * 1e-3)) / (1024 * 1024 * 1024);
    float util = bw / peakBW * 100;
    std::cout << "   " << THREADS_PER_ROW << " threads/row (" << ROWS_PER_WARP << " rows/warp): "
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

    std::cout << "Lane allocation strategies:\n";
    runLaneTest<2>(rows, cols, avgNnz, iterations, matrix, d_x, d_y, peakBW, dataBytes);
    runLaneTest<4>(rows, cols, avgNnz, iterations, matrix, d_x, d_y, peakBW, dataBytes);
    runLaneTest<8>(rows, cols, avgNnz, iterations, matrix, d_x, d_y, peakBW, dataBytes);
    runLaneTest<16>(rows, cols, avgNnz, iterations, matrix, d_x, d_y, peakBW, dataBytes);
    runLaneTest<32>(rows, cols, avgNnz, iterations, matrix, d_x, d_y, peakBW, dataBytes);

    // Adaptive warp test
    std::cout << "Adaptive Warp kernel: ";
    GpuTimer timer;
    int blockSize = 256;
    int gridSize = (rows + 16 * (blockSize / WARP_SIZE) - 1) / (16 * (blockSize / WARP_SIZE));
    float totalTime = 0;
    for (int i = 0; i < iterations; i++) {
        cudaMemset(d_y, 0, rows * sizeof(float));
        cudaDeviceSynchronize();
        timer.start();
        spmv_adaptive_warp_kernel<float, 256><<<gridSize, blockSize>>>(
            rows, matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values, d_x, d_y);
        timer.stop();
        totalTime += timer.elapsed_ms();
    }
    float avgTime = totalTime / iterations;
    float bw = (dataBytes / (avgTime * 1e-3)) / (1024 * 1024 * 1024);
    float util = bw / peakBW * 100;
    std::cout << avgTime << " ms, " << bw << " GB/s, " << util << "%\n";

    delete[] h_x;
    cudaFree(d_x);
    cudaFree(d_y);
}

int main(int argc, char** argv) {
    std::cout << "=== Lane Allocation Test for Mars X201 ===\n";
    std::cout << "WARP_SIZE = " << WARP_SIZE << "\n";

    int iterations = 10;

    // Test different matrix sizes
    runTest(100000, 1000, 4, iterations);
    runTest(500000, 1000, 4, iterations);
    runTest(1000000, 1000, 4, iterations);

    // Test different avgNnz values
    runTest(500000, 1000, 6, iterations);
    runTest(500000, 1000, 8, iterations);

    return 0;
}