/**
 * @file test_virtual_warp_size.cu
 * @brief Test different virtual warp sizes for Mars X201
 *
 * Fixed version - correct grid calculation and shuffle operations
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

// Virtual warp kernel - fixed version
// Each virtual warp handles 1 row
template<typename FloatType, int BLOCK_SIZE, int VIRTUAL_WARP>
__global__ void spmv_virtual_warp_kernel(
    int numRows,
    int numCols,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y)
{
    // Global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Virtual warp ID (each virtual warp handles 1 row)
    int virtualWarpId = tid / VIRTUAL_WARP;
    int laneInVirtual = tid % VIRTUAL_WARP;

    if (virtualWarpId >= numRows) return;

    FloatType sum = static_cast<FloatType>(0);
    int rowStart = rowPtr[virtualWarpId];
    int rowEnd = rowPtr[virtualWarpId + 1];

    // Each thread in virtual warp processes part of the row
    for (int idx = rowStart + laneInVirtual; idx < rowEnd; idx += VIRTUAL_WARP) {
        sum += values[idx] * x[colIdx[idx]];
    }

    // Warp reduce within virtual warp with limited width
    #pragma unroll
    for (int offset = VIRTUAL_WARP / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset, VIRTUAL_WARP);
    }

    if (laneInVirtual == 0) {
        y[virtualWarpId] = sum;
    }
}

template<int VIRTUAL_WARP>
void testVirtualWarp(int rows, int cols, int avgNnz, int iterations,
                      const CSRMatrix<float>& matrix,
                      float* d_x, float* d_y, float peakBW, size_t dataBytes) {
    GpuTimer timer;
    int blockSize = 256;

    // Correct grid calculation:
    // - Each virtual warp handles 1 row
    // - Need 'rows' virtual warps
    // - Each block has blockSize/VIRTUAL_WARP virtual warps
    int virtualWarpsPerBlock = blockSize / VIRTUAL_WARP;
    int gridSize = (rows + virtualWarpsPerBlock - 1) / virtualWarpsPerBlock;

    float totalTime = 0;
    for (int i = 0; i < iterations; i++) {
        cudaMemset(d_y, 0, rows * sizeof(float));
        cudaDeviceSynchronize();
        timer.start();
        spmv_virtual_warp_kernel<float, 256, VIRTUAL_WARP><<<gridSize, blockSize>>>(
            rows, cols, matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values, d_x, d_y);
        timer.stop();
        totalTime += timer.elapsed_ms();
    }
    float avgTime = totalTime / iterations;
    float bw = (dataBytes / (avgTime * 1e-3)) / (1024 * 1024 * 1024);
    float util = bw / peakBW * 100;
    std::cout << "   Warp=" << VIRTUAL_WARP << ": Time=" << avgTime << " ms, BW=" << bw
              << " GB/s, Util=" << util << "%\n";
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

    std::cout << "Testing different virtual warp sizes:\n";

    // Test different virtual warp sizes
    testVirtualWarp<4>(rows, cols, avgNnz, iterations, matrix, d_x, d_y, peakBW, dataBytes);
    testVirtualWarp<8>(rows, cols, avgNnz, iterations, matrix, d_x, d_y, peakBW, dataBytes);
    testVirtualWarp<16>(rows, cols, avgNnz, iterations, matrix, d_x, d_y, peakBW, dataBytes);
    testVirtualWarp<32>(rows, cols, avgNnz, iterations, matrix, d_x, d_y, peakBW, dataBytes);

    delete[] h_x;
    cudaFree(d_x);
    cudaFree(d_y);
}

int main(int argc, char** argv) {
    std::cout << "=== Virtual Warp Size Test ===\n";
    std::cout << "WARP_SIZE = " << WARP_SIZE << "\n";

    int rows = 1000000;
    int cols = 1000;
    int iterations = 20;

    runTest(rows, cols, 4, iterations);
    runTest(rows, cols, 6, iterations);
    runTest(rows, cols, 8, iterations);
    runTest(rows, cols, 10, iterations);

    return 0;
}