/**
 * @file test_warp_optimization.cu
 * @brief Test different warp utilization strategies for Mars X201
 *
 * Key insight: Mars X201 has warp size=64, which causes low utilization
 * for very sparse matrices (avgNnz=4 means only 4/64 threads working).
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

// ==================== Kernel Variants ====================

// Kernel 1: Each thread processes multiple rows (light kernel)
template<typename FloatType, int BLOCK_SIZE>
__global__ void spmv_light_kernel(
    int numRows,
    int numCols,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y,
    int rowsPerThread)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int myRowStart = tid * rowsPerThread;
    int myRowEnd = min(myRowStart + rowsPerThread, numRows);

    for (int row = myRowStart; row < myRowEnd; row++) {
        FloatType sum = static_cast<FloatType>(0);
        for (int idx = rowPtr[row]; idx < rowPtr[row + 1]; idx++) {
            sum += values[idx] * x[colIdx[idx]];
        }
        y[row] = sum;
    }
}

// Kernel 2: Warp-cooperative with stride (virtual warp)
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
    // Calculate virtual warp ID and lane
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int virtualWarpId = tid / VIRTUAL_WARP;
    int laneInVirtual = tid % VIRTUAL_WARP;

    int myRow = virtualWarpId;
    if (myRow >= numRows) return;

    FloatType sum = static_cast<FloatType>(0);
    int rowStart = rowPtr[myRow];
    int rowEnd = rowPtr[myRow + 1];

    // Each thread in virtual warp processes part of the row
    for (int idx = rowStart + laneInVirtual; idx < rowEnd; idx += VIRTUAL_WARP) {
        sum += values[idx] * x[colIdx[idx]];
    }

    // Warp reduce within virtual warp - use mask to limit to virtual warp
    unsigned int mask = (1u << VIRTUAL_WARP) - 1u;
    int laneBase = (tid / VIRTUAL_WARP) * VIRTUAL_WARP;
    mask = mask << (tid - laneBase);
    // Actually, for simplicity, use a different approach:
    // Manual reduction with proper masking
    #pragma unroll
    for (int offset = VIRTUAL_WARP / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset, VIRTUAL_WARP);
    }

    if (laneInVirtual == 0) {
        y[myRow] = sum;
    }
}

// Kernel 3: Scalar kernel (one thread per row)
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

// ==================== Test Functions ====================

void runComparison(int rows, int cols, int avgNnz, int iterations) {
    std::cout << "\n=== Test: rows=" << rows << ", cols=" << cols
              << ", avgNnz=" << avgNnz << " ===\n";

    // Generate matrix
    CSRMatrix<float> matrix;
    int nnz = rows * avgNnz;
    generateRandomMatrix<float>(rows, cols, nnz, matrix);
    matrix.allocateDevice();
    matrix.copyToDevice();
    cudaDeviceSynchronize();

    // Generate x vector
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

    // Test 1: Scalar kernel (baseline)
    std::cout << "1. Scalar kernel (1 thread/row):\n";
    {
        int blockSize = 256;
        int gridSize = (rows + blockSize - 1) / blockSize;
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
        std::cout << "   Time: " << avgTime << " ms, BW: " << bw << " GB/s, Util: " << util << "%\n";
    }

    // Test 2: Light kernel (each thread processes multiple rows)
    std::cout << "2. Light kernel (multiple rows/thread):\n";
    {
        int blockSize = 256;
        int numSMs = (WARP_SIZE == 64) ? 104 : 128;
        int targetThreads = numSMs * blockSize * 4;
        int rowsPerThread = max(1, rows / targetThreads);
        int gridSize = (rows + rowsPerThread * blockSize - 1) / (rowsPerThread * blockSize);

        totalTime = 0;
        for (int i = 0; i < iterations; i++) {
            cudaMemset(d_y, 0, rows * sizeof(float));
            cudaDeviceSynchronize();
            timer.start();
            spmv_light_kernel<float, 256><<<gridSize, blockSize>>>(
                rows, cols, matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values, d_x, d_y, rowsPerThread);
            timer.stop();
            totalTime += timer.elapsed_ms();
        }
        float avgTime = totalTime / iterations;
        float bw = (dataBytes / (avgTime * 1e-3)) / (1024 * 1024 * 1024);
        float util = bw / peakBW * 100;
        std::cout << "   Time: " << avgTime << " ms, BW: " << bw << " GB/s, Util: " << util << "%\n";
        std::cout << "   Config: " << gridSize << " blocks, " << rowsPerThread << " rows/thread\n";
    }

    // Test 3: Virtual warp kernel (warp=16 for avgNnz=4)
    if (WARP_SIZE == 64 && avgNnz <= 8) {
        std::cout << "3. Virtual warp kernel (warp=16):\n";
        int virtualWarp = 16;
        int blockSize = 256;
        // Each virtual warp handles 1 row
        int virtualWarpsPerBlock = blockSize / virtualWarp;
        int gridSize = (rows + virtualWarpsPerBlock - 1) / virtualWarpsPerBlock;

        totalTime = 0;
        for (int i = 0; i < iterations; i++) {
            cudaMemset(d_y, 0, rows * sizeof(float));
            cudaDeviceSynchronize();
            timer.start();
            spmv_virtual_warp_kernel<float, 256, 16><<<gridSize, blockSize>>>(
                rows, cols, matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values, d_x, d_y);
            timer.stop();
            totalTime += timer.elapsed_ms();
        }
        float avgTime = totalTime / iterations;
        float bw = (dataBytes / (avgTime * 1e-3)) / (1024 * 1024 * 1024);
        float util = bw / peakBW * 100;
        std::cout << "   Time: " << avgTime << " ms, BW: " << bw << " GB/s, Util: " << util << "%\n";
    }

    // Cleanup
    delete[] h_x;
    cudaFree(d_x);
    cudaFree(d_y);
}

int main(int argc, char** argv) {
    std::cout << "=== Warp Optimization Test ===\n";
    std::cout << "WARP_SIZE = " << WARP_SIZE << "\n";

    int rows = 1000000;
    int cols = 1000;
    int iterations = 20;

    runComparison(rows, cols, 4, iterations);
    runComparison(rows, cols, 6, iterations);
    runComparison(rows, cols, 8, iterations);
    runComparison(rows, cols, 10, iterations);

    return 0;
}