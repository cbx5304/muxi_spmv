/**
 * @file test_partition_params.cu
 * @brief Test different elementsPerPartition values for merge-based kernel
 */

#include <iostream>
#include <vector>
#include <cmath>

#include "formats/sparse_formats.h"
#include "spmv/csr5/spmv_csr5.cuh"
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

// Forward declaration of internal function
template<typename FloatType, int BlockSize, int WarpSize, int ElementsPerPartition>
__global__ void spmv_merge_based_kernel_tunable(
    int numRows, int numCols, int nnz,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* y,
    const int* __restrict__ mergePathPos,
    int numPartitions);

void runTest(int rows, int cols, int avgNnz, int iterations) {
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

    // Test baseline (current implementation)
    std::cout << "Baseline merge-based kernel:\n";
    float totalTime = 0;
    for (int i = 0; i < iterations; i++) {
        cudaMemset(d_y, 0, rows * sizeof(float));
        cudaDeviceSynchronize();
        timer.start();
        spmv_merge_based<float>(matrix, d_x, d_y, 0);
        timer.stop();
        totalTime += timer.elapsed_ms();
    }
    float avgTime = totalTime / iterations;
    float bw = (dataBytes / (avgTime * 1e-3)) / (1024 * 1024 * 1024);
    float baselineUtil = bw / peakBW * 100;
    std::cout << "  Time: " << avgTime << " ms, BW: " << bw << " GB/s, Util: " << baselineUtil << "%\n";

    // Cleanup
    delete[] h_x;
    cudaFree(d_x);
    cudaFree(d_y);
}

int main(int argc, char** argv) {
    std::cout << "=== Partition Parameters Test ===\n";
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