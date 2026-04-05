/**
 * @file test_kernel_comparison.cu
 * @brief Compare merge-based kernel with column sorting for very sparse matrices
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

#include "formats/sparse_formats.h"
#include "spmv/csr5/spmv_csr5.cuh"
#include "generators/matrix_generator.h"
#include "preprocessing/column_reorder.cuh"

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

void runTest(int rows, int cols, int avgNnz, int iterations) {
    std::cout << "\n=== Test: rows=" << rows << ", cols=" << cols
              << ", avgNnz=" << avgNnz << " ===\n";

    // Generate matrix (creates host data)
    CSRMatrix<float> matrix;
    int nnz = rows * avgNnz;
    generateRandomMatrix<float>(rows, cols, nnz, matrix);

    // Allocate device memory and copy initial data
    matrix.allocateDevice();
    matrix.copyToDevice();
    cudaDeviceSynchronize();  // Ensure copy completes

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

    // Test 1: Merge-based kernel (original)
    std::cout << "\n1. Merge-based (original):\n";
    totalTime = 0;
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
    float origUtil = bw / peakBW * 100;
    std::cout << "   Time: " << avgTime << " ms, BW: " << bw << " GB/s, Util: " << origUtil << "%\n";

    // Test 2: Merge-based with column sorting
    std::cout << "\n2. Merge-based + column sorting:\n";

    // Create a copy of the matrix for sorting
    CSRMatrix<float> sortedMatrix;
    sortedMatrix.numRows = rows;
    sortedMatrix.numCols = cols;
    sortedMatrix.nnz = nnz;
    sortedMatrix.allocateHost(rows, cols, nnz);

    // Copy from original host data
    memcpy(sortedMatrix.rowPtr, matrix.rowPtr, (rows + 1) * sizeof(int));
    memcpy(sortedMatrix.colIdx, matrix.colIdx, nnz * sizeof(int));
    memcpy(sortedMatrix.values, matrix.values, nnz * sizeof(float));

    // Sort columns within rows
    sortColumnsWithinRows(sortedMatrix);

    // Allocate and copy to device
    sortedMatrix.allocateDevice();
    sortedMatrix.copyToDevice();
    cudaDeviceSynchronize();  // CRITICAL: Ensure copy completes before kernel

    totalTime = 0;
    for (int i = 0; i < iterations; i++) {
        cudaMemset(d_y, 0, rows * sizeof(float));
        cudaDeviceSynchronize();
        timer.start();
        spmv_merge_based<float>(sortedMatrix, d_x, d_y, 0);
        timer.stop();
        totalTime += timer.elapsed_ms();
    }
    avgTime = totalTime / iterations;
    bw = (dataBytes / (avgTime * 1e-3)) / (1024 * 1024 * 1024);
    float sortedUtil = bw / peakBW * 100;
    std::cout << "   Time: " << avgTime << " ms, BW: " << bw << " GB/s, Util: " << sortedUtil << "%\n";

    // Summary
    float improvement = (sortedUtil - origUtil) / origUtil * 100;
    std::cout << "\n   Improvement: " << improvement << "%\n";

    // Cleanup
    delete[] h_x;
    cudaFree(d_x);
    cudaFree(d_y);
}

int main(int argc, char** argv) {
    std::cout << "=== Kernel Comparison Test ===\n";
    std::cout << "WARP_SIZE = " << WARP_SIZE << "\n";

    int rows = 1000000;
    int cols = 1000;
    int iterations = 20;

    // Test different sparsity levels
    runTest(rows, cols, 4, iterations);
    runTest(rows, cols, 6, iterations);
    runTest(rows, cols, 8, iterations);
    runTest(rows, cols, 10, iterations);

    return 0;
}