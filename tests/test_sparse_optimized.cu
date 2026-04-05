/**
 * @file test_sparse_optimized.cu
 * @brief Test sparse optimized kernel vs merge-based kernel
 */

#include <iostream>
#include <cmath>

#include "formats/sparse_formats.h"
#include "spmv/csr5/spmv_csr5.cuh"
#include "spmv/csr/spmv_sparse_optimized.cuh"
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

    float* d_x, *d_y1, *d_y2;
    cudaMalloc(&d_x, cols * sizeof(float));
    cudaMalloc(&d_y1, rows * sizeof(float));
    cudaMalloc(&d_y2, rows * sizeof(float));
    cudaMemcpy(d_x, h_x, cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    float peakBW = (WARP_SIZE == 64) ? 1843.0f : 1008.0f;
    size_t dataBytes = rows * sizeof(int) * 2 + matrix.nnz * sizeof(int) +
                       matrix.nnz * sizeof(float) * 2 + rows * sizeof(float);

    GpuTimer timer;

    // Test 1: Standard merge-based
    std::cout << "1. Merge-based (standard):\n";
    float totalTime = 0;
    for (int i = 0; i < iterations; i++) {
        cudaMemset(d_y1, 0, rows * sizeof(float));
        cudaDeviceSynchronize();
        timer.start();
        spmv_merge_based<float>(matrix, d_x, d_y1, 0);
        timer.stop();
        totalTime += timer.elapsed_ms();
    }
    float avgTime1 = totalTime / iterations;
    float bw1 = (dataBytes / (avgTime1 * 1e-3)) / (1024 * 1024 * 1024);
    float util1 = bw1 / peakBW * 100;
    std::cout << "   Time: " << avgTime1 << " ms, BW: " << bw1 << " GB/s, Util: " << util1 << "%\n";

    // Test 2: Sparse optimized kernel
    std::cout << "2. Sparse optimized:\n";
    totalTime = 0;
    for (int i = 0; i < iterations; i++) {
        cudaMemset(d_y2, 0, rows * sizeof(float));
        cudaDeviceSynchronize();
        timer.start();
        spmv_sparse_optimized<float>(matrix, d_x, d_y2, 0);
        timer.stop();
        totalTime += timer.elapsed_ms();
    }
    float avgTime2 = totalTime / iterations;
    float bw2 = (dataBytes / (avgTime2 * 1e-3)) / (1024 * 1024 * 1024);
    float util2 = bw2 / peakBW * 100;
    std::cout << "   Time: " << avgTime2 << " ms, BW: " << bw2 << " GB/s, Util: " << util2 << "%\n";

    // Compare
    float improvement = (util2 - util1) / util1 * 100;
    std::cout << "   Improvement: " << improvement << "%\n";

    // Cleanup
    delete[] h_x;
    cudaFree(d_x);
    cudaFree(d_y1);
    cudaFree(d_y2);
}

int main(int argc, char** argv) {
    std::cout << "=== Sparse Optimized Kernel Test ===\n";
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