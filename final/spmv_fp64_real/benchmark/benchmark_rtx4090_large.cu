/**
 * @file benchmark_rtx4090_large.cu
 * @brief Test optimized spmv_fp64 library on RTX 4090 with large random matrices
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>

#include "spmv_fp64.h"

// Generate CSR matrix with random columns (more realistic)
void generateRandomMatrix(int rows, int cols, int avgNnz,
                          std::vector<int>& rowPtr,
                          std::vector<int>& colIdx,
                          std::vector<double>& values) {
    int nnz = rows * avgNnz;
    rowPtr.resize(rows + 1);
    colIdx.resize(nnz);
    values.resize(nnz);

    rowPtr[0] = 0;
    for (int i = 0; i < rows; i++) {
        rowPtr[i + 1] = rowPtr[i] + avgNnz;
        for (int j = 0; j < avgNnz; j++) {
            colIdx[rowPtr[i] + j] = rand() % cols;  // Random column
            values[rowPtr[i] + j] = (double)(rand() % 100) / 100.0;
        }
    }
}

double calculateBandwidth(int nnz, int rows, double timeMs) {
    double bytes = (double)nnz * 20.0 + (double)rows * 8.0;
    return bytes / (timeMs * 1e6);
}

int main() {
    std::cout << "===== Optimized SpMV FP64 Benchmark (RTX 4090 - Large) =====" << std::endl;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << ", Warp Size: " << prop.warpSize << std::endl;
    double peakBW = 1008.0;  // RTX 4090 peak bandwidth
    std::cout << "Peak Bandwidth: " << peakBW << " GB/s" << std::endl;

    if (spmv_fp64_check_license() != SPMV_FP64_SUCCESS) {
        std::cerr << "License expired!" << std::endl;
        return 1;
    }
    std::cout << "License valid until: " << spmv_fp64_get_license_expiry() << std::endl;
    std::cout << std::endl;

    srand(12345);
    int rows = 500000;
    int cols = 500000;
    int avgNnz = 85;
    int iterations = 50;

    std::vector<int> rowPtr, colIdx;
    std::vector<double> values;
    generateRandomMatrix(rows, cols, avgNnz, rowPtr, colIdx, values);
    int nnz = rows * avgNnz;

    std::cout << "Matrix: " << rows << " rows, " << nnz << " nnz, avgNnz=" << avgNnz << std::endl;
    std::cout << std::endl;

    int *d_rowPtr, *d_colIdx;
    double *d_values, *d_x, *d_y;

    cudaMalloc(&d_rowPtr, (rows + 1) * sizeof(int));
    cudaMalloc(&d_colIdx, nnz * sizeof(int));
    cudaMalloc(&d_values, nnz * sizeof(double));
    cudaMalloc(&d_x, cols * sizeof(double));
    cudaMalloc(&d_y, rows * sizeof(double));

    cudaMemcpy(d_rowPtr, rowPtr.data(), (rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdx, colIdx.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values.data(), nnz * sizeof(double), cudaMemcpyHostToDevice);

    std::vector<double> x(cols, 1.0);
    cudaMemcpy(d_x, x.data(), cols * sizeof(double), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup (multiple iterations)
    for (int i = 0; i < 10; i++) {
        spmv_fp64_execute_direct(rows, nnz, d_rowPtr, d_colIdx, d_values, d_x, d_y, 0);
    }
    cudaDeviceSynchronize();

    // Benchmark
    double minTime = 1e9, totalTime = 0;
    for (int i = 0; i < iterations; i++) {
        cudaEventRecord(start);
        spmv_fp64_execute_direct(rows, nnz, d_rowPtr, d_colIdx, d_values, d_x, d_y, 0);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float timeMs = 0;
        cudaEventElapsedTime(&timeMs, start, stop);
        minTime = std::min(minTime, (double)timeMs);
        totalTime += timeMs;
    }

    double avgTime = totalTime / iterations;
    double bw = calculateBandwidth(nnz, rows, minTime);
    double util = bw / peakBW * 100;

    std::cout << "| Metric | Value |" << std::endl;
    std::cout << "|--------|-------|" << std::endl;
    std::cout << "| Best Time | " << minTime << " ms |" << std::endl;
    std::cout << "| Avg Time | " << avgTime << " ms |" << std::endl;
    std::cout << "| Bandwidth | " << bw << " GB/s |" << std::endl;
    std::cout << "| Utilization | " << util << "% |" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_rowPtr);
    cudaFree(d_colIdx);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);

    std::cout << std::endl;
    std::cout << "===== Benchmark Complete =====" << std::endl;

    return 0;
}