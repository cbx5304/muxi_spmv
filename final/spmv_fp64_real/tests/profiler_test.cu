/**
 * @file profiler_test.cu
 * @brief Profiler test with CUDA Events for accurate timing
 */

#include "spmv_fp64.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main() {
    printf("\n========================================\n");
    printf("  SPMV FP64 Profiler Test (CUDA Events)\n");
    printf("========================================\n\n");

    // GPU info
    int warpSize;
    const char* gpuName;
    size_t gpuMem;
    spmv_fp64_status_t status = spmv_fp64_get_device_info(&warpSize, &gpuName, &gpuMem);

    printf("GPU: %s\n", gpuName);
    printf("Warp Size: %d\n", warpSize);
    printf("Memory: %.2f GB\n", gpuMem / 1e9);

    double theoreticalBW;
    spmv_fp64_get_theoretical_bandwidth(&theoreticalBW);
    printf("Theoretical BW: %.1f GB/s\n\n", theoreticalBW);

    // Large matrix test
    int numRows = 1000000;
    int numCols = numRows;
    int avgNnz = 10;
    int nnz = numRows * avgNnz;

    printf("Matrix: %d rows, avgNnz=%d, total nnz=%d\n\n", numRows, avgNnz, nnz);

    // Generate CSR with band pattern (column locality)
    int* h_rowPtr = (int*)malloc((numRows + 1) * sizeof(int));
    int* h_colIdx = (int*)malloc(nnz * sizeof(int));
    double* h_values = (double*)malloc(nnz * sizeof(double));

    h_rowPtr[0] = 0;
    for (int i = 0; i < numRows; i++) {
        h_rowPtr[i + 1] = h_rowPtr[i] + avgNnz;
        int base = h_rowPtr[i];
        int baseCol = (i * numCols / numRows) % numCols;
        for (int j = 0; j < avgNnz; j++) {
            // Band pattern + some randomness
            if (j < avgNnz / 2) {
                h_colIdx[base + j] = (baseCol + j) % numCols;
            } else {
                h_colIdx[base + j] = rand() % numCols;
            }
            h_values[base + j] = 1.0 + j * 0.1;
        }
    }

    // Device memory
    int* d_rowPtr, *d_colIdx;
    double* d_values, *d_x, *d_y;
    cudaMalloc(&d_rowPtr, (numRows + 1) * sizeof(int));
    cudaMalloc(&d_colIdx, nnz * sizeof(int));
    cudaMalloc(&d_values, nnz * sizeof(double));
    cudaMalloc(&d_x, numCols * sizeof(double));
    cudaMalloc(&d_y, numRows * sizeof(double));

    cudaMemcpy(d_rowPtr, h_rowPtr, (numRows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdx, h_colIdx, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values, nnz * sizeof(double), cudaMemcpyHostToDevice);

    // Initialize x
    double* h_x = (double*)malloc(numCols * sizeof(double));
    for (int i = 0; i < numCols; i++) h_x[i] = 1.0;
    cudaMemcpy(d_x, h_x, numCols * sizeof(double), cudaMemcpyHostToDevice);

    // CUDA Events for accurate timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    printf("Warmup runs (10 iterations)...\n");
    for (int w = 0; w < 10; w++) {
        spmv_fp64_execute_direct(numRows, nnz, d_rowPtr, d_colIdx, d_values, d_x, d_y, 0);
    }
    cudaDeviceSynchronize();

    // Benchmark with CUDA Events
    printf("Benchmark runs (100 iterations)...\n\n");
    int iterations = 100;
    float total_time = 0.0f;

    cudaEventRecord(start, 0);
    for (int i = 0; i < iterations; i++) {
        spmv_fp64_execute_direct(numRows, nnz, d_rowPtr, d_colIdx, d_values, d_x, d_y, 0);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&total_time, start, stop);
    float avg_time_ms = total_time / iterations;

    // Calculate bandwidth
    // FP64 SpMV: values(8B) + colIdx(4B) + x(col)(8B) per nnz = 20 bytes per nnz
    // Plus y output: numRows * 8B
    double bytes_per_iter = (double)nnz * 20.0 + (double)numRows * 8.0;
    double bandwidth = bytes_per_iter / (avg_time_ms * 1e6);  // GB/s
    double utilization = (bandwidth / theoreticalBW) * 100.0;

    printf("=== Performance Results ===\n");
    printf("Kernel time (avg): %.3f ms\n", avg_time_ms);
    printf("Bandwidth: %.1f GB/s\n", bandwidth);
    printf("Utilization: %.1f%%\n", utilization);
    printf("Kernel type: %s\n", warpSize == 64 ? "TPR=8" : "__ldg");

    // Verify correctness
    printf("\n=== Correctness Check ===\n");
    cudaMemcpy(d_x, h_x, numCols * sizeof(double), cudaMemcpyHostToDevice);
    spmv_fp64_execute_direct(numRows, nnz, d_rowPtr, d_colIdx, d_values, d_x, d_y, 0);
    cudaDeviceSynchronize();

    double* h_y = (double*)malloc(numRows * sizeof(double));
    cudaMemcpy(h_y, d_y, numRows * sizeof(double), cudaMemcpyDeviceToHost);

    // Reference computation on CPU (first 100 rows)
    int errors = 0;
    for (int i = 0; i < 100; i++) {
        double ref = 0.0;
        for (int j = h_rowPtr[i]; j < h_rowPtr[i + 1]; j++) {
            ref += h_values[j] * h_x[h_colIdx[j]];
        }
        double err = fabs(h_y[i] - ref);
        double rel_err = fabs(ref) > 1e-10 ? err / fabs(ref) : err;
        if (rel_err > 1e-10) {
            errors++;
            if (errors <= 5) {
                printf("Row %d: ref=%.6f, gpu=%.6f, err=%.2e\n", i, ref, h_y[i], rel_err);
            }
        }
    }
    printf("Correctness: %s (%d errors in 100 sample rows)\n",
           errors == 0 ? "PASSED" : "FAILED", errors);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_rowPtr);
    cudaFree(d_colIdx);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);
    free(h_rowPtr);
    free(h_colIdx);
    free(h_values);
    free(h_x);
    free(h_y);

    printf("\n========================================\n");
    printf("  Test Complete\n");
    printf("========================================\n\n");

    return 0;
}