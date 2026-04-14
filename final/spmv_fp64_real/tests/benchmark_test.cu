/**
 * @file benchmark_test.cu
 * @brief Benchmark test with larger matrices
 */

#include "spmv_fp64.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main() {
    printf("\n========================================\n");
    printf("  SPMV FP64 Benchmark Test\n");
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

    // Test sizes (larger matrices for realistic bandwidth)
    int sizes[] = {10000, 50000, 100000, 500000, 1000000};
    int numSizes = 5;
    int avgNnz = 10;

    printf("=== Performance Benchmark ===\n\n");

    double totalBW = 0.0;
    double totalUtil = 0.0;
    int count = 0;

    for (int t = 0; t < numSizes; t++) {
        int numRows = sizes[t];
        int numCols = numRows;
        int nnz = numRows * avgNnz;

        // Generate simple CSR
        int* h_rowPtr = (int*)malloc((numRows + 1) * sizeof(int));
        int* h_colIdx = (int*)malloc(nnz * sizeof(int));
        double* h_values = (double*)malloc(nnz * sizeof(double));

        h_rowPtr[0] = 0;
        for (int i = 0; i < numRows; i++) {
            h_rowPtr[i + 1] = h_rowPtr[i] + avgNnz;
            int base = h_rowPtr[i];
            for (int j = 0; j < avgNnz; j++) {
                h_colIdx[base + j] = (i + j) % numCols;  // Band pattern for locality
                h_values[base + j] = 1.0 + j * 0.1;
            }
        }

        // Pinned memory for vectors
        double* h_x, *h_y;
        spmv_fp64_alloc_pinned((void**)&h_x, numCols * sizeof(double));
        spmv_fp64_alloc_pinned((void**)&h_y, numRows * sizeof(double));

        for (int i = 0; i < numCols; i++) h_x[i] = 1.0;

        // Create matrix
        spmv_fp64_matrix_handle_t matrix;
        spmv_fp64_opts_t opts = SPMV_FP64_BENCHMARK_OPTS;

        status = spmv_fp64_create_matrix(&matrix, numRows, numCols, nnz,
                                          h_rowPtr, h_colIdx, h_values, &opts);

        if (status != SPMV_FP64_SUCCESS) {
            printf("[%d] Create failed\n", t);
            continue;
        }

        // Warmup (skip timing for first few)
        spmv_fp64_opts_t warmup_opts = SPMV_FP64_DEFAULT_OPTS;
        for (int w = 0; w < 5; w++) {
            spmv_fp64_execute(matrix, h_x, h_y, &warmup_opts, NULL);
        }

        // Benchmark
        spmv_fp64_stats_t stats;
        status = spmv_fp64_execute(matrix, h_x, h_y, &opts, &stats);

        if (status == SPMV_FP64_SUCCESS) {
            printf("[%d] rows=%d, nnz=%d\n", t, numRows, nnz);
            printf("    Kernel time: %.3f ms\n", stats.kernel_time_ms);
            printf("    Bandwidth: %.1f GB/s\n", stats.bandwidth_gbps);
            printf("    Utilization: %.1f%%\n\n", stats.utilization_pct);

            // Only count larger matrices (>= 50000 rows)
            if (numRows >= 50000) {
                totalBW += stats.bandwidth_gbps;
                totalUtil += stats.utilization_pct;
                count++;
            }
        }

        spmv_fp64_destroy_matrix(matrix);
        spmv_fp64_free_pinned(h_x);
        spmv_fp64_free_pinned(h_y);
        free(h_rowPtr);
        free(h_colIdx);
        free(h_values);
    }

    if (count > 0) {
        printf("=== Summary (large matrices) ===\n");
        printf("Average Bandwidth: %.1f GB/s\n", totalBW / count);
        printf("Average Utilization: %.1f%%\n", totalUtil / count);
        printf("\n");
    }

    printf("========================================\n");
    printf("  Benchmark Complete\n");
    printf("========================================\n\n");

    return 0;
}