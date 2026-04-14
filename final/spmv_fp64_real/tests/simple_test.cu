/**
 * @file simple_test.cu
 * @brief Simple validation test
 */

#include "spmv_fp64.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    printf("=== Simple SPMV Test ===\n");

    // Step 1: Version
    printf("Library version: %s\n", spmv_fp64_get_version());

    // Step 2: GPU info
    int warpSize;
    const char* gpuName;
    size_t gpuMem;
    spmv_fp64_status_t status = spmv_fp64_get_device_info(&warpSize, &gpuName, &gpuMem);
    printf("Status: %d\n", status);
    if (status != SPMV_FP64_SUCCESS) {
        printf("Error: %s\n", spmv_fp64_get_error_string(status));
        return 1;
    }
    printf("GPU: %s, WarpSize: %d, Memory: %.2f GB\n", gpuName, warpSize, gpuMem / 1e9);

    // Step 3: Small matrix test
    int numRows = 1000;
    int numCols = 1000;
    int avgNnz = 10;
    int nnz = numRows * avgNnz;

    // Generate CSR on host
    int* h_rowPtr = (int*)malloc((numRows + 1) * sizeof(int));
    int* h_colIdx = (int*)malloc(nnz * sizeof(int));
    double* h_values = (double*)malloc(nnz * sizeof(double));

    h_rowPtr[0] = 0;
    for (int i = 0; i < numRows; i++) {
        h_rowPtr[i + 1] = h_rowPtr[i] + avgNnz;
    }
    for (int i = 0; i < nnz; i++) {
        h_colIdx[i] = i % numCols;
        h_values[i] = 1.0;
    }

    double* h_x = (double*)malloc(numCols * sizeof(double));
    double* h_y = (double*)malloc(numRows * sizeof(double));
    for (int i = 0; i < numCols; i++) h_x[i] = 1.0;

    // Create matrix
    printf("\nCreating matrix...\n");
    spmv_fp64_opts_t opts = SPMV_FP64_DEFAULT_OPTS;
    spmv_fp64_matrix_handle_t matrix;

    status = spmv_fp64_create_matrix(&matrix, numRows, numCols, nnz,
                                      h_rowPtr, h_colIdx, h_values, &opts);
    printf("Create status: %d\n", status);

    if (status != SPMV_FP64_SUCCESS) {
        printf("Error: %s\n", spmv_fp64_get_error_string(status));
        return 1;
    }

    // Execute
    printf("Executing SpMV...\n");
    status = spmv_fp64_execute(matrix, h_x, h_y, &opts, NULL);
    printf("Execute status: %d\n", status);

    if (status != SPMV_FP64_SUCCESS) {
        printf("Error: %s\n", spmv_fp64_get_error_string(status));
        spmv_fp64_destroy_matrix(matrix);
        return 1;
    }

    // Verify result (should be avgNnz * 1.0 = 10.0 for each row)
    printf("\nVerifying result...\n");
    double expected = avgNnz * 1.0;
    int errors = 0;
    for (int i = 0; i < numRows; i++) {
        if (fabs(h_y[i] - expected) > 1e-6) {
            errors++;
            if (errors < 5) {
                printf("Row %d: expected %.1f, got %.1f\n", i, expected, h_y[i]);
            }
        }
    }
    printf("Errors: %d/%d\n", errors, numRows);

    // Benchmark
    printf("\nBenchmark...\n");
    spmv_fp64_opts_t bench_opts = SPMV_FP64_BENCHMARK_OPTS;
    spmv_fp64_stats_t stats;
    status = spmv_fp64_execute(matrix, h_x, h_y, &bench_opts, &stats);
    printf("Benchmark status: %d\n", status);

    if (status == SPMV_FP64_SUCCESS) {
        printf("Kernel time: %.3f ms\n", stats.kernel_time_ms);
        printf("Bandwidth: %.1f GB/s\n", stats.bandwidth_gbps);
        printf("Utilization: %.1f%%\n", stats.utilization_pct);
    }

    // Cleanup
    spmv_fp64_destroy_matrix(matrix);
    free(h_rowPtr);
    free(h_colIdx);
    free(h_values);
    free(h_x);
    free(h_y);

    printf("\n=== Test Complete ===\n");
    return (errors == 0) ? 0 : 1;
}