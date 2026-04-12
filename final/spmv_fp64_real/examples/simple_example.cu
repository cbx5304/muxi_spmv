/**
 * @file simple_example.cu
 * @brief Basic usage example for SPMV FP64 library
 */

#include "spmv_fp64.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv)
{
    printf("=== SPMV FP64 Library Example ===\n");
    printf("Version: %s\n\n", spmv_fp64_get_version());

    // Get GPU info
    int warpSize;
    const char* gpuName;
    size_t totalMem;

    spmv_fp64_status_t status = spmv_fp64_get_device_info(&warpSize, &gpuName, &totalMem);
    if (status != SPMV_FP64_SUCCESS) {
        fprintf(stderr, "Error: %s\n", spmv_fp64_get_error_string(status));
        return 1;
    }

    printf("GPU: %s\n", gpuName);
    printf("Warp Size: %d\n", warpSize);
    printf("Total Memory: %.2f GB\n\n", totalMem / 1e9);

    // Determine matrix file
    const char* matrixFile = "matrix.mtx";
    if (argc > 1) {
        matrixFile = argv[1];
    }

    printf("Loading matrix: %s\n", matrixFile);

    // Load matrix with benchmark mode
    spmv_fp64_opts_t opts = SPMV_FP64_BENCHMARK_OPTS;

    spmv_fp64_matrix_handle_t matrix;
    status = spmv_fp64_create_matrix_from_file(&matrix, matrixFile, &opts);
    if (status != SPMV_FP64_SUCCESS) {
        fprintf(stderr, "Error loading matrix: %s\n", spmv_fp64_get_error_string(status));
        return 1;
    }

    // Get matrix info
    int numRows, numCols, nnz;
    spmv_fp64_get_matrix_info(matrix, &numRows, &numCols, &nnz);

    printf("Matrix: %d rows, %d cols, %d nnz\n", numRows, numCols, nnz);
    printf("avgNnzPerRow: %.2f\n\n", (double)nnz / numRows);

    // Allocate pinned memory for vectors (recommended!)
    double* x;
    double* y;

    status = spmv_fp64_alloc_pinned((void**)&x, numCols * sizeof(double));
    if (status != SPMV_FP64_SUCCESS) {
        fprintf(stderr, "Error allocating x: %s\n", spmv_fp64_get_error_string(status));
        spmv_fp64_destroy_matrix(matrix);
        return 1;
    }

    status = spmv_fp64_alloc_pinned((void**)&y, numRows * sizeof(double));
    if (status != SPMV_FP64_SUCCESS) {
        fprintf(stderr, "Error allocating y: %s\n", spmv_fp64_get_error_string(status));
        spmv_fp64_free_pinned(x);
        spmv_fp64_destroy_matrix(matrix);
        return 1;
    }

    // Initialize input vector (all ones for simple verification)
    for (int i = 0; i < numCols; i++) {
        x[i] = 1.0;
    }

    // Execute SpMV
    printf("Running SpMV benchmark...\n");

    spmv_fp64_stats_t stats;
    status = spmv_fp64_execute(matrix, x, y, &opts, &stats);
    if (status != SPMV_FP64_SUCCESS) {
        fprintf(stderr, "Error: %s\n", spmv_fp64_get_error_string(status));
        spmv_fp64_free_pinned(x);
        spmv_fp64_free_pinned(y);
        spmv_fp64_destroy_matrix(matrix);
        return 1;
    }

    // Print results
    printf("\n=== Performance Results ===\n");
    printf("Kernel+Copy Time: %.3f ms\n", stats.kernel_time_ms);
    printf("Bandwidth:        %.1f GB/s\n", stats.bandwidth_gbps);
    printf("Utilization:      %.1f%%\n", stats.utilization_pct);
    printf("Theoretical BW:   %.1f GB/s\n", stats.theoretical_bw);
    printf("Warp Size:        %d\n", stats.warp_size);
    printf("Optimal TPR:      %d\n", stats.optimal_tpr);

    // Simple verification
    printf("\n=== Verification ===\n");

    // Sum of y should approximate nnz (since x[i]=1.0)
    double sumY = 0.0;
    for (int r = 0; r < numRows; r++) {
        sumY += y[r];
    }
    double expectedSum = (double)nnz;
    double relError = fabs(sumY - expectedSum) / expectedSum;

    printf("Output sum:   %.1f\n", sumY);
    printf("Expected sum: %.1f\n", expectedSum);
    printf("Rel error:    %.2e\n", relError);

    if (relError < 0.001) {
        printf("Verification: PASSED\n");
    } else {
        printf("Verification: FAILED (error too large)\n");
    }

    // Cleanup
    spmv_fp64_free_pinned(x);
    spmv_fp64_free_pinned(y);
    spmv_fp64_destroy_matrix(matrix);

    printf("\n=== Example Complete ===\n");

    return 0;
}