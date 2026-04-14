/**
 * @file benchmark_real_matrices.cu
 * @brief Benchmark FP64 SpMV library with real matrices
 */

#include "spmv_fp64.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>

int main(int argc, char** argv)
{
    printf("=== FP64 SpMV Library - Real Matrix Benchmark ===\n");
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

    // Matrix directory
    const char* matrixDir = argc > 1 ? argv[1] : "~/cbx/spmv_muxi/real_cases/mtx";
    printf("Matrix directory: %s\n\n", matrixDir);

    // Benchmark options
    spmv_fp64_opts_t opts = SPMV_FP64_BENCHMARK_OPTS;

    // Results table header
    printf("| Matrix | Rows | NNZ | avgNnz | Kernel Time (ms) | Bandwidth (GB/s) | Utilization (%) | Status |\n");
    printf("|--------|------|-----|--------|------------------|------------------|-----------------|--------|\n");

    // Test matrices p0 through p9
    for (int i = 0; i <= 9; i++) {
        char matrixFile[512];
        snprintf(matrixFile, sizeof(matrixFile), "%s/p%d_A", matrixDir, i);

        // Load matrix
        spmv_fp64_matrix_handle_t matrix;
        status = spmv_fp64_create_matrix_from_file(&matrix, matrixFile, &opts);

        if (status != SPMV_FP64_SUCCESS) {
            printf("| p%d_A | - | - | - | - | - | - | LOAD FAILED |\n", i);
            continue;
        }

        // Get matrix info
        int numRows, numCols, nnz;
        spmv_fp64_get_matrix_info(matrix, &numRows, &numCols, &nnz);
        double avgNnz = (double)nnz / numRows;

        // Allocate pinned vectors
        double* x;
        double* y;

        status = spmv_fp64_alloc_pinned((void**)&x, numCols * sizeof(double));
        if (status != SPMV_FP64_SUCCESS) {
            printf("| p%d_A | %d | %d | %.2f | - | - | - | ALLOC FAILED |\n", i, numRows, nnz, avgNnz);
            spmv_fp64_destroy_matrix(matrix);
            continue;
        }

        status = spmv_fp64_alloc_pinned((void**)&y, numRows * sizeof(double));
        if (status != SPMV_FP64_SUCCESS) {
            spmv_fp64_free_pinned(x);
            spmv_fp64_destroy_matrix(matrix);
            printf("| p%d_A | %d | %d | %.2f | - | - | - | ALLOC FAILED |\n", i, numRows, nnz, avgNnz);
            continue;
        }

        // Initialize x with ones
        for (int j = 0; j < numCols; j++) {
            x[j] = 1.0;
        }

        // Execute SpMV with benchmark
        spmv_fp64_stats_t stats;
        status = spmv_fp64_execute(matrix, x, y, &opts, &stats);

        if (status != SPMV_FP64_SUCCESS) {
            printf("| p%d_A | %d | %d | %.2f | - | - | - | EXEC FAILED |\n", i, numRows, nnz, avgNnz);
        } else {
            printf("| p%d_A | %d | %d | %.2f | %.3f | %.1f | %.1f | PASS |\n",
                   i, numRows, nnz, avgNnz,
                   stats.kernel_time_ms, stats.bandwidth_gbps, stats.utilization_pct);
        }

        // Cleanup
        spmv_fp64_free_pinned(x);
        spmv_fp64_free_pinned(y);
        spmv_fp64_destroy_matrix(matrix);
    }

    printf("\n=== Benchmark Complete ===\n");

    return 0;
}