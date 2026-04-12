/**
 * @file benchmark_matrices.cu
 * @brief Benchmark FP64 SpMV library with synthetic matrices
 */

#include "spmv_fp64.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv)
{
    printf("=== FP64 SpMV Library - Matrix Benchmark ===\n");
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
    const char* matrixDir = argc > 1 ? argv[1] : "./matrices";
    printf("Matrix directory: %s\n\n", matrixDir);

    // Benchmark options
    spmv_fp64_opts_t opts = SPMV_FP64_BENCHMARK_OPTS;

    // Results table header
    printf("%-10s %-10s %-12s %-10s %-18s %-18s %-15s %-8s\n",
           "Matrix", "Rows", "NNZ", "avgNnz", "Time(ms)", "BW(GB/s)", "Util(%)", "Status");
    printf("%-10s %-10s %-12s %-10s %-18s %-18s %-15s %-8s\n",
           "--------", "------", "-----", "--------", "--------", "--------", "--------", "--------");

    // Test matrices p0 through p9
    double total_bw = 0.0;
    double total_time = 0.0;
    int success_count = 0;

    for (int i = 0; i <= 9; i++) {
        char matrixFile[512];
        snprintf(matrixFile, sizeof(matrixFile), "%s/p%d_A.mtx", matrixDir, i);

        // Load matrix
        spmv_fp64_matrix_handle_t matrix;
        status = spmv_fp64_create_matrix_from_file(&matrix, matrixFile, &opts);

        if (status != SPMV_FP64_SUCCESS) {
            printf("p%d_A     LOAD FAILED\n", i);
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
            printf("p%d_A     ALLOC FAILED\n", i);
            spmv_fp64_destroy_matrix(matrix);
            continue;
        }

        status = spmv_fp64_alloc_pinned((void**)&y, numRows * sizeof(double));
        if (status != SPMV_FP64_SUCCESS) {
            spmv_fp64_free_pinned(x);
            spmv_fp64_destroy_matrix(matrix);
            printf("p%d_A     ALLOC FAILED\n", i);
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
            printf("p%d_A     EXEC FAILED\n", i);
        } else {
            printf("p%d_A     %-10d %-12d %-10.2f %-18.3f %-18.1f %-15.1f PASS\n",
                   i, numRows, nnz, avgNnz,
                   stats.kernel_time_ms, stats.bandwidth_gbps, stats.utilization_pct);

            total_bw += stats.bandwidth_gbps;
            total_time += stats.kernel_time_ms;
            success_count++;
        }

        // Cleanup
        spmv_fp64_free_pinned(x);
        spmv_fp64_free_pinned(y);
        spmv_fp64_destroy_matrix(matrix);
    }

    printf("\n=== Summary ===\n");
    if (success_count > 0) {
        printf("Tests passed: %d/10\n", success_count);
        printf("Average bandwidth: %.1f GB/s\n", total_bw / success_count);
        printf("Average time: %.3f ms\n", total_time / success_count);
        printf("Optimal kernel: %s (TPR=%d)\n",
               warpSize == 64 ? "TPR" : "__ldg",
               warpSize == 64 ? 8 : 1);
    }
    printf("=== Benchmark Complete ===\n");

    return 0;
}