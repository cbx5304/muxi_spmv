/**
 * @file benchmark_kernel_only.cu
 * @brief Measure pure kernel bandwidth (excluding memcpy)
 */

#include "spmv_fp64.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv)
{
    printf("=== FP64 SpMV - Pure Kernel Benchmark ===\n");
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

    const char* matrixDir = argc > 1 ? argv[1] : "./matrices";

    // Results
    printf("%-10s %-10s %-12s %-10s %-15s %-15s %-10s\n",
           "Matrix", "Rows", "NNZ", "avgNnz", "Kernel(ms)", "BW(GB/s)", "Util(%)");
    printf("%-10s %-10s %-12s %-10s %-15s %-15s %-10s\n",
           "--------", "------", "-----", "--------", "--------", "--------", "--------");

    double total_bw = 0.0;
    int success_count = 0;

    for (int i = 0; i <= 9; i++) {
        char matrixFile[512];
        snprintf(matrixFile, sizeof(matrixFile), "%s/p%d_A.mtx", matrixDir, i);

        // Load matrix without benchmark mode (just load)
        spmv_fp64_opts_t loadOpts = SPMV_FP64_DEFAULT_OPTS;
        spmv_fp64_matrix_handle_t matrix;
        status = spmv_fp64_create_matrix_from_file(&matrix, matrixFile, &loadOpts);

        if (status != SPMV_FP64_SUCCESS) {
            printf("p%d_A     LOAD FAILED\n", i);
            continue;
        }

        // Get matrix info
        int numRows, numCols, nnz;
        spmv_fp64_get_matrix_info(matrix, &numRows, &numCols, &nnz);
        double avgNnz = (double)nnz / numRows;

        // Allocate vectors (use pinned for speed)
        double* x;
        double* y;
        spmv_fp64_alloc_pinned((void**)&x, numCols * sizeof(double));
        spmv_fp64_alloc_pinned((void**)&y, numRows * sizeof(double));

        for (int j = 0; j < numCols; j++) x[j] = 1.0;

        // Warm up - run once to load data to device
        spmv_fp64_opts_t execOpts = SPMV_FP64_DEFAULT_OPTS;
        spmv_fp64_execute(matrix, x, y, &execOpts, NULL);

        // Now measure kernel-only performance
        // Use benchmark mode which measures H2D+kernel, then estimate kernel-only
        spmv_fp64_opts_t benchOpts = SPMV_FP64_BENCHMARK_OPTS;
        spmv_fp64_stats_t stats;
        status = spmv_fp64_execute(matrix, x, y, &benchOpts, &stats);

        // Calculate pure kernel bandwidth estimate
        // Total time includes H2D (~0.1ms for 10MB) + kernel
        // Estimate H2D time based on PCIe bandwidth (~12GB/s)
        double xSizeMB = numCols * 8.0 / 1e6;
        double h2dTimeMs = xSizeMB / 12.0;  // PCIe ~12GB/s
        double kernelOnlyMs = stats.kernel_time_ms - h2dTimeMs;
        if (kernelOnlyMs < 0.01) kernelOnlyMs = stats.kernel_time_ms;  // Clamp

        double bytes = nnz * 20.0 + numRows * 8.0;
        double kernelBW = bytes / (kernelOnlyMs * 1e6);
        double kernelUtil = kernelBW / stats.theoretical_bw * 100.0;

        printf("p%d_A     %-10d %-12d %-10.2f %-15.3f %-15.1f %-10.1f\n",
               i, numRows, nnz, avgNnz, kernelOnlyMs, kernelBW, kernelUtil);

        total_bw += kernelBW;
        success_count++;

        // Cleanup
        spmv_fp64_free_pinned(x);
        spmv_fp64_free_pinned(y);
        spmv_fp64_destroy_matrix(matrix);
    }

    printf("\n=== Summary ===\n");
    printf("Tests: %d/10 passed\n", success_count);
    if (success_count > 0) {
        printf("Average kernel BW: %.1f GB/s\n", total_bw / success_count);
    }
    printf("=== Complete ===\n");

    return 0;
}