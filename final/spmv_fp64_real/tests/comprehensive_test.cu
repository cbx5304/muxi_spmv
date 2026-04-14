/**
 * @file comprehensive_test.cu
 * @brief Comprehensive test for spmv_fp64 library - Correctness + Performance
 */

#include "spmv_fp64.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>

// ==================== Reference Implementation ====================

void reference_spmv(
    int numRows,
    int numCols,
    const int* rowPtr,
    const int* colIdx,
    const double* values,
    const double* x,
    double* y)
{
    for (int i = 0; i < numRows; i++) {
        double sum = 0.0;
        for (int j = rowPtr[i]; j < rowPtr[i + 1]; j++) {
            sum += values[j] * x[colIdx[j]];
        }
        y[i] = sum;
    }
}

// ==================== Test Matrix Generation ====================

void generate_test_csr(
    int numRows,
    int numCols,
    int avgNnzPerRow,
    int** rowPtr,
    int** colIdx,
    double** values,
    int* nnz)
{
    *rowPtr = (int*)malloc((numRows + 1) * sizeof(int));
    (*rowPtr)[0] = 0;

    for (int i = 0; i < numRows; i++) {
        int rowNnz = avgNnzPerRow + (i % 3);  // Vary slightly
        (*rowPtr)[i + 1] = (*rowPtr)[i] + rowNnz;
    }

    *nnz = (*rowPtr)[numRows];
    *colIdx = (int*)malloc(*nnz * sizeof(int));
    *values = (double*)malloc(*nnz * sizeof(double));

    // Fill with locality pattern (band + random)
    for (int i = 0; i < numRows; i++) {
        int rowStart = (*rowPtr)[i];
        int rowEnd = (*rowPtr)[i + 1];
        int rowNnz = rowEnd - rowStart;
        int baseCol = (i * numCols / numRows) % numCols;

        for (int j = 0; j < rowNnz; j++) {
            if (j < rowNnz / 2) {
                (*colIdx)[rowStart + j] = (baseCol + j) % numCols;
            } else {
                (*colIdx)[rowStart + j] = rand() % numCols;
            }
            (*values)[rowStart + j] = 1.0 + (rand() % 100) / 100.0;
        }
    }
}

// ==================== Error Calculation ====================

double calculate_max_error(const double* ref, const double* result, int n) {
    double max_err = 0.0;
    for (int i = 0; i < n; i++) {
        double err = fabs(ref[i] - result[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

double calculate_rel_error(const double* ref, const double* result, int n) {
    double max_rel = 0.0;
    for (int i = 0; i < n; i++) {
        if (fabs(ref[i]) > 1e-10) {
            double rel = fabs(ref[i] - result[i]) / fabs(ref[i]);
            if (rel > max_rel) max_rel = rel;
        }
    }
    return max_rel;
}

// ==================== Main ====================

int main(int argc, char** argv) {
    printf("\n========================================\n");
    printf("  SPMV FP64 Library Comprehensive Test\n");
    printf("========================================\n\n");

    printf("Library Version: %s\n\n", spmv_fp64_get_version());

    // GPU Info
    int warpSize;
    const char* gpuName;
    size_t gpuMem;
    spmv_fp64_status_t status = spmv_fp64_get_device_info(&warpSize, &gpuName, &gpuMem);

    if (status != SPMV_FP64_SUCCESS) {
        fprintf(stderr, "Error getting GPU info\n");
        return 1;
    }

    printf("=== GPU Information ===\n");
    printf("GPU: %s\n", gpuName);
    printf("Warp Size: %d\n", warpSize);
    printf("Memory: %.2f GB\n", gpuMem / 1e9);

    double theoreticalBW;
    spmv_fp64_get_theoretical_bandwidth(&theoreticalBW);
    printf("Theoretical BW: %.1f GB/s\n\n", theoreticalBW);

    // Test sizes
    int sizes[] = {1000, 5000, 10000, 50000, 100000};
    int numSizes = 5;
    int avgNnz = 10;

    // ==================== Part 1: Correctness Tests ====================
    printf("=== Part 1: Correctness Validation ===\n\n");

    int passCount = 0;
    double tolerance = 1e-12;  // FP64 tolerance

    for (int t = 0; t < numSizes; t++) {
        int numRows = sizes[t];
        int numCols = numRows;

        // Generate CSR
        int* h_rowPtr, *h_colIdx;
        double* h_values;
        int nnz;
        generate_test_csr(numRows, numCols, avgNnz, &h_rowPtr, &h_colIdx, &h_values, &nnz);

        // Input/output vectors
        double* h_x = (double*)malloc(numCols * sizeof(double));
        double* h_y_ref = (double*)malloc(numRows * sizeof(double));
        double* h_y_lib = (double*)malloc(numRows * sizeof(double));

        for (int i = 0; i < numCols; i++) h_x[i] = 1.0 + (i % 10) * 0.1;

        // Reference
        reference_spmv(numRows, numCols, h_rowPtr, h_colIdx, h_values, h_x, h_y_ref);

        // Library
        spmv_fp64_matrix_handle_t matrix;
        spmv_fp64_opts_t opts = SPMV_FP64_DEFAULT_OPTS;

        status = spmv_fp64_create_matrix(&matrix, numRows, numCols, nnz,
                                          h_rowPtr, h_colIdx, h_values, &opts);

        if (status != SPMV_FP64_SUCCESS) {
            printf("[%d] FAILED: create_matrix error\n", t);
            free(h_rowPtr); free(h_colIdx); free(h_values);
            free(h_x); free(h_y_ref); free(h_y_lib);
            continue;
        }

        status = spmv_fp64_execute(matrix, h_x, h_y_lib, &opts, NULL);

        if (status != SPMV_FP64_SUCCESS) {
            printf("[%d] FAILED: execute error\n", t);
            spmv_fp64_destroy_matrix(matrix);
            free(h_rowPtr); free(h_colIdx); free(h_values);
            free(h_x); free(h_y_ref); free(h_y_lib);
            continue;
        }

        // Check error
        double maxErr = calculate_max_error(h_y_ref, h_y_lib, numRows);
        double relErr = calculate_rel_error(h_y_ref, h_y_lib, numRows);

        if (relErr < tolerance) {
            printf("[%d] PASSED: rows=%d, nnz=%d, max_err=%.2e, rel_err=%.2e\n",
                   t, numRows, nnz, maxErr, relErr);
            passCount++;
        } else {
            printf("[%d] FAILED: rows=%d, nnz=%d, max_err=%.2e, rel_err=%.2e (tol=%.2e)\n",
                   t, numRows, nnz, maxErr, relErr, tolerance);
        }

        spmv_fp64_destroy_matrix(matrix);
        free(h_rowPtr); free(h_colIdx); free(h_values);
        free(h_x); free(h_y_ref); free(h_y_lib);
    }

    printf("\nCorrectness Summary: %d/%d tests passed\n\n", passCount, numSizes);

    // ==================== Part 2: Performance Benchmark ====================
    printf("=== Part 2: Performance Benchmark ===\n\n");

    double totalBW = 0.0;
    double totalUtil = 0.0;
    int bwCount = 0;

    for (int t = 0; t < numSizes; t++) {
        int numRows = sizes[t];
        int numCols = numRows;

        // Generate CSR
        int* h_rowPtr, *h_colIdx;
        double* h_values;
        int nnz;
        generate_test_csr(numRows, numCols, avgNnz, &h_rowPtr, &h_colIdx, &h_values, &nnz);

        // Pinned vectors for best performance
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
            printf("[%d] Performance test skipped\n", t);
            continue;
        }

        // Warmup (5 runs)
        for (int w = 0; w < 5; w++) {
            spmv_fp64_execute(matrix, h_x, h_y, &opts, NULL);
        }

        // Benchmark
        spmv_fp64_stats_t stats;
        status = spmv_fp64_execute(matrix, h_x, h_y, &opts, &stats);

        if (status == SPMV_FP64_SUCCESS) {
            printf("[%d] rows=%d, nnz=%d\n", t, numRows, nnz);
            printf("    Kernel time: %.3f ms\n", stats.kernel_time_ms);
            printf("    Bandwidth: %.1f GB/s\n", stats.bandwidth_gbps);
            printf("    Utilization: %.1f%%\n", stats.utilization_pct);
            printf("    Warp size: %d, TPR: %d\n\n", stats.warp_size, stats.optimal_tpr);

            totalBW += stats.bandwidth_gbps;
            totalUtil += stats.utilization_pct;
            bwCount++;
        }

        spmv_fp64_destroy_matrix(matrix);
        spmv_fp64_free_pinned(h_x);
        spmv_fp64_free_pinned(h_y);
        free(h_rowPtr); free(h_colIdx); free(h_values);
    }

    if (bwCount > 0) {
        printf("=== Performance Summary ===\n");
        printf("Average Bandwidth: %.1f GB/s\n", totalBW / bwCount);
        printf("Average Utilization: %.1f%%\n", totalUtil / bwCount);
        printf("\n");
    }

    // ==================== Part 3: API Coverage Tests ====================
    printf("=== Part 3: API Coverage Tests ===\n\n");

    int testRows = 5000;
    int testCols = 5000;
    int testNnz;
    int* testRowPtr, *testColIdx;
    double* testValues;
    generate_test_csr(testRows, testCols, avgNnz, &testRowPtr, &testColIdx, &testValues, &testNnz);

    double* testX = (double*)malloc(testCols * sizeof(double));
    double* testY = (double*)malloc(testRows * sizeof(double));
    double* testYRef = (double*)malloc(testRows * sizeof(double));
    double* testY2 = (double*)malloc(testRows * sizeof(double));

    for (int i = 0; i < testCols; i++) testX[i] = 1.0;

    // Reference
    reference_spmv(testRows, testCols, testRowPtr, testColIdx, testValues, testX, testYRef);

    // Test 3.1: Device pointer mode
    {
        printf("Test 3.1: Device pointer mode (spmv_fp64_create_matrix_device)\n");

        int* d_rowPtr, *d_colIdx;
        double* d_values, *d_x, *d_y;
        cudaMalloc(&d_rowPtr, (testRows + 1) * sizeof(int));
        cudaMalloc(&d_colIdx, testNnz * sizeof(int));
        cudaMalloc(&d_values, testNnz * sizeof(double));
        cudaMalloc(&d_x, testCols * sizeof(double));
        cudaMalloc(&d_y, testRows * sizeof(double));

        cudaMemcpy(d_rowPtr, testRowPtr, (testRows + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_colIdx, testColIdx, testNnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_values, testValues, testNnz * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, testX, testCols * sizeof(double), cudaMemcpyHostToDevice);

        spmv_fp64_matrix_handle_t mat;
        spmv_fp64_opts_t opts = SPMV_FP64_DEFAULT_OPTS;

        status = spmv_fp64_create_matrix_device(&mat, testRows, testCols, testNnz,
                                                 d_rowPtr, d_colIdx, d_values, &opts);

        if (status == SPMV_FP64_SUCCESS) {
            status = spmv_fp64_execute_device(mat, d_x, d_y, &opts, NULL);
            cudaDeviceSynchronize();
            cudaMemcpy(testY, d_y, testRows * sizeof(double), cudaMemcpyDeviceToHost);

            double err = calculate_rel_error(testYRef, testY, testRows);
            printf("  Result: %s (rel_err=%.2e)\n", err < tolerance ? "PASSED" : "FAILED", err);
        } else {
            printf("  Result: FAILED (create error)\n");
        }

        spmv_fp64_destroy_matrix(mat);
        cudaFree(d_rowPtr); cudaFree(d_colIdx); cudaFree(d_values);
        cudaFree(d_x); cudaFree(d_y);
    }

    // Test 3.2: Direct execution
    {
        printf("Test 3.2: Direct execution (spmv_fp64_execute_direct)\n");

        int* d_rowPtr, *d_colIdx;
        double* d_values, *d_x, *d_y;
        cudaMalloc(&d_rowPtr, (testRows + 1) * sizeof(int));
        cudaMalloc(&d_colIdx, testNnz * sizeof(int));
        cudaMalloc(&d_values, testNnz * sizeof(double));
        cudaMalloc(&d_x, testCols * sizeof(double));
        cudaMalloc(&d_y, testRows * sizeof(double));

        cudaMemcpy(d_rowPtr, testRowPtr, (testRows + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_colIdx, testColIdx, testNnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_values, testValues, testNnz * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, testX, testCols * sizeof(double), cudaMemcpyHostToDevice);

        status = spmv_fp64_execute_direct(testRows, testNnz,
                                          d_rowPtr, d_colIdx, d_values, d_x, d_y, 0);
        cudaDeviceSynchronize();
        cudaMemcpy(testY, d_y, testRows * sizeof(double), cudaMemcpyDeviceToHost);

        double err = calculate_rel_error(testYRef, testY, testRows);
        printf("  Result: %s (rel_err=%.2e)\n", err < tolerance ? "PASSED" : "FAILED", err);

        cudaFree(d_rowPtr); cudaFree(d_colIdx); cudaFree(d_values);
        cudaFree(d_x); cudaFree(d_y);
    }

    // Test 3.3: Direct scaled
    {
        printf("Test 3.3: Direct scaled (spmv_fp64_execute_direct_scaled, alpha=2.0)\n");

        int* d_rowPtr, *d_colIdx;
        double* d_values, *d_x, *d_y;
        cudaMalloc(&d_rowPtr, (testRows + 1) * sizeof(int));
        cudaMalloc(&d_colIdx, testNnz * sizeof(int));
        cudaMalloc(&d_values, testNnz * sizeof(double));
        cudaMalloc(&d_x, testCols * sizeof(double));
        cudaMalloc(&d_y, testRows * sizeof(double));

        cudaMemcpy(d_rowPtr, testRowPtr, (testRows + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_colIdx, testColIdx, testNnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_values, testValues, testNnz * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, testX, testCols * sizeof(double), cudaMemcpyHostToDevice);

        double alpha = 2.0;
        status = spmv_fp64_execute_direct_scaled(alpha, testRows, testNnz,
                                                  d_rowPtr, d_colIdx, d_values, d_x, d_y, 0);
        cudaDeviceSynchronize();
        cudaMemcpy(testY, d_y, testRows * sizeof(double), cudaMemcpyDeviceToHost);

        // Reference with scaling
        for (int i = 0; i < testRows; i++) testY2[i] = alpha * testYRef[i];

        double err = calculate_rel_error(testY2, testY, testRows);
        printf("  Result: %s (rel_err=%.2e)\n", err < tolerance ? "PASSED" : "FAILED", err);

        cudaFree(d_rowPtr); cudaFree(d_colIdx); cudaFree(d_values);
        cudaFree(d_x); cudaFree(d_y);
    }

    // Test 3.4: Direct general
    {
        printf("Test 3.4: Direct general (spmv_fp64_execute_direct_general, alpha=1.5, beta=0.5)\n");

        int* d_rowPtr, *d_colIdx;
        double* d_values, *d_x, *d_y;
        cudaMalloc(&d_rowPtr, (testRows + 1) * sizeof(int));
        cudaMalloc(&d_colIdx, testNnz * sizeof(int));
        cudaMalloc(&d_values, testNnz * sizeof(double));
        cudaMalloc(&d_x, testCols * sizeof(double));
        cudaMalloc(&d_y, testRows * sizeof(double));

        cudaMemcpy(d_rowPtr, testRowPtr, (testRows + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_colIdx, testColIdx, testNnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_values, testValues, testNnz * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, testX, testCols * sizeof(double), cudaMemcpyHostToDevice);

        // Initialize y_old = 1.0
        double* y_old = (double*)malloc(testRows * sizeof(double));
        for (int i = 0; i < testRows; i++) y_old[i] = 1.0;
        cudaMemcpy(d_y, y_old, testRows * sizeof(double), cudaMemcpyHostToDevice);

        double alpha = 1.5;
        double beta = 0.5;
        status = spmv_fp64_execute_direct_general(alpha, beta, testRows, testNnz,
                                                   d_rowPtr, d_colIdx, d_values, d_x, d_y, 0);
        cudaDeviceSynchronize();
        cudaMemcpy(testY, d_y, testRows * sizeof(double), cudaMemcpyDeviceToHost);

        // Reference: y = alpha * A*x + beta * y_old
        for (int i = 0; i < testRows; i++) testY2[i] = alpha * testYRef[i] + beta * y_old[i];

        double err = calculate_rel_error(testY2, testY, testRows);
        printf("  Result: %s (rel_err=%.2e)\n", err < tolerance ? "PASSED" : "FAILED", err);

        free(y_old);
        cudaFree(d_rowPtr); cudaFree(d_colIdx); cudaFree(d_values);
        cudaFree(d_x); cudaFree(d_y);
    }

    // Cleanup
    free(testRowPtr); free(testColIdx); free(testValues);
    free(testX); free(testY); free(testYRef); free(testY2);

    printf("\n========================================\n");
    printf("        All Tests Complete!\n");
    printf("========================================\n\n");

    return 0;
}