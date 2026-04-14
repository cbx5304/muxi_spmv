/**
 * @file test_new_api.cu
 * @brief Test new API interfaces for SPMV FP64 library
 */

#include "spmv_fp64.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Create a simple synthetic CSR matrix
void create_synthetic_csr(int numRows, int avgNnz,
                          int** rowPtr, int** colIdx, double** values,
                          int* nnz) {
    *nnz = numRows * avgNnz;
    *rowPtr = (int*)malloc((numRows + 1) * sizeof(int));
    *colIdx = (int*)malloc(*nnz * sizeof(int));
    *values = (double*)malloc(*nnz * sizeof(double));

    // Fill rowPtr
    (*rowPtr)[0] = 0;
    for (int i = 0; i < numRows; i++) {
        (*rowPtr)[i + 1] = (*rowPtr)[i] + avgNnz;
    }

    // Fill colIdx and values with simple pattern
    int cols = numRows;  // Square matrix
    for (int i = 0; i < *nnz; i++) {
        (*colIdx)[i] = (i % cols);  // Simple pattern
        (*values)[i] = 1.0;
    }
}

int main() {
    printf("=== SPMV FP64 New API Test ===\n");
    printf("Version: %s\n\n", spmv_fp64_get_version());

    // Get GPU info
    int warpSize;
    const char* gpuName;
    spmv_fp64_get_device_info(&warpSize, NULL, NULL);
    printf("Warp Size: %d\n", warpSize);

    // Create synthetic matrix
    int numRows = 10000;
    int avgNnz = 10;
    int* h_rowPtr, *h_colIdx;
    double* h_values;
    int nnz;

    create_synthetic_csr(numRows, avgNnz, &h_rowPtr, &h_colIdx, &h_values, &nnz);
    printf("Matrix: %d rows, %d nnz, avgNnz=%d\n\n", numRows, nnz, avgNnz);

    // Allocate host vectors
    double* h_x = (double*)malloc(numRows * sizeof(double));
    double* h_y = (double*)malloc(numRows * sizeof(double));
    for (int i = 0; i < numRows; i++) {
        h_x[i] = 1.0;
        h_y[i] = 0.0;
    }

    // ==================== Test 1: Host CSR API ====================
    printf("=== Test 1: Host CSR API (spmv_fp64_create_matrix) ===\n");

    spmv_fp64_matrix_handle_t mat1;
    spmv_fp64_opts_t opts = SPMV_FP64_BENCHMARK_OPTS;

    spmv_fp64_status_t status = spmv_fp64_create_matrix(
        &mat1, numRows, numRows, nnz, h_rowPtr, h_colIdx, h_values, &opts);

    if (status != SPMV_FP64_SUCCESS) {
        printf("Test 1 FAILED: %s\n", spmv_fp64_get_error_string(status));
        return 1;
    }

    spmv_fp64_stats_t stats1;
    status = spmv_fp64_execute(mat1, h_x, h_y, &opts, &stats1);

    if (status != SPMV_FP64_SUCCESS) {
        printf("Test 1 FAILED: %s\n", spmv_fp64_get_error_string(status));
        spmv_fp64_destroy_matrix(mat1);
        return 1;
    }

    printf("Test 1 PASSED: BW=%.1f GB/s, Util=%.1f%%\n",
           stats1.bandwidth_gbps, stats1.utilization_pct);

    // Verify result
    double sum1 = 0.0;
    for (int i = 0; i < numRows; i++) sum1 += h_y[i];
    printf("Result sum: %.1f (expected: %.1f)\n\n", sum1, (double)nnz);

    spmv_fp64_destroy_matrix(mat1);

    // ==================== Test 2: Device CSR API ====================
    printf("=== Test 2: Device CSR API (spmv_fp64_create_matrix_device) ===\n");

    // Allocate device CSR arrays
    int* d_rowPtr, *d_colIdx;
    double* d_values;
    cudaMalloc(&d_rowPtr, (numRows + 1) * sizeof(int));
    cudaMalloc(&d_colIdx, nnz * sizeof(int));
    cudaMalloc(&d_values, nnz * sizeof(double));

    cudaMemcpy(d_rowPtr, h_rowPtr, (numRows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdx, h_colIdx, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values, nnz * sizeof(double), cudaMemcpyHostToDevice);

    // Allocate device vectors
    double* d_x, *d_y;
    cudaMalloc(&d_x, numRows * sizeof(double));
    cudaMalloc(&d_y, numRows * sizeof(double));
    cudaMemcpy(d_x, h_x, numRows * sizeof(double), cudaMemcpyHostToDevice);

    // Create matrix handle with device pointers (zero-copy)
    spmv_fp64_matrix_handle_t mat2;
    status = spmv_fp64_create_matrix_device(
        &mat2, numRows, numRows, nnz, d_rowPtr, d_colIdx, d_values, &opts);

    if (status != SPMV_FP64_SUCCESS) {
        printf("Test 2 FAILED: %s\n", spmv_fp64_get_error_string(status));
        cudaFree(d_rowPtr);
        return 1;
    }

    // Execute with device vectors (no H2D/D2H copies)
    spmv_fp64_stats_t stats2;
    status = spmv_fp64_execute_device(mat2, d_x, d_y, &opts, &stats2);

    if (status != SPMV_FP64_SUCCESS) {
        printf("Test 2 FAILED: %s\n", spmv_fp64_get_error_string(status));
        spmv_fp64_destroy_matrix(mat2);
        cudaFree(d_rowPtr);
        return 1;
    }

    printf("Test 2 PASSED: BW=%.1f GB/s, Util=%.1f%%\n",
           stats2.bandwidth_gbps, stats2.utilization_pct);

    // Copy result back and verify
    cudaMemcpy(h_y, d_y, numRows * sizeof(double), cudaMemcpyDeviceToHost);
    double sum2 = 0.0;
    for (int i = 0; i < numRows; i++) sum2 += h_y[i];
    printf("Result sum: %.1f (expected: %.1f)\n\n", sum2, (double)nnz);

    // Clean up (user manages device memory in zero-copy mode)
    spmv_fp64_destroy_matrix(mat2);
    cudaFree(d_rowPtr);
    cudaFree(d_colIdx);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);

    // ==================== Test 3: Direct Execution API ====================
    printf("=== Test 3: Direct Execution API (spmv_fp64_execute_direct) ===\n");

    // Re-allocate device CSR and vectors
    cudaMalloc(&d_rowPtr, (numRows + 1) * sizeof(int));
    cudaMalloc(&d_colIdx, nnz * sizeof(int));
    cudaMalloc(&d_values, nnz * sizeof(double));
    cudaMalloc(&d_x, numRows * sizeof(double));
    cudaMalloc(&d_y, numRows * sizeof(double));

    cudaMemcpy(d_rowPtr, h_rowPtr, (numRows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdx, h_colIdx, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values, nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, numRows * sizeof(double), cudaMemcpyHostToDevice);

    // Execute directly without handle creation
    status = spmv_fp64_execute_direct(numRows, nnz, d_rowPtr, d_colIdx, d_values, d_x, d_y, 0);

    if (status != SPMV_FP64_SUCCESS) {
        printf("Test 3 FAILED: %s\n", spmv_fp64_get_error_string(status));
        cudaFree(d_rowPtr);
        return 1;
    }

    cudaDeviceSynchronize();

    // Copy result back and verify
    cudaMemcpy(h_y, d_y, numRows * sizeof(double), cudaMemcpyDeviceToHost);
    double sum3 = 0.0;
    for (int i = 0; i < numRows; i++) sum3 += h_y[i];
    printf("Test 3 PASSED: Result sum=%.1f (expected: %.1f)\n\n", sum3, (double)nnz);

    cudaFree(d_rowPtr);
    cudaFree(d_colIdx);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);

    // ==================== Test 4: Error Handling ====================
    printf("=== Test 4: Error Handling ===\n");

    // Try to call execute (host mode) on a device-only matrix
    cudaMalloc(&d_rowPtr, (numRows + 1) * sizeof(int));
    cudaMalloc(&d_colIdx, nnz * sizeof(int));
    cudaMalloc(&d_values, nnz * sizeof(double));

    spmv_fp64_matrix_handle_t mat4;
    spmv_fp64_create_matrix_device(&mat4, numRows, numRows, nnz, d_rowPtr, d_colIdx, d_values, &opts);

    // This should fail because device-only matrix doesn't have internal vectors
    status = spmv_fp64_execute(mat4, h_x, h_y, &opts, NULL);

    if (status == SPMV_FP64_ERROR_NOT_SUPPORTED) {
        printf("Test 4 PASSED: Correctly returned ERROR_NOT_SUPPORTED for host execute on device-only matrix\n");
    } else {
        printf("Test 4 FAILED: Expected ERROR_NOT_SUPPORTED, got %d\n", status);
    }

    spmv_fp64_destroy_matrix(mat4);
    cudaFree(d_rowPtr);
    cudaFree(d_colIdx);
    cudaFree(d_values);

    // Cleanup
    free(h_rowPtr);
    free(h_colIdx);
    free(h_values);
    free(h_x);
    free(h_y);

    printf("\n=== All Tests Complete ===\n");
    return 0;
}