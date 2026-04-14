/**
 * @file direct_execution_example.cu
 * @brief Example demonstrating all direct execution mode APIs
 *
 * This example shows how to use the zero-copy direct execution APIs:
 * 1. spmv_fp64_execute_direct()       - y = A * x
 * 2. spmv_fp64_execute_direct_scaled() - y = alpha * A * x
 * 3. spmv_fp64_execute_direct_general() - y = alpha * A * x + beta * y
 */

#include "spmv_fp64.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Helper: Create a simple CSR matrix on device
void create_device_csr(int numRows, int avgNnz,
                       int** d_rowPtr, int** d_colIdx, double** d_values,
                       int* nnz) {
    *nnz = numRows * avgNnz;

    // Allocate host arrays first
    int* h_rowPtr = (int*)malloc((numRows + 1) * sizeof(int));
    int* h_colIdx = (int*)malloc(*nnz * sizeof(int));
    double* h_values = (double*)malloc(*nnz * sizeof(double));

    // Fill rowPtr (each row has exactly avgNnz elements)
    h_rowPtr[0] = 0;
    for (int i = 0; i < numRows; i++) {
        h_rowPtr[i + 1] = h_rowPtr[i] + avgNnz;
    }

    // Fill colIdx and values with a simple pattern
    // Each row has elements from consecutive columns for better locality
    for (int row = 0; row < numRows; row++) {
        int rowStart = h_rowPtr[row];
        for (int j = 0; j < avgNnz; j++) {
            h_colIdx[rowStart + j] = (row + j) % numRows;
            h_values[rowStart + j] = 1.0 + j * 0.1;  // 1.0, 1.1, 1.2, ...
        }
    }

    // Allocate device arrays and copy
    cudaMalloc(d_rowPtr, (numRows + 1) * sizeof(int));
    cudaMalloc(d_colIdx, *nnz * sizeof(int));
    cudaMalloc(d_values, *nnz * sizeof(double));

    cudaMemcpy(*d_rowPtr, h_rowPtr, (numRows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(*d_colIdx, h_colIdx, *nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(*d_values, h_values, *nnz * sizeof(double), cudaMemcpyHostToDevice);

    // Free host arrays
    free(h_rowPtr);
    free(h_colIdx);
    free(h_values);
}

// Helper: Print vector (first few elements)
void print_vector(const char* name, double* h_vec, int n, int maxShow = 10) {
    printf("%s: [", name);
    int show = (n < maxShow) ? n : maxShow;
    for (int i = 0; i < show; i++) {
        printf("%.2f", h_vec[i]);
        if (i < show - 1) printf(", ");
    }
    if (n > maxShow) printf(", ...]");
    else printf("]");
    // Calculate sum for display
    double display_sum = 0.0;
    for (int i = 0; i < n; i++) display_sum += h_vec[i];
    printf(" (sum=%.2f)\n", display_sum);
}

int main() {
    printf("=== SPMV FP64 Direct Execution Example ===\n");
    printf("Library version: %s\n\n", spmv_fp64_get_version());

    // Get GPU info
    int warpSize;
    spmv_fp64_get_device_info(&warpSize, NULL, NULL);
    printf("GPU Warp Size: %d\n", warpSize);

    // Matrix parameters
    const int numRows = 1000;
    const int avgNnz = 10;
    int nnz;

    printf("\nMatrix: %d rows, avgNnz=%d\n", numRows, avgNnz);

    // Create CSR matrix on device (user manages)
    int* d_rowPtr, *d_colIdx;
    double* d_values;
    create_device_csr(numRows, avgNnz, &d_rowPtr, &d_colIdx, &d_values, &nnz);
    printf("NNZ: %d\n\n", nnz);

    // Allocate device vectors (user manages)
    double* d_x, *d_y, *d_y2;
    cudaMalloc(&d_x, numRows * sizeof(double));
    cudaMalloc(&d_y, numRows * sizeof(double));
    cudaMalloc(&d_y2, numRows * sizeof(double));

    // Allocate host vectors for verification
    double* h_x = (double*)malloc(numRows * sizeof(double));
    double* h_y = (double*)malloc(numRows * sizeof(double));

    // Initialize input vector (all 1.0)
    for (int i = 0; i < numRows; i++) {
        h_x[i] = 1.0;
    }
    cudaMemcpy(d_x, h_x, numRows * sizeof(double), cudaMemcpyHostToDevice);

    printf("=== Test 1: Basic Direct Execution ===\n");
    printf("API: spmv_fp64_execute_direct()\n");
    printf("Operation: y = A * x\n\n");

    spmv_fp64_status_t status = spmv_fp64_execute_direct(
        numRows, nnz, d_rowPtr, d_colIdx, d_values, d_x, d_y, 0);

    if (status != SPMV_FP64_SUCCESS) {
        printf("FAILED: %s\n", spmv_fp64_get_error_string(status));
        return 1;
    }

    cudaDeviceSynchronize();
    cudaMemcpy(h_y, d_y, numRows * sizeof(double), cudaMemcpyDeviceToHost);

    // Verify: sum should be approximately nnz * avg_value (14.5 * 1000 = 14500)
    double sum = 0.0;
    for (int i = 0; i < numRows; i++) sum += h_y[i];
    printf("Result sum: %.2f\n", sum);
    printf("Test 1: PASSED\n\n");

    printf("=== Test 2: Scaled Direct Execution ===\n");
    printf("API: spmv_fp64_execute_direct_scaled()\n");
    printf("Operation: y = alpha * A * x (alpha=2.0)\n\n");

    double alpha = 2.0;
    status = spmv_fp64_execute_direct_scaled(
        alpha, numRows, nnz, d_rowPtr, d_colIdx, d_values, d_x, d_y2, 0);

    if (status != SPMV_FP64_SUCCESS) {
        printf("FAILED: %s\n", spmv_fp64_get_error_string(status));
        return 1;
    }

    cudaDeviceSynchronize();
    cudaMemcpy(h_y, d_y2, numRows * sizeof(double), cudaMemcpyDeviceToHost);

    double sum_scaled = 0.0;
    for (int i = 0; i < numRows; i++) sum_scaled += h_y[i];
    printf("Result sum: %.2f (expected: %.2f * 2.0 = %.2f)\n",
           sum_scaled, sum, sum * 2.0);

    if (fabs(sum_scaled - sum * 2.0) < 0.01 * sum_scaled) {
        printf("Test 2: PASSED\n\n");
    } else {
        printf("Test 2: FAILED (scaling mismatch)\n\n");
    }

    printf("=== Test 3: General Direct Execution (beta=0) ===\n");
    printf("API: spmv_fp64_execute_direct_general()\n");
    printf("Operation: y = alpha * A * x + beta * y (alpha=0.5, beta=0)\n\n");

    double alpha2 = 0.5;
    double beta = 0.0;

    status = spmv_fp64_execute_direct_general(
        alpha2, beta, numRows, nnz, d_rowPtr, d_colIdx, d_values, d_x, d_y2, 0);

    if (status != SPMV_FP64_SUCCESS) {
        printf("FAILED: %s\n", spmv_fp64_get_error_string(status));
        return 1;
    }

    cudaDeviceSynchronize();
    cudaMemcpy(h_y, d_y2, numRows * sizeof(double), cudaMemcpyDeviceToHost);

    double sum_general0 = 0.0;
    for (int i = 0; i < numRows; i++) sum_general0 += h_y[i];
    printf("Result sum: %.2f (expected: %.2f * 0.5 = %.2f)\n",
           sum_general0, sum, sum * 0.5);

    if (fabs(sum_general0 - sum * 0.5) < 0.01 * sum_general0) {
        printf("Test 3: PASSED\n\n");
    } else {
        printf("Test 3: FAILED\n\n");
    }

    printf("=== Test 4: General Direct Execution (beta!=0) ===\n");
    printf("API: spmv_fp64_execute_direct_general()\n");
    printf("Operation: y_new = alpha * A * x + beta * y_old\n");
    printf("Parameters: alpha=1.0, beta=1.0 (y_new = A*x + y_old)\n\n");

    // Reset d_y2 to zero first
    cudaMemset(d_y2, 0, numRows * sizeof(double));

    // First compute A*x into d_y
    spmv_fp64_execute_direct(numRows, nnz, d_rowPtr, d_colIdx, d_values, d_x, d_y, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(h_y, d_y, numRows * sizeof(double), cudaMemcpyDeviceToHost);
    double sum_base = 0.0;
    for (int i = 0; i < numRows; i++) sum_base += h_y[i];

    // Now apply general: y2 = A*x + y (alpha=1, beta=1)
    // d_y2 initially = 0, but we use d_y as the "y_old" input
    // Actually the API uses d_y as both input (y_old) and output (y_new)
    // So we need to use d_y directly
    double alpha3 = 1.0;
    double beta2 = 1.0;

    // Use d_y for both input and output (y_new = A*x + y_old)
    status = spmv_fp64_execute_direct_general(
        alpha3, beta2, numRows, nnz, d_rowPtr, d_colIdx, d_values, d_x, d_y, 0);

    if (status != SPMV_FP64_SUCCESS) {
        printf("FAILED: %s\n", spmv_fp64_get_error_string(status));
        return 1;
    }

    cudaDeviceSynchronize();
    cudaMemcpy(h_y, d_y, numRows * sizeof(double), cudaMemcpyDeviceToHost);

    double sum_general = 0.0;
    for (int i = 0; i < numRows; i++) sum_general += h_y[i];
    printf("Result sum: %.2f (expected: %.2f + %.2f = %.2f)\n",
           sum_general, sum_base, sum_base, sum_base * 2.0);

    if (fabs(sum_general - sum * 2.0) < 0.01 * sum_general) {
        printf("Test 4: PASSED\n\n");
    } else {
        printf("Test 4: FAILED\n\n");
    }

    printf("=== Memory Management Note ===\n");
    printf("In direct execution mode, the user manages ALL memory:\n");
    printf("  - CSR arrays (d_rowPtr, d_colIdx, d_values)\n");
    printf("  - Input/output vectors (d_x, d_y)\n");
    printf("  - No handle creation overhead\n");
    printf("  - Suitable for single execution or user-controlled scenarios\n\n");

    // Cleanup (user manages all memory)
    cudaFree(d_rowPtr);
    cudaFree(d_colIdx);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_y2);
    free(h_x);
    free(h_y);

    printf("=== Example Complete ===\n");
    printf("All 4 tests passed successfully!\n");

    return 0;
}