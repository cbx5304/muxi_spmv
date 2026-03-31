/**
 * @file test_spmv.cu
 * @brief Basic SpMV test
 */

#include "spmv/csr/spmv_csr.cuh"
#include "api/spmv_api.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>

using namespace muxi_spmv;

// Generate a simple test matrix (diagonal)
void generateDiagonalMatrix(int n, CSRMatrix<float>& matrix) {
    matrix.numRows = n;
    matrix.numCols = n;
    matrix.nnz = n;

    matrix.allocateHost(n, n, n);

    for (int i = 0; i <= n; i++) {
        matrix.rowPtr[i] = i;
    }

    for (int i = 0; i < n; i++) {
        matrix.colIdx[i] = i;
        matrix.values[i] = 1.0f;
    }
}

// Verify result
bool verifyResult(const float* y, const float* expected, int n, float tol = 1e-5f) {
    for (int i = 0; i < n; i++) {
        float diff = fabsf(y[i] - expected[i]);
        if (diff > tol) {
            printf("Mismatch at index %d: got %f, expected %f\n", i, y[i], expected[i]);
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    printf("=== SpMV CSR Basic Test ===\n\n");

    int n = 1024;
    if (argc > 1) {
        n = atoi(argv[1]);
    }

    printf("Testing with %dx%d diagonal matrix\n", n, n);

    // Generate test matrix
    CSRMatrix<float> matrix;
    generateDiagonalMatrix(n, matrix);

    // Allocate and initialize vectors
    float* h_x = (float*)malloc(n * sizeof(float));
    float* h_y = (float*)malloc(n * sizeof(float));
    float* h_expected = (float*)malloc(n * sizeof(float));

    for (int i = 0; i < n; i++) {
        h_x[i] = static_cast<float>(i + 1);
        h_y[i] = 0.0f;
        h_expected[i] = h_x[i];  // Diagonal matrix * x = x
    }

    // Allocate device memory
    matrix.allocateDevice();
    float *d_x, *d_y;
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_y, n * sizeof(float));

    // Copy to device
    matrix.copyToDevice();
    cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, n * sizeof(float), cudaMemcpyHostToDevice);

    // Create handle and execute
    spmv_handle_t* handle;
    spmv_status_t status = spmv_create_handle(&handle);
    if (status != SPMV_SUCCESS) {
        printf("Failed to create handle: %d\n", status);
        return 1;
    }

    printf("Device: Warp size = %d\n", handle->warpSize);

    // Execute SpMV
    spmv_opts_t opts = spmv_default_opts();
    opts.sync = 1;

    status = spmv_csr<float>(matrix, d_x, d_y, 1.0f, 0.0f, opts);
    if (status != SPMV_SUCCESS) {
        printf("SpMV failed: %d\n", status);
        return 1;
    }

    // Copy result back
    cudaMemcpy(h_y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify
    bool passed = verifyResult(h_y, h_expected, n);
    printf("\nTest: %s\n", passed ? "PASSED" : "FAILED");

    // Cleanup
    spmv_destroy_handle(handle);
    cudaFree(d_x);
    cudaFree(d_y);
    free(h_x);
    free(h_y);
    free(h_expected);

    return passed ? 0 : 1;
}