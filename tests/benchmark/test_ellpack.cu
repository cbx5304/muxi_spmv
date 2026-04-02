/**
 * @file test_ellpack.cu
 * @brief Quick test for ELLPACK format performance
 */

#include "utils/common.h"
#include "formats/sparse_formats.h"
#include "generators/matrix_generator.h"
#include "spmv/csr/spmv_csr.cuh"
#include "spmv/ellpack/spmv_ellpack.cuh"

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

using namespace muxi_spmv;
using namespace muxi_spmv::generators;

template<typename FloatType>
void testEllpack(int numRows, int numCols, double sparsity) {
    printf("=== ELLPACK Format Test ===\n");
    printf("Matrix: %d x %d, Sparsity: %.4f%%\n\n", numRows, numCols, sparsity * 100);

    // Generate random sparse matrix
    MatrixGenConfig config;
    config.numRows = numRows;
    config.numCols = numCols;
    config.sparsity = sparsity;
    config.type = MatrixType::RANDOM_UNIFORM;

    MatrixGenerator<FloatType>* generator = createGenerator<FloatType>(MatrixType::RANDOM_UNIFORM);
    CSRMatrix<FloatType> csrMatrix;
    generator->generate(config, csrMatrix);
    delete generator;

    printf("Generated CSR matrix: %d NNZ, avgNnzPerRow = %.1f\n",
           csrMatrix.nnz, (double)csrMatrix.nnz / csrMatrix.numRows);

    // Allocate device memory for CSR
    csrMatrix.allocateDevice();
    csrMatrix.copyToDevice();

    // Convert to ELLPACK
    ELLPACKMatrix<FloatType> ellpackMatrix;
    csr_to_ellpack(csrMatrix, ellpackMatrix);

    int totalElements = ellpackMatrix.numRows * ellpackMatrix.numElements;
    printf("ELLPACK matrix: %d elements per row, total = %d, overhead = %.1f%%\n",
           ellpackMatrix.numElements, totalElements,
           100.0 * (totalElements - csrMatrix.nnz) / csrMatrix.nnz);

    // Allocate vectors
    FloatType* d_x;
    FloatType* d_y_csr;
    FloatType* d_y_ellpack;
    cudaMalloc(&d_x, numCols * sizeof(FloatType));
    cudaMalloc(&d_y_csr, numRows * sizeof(FloatType));
    cudaMalloc(&d_y_ellpack, numRows * sizeof(FloatType));

    // Initialize x
    FloatType* h_x = new FloatType[numCols];
    for (int i = 0; i < numCols; i++) {
        h_x[i] = static_cast<FloatType>(1.0);
    }
    cudaMemcpy(d_x, h_x, numCols * sizeof(FloatType), cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Test CSR performance
    printf("\n--- CSR SpMV ---\n");
    spmv_opts_t opts = spmv_default_opts();

    // Warmup
    for (int i = 0; i < 5; i++) {
        spmv_csr(csrMatrix, d_x, d_y_csr, 1.0f, 0.0f, opts);
    }
    cudaDeviceSynchronize();

    // Measure CSR
    std::vector<float> csrTimes;
    for (int i = 0; i < 20; i++) {
        cudaEventRecord(start);
        spmv_csr(csrMatrix, d_x, d_y_csr, 1.0f, 0.0f, opts);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float timeMs;
        cudaEventElapsedTime(&timeMs, start, stop);
        csrTimes.push_back(timeMs);
    }

    std::sort(csrTimes.begin(), csrTimes.end());
    float csrMedian = csrTimes[csrTimes.size() / 2];
    printf("CSR Time (median): %.4f ms\n", csrMedian);
    printf("CSR Bandwidth: %.2f GB/s\n",
           (csrMatrix.nnz * sizeof(FloatType) * 3 + csrMatrix.numRows * sizeof(int)) / csrMedian / 1e6);

    // Test ELLPACK performance
    printf("\n--- ELLPACK SpMV ---\n");

    // Warmup
    for (int i = 0; i < 5; i++) {
        spmv_ellpack(ellpackMatrix, d_x, d_y_ellpack, 0);
    }
    cudaDeviceSynchronize();

    // Measure ELLPACK
    std::vector<float> ellpackTimes;
    for (int i = 0; i < 20; i++) {
        cudaEventRecord(start);
        spmv_ellpack(ellpackMatrix, d_x, d_y_ellpack, 0);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float timeMs;
        cudaEventElapsedTime(&timeMs, start, stop);
        ellpackTimes.push_back(timeMs);
    }

    std::sort(ellpackTimes.begin(), ellpackTimes.end());
    float ellpackMedian = ellpackTimes[ellpackTimes.size() / 2];
    printf("ELLPACK Time (median): %.4f ms\n", ellpackMedian);
    printf("ELLPACK Bandwidth: %.2f GB/s\n",
           (totalElements * sizeof(FloatType) * 2 + totalElements * sizeof(int)) / ellpackMedian / 1e6);

    // Compare
    printf("\n--- Comparison ---\n");
    printf("Speedup: %.2fx\n", csrMedian / ellpackMedian);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_x);
    cudaFree(d_y_csr);
    cudaFree(d_y_ellpack);
    delete[] h_x;

    // Free ELLPACK
    if (ellpackMatrix.d_colIdx) cudaFree(ellpackMatrix.d_colIdx);
    if (ellpackMatrix.d_values) cudaFree(ellpackMatrix.d_values);
}

int main(int argc, char** argv) {
    int numRows = 1000000;
    int numCols = 1000;
    double sparsity = 0.01;

    // Parse simple args
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--rows") == 0 && i + 1 < argc) {
            numRows = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--cols") == 0 && i + 1 < argc) {
            numCols = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--sparsity") == 0 && i + 1 < argc) {
            sparsity = atof(argv[++i]);
        }
    }

    testEllpack<float>(numRows, numCols, sparsity);

    return 0;
}