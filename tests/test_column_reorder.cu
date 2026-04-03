/**
 * @file test_column_reorder.cu
 * @brief Test column reordering optimization
 *
 * This test verifies that column reordering can improve SpMV performance
 * by improving x-vector cache hit rate.
 */

#include <iostream>
#include <vector>
#include <cmath>

#include "formats/sparse_formats.h"
#include "spmv/csr/spmv_csr.cuh"
#include "spmv/csr5/spmv_csr5.cuh"
#include "generators/matrix_generator.h"
#include "preprocessing/column_reorder.cuh"

using namespace muxi_spmv;
using namespace muxi_spmv::generators;

// GPU Timer
class GpuTimer {
public:
    GpuTimer() { cudaEventCreate(&start_); cudaEventCreate(&stop_); }
    ~GpuTimer() { cudaEventDestroy(start_); cudaEventDestroy(stop_); }
    void start() { cudaEventRecord(start_, 0); }
    void stop() { cudaEventRecord(stop_, 0); cudaEventSynchronize(stop_); }
    float elapsed_ms() { float ms; cudaEventElapsedTime(&ms, start_, stop_); return ms; }
private:
    cudaEvent_t start_, stop_;
};

void printSeparator() {
    std::cout << "========================================\n";
}

int main(int argc, char** argv) {
    printSeparator();
    std::cout << "Column Reordering Optimization Test\n";
    std::cout << "WARP_SIZE = " << WARP_SIZE << "\n";
    printSeparator();

    int rows = 1000000;
    int cols = 1000;
    int avgNnz = 10;
    int iterations = 30;

    if (argc >= 4) {
        rows = atoi(argv[1]);
        cols = atoi(argv[2]);
        avgNnz = atoi(argv[3]);
    }

    std::cout << "\nConfiguration:\n";
    std::cout << "  Rows: " << rows << "\n";
    std::cout << "  Columns: " << cols << "\n";
    std::cout << "  Avg NNZ/Row: " << avgNnz << "\n";
    std::cout << "  Iterations: " << iterations << "\n";

    // Generate random matrix
    std::cout << "\nGenerating random matrix...\n";
    CSRMatrix<float> matrix;
    int nnz = rows * avgNnz;
    generateRandomMatrix<float>(rows, cols, nnz, matrix);
    matrix.copyToDevice();

    std::cout << "  Actual NNZ: " << matrix.nnz << "\n";

    // Generate x vector
    float* h_x = new float[cols];
    float* h_y_ref = new float[rows];
    for (int i = 0; i < cols; i++) h_x[i] = rand() / (float)RAND_MAX;

    // Compute reference result on CPU
    std::cout << "\nComputing CPU reference...\n";
    int* h_rowPtr = new int[rows + 1];
    int* h_colIdx = new int[matrix.nnz];
    float* h_values = new float[matrix.nnz];

    cudaMemcpy(h_rowPtr, matrix.d_rowPtr, (rows + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_colIdx, matrix.d_colIdx, matrix.nnz * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_values, matrix.d_values, matrix.nnz * sizeof(float), cudaMemcpyDeviceToHost);

    for (int row = 0; row < rows; row++) {
        float sum = 0;
        for (int idx = h_rowPtr[row]; idx < h_rowPtr[row + 1]; idx++) {
            sum += h_values[idx] * h_x[h_colIdx[idx]];
        }
        h_y_ref[row] = sum;
    }

    // Test 1: Original matrix
    std::cout << "\n=== Test 1: Original Random Matrix ===\n";

    float* d_x, *d_y;
    cudaMalloc(&d_x, cols * sizeof(float));
    cudaMalloc(&d_y, rows * sizeof(float));
    cudaMemcpy(d_x, h_x, cols * sizeof(float), cudaMemcpyHostToDevice);

    GpuTimer timer;
    float totalTime = 0;

    // Warmup
    cudaMemset(d_y, 0, rows * sizeof(float));
    spmv_merge_based<float>(matrix, d_x, d_y, 0);
    cudaDeviceSynchronize();

    for (int i = 0; i < iterations; i++) {
        cudaMemset(d_y, 0, rows * sizeof(float));
        timer.start();
        spmv_merge_based<float>(matrix, d_x, d_y, 0);
        timer.stop();
        totalTime += timer.elapsed_ms();
    }

    float avgTimeOriginal = totalTime / iterations;
    size_t dataBytes = rows * sizeof(int) * 2 + matrix.nnz * sizeof(int) +
                       matrix.nnz * sizeof(float) * 2 + rows * sizeof(float);
    float peakBW = (WARP_SIZE == 64) ? 1843.0f : 1008.0f;
    float bwOriginal = (dataBytes / (avgTimeOriginal * 1e-3)) / (1024 * 1024 * 1024);
    float utilOriginal = bwOriginal / peakBW * 100;

    std::cout << "  Time: " << avgTimeOriginal << " ms\n";
    std::cout << "  Bandwidth: " << bwOriginal << " GB/s\n";
    std::cout << "  Utilization: " << utilOriginal << " %\n";

    // Test 2: Sorted columns within rows
    std::cout << "\n=== Test 2: Sorted Columns Within Rows ===\n";

    CSRMatrix<float> sortedMatrix;
    sortedMatrix.numRows = rows;
    sortedMatrix.numCols = cols;
    sortedMatrix.nnz = matrix.nnz;
    sortedMatrix.allocateHost(rows, cols, matrix.nnz);

    // Copy data
    for (int i = 0; i <= rows; i++) sortedMatrix.rowPtr[i] = h_rowPtr[i];
    for (int i = 0; i < matrix.nnz; i++) {
        sortedMatrix.colIdx[i] = h_colIdx[i];
        sortedMatrix.values[i] = h_values[i];
    }

    // Sort columns within each row
    sortColumnsWithinRows(sortedMatrix);
    sortedMatrix.copyToDevice();

    totalTime = 0;
    for (int i = 0; i < iterations; i++) {
        cudaMemset(d_y, 0, rows * sizeof(float));
        timer.start();
        spmv_merge_based<float>(sortedMatrix, d_x, d_y, 0);
        timer.stop();
        totalTime += timer.elapsed_ms();
    }

    float avgTimeSorted = totalTime / iterations;
    float bwSorted = (dataBytes / (avgTimeSorted * 1e-3)) / (1024 * 1024 * 1024);
    float utilSorted = bwSorted / peakBW * 100;

    std::cout << "  Time: " << avgTimeSorted << " ms\n";
    std::cout << "  Bandwidth: " << bwSorted << " GB/s\n";
    std::cout << "  Utilization: " << utilSorted << " %\n";
    std::cout << "  Improvement: " << ((avgTimeOriginal - avgTimeSorted) / avgTimeOriginal * 100) << " %\n";

    // Verify correctness
    cudaMemset(d_y, 0, rows * sizeof(float));
    spmv_merge_based<float>(sortedMatrix, d_x, d_y, 0);
    cudaDeviceSynchronize();

    float* h_y = new float[rows];
    cudaMemcpy(h_y, d_y, rows * sizeof(float), cudaMemcpyDeviceToHost);

    bool correct = true;
    for (int i = 0; i < rows && correct; i++) {
        float diff = fabs(h_y[i] - h_y_ref[i]);
        float tol = max(fabs(h_y_ref[i]) * 1e-4f, 1e-6f);
        if (diff > tol) correct = false;
    }
    std::cout << "  Correctness: " << (correct ? "PASS" : "FAIL") << "\n";

    // Test 3: Banded matrix (for comparison)
    std::cout << "\n=== Test 3: Banded Matrix (Reference) ===\n";

    CSRMatrix<float> bandedMatrix;
    generateBandedMatrix<float>(rows, 10, bandedMatrix);  // bandwidth=10
    bandedMatrix.copyToDevice();

    totalTime = 0;
    for (int i = 0; i < iterations; i++) {
        cudaMemset(d_y, 0, rows * sizeof(float));
        timer.start();
        spmv_merge_based<float>(bandedMatrix, d_x, d_y, 0);
        timer.stop();
        totalTime += timer.elapsed_ms();
    }

    float avgTimeBanded = totalTime / iterations;
    size_t bandedDataBytes = rows * sizeof(int) * 2 + bandedMatrix.nnz * sizeof(int) +
                             bandedMatrix.nnz * sizeof(float) * 2 + rows * sizeof(float);
    float bwBanded = (bandedDataBytes / (avgTimeBanded * 1e-3)) / (1024 * 1024 * 1024);
    float utilBanded = bwBanded / peakBW * 100;

    std::cout << "  Time: " << avgTimeBanded << " ms\n";
    std::cout << "  Bandwidth: " << bwBanded << " GB/s\n";
    std::cout << "  Utilization: " << utilBanded << " %\n";
    std::cout << "  NNZ: " << bandedMatrix.nnz << "\n";

    // Summary
    printSeparator();
    std::cout << "Summary\n";
    printSeparator();

    std::cout << "\n| Matrix Type | Time (ms) | BW (GB/s) | Util (%) |\n";
    std::cout << "|-------------|-----------|-----------|----------|\n";
    std::cout << "| Random | " << avgTimeOriginal << " | " << bwOriginal << " | " << utilOriginal << " |\n";
    std::cout << "| Sorted cols | " << avgTimeSorted << " | " << bwSorted << " | " << utilSorted << " |\n";
    std::cout << "| Banded | " << avgTimeBanded << " | " << bwBanded << " | " << utilBanded << " |\n";

    std::cout << "\nKey Insight:\n";
    std::cout << "  - Banded matrix achieves " << (utilBanded / utilOriginal) << "x better utilization\n";
    std::cout << "  - Column sorting improvement: " << ((avgTimeOriginal - avgTimeSorted) / avgTimeOriginal * 100) << " %\n";
    std::cout << "  - Potential for column reordering: " << (utilBanded - utilOriginal) << " % utilization gap\n";

    // Cleanup
    delete[] h_x;
    delete[] h_y_ref;
    delete[] h_rowPtr;
    delete[] h_colIdx;
    delete[] h_values;
    delete[] h_y;
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}