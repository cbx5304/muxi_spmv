/**
 * @file test_global_reorder.cu
 * @brief Test global column reordering optimization
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

int main(int argc, char** argv) {
    int rows = 1000000;
    int cols = 1000;
    int avgNnz = 10;
    int iterations = 20;

    if (argc >= 4) {
        rows = atoi(argv[1]);
        cols = atoi(argv[2]);
        avgNnz = atoi(argv[3]);
    }

    std::cout << "=== Global Column Reordering Test ===\n";
    std::cout << "Rows: " << rows << ", Cols: " << cols << ", avgNnz: " << avgNnz << "\n\n";

    // Generate matrix
    CSRMatrix<float> matrix;
    int nnz = rows * avgNnz;
    generateRandomMatrix<float>(rows, cols, nnz, matrix);

    // Copy to host for preprocessing
    int* h_rowPtr = new int[rows + 1];
    int* h_colIdx = new int[matrix.nnz];
    float* h_values = new float[matrix.nnz];

    // Need to copy from matrix to host first
    matrix.allocateDevice();
    matrix.copyToDevice();
    cudaDeviceSynchronize();

    cudaMemcpy(h_rowPtr, matrix.d_rowPtr, (rows + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_colIdx, matrix.d_colIdx, matrix.nnz * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_values, matrix.d_values, matrix.nnz * sizeof(float), cudaMemcpyDeviceToHost);

    // Set host pointers for preprocessing
    matrix.rowPtr = h_rowPtr;
    matrix.colIdx = h_colIdx;
    matrix.values = h_values;

    // Generate x vector
    float* h_x = new float[cols];
    for (int i = 0; i < cols; i++) h_x[i] = rand() / (float)RAND_MAX;

    // Test 1: Original matrix
    std::cout << "=== Test 1: Original Matrix ===\n";
    CSRMatrix<float> origMatrix;
    origMatrix.numRows = rows;
    origMatrix.numCols = cols;
    origMatrix.nnz = matrix.nnz;
    origMatrix.allocateHost(rows, cols, matrix.nnz);
    memcpy(origMatrix.rowPtr, h_rowPtr, (rows + 1) * sizeof(int));
    memcpy(origMatrix.colIdx, h_colIdx, matrix.nnz * sizeof(int));
    memcpy(origMatrix.values, h_values, matrix.nnz * sizeof(float));
    origMatrix.allocateDevice();
    origMatrix.copyToDevice();
    cudaDeviceSynchronize();

    float* d_x, *d_y;
    cudaMalloc(&d_x, cols * sizeof(float));
    cudaMalloc(&d_y, rows * sizeof(float));
    cudaMemcpy(d_x, h_x, cols * sizeof(float), cudaMemcpyHostToDevice);

    GpuTimer timer;
    float totalTime = 0;
    for (int i = 0; i < iterations; i++) {
        cudaMemset(d_y, 0, rows * sizeof(float));
        timer.start();
        spmv_merge_based<float>(origMatrix, d_x, d_y, 0);
        timer.stop();
        totalTime += timer.elapsed_ms();
    }

    float avgTime1 = totalTime / iterations;
    float peakBW = (WARP_SIZE == 64) ? 1843.0f : 1008.0f;
    size_t dataBytes = rows * sizeof(int) * 2 + matrix.nnz * sizeof(int) + matrix.nnz * sizeof(float) * 2 + rows * sizeof(float);
    float bw1 = (dataBytes / (avgTime1 * 1e-3)) / (1024 * 1024 * 1024);
    float util1 = bw1 / peakBW * 100;

    std::cout << "Time: " << avgTime1 << " ms, BW: " << bw1 << " GB/s, Util: " << util1 << "%\n\n";

    // Test 2: Simple column sort within rows
    std::cout << "=== Test 2: Sorted Columns Within Rows ===\n";
    CSRMatrix<float> sortedMatrix;
    sortedMatrix.numRows = rows;
    sortedMatrix.numCols = cols;
    sortedMatrix.nnz = matrix.nnz;
    sortedMatrix.allocateHost(rows, cols, matrix.nnz);
    memcpy(sortedMatrix.rowPtr, h_rowPtr, (rows + 1) * sizeof(int));
    memcpy(sortedMatrix.colIdx, h_colIdx, matrix.nnz * sizeof(int));
    memcpy(sortedMatrix.values, h_values, matrix.nnz * sizeof(float));

    sortColumnsWithinRows(sortedMatrix);
    sortedMatrix.allocateDevice();
    sortedMatrix.copyToDevice();
    cudaDeviceSynchronize();

    totalTime = 0;
    for (int i = 0; i < iterations; i++) {
        cudaMemset(d_y, 0, rows * sizeof(float));
        timer.start();
        spmv_merge_based<float>(sortedMatrix, d_x, d_y, 0);
        timer.stop();
        totalTime += timer.elapsed_ms();
    }

    float avgTime2 = totalTime / iterations;
    float bw2 = (dataBytes / (avgTime2 * 1e-3)) / (1024 * 1024 * 1024);
    float util2 = bw2 / peakBW * 100;

    std::cout << "Time: " << avgTime2 << " ms, BW: " << bw2 << " GB/s, Util: " << util2 << "%\n";
    std::cout << "Improvement: " << ((avgTime1 - avgTime2) / avgTime1 * 100) << "%\n\n";

    // Test 3: Global column reordering
    std::cout << "=== Test 3: Global Column Reordering ===\n";
    CSRMatrix<float> reorderedMatrix;
    std::vector<int> forwardMap;
    ColumnReorderStats stats;

    reorderColumns(matrix, reorderedMatrix, forwardMap, stats);

    std::cout << "Avg distance before: " << stats.avgDistanceBefore << "\n";
    std::cout << "Avg distance after: " << stats.avgDistanceAfter << "\n";
    std::cout << "Improvement ratio: " << stats.improvementRatio << "x\n";

    // Reorder x vector
    float* h_x_reordered = new float[cols];
    reorderXVector(h_x, h_x_reordered, forwardMap);

    reorderedMatrix.allocateDevice();
    reorderedMatrix.copyToDevice();
    cudaDeviceSynchronize();

    float* d_x_reordered;
    cudaMalloc(&d_x_reordered, cols * sizeof(float));
    cudaMemcpy(d_x_reordered, h_x_reordered, cols * sizeof(float), cudaMemcpyHostToDevice);

    totalTime = 0;
    for (int i = 0; i < iterations; i++) {
        cudaMemset(d_y, 0, rows * sizeof(float));
        timer.start();
        spmv_merge_based<float>(reorderedMatrix, d_x_reordered, d_y, 0);
        timer.stop();
        totalTime += timer.elapsed_ms();
    }

    float avgTime3 = totalTime / iterations;
    float bw3 = (dataBytes / (avgTime3 * 1e-3)) / (1024 * 1024 * 1024);
    float util3 = bw3 / peakBW * 100;

    std::cout << "Time: " << avgTime3 << " ms, BW: " << bw3 << " GB/s, Util: " << util3 << "%\n";
    std::cout << "Improvement vs original: " << ((avgTime1 - avgTime3) / avgTime1 * 100) << "%\n";
    std::cout << "Improvement vs sorted: " << ((avgTime2 - avgTime3) / avgTime2 * 100) << "%\n\n";

    // Summary
    std::cout << "=== Summary ===\n";
    std::cout << "| Method | Time (ms) | BW (GB/s) | Util (%) | Improvement |\n";
    std::cout << "|--------|-----------|-----------|----------|-------------|\n";
    std::cout << "| Original | " << avgTime1 << " | " << bw1 << " | " << util1 << " | - |\n";
    std::cout << "| Sorted cols | " << avgTime2 << " | " << bw2 << " | " << util2 << " | " << ((avgTime1 - avgTime2) / avgTime1 * 100) << "% |\n";
    std::cout << "| Global reorder | " << avgTime3 << " | " << bw3 << " | " << util3 << " | " << ((avgTime1 - avgTime3) / avgTime1 * 100) << "% |\n";

    // Cleanup
    delete[] h_x;
    delete[] h_x_reordered;
    cudaFree(d_x);
    cudaFree(d_x_reordered);
    cudaFree(d_y);

    return 0;
}