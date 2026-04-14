/**
 * @file benchmark_optimized.cu
 * @brief Comprehensive benchmark for optimized spmv_fp64 library
 *
 * Tests adaptive TPR optimization for avgNnz~85 matrices
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <cuda_runtime.h>

// Include the spmv_fp64 header
#include "spmv_fp64.h"

struct CSRMatrix {
    int rows, cols, nnz;
    std::vector<int> rowPtr;
    std::vector<int> colIdx;
    std::vector<double> values;
};

bool loadMatrixMarket(const std::string& filename, CSRMatrix& mat) {
    std::ifstream file(filename);
    if (!file.is_open()) return false;

    std::string line;
    while (std::getline(file, line)) {
        if (line[0] != '%') break;
    }

    std::istringstream iss(line);
    iss >> mat.rows >> mat.cols >> mat.nnz;

    std::vector<std::tuple<int, int, double>> entries;
    entries.reserve(mat.nnz);

    int row, col;
    double val;
    while (file >> row >> col >> val) {
        entries.push_back({row - 1, col - 1, val});
    }

    std::vector<int> nnzPerRow(mat.rows, 0);
    for (const auto& e : entries) nnzPerRow[std::get<0>(e)]++;

    mat.rowPtr.resize(mat.rows + 1, 0);
    for (int i = 0; i < mat.rows; i++)
        mat.rowPtr[i + 1] = mat.rowPtr[i] + nnzPerRow[i];

    mat.colIdx.resize(mat.nnz);
    mat.values.resize(mat.nnz);
    std::vector<int> current(mat.rows, 0);
    for (const auto& e : entries) {
        int r = std::get<0>(e);
        int pos = mat.rowPtr[r] + current[r];
        mat.colIdx[pos] = std::get<1>(e);
        mat.values[pos] = std::get<2>(e);
        current[r]++;
    }
    return true;
}

bool loadVectorMM(const std::string& filename, std::vector<double>& vec) {
    std::ifstream file(filename);
    if (!file.is_open()) return false;

    std::string line;
    while (std::getline(file, line)) {
        if (line[0] != '%') break;
    }

    int rows, cols;
    std::istringstream iss(line);
    iss >> rows >> cols;

    vec.clear();
    vec.reserve(rows);
    double val;
    for (int i = 0; i < rows && file >> val; i++)
        vec.push_back(val);
    return true;
}

double calculateBandwidth(int nnz, int rows, double timeMs) {
    double bytes = (double)nnz * 20.0 + (double)rows * 8.0;
    return bytes / (timeMs * 1e6);
}

int main() {
    std::cout << "===== Optimized SpMV FP64 Benchmark =====" << std::endl;

    // Check GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << ", Warp Size: " << prop.warpSize << std::endl;

    // Check license
    if (spmv_fp64_check_license() != SPMV_FP64_SUCCESS) {
        std::cerr << "License expired!" << std::endl;
        return 1;
    }
    std::cout << "License valid until: " << spmv_fp64_get_license_expiry() << std::endl;
    std::cout << std::endl;

    std::vector<std::string> testCases = {"pressure_0", "pressure_10", "pressure_50"};
    std::string caseDir = "/home/chenbinxiangc/spmv_comp/cases/shunyun";
    int iterations = 20;

    std::cout << "| Test | Rows | NNZ | avgNnz | Best(ms) | Avg(ms) | BW(GB/s) | Util% | Optimal TPR |" << std::endl;
    std::cout << "|------|------|-----|--------|----------|---------|----------|-------|-------------|" << std::endl;

    for (const auto& testCase : testCases) {
        CSRMatrix mat;
        if (!loadMatrixMarket(caseDir + "/" + testCase + ".mat", mat)) {
            std::cerr << "Failed to load " << testCase << std::endl;
            continue;
        }

        std::vector<double> x;
        loadVectorMM(caseDir + "/" + testCase + ".rhs", x);

        double avgNnz = (double)mat.nnz / mat.rows;
        std::cout << "Matrix: " << testCase << " - avgNnz=" << avgNnz << std::endl;

        // Device memory
        int *d_rowPtr, *d_colIdx;
        double *d_values, *d_x, *d_y;

        cudaMalloc(&d_rowPtr, (mat.rows + 1) * sizeof(int));
        cudaMalloc(&d_colIdx, mat.nnz * sizeof(int));
        cudaMalloc(&d_values, mat.nnz * sizeof(double));
        cudaMalloc(&d_x, mat.cols * sizeof(double));
        cudaMalloc(&d_y, mat.rows * sizeof(double));

        cudaMemcpy(d_rowPtr, mat.rowPtr.data(), (mat.rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_colIdx, mat.colIdx.data(), mat.nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_values, mat.values.data(), mat.nnz * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, x.data(), mat.cols * sizeof(double), cudaMemcpyHostToDevice);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Warmup
        spmv_fp64_execute_direct(mat.rows, mat.nnz, d_rowPtr, d_colIdx, d_values, d_x, d_y, 0);
        cudaDeviceSynchronize();

        // Benchmark
        double minTime = 1e9, totalTime = 0;
        for (int i = 0; i < iterations; i++) {
            cudaEventRecord(start);
            spmv_fp64_execute_direct(mat.rows, mat.nnz, d_rowPtr, d_colIdx, d_values, d_x, d_y, 0);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float timeMs = 0;
            cudaEventElapsedTime(&timeMs, start, stop);
            minTime = std::min(minTime, (double)timeMs);
            totalTime += timeMs;
        }

        double avgTime = totalTime / iterations;
        double bw = calculateBandwidth(mat.nnz, mat.rows, minTime);
        double peakBW = 1843.2;  // Mars X201 peak
        double util = bw / peakBW * 100;
        int optimalTPR = (avgNnz >= 32) ? 32 : 8;

        std::cout << "| " << testCase << " | " << mat.rows << " | " << mat.nnz << " | "
                  << avgNnz << " | " << minTime << " | " << avgTime << " | " << bw
                  << " | " << util << "% | TPR=" << optimalTPR << " |" << std::endl;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_rowPtr);
        cudaFree(d_colIdx);
        cudaFree(d_values);
        cudaFree(d_x);
        cudaFree(d_y);
    }

    std::cout << std::endl;
    std::cout << "===== Benchmark Complete =====" << std::endl;

    return 0;
}