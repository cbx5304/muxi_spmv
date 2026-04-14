/**
 * @file benchmark_detailed.cpp
 * @brief Detailed SpMV Library Comparison with Multiple Iterations
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

// Library 1: nctigpu_spmv
#include "spmv.h"

// Library 2: spmv_fp64
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
    double bytes = (double)nnz * (8.0 + 4.0 + 8.0) + (double)rows * 8.0;
    return bytes / (timeMs * 1e-3) / 1e9;
}

void compareOutputs(const std::vector<double>& y1, const std::vector<double>& y2,
                    double& maxDiff, double& avgDiff) {
    maxDiff = 0.0;
    avgDiff = 0.0;
    int n = y1.size();
    for (int i = 0; i < n; i++) {
        double diff = std::abs(y1[i] - y2[i]);
        maxDiff = std::max(maxDiff, diff);
        avgDiff += diff;
    }
    avgDiff /= n;
}

int main() {
    std::cout << "===== Detailed SpMV Library Comparison Benchmark =====" << std::endl;
    std::cout << "Platform: Mars X201 (Warp=64)" << std::endl;

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << ", Warp: " << prop.warpSize << std::endl;

    // License check
    if (spmv_fp64_check_license() != SPMV_FP64_SUCCESS) {
        std::cerr << "spmv_fp64 license expired!" << std::endl;
        return 1;
    }
    std::cout << "License valid until: " << spmv_fp64_get_license_expiry() << std::endl;
    std::cout << std::endl;

    std::vector<std::string> testCases = {"pressure_0", "pressure_10", "pressure_50"};
    std::string caseDir = "/home/chenbinxiangc/spmv_comp/cases/shunyun";
    int iterations = 10;

    std::cout << "Iterations per test: " << iterations << std::endl;
    std::cout << std::endl;

    std::cout << "| Test | Rows | NNZ | avgNnz | Lib | Best(ms) | Avg(ms) | BW(GB/s) | E2E(ms) |" << std::endl;
    std::cout << "|------|------|-----|--------|-----|----------|---------|----------|---------|" << std::endl;

    for (const auto& testCase : testCases) {
        CSRMatrix mat;
        if (!loadMatrixMarket(caseDir + "/" + testCase + ".mat", mat)) continue;

        std::vector<double> x;
        loadVectorMM(caseDir + "/" + testCase + ".rhs", x);

        // Device memory
        int *d_rowPtr, *d_colIdx;
        double *d_values, *d_x, *d_y1, *d_y2;
        cudaMalloc(&d_rowPtr, (mat.rows + 1) * sizeof(int));
        cudaMalloc(&d_colIdx, mat.nnz * sizeof(int));
        cudaMalloc(&d_values, mat.nnz * sizeof(double));
        cudaMalloc(&d_x, mat.cols * sizeof(double));
        cudaMalloc(&d_y1, mat.rows * sizeof(double));
        cudaMalloc(&d_y2, mat.rows * sizeof(double));

        cudaMemcpy(d_rowPtr, mat.rowPtr.data(), (mat.rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_colIdx, mat.colIdx.data(), mat.nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_values, mat.values.data(), mat.nnz * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, x.data(), mat.cols * sizeof(double), cudaMemcpyHostToDevice);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Library 1 benchmark
        double minTime1 = 1e9, totalTime1 = 0, totalE2E1 = 0;
        for (int i = 0; i < iterations; i++) {
            auto cpuStart = std::chrono::high_resolution_clock::now();

            ncti::sparse::gpu::nctigpuCsrMatDescr_t<double, int, int> matA;
            matA.rows = mat.rows; matA.cols = mat.cols; matA.nnz = mat.nnz;
            matA.rowPtr = d_rowPtr; matA.colInd = d_colIdx; matA.values = d_values;

            ncti::sparse::gpu::nctigpuDnVecDescr_t<const double, int> vecX;
            vecX.size = mat.cols; vecX.values = d_x;

            ncti::sparse::gpu::nctigpuDnVecDescr_t<double, int> vecY;
            vecY.size = mat.rows; vecY.values = d_y1;

            double alpha = 1.0, beta = 0.0;

            cudaEventRecord(start);
            ncti::sparse::gpu::nctigpuSpMV<double, int, int>(&alpha, &matA, &vecX, &beta, &vecY);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float timeMs;
            cudaEventElapsedTime(&timeMs, start, stop);
            minTime1 = std::min(minTime1, (double)timeMs);
            totalTime1 += timeMs;

            auto cpuEnd = std::chrono::high_resolution_clock::now();
            totalE2E1 += std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();
        }

        // Library 2 benchmark
        double minTime2 = 1e9, totalTime2 = 0, totalE2E2 = 0;
        for (int i = 0; i < iterations; i++) {
            auto cpuStart = std::chrono::high_resolution_clock::now();

            cudaEventRecord(start);
            spmv_fp64_execute_direct(mat.rows, mat.nnz, d_rowPtr, d_colIdx, d_values, d_x, d_y2, 0);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float timeMs;
            cudaEventElapsedTime(&timeMs, start, stop);
            minTime2 = std::min(minTime2, (double)timeMs);
            totalTime2 += timeMs;

            auto cpuEnd = std::chrono::high_resolution_clock::now();
            totalE2E2 += std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();
        }

        // Get outputs for comparison
        std::vector<double> y1(mat.rows), y2(mat.rows);
        cudaMemcpy(y1.data(), d_y1, mat.rows * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(y2.data(), d_y2, mat.rows * sizeof(double), cudaMemcpyDeviceToHost);

        double maxDiff, avgDiff;
        compareOutputs(y1, y2, maxDiff, avgDiff);

        // Print results
        double avgNnz = (double)mat.nnz / mat.rows;
        double bw1 = calculateBandwidth(mat.nnz, mat.rows, minTime1);
        double bw2 = calculateBandwidth(mat.nnz, mat.rows, minTime2);

        std::cout << "| " << testCase << " | " << mat.rows << " | " << mat.nnz << " | "
                  << avgNnz << " | nctigpu | " << minTime1 << " | " << totalTime1/iterations
                  << " | " << bw1 << " | " << totalE2E1/iterations << " |" << std::endl;
        std::cout << "| " << testCase << " | " << mat.rows << " | " << mat.nnz << " | "
                  << avgNnz << " | spmv_fp64 | " << minTime2 << " | " << totalTime2/iterations
                  << " | " << bw2 << " | " << totalE2E2/iterations << " |" << std::endl;

        std::cout << "  Cross-library max diff: " << maxDiff << ", avg diff: " << avgDiff << std::endl;
        std::cout << std::endl;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_rowPtr); cudaFree(d_colIdx); cudaFree(d_values);
        cudaFree(d_x); cudaFree(d_y1); cudaFree(d_y2);
    }

    std::cout << "===== Benchmark Complete =====" << std::endl;
    return 0;
}