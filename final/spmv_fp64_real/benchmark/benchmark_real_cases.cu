/**
 * @file benchmark_real_cases.cu
 * @brief Compare nctigpu (spmv_1) and optimized spmv_fp64 with real test cases
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <cuda_runtime.h>

// Optimized spmv_fp64
#include "spmv_fp64.h"

// nctigpu from spmv_1
#include "/home/chenbinxiangc/spmv_comp/spmv_1/nctigpu_spmv/include/spmv.h"

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
    std::cout << "===== SpMV Library Comparison - Real Test Cases =====" << std::endl;
    std::cout << "Testing: nctigpu (spmv_1) vs optimized spmv_fp64" << std::endl;
    std::cout << std::endl;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << ", Warp Size: " << prop.warpSize << std::endl;
    double peakBW = 1843.2;  // Mars X201
    std::cout << "Peak Bandwidth: " << peakBW << " GB/s" << std::endl;
    std::cout << std::endl;

    // Check optimized library license
    if (spmv_fp64_check_license() != SPMV_FP64_SUCCESS) {
        std::cerr << "Optimized spmv_fp64 license expired!" << std::endl;
        return 1;
    }
    std::cout << "Optimized spmv_fp64: License valid until " << spmv_fp64_get_license_expiry() << std::endl;
    std::cout << std::endl;

    std::string caseDir = "/home/chenbinxiangc/spmv_muxi/real_cases/mtx";
    std::vector<std::string> testCases = {"p0", "p1", "p2", "p3", "p4", "p5", "p6"};
    int iterations = 30;

    // Summary table header
    std::cout << "| Case | Rows | NNZ | avgNnz | nctigpu(ms) | opt(ms) | ncti BW(GB/s) | opt BW(GB/s) | ncti Util | opt Util | Speedup | Max Err |" << std::endl;
    std::cout << "|------|------|-----|--------|-------------|---------|---------------|--------------|-----------|----------|---------|---------|" << std::endl;

    double totalNctiBW = 0, totalOptBW = 0;
    int validTests = 0;

    for (const auto& testCase : testCases) {
        CSRMatrix mat;
        std::string matFile = caseDir + "/" + testCase + "_A";
        std::string vecFile = caseDir + "/" + testCase + "_b";

        if (!loadMatrixMarket(matFile, mat)) {
            std::cerr << "Failed to load " << testCase << std::endl;
            continue;
        }

        std::vector<double> x;
        loadVectorMM(vecFile, x);

        double avgNnz = (double)mat.nnz / mat.rows;

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

        std::vector<double> ref_y(mat.rows);
        double nctiTime = 0, optTime = 0;
        double nctiBW = 0, optBW = 0;
        double maxDiff = 0;

        // ===== Test 1: Optimized spmv_fp64 =====
        {
            // Warmup
            for (int w = 0; w < 5; w++) {
                spmv_fp64_execute_direct(mat.rows, mat.nnz, d_rowPtr, d_colIdx, d_values, d_x, d_y, 0);
            }
            cudaDeviceSynchronize();

            // Benchmark
            double minTime = 1e9;
            for (int i = 0; i < iterations; i++) {
                cudaEventRecord(start);
                spmv_fp64_execute_direct(mat.rows, mat.nnz, d_rowPtr, d_colIdx, d_values, d_x, d_y, 0);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);

                float timeMs = 0;
                cudaEventElapsedTime(&timeMs, start, stop);
                minTime = std::min(minTime, (double)timeMs);
            }

            optTime = minTime;
            optBW = calculateBandwidth(mat.nnz, mat.rows, optTime);
            cudaMemcpy(ref_y.data(), d_y, mat.rows * sizeof(double), cudaMemcpyDeviceToHost);
        }

        // ===== Test 2: nctigpu (spmv_1) =====
        {
            ncti::sparse::gpu::nctigpuCsrMatDescr_t<double, int, int> nctiMat;
            nctiMat.rows = mat.rows;
            nctiMat.cols = mat.cols;
            nctiMat.nnz = mat.nnz;
            nctiMat.rowPtr = d_rowPtr;
            nctiMat.colInd = d_colIdx;
            nctiMat.values = d_values;

            ncti::sparse::gpu::nctigpuDnVecDescr_t<const double, int> vecX;
            vecX.size = mat.cols;
            vecX.values = d_x;

            ncti::sparse::gpu::nctigpuDnVecDescr_t<double, int> vecY;
            vecY.size = mat.rows;
            vecY.values = d_y;

            double alpha = 1.0, beta = 0.0;

            // Warmup
            for (int w = 0; w < 5; w++) {
                ncti::sparse::gpu::nctigpuSpMV<double, int, int>(&alpha, &nctiMat, &vecX, &beta, &vecY);
            }
            cudaDeviceSynchronize();

            // Benchmark
            double minTime = 1e9;
            for (int i = 0; i < iterations; i++) {
                cudaEventRecord(start);
                ncti::sparse::gpu::nctigpuSpMV<double, int, int>(&alpha, &nctiMat, &vecX, &beta, &vecY);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);

                float timeMs = 0;
                cudaEventElapsedTime(&timeMs, start, stop);
                minTime = std::min(minTime, (double)timeMs);
            }

            nctiTime = minTime;
            nctiBW = calculateBandwidth(mat.nnz, mat.rows, nctiTime);

            // Compare accuracy
            std::vector<double> ncti_y(mat.rows);
            cudaMemcpy(ncti_y.data(), d_y, mat.rows * sizeof(double), cudaMemcpyDeviceToHost);

            for (int i = 0; i < mat.rows; i++) {
                maxDiff = std::max(maxDiff, std::abs(ncti_y[i] - ref_y[i]));
            }
        }

        // Calculate metrics
        double speedup = nctiTime / optTime;
        double nctiUtil = nctiBW / peakBW * 100;
        double optUtil = optBW / peakBW * 100;

        totalNctiBW += nctiBW;
        totalOptBW += optBW;
        validTests++;

        // Output row
        std::cout << "| " << testCase << " | " << mat.rows << " | " << mat.nnz << " | "
                  << avgNnz << " | " << nctiTime << " | " << optTime << " | "
                  << nctiBW << " | " << optBW << " | " << nctiUtil << "% | "
                  << optUtil << "% | " << speedup << "x | " << maxDiff << " |" << std::endl;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_rowPtr);
        cudaFree(d_colIdx);
        cudaFree(d_values);
        cudaFree(d_x);
        cudaFree(d_y);
    }

    // Summary
    std::cout << std::endl;
    std::cout << "===== Summary =====" << std::endl;
    std::cout << "Average nctigpu bandwidth: " << (totalNctiBW / validTests) << " GB/s" << std::endl;
    std::cout << "Average optimized bandwidth: " << (totalOptBW / validTests) << " GB/s" << std::endl;
    std::cout << "Average speedup: " << ((totalNctiBW / validTests) / (totalOptBW / validTests)) << "x (opt/ncti)" << std::endl;
    std::cout << std::endl;
    std::cout << "===== Benchmark Complete =====" << std::endl;

    return 0;
}