/**
 * @file benchmark_comparison.cu
 * @brief Comprehensive comparison of three SpMV libraries
 *
 * Tests: nctigpu, old spmv_fp64, new optimized spmv_fp64
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <cuda_runtime.h>

// Include the optimized spmv_fp64 header
#include "spmv_fp64.h"

// nctigpu header
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
    std::cout << "===== SpMV Library Comparison Benchmark =====" << std::endl;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << ", Warp Size: " << prop.warpSize << std::endl;
    std::cout << "Peak Bandwidth: 1843.2 GB/s (Mars X201)" << std::endl;
    std::cout << std::endl;

    std::vector<std::string> testCases = {"pressure_0", "pressure_10", "pressure_50"};
    std::string caseDir = "/home/chenbinxiangc/spmv_comp/cases/shunyun";
    int iterations = 20;

    // Check optimized spmv_fp64 license
    if (spmv_fp64_check_license() != SPMV_FP64_SUCCESS) {
        std::cerr << "Optimized spmv_fp64 license expired!" << std::endl;
        return 1;
    }
    std::cout << "Optimized spmv_fp64 license valid until: " << spmv_fp64_get_license_expiry() << std::endl;
    std::cout << std::endl;

    std::cout << "| Test | avgNnz | Library | Best(ms) | BW(GB/s) | Util% | Max Err vs Opt |" << std::endl;
    std::cout << "|------|--------|---------|----------|----------|-------|----------------|" << std::endl;

    for (const auto& testCase : testCases) {
        CSRMatrix mat;
        if (!loadMatrixMarket(caseDir + "/" + testCase + ".mat", mat)) {
            std::cerr << "Failed to load " << testCase << std::endl;
            continue;
        }

        std::vector<double> x;
        loadVectorMM(caseDir + "/" + testCase + ".rhs", x);

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

        // Reference output (optimized spmv_fp64)
        std::vector<double> ref_y(mat.rows);

        // ========== Test 1: Optimized spmv_fp64 ==========
        {
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

            double bw = calculateBandwidth(mat.nnz, mat.rows, minTime);
            double util = bw / 1843.2 * 100;

            // Store reference
            cudaMemcpy(ref_y.data(), d_y, mat.rows * sizeof(double), cudaMemcpyDeviceToHost);

            std::cout << "| " << testCase << " | " << avgNnz << " | optimized_spmv_fp64 | "
                      << minTime << " | " << bw << " | " << util << "% | ref |" << std::endl;
        }

        // ========== Test 2: nctigpu ==========
        {
            // Create nctigpu sparse matrix descriptor
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
            ncti::sparse::gpu::nctigpuSpMV<double, int, int>(&alpha, &nctiMat, &vecX, &beta, &vecY);
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

            double bw = calculateBandwidth(mat.nnz, mat.rows, minTime);
            double util = bw / 1843.2 * 100;

            // Compare with optimized reference
            std::vector<double> y_ncti(mat.rows);
            cudaMemcpy(y_ncti.data(), d_y, mat.rows * sizeof(double), cudaMemcpyDeviceToHost);

            double maxDiff = 0;
            for (int i = 0; i < mat.rows; i++) {
                maxDiff = std::max(maxDiff, std::abs(y_ncti[i] - ref_y[i]));
            }

            std::cout << "| " << testCase << " | " << avgNnz << " | nctigpu | "
                      << minTime << " | " << bw << " | " << util << "% | " << maxDiff << " |" << std::endl;
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_rowPtr);
        cudaFree(d_colIdx);
        cudaFree(d_values);
        cudaFree(d_x);
        cudaFree(d_y);
    }

    std::cout << std::endl;
    std::cout << "===== Comparison Complete =====" << std::endl;

    return 0;
}