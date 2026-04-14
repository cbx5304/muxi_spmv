/**
 * @file fp32_comparison_test.cu
 * @brief Compare nctigpu (spmv_1) and optimized spmv_fp32 with real test cases
 *
 * Test metrics:
 * - CPU end-to-end time (including H2D, kernel, D2H)
 * - Kernel-only time
 * - CPU end-to-end bandwidth
 * - Kernel-only bandwidth
 * - Accuracy comparison
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

// Optimized spmv_fp32
#include "spmv_fp32.h"

// nctigpu from spmv_1
#include "/home/chenbinxiangc/spmv_comp/spmv_1/nctigpu_spmv/include/spmv.h"

struct CSRMatrix {
    int rows, cols, nnz;
    std::vector<int> rowPtr;
    std::vector<int> colIdx;
    std::vector<float> values;
};

struct TestResult {
    std::string caseName;
    int rows, nnz;
    double avgNnz;

    // nctigpu results
    double ncti_e2e_time_ms;
    double ncti_kernel_time_ms;
    double ncti_e2e_bw_gbps;
    double ncti_kernel_bw_gbps;

    // optimized results
    double opt_e2e_time_ms;
    double opt_kernel_time_ms;
    double opt_e2e_bw_gbps;
    double opt_kernel_bw_gbps;

    // comparison
    double e2e_speedup;
    double kernel_speedup;
    double max_diff;

    // hcTracer data
    double ncti_hc_time_us;
    double opt_hc_time_us;
};

// Load MatrixMarket format matrix
bool loadMTXMatrix(const std::string& filename, CSRMatrix& mat) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return false;
    }

    std::string line;
    // Skip header lines
    while (std::getline(file, line)) {
        if (line[0] != '%') break;
    }

    // Parse dimensions
    std::istringstream iss(line);
    iss >> mat.rows >> mat.cols >> mat.nnz;

    std::cout << "Matrix: " << mat.rows << " rows, " << mat.cols << " cols, " << mat.nnz << " nnz" << std::endl;

    // Read COO entries
    std::vector<std::tuple<int, int, double>> entries;
    entries.reserve(mat.nnz);

    int row, col;
    double val;
    while (file >> row >> col >> val) {
        entries.push_back({row - 1, col - 1, val});  // MTX is 1-indexed
    }

    // Convert COO to CSR
    mat.rowPtr.resize(mat.rows + 1, 0);
    mat.colIdx.resize(mat.nnz);
    mat.values.resize(mat.nnz);

    // Count nnz per row
    for (const auto& e : entries) {
        mat.rowPtr[std::get<0>(e) + 1]++;
    }

    // Cumulative sum
    for (int i = 0; i < mat.rows; i++) {
        mat.rowPtr[i + 1] += mat.rowPtr[i];
    }

    // Fill CSR arrays
    std::vector<int> current(mat.rows, 0);
    for (const auto& e : entries) {
        int r = std::get<0>(e);
        int pos = mat.rowPtr[r] + current[r];
        mat.colIdx[pos] = std::get<1>(e);
        mat.values[pos] = (float)std::get<2>(e);
        current[r]++;
    }

    return true;
}

// Load MatrixMarket format vector (array type)
bool loadMTXVector(const std::string& filename, std::vector<float>& vec) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return false;
    }

    std::string line;
    // Skip header lines
    while (std::getline(file, line)) {
        if (line[0] != '%') break;
    }

    // Parse dimensions
    int rows, cols;
    std::istringstream iss(line);
    iss >> rows >> cols;

    vec.clear();
    vec.reserve(rows);

    double val;
    for (int i = 0; i < rows && file >> val; i++) {
        vec.push_back((float)val);
    }

    return true;
}

// Calculate bandwidth for FP32 SpMV
double calculateBandwidth(int nnz, int rows, double timeMs) {
    // FP32: values(4B) + colIdx(4B) + x[col](4B) = 12B per nnz
    // Plus y output: rows * 4B
    double bytes = (double)nnz * 12.0 + (double)rows * 4.0;
    return bytes / (timeMs * 1e6);  // GB/s
}

// Run nctigpu SpMV test
void runNctigpuTest(const CSRMatrix& mat, const std::vector<float>& x,
                    std::vector<float>& y_ncti, TestResult& result) {

    // Device memory
    int *d_rowPtr, *d_colIdx;
    float *d_values, *d_x, *d_y;

    cudaMalloc(&d_rowPtr, (mat.rows + 1) * sizeof(int));
    cudaMalloc(&d_colIdx, mat.nnz * sizeof(int));
    cudaMalloc(&d_values, mat.nnz * sizeof(float));
    cudaMalloc(&d_x, mat.cols * sizeof(float));
    cudaMalloc(&d_y, mat.rows * sizeof(float));

    // Setup nctigpu descriptors
    ncti::sparse::gpu::nctigpuCsrMatDescr_t<float, int, int> nctiMat;
    nctiMat.rows = mat.rows;
    nctiMat.cols = mat.cols;
    nctiMat.nnz = mat.nnz;
    nctiMat.rowPtr = d_rowPtr;
    nctiMat.colInd = d_colIdx;
    nctiMat.values = d_values;

    ncti::sparse::gpu::nctigpuDnVecDescr_t<const float, int> vecX;
    vecX.size = mat.cols;
    vecX.values = d_x;

    ncti::sparse::gpu::nctigpuDnVecDescr_t<float, int> vecY;
    vecY.size = mat.rows;
    vecY.values = d_y;

    float alpha = 1.0f, beta = 0.0f;

    // Warmup
    for (int w = 0; w < 5; w++) {
        cudaMemcpy(d_rowPtr, mat.rowPtr.data(), (mat.rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_colIdx, mat.colIdx.data(), mat.nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_values, mat.values.data(), mat.nnz * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, x.data(), mat.cols * sizeof(float), cudaMemcpyHostToDevice);

        ncti::sparse::gpu::nctigpuSpMV<float, int, int>(&alpha, &nctiMat, &vecX, &beta, &vecY);
        cudaDeviceSynchronize();
    }

    // Measure end-to-end time (including H2D/D2H)
    const int iterations = 30;

    auto e2e_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        cudaMemcpy(d_rowPtr, mat.rowPtr.data(), (mat.rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_colIdx, mat.colIdx.data(), mat.nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_values, mat.values.data(), mat.nnz * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, x.data(), mat.cols * sizeof(float), cudaMemcpyHostToDevice);

        ncti::sparse::gpu::nctigpuSpMV<float, int, int>(&alpha, &nctiMat, &vecX, &beta, &vecY);
        cudaDeviceSynchronize();

        cudaMemcpy(y_ncti.data(), d_y, mat.rows * sizeof(float), cudaMemcpyDeviceToHost);
    }
    auto e2e_end = std::chrono::high_resolution_clock::now();

    result.ncti_e2e_time_ms = std::chrono::duration<double>(e2e_end - e2e_start).count() * 1000 / iterations;
    result.ncti_e2e_bw_gbps = calculateBandwidth(mat.nnz, mat.rows, result.ncti_e2e_time_ms);

    // Measure kernel-only time (data already on device)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float minKernelTime = 1e9f;
    for (int i = 0; i < iterations; i++) {
        cudaEventRecord(start);
        ncti::sparse::gpu::nctigpuSpMV<float, int, int>(&alpha, &nctiMat, &vecX, &beta, &vecY);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float timeMs = 0;
        cudaEventElapsedTime(&timeMs, start, stop);
        minKernelTime = std::min(minKernelTime, timeMs);
    }

    result.ncti_kernel_time_ms = minKernelTime;
    result.ncti_kernel_bw_gbps = calculateBandwidth(mat.nnz, mat.rows, result.ncti_kernel_time_ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Final result for accuracy comparison
    cudaMemcpy(y_ncti.data(), d_y, mat.rows * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_rowPtr);
    cudaFree(d_colIdx);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);
}

// Run optimized spmv_fp32 test
void runOptimizedTest(const CSRMatrix& mat, const std::vector<float>& x,
                      std::vector<float>& y_opt, TestResult& result) {

    // Device memory
    int *d_rowPtr, *d_colIdx;
    float *d_values, *d_x, *d_y;

    cudaMalloc(&d_rowPtr, (mat.rows + 1) * sizeof(int));
    cudaMalloc(&d_colIdx, mat.nnz * sizeof(int));
    cudaMalloc(&d_values, mat.nnz * sizeof(float));
    cudaMalloc(&d_x, mat.cols * sizeof(float));
    cudaMalloc(&d_y, mat.rows * sizeof(float));

    // Warmup
    for (int w = 0; w < 5; w++) {
        cudaMemcpy(d_rowPtr, mat.rowPtr.data(), (mat.rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_colIdx, mat.colIdx.data(), mat.nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_values, mat.values.data(), mat.nnz * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, x.data(), mat.cols * sizeof(float), cudaMemcpyHostToDevice);

        spmv_fp32_execute_direct(mat.rows, mat.nnz, d_rowPtr, d_colIdx, d_values, d_x, d_y, 0);
        cudaDeviceSynchronize();
    }

    // Measure end-to-end time
    const int iterations = 30;

    auto e2e_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        cudaMemcpy(d_rowPtr, mat.rowPtr.data(), (mat.rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_colIdx, mat.colIdx.data(), mat.nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_values, mat.values.data(), mat.nnz * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, x.data(), mat.cols * sizeof(float), cudaMemcpyHostToDevice);

        spmv_fp32_execute_direct(mat.rows, mat.nnz, d_rowPtr, d_colIdx, d_values, d_x, d_y, 0);
        cudaDeviceSynchronize();

        cudaMemcpy(y_opt.data(), d_y, mat.rows * sizeof(float), cudaMemcpyDeviceToHost);
    }
    auto e2e_end = std::chrono::high_resolution_clock::now();

    result.opt_e2e_time_ms = std::chrono::duration<double>(e2e_end - e2e_start).count() * 1000 / iterations;
    result.opt_e2e_bw_gbps = calculateBandwidth(mat.nnz, mat.rows, result.opt_e2e_time_ms);

    // Measure kernel-only time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float minKernelTime = 1e9f;
    for (int i = 0; i < iterations; i++) {
        cudaEventRecord(start);
        spmv_fp32_execute_direct(mat.rows, mat.nnz, d_rowPtr, d_colIdx, d_values, d_x, d_y, 0);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float timeMs = 0;
        cudaEventElapsedTime(&timeMs, start, stop);
        minKernelTime = std::min(minKernelTime, timeMs);
    }

    result.opt_kernel_time_ms = minKernelTime;
    result.opt_kernel_bw_gbps = calculateBandwidth(mat.nnz, mat.rows, result.opt_kernel_time_ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Final result for accuracy comparison
    cudaMemcpy(y_opt.data(), d_y, mat.rows * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_rowPtr);
    cudaFree(d_colIdx);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  FP32 SpMV Library Comparison Test" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Warp Size: " << prop.warpSize << std::endl;
    double peakBW = 1843.2;  // Mars X201
    std::cout << "Peak Bandwidth: " << peakBW << " GB/s" << std::endl;
    std::cout << std::endl;

    // Check licenses
    if (spmv_fp32_check_license() != SPMV_FP32_SUCCESS) {
        std::cerr << "spmv_fp32 license expired!" << std::endl;
        return 1;
    }
    std::cout << "spmv_fp32 license valid until: " << spmv_fp32_get_license_expiry() << std::endl;
    std::cout << std::endl;

    // Test cases
    std::string caseDir = "/home/chenbinxiangc/spmv_comp/cases/shunyun";
    std::vector<std::string> testCases = {"pressure_0", "pressure_10", "pressure_50"};

    std::vector<TestResult> results;

    // Table header
    std::cout << "| Case | Rows | NNZ | avgNnz | ncti E2E(ms) | opt E2E(ms) | ncti Kern(ms) | opt Kern(ms) | ncti E2E BW | opt E2E BW | ncti Kern BW | opt Kern BW | E2E Speedup | Kern Speedup | Max Diff |" << std::endl;
    std::cout << "|------|------|-----|--------|--------------|-------------|---------------|--------------|-------------|------------|--------------|-------------|-------------|--------------|----------|" << std::endl;

    for (const auto& testCase : testCases) {
        CSRMatrix mat;
        std::string matFile = caseDir + "/" + testCase + ".mat";

        if (!loadMTXMatrix(matFile, mat)) {
            std::cerr << "Failed to load " << testCase << std::endl;
            continue;
        }

        std::vector<float> x;
        std::string rhsFile = caseDir + "/" + testCase + ".rhs";
        if (!loadMTXVector(rhsFile, x)) {
            std::cerr << "Failed to load rhs for " << testCase << std::endl;
            continue;
        }

        // Ensure x has correct size
        if (x.size() < mat.cols) {
            x.resize(mat.cols, 1.0f);
        }

        TestResult result;
        result.caseName = testCase;
        result.rows = mat.rows;
        result.nnz = mat.nnz;
        result.avgNnz = (double)mat.nnz / mat.rows;

        std::vector<float> y_ncti(mat.rows);
        std::vector<float> y_opt(mat.rows);

        // Run tests
        runNctigpuTest(mat, x, y_ncti, result);
        runOptimizedTest(mat, x, y_opt, result);

        // Calculate speedups
        result.e2e_speedup = result.ncti_e2e_time_ms / result.opt_e2e_time_ms;
        result.kernel_speedup = result.ncti_kernel_time_ms / result.opt_kernel_time_ms;

        // Compare accuracy
        result.max_diff = 0.0;
        for (int i = 0; i < mat.rows; i++) {
            double diff = std::abs(y_ncti[i] - y_opt[i]);
            if (diff > result.max_diff) {
                result.max_diff = diff;
            }
        }

        results.push_back(result);

        // Output row
        std::cout << "| " << testCase << " | " << mat.rows << " | " << mat.nnz << " | "
                  << result.avgNnz << " | " << result.ncti_e2e_time_ms << " | " << result.opt_e2e_time_ms << " | "
                  << result.ncti_kernel_time_ms << " | " << result.opt_kernel_time_ms << " | "
                  << result.ncti_e2e_bw_gbps << " | " << result.opt_e2e_bw_gbps << " | "
                  << result.ncti_kernel_bw_gbps << " | " << result.opt_kernel_bw_gbps << " | "
                  << result.e2e_speedup << "x | " << result.kernel_speedup << "x | "
                  << result.max_diff << " |" << std::endl;
    }

    // Summary
    std::cout << std::endl;
    std::cout << "===== Summary =====" << std::endl;

    double avgNctiE2E = 0, avgOptE2E = 0;
    double avgNctiKern = 0, avgOptKern = 0;
    double avgNctiE2EBW = 0, avgOptE2EBW = 0;
    double avgNctiKernBW = 0, avgOptKernBW = 0;
    double avgE2ESpeedup = 0, avgKernSpeedup = 0;

    for (const auto& r : results) {
        avgNctiE2E += r.ncti_e2e_time_ms;
        avgOptE2E += r.opt_e2e_time_ms;
        avgNctiKern += r.ncti_kernel_time_ms;
        avgOptKern += r.opt_kernel_time_ms;
        avgNctiE2EBW += r.ncti_e2e_bw_gbps;
        avgOptE2EBW += r.opt_e2e_bw_gbps;
        avgNctiKernBW += r.ncti_kernel_bw_gbps;
        avgOptKernBW += r.opt_kernel_bw_gbps;
        avgE2ESpeedup += r.e2e_speedup;
        avgKernSpeedup += r.kernel_speedup;
    }

    int n = results.size();
    std::cout << "Average nctigpu E2E time: " << (avgNctiE2E/n) << " ms" << std::endl;
    std::cout << "Average optimized E2E time: " << (avgOptE2E/n) << " ms" << std::endl;
    std::cout << "Average nctigpu kernel time: " << (avgNctiKern/n) << " ms" << std::endl;
    std::cout << "Average optimized kernel time: " << (avgOptKern/n) << " ms" << std::endl;
    std::cout << std::endl;
    std::cout << "Average nctigpu E2E bandwidth: " << (avgNctiE2EBW/n) << " GB/s" << std::endl;
    std::cout << "Average optimized E2E bandwidth: " << (avgOptE2EBW/n) << " GB/s" << std::endl;
    std::cout << "Average nctigpu kernel bandwidth: " << (avgNctiKernBW/n) << " GB/s" << std::endl;
    std::cout << "Average optimized kernel bandwidth: " << (avgOptKernBW/n) << " GB/s" << std::endl;
    std::cout << std::endl;
    std::cout << "Average E2E speedup: " << (avgE2ESpeedup/n) << "x" << std::endl;
    std::cout << "Average kernel speedup: " << (avgKernSpeedup/n) << "x" << std::endl;
    std::cout << std::endl;

    // Detailed comparison table
    std::cout << "===== Detailed Performance Comparison =====" << std::endl;
    std::cout << std::endl;

    std::cout << "| Metric | nctigpu | optimized | Improvement |" << std::endl;
    std::cout << "|--------|---------|-----------|-------------|" << std::endl;
    std::cout << "| E2E Time (avg) | " << (avgNctiE2E/n) << " ms | " << (avgOptE2E/n) << " ms | " << ((avgE2ESpeedup/n)-1)*100 << "% |" << std::endl;
    std::cout << "| Kernel Time (avg) | " << (avgNctiKern/n) << " ms | " << (avgOptKern/n) << " ms | " << ((avgKernSpeedup/n)-1)*100 << "% |" << std::endl;
    std::cout << "| E2E BW (avg) | " << (avgNctiE2EBW/n) << " GB/s | " << (avgOptE2EBW/n) << " GB/s | +" << ((avgOptE2EBW-avgNctiE2EBW)/avgNctiE2EBW*100/n) << "% |" << std::endl;
    std::cout << "| Kernel BW (avg) | " << (avgNctiKernBW/n) << " GB/s | " << (avgOptKernBW/n) << " GB/s | +" << ((avgOptKernBW-avgNctiKernBW)/avgNctiKernBW*100/n) << "% |" << std::endl;

    std::cout << std::endl;
    std::cout << "===== Test Complete =====" << std::endl;

    return 0;
}