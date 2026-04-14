/**
 * @file benchmark.cpp
 * @brief SpMV Library Comparison Benchmark for Mars X201
 *
 * Compares two SpMV libraries:
 * - Library 1: nctigpu_spmv (ncti::sparse::gpu)
 * - Library 2: spmv_fp64 (C API)
 *
 * Test cases from /home/chenbinxiangc/spmv_comp/cases/shunyun/
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

// Matrix Market format loader for coordinate matrix
struct CSRMatrix {
    int rows, cols, nnz;
    std::vector<int> rowPtr;
    std::vector<int> colIdx;
    std::vector<double> values;
};

bool loadMatrixMarket(const std::string& filename, CSRMatrix& mat) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Cannot open matrix file: " << filename << std::endl;
        return false;
    }

    std::string line;
    // Skip header comments
    while (std::getline(file, line)) {
        if (line[0] != '%') break;
    }

    // Parse dimensions
    std::istringstream iss(line);
    iss >> mat.rows >> mat.cols >> mat.nnz;

    // Count actual nnz per row to build rowPtr
    std::vector<int> nnzPerRow(mat.rows, 0);

    // Read all entries first to count nnz per row
    std::vector<std::tuple<int, int, double>> entries;
    entries.reserve(mat.nnz);

    int row, col;
    double val;
    while (file >> row >> col >> val) {
        // Matrix Market uses 1-based indexing
        entries.push_back({row - 1, col - 1, val});
        nnzPerRow[row - 1]++;
    }

    // Build CSR format
    mat.rowPtr.resize(mat.rows + 1, 0);
    mat.colIdx.resize(mat.nnz);
    mat.values.resize(mat.nnz);

    // Build rowPtr
    for (int i = 0; i < mat.rows; i++) {
        mat.rowPtr[i + 1] = mat.rowPtr[i] + nnzPerRow[i];
    }

    // Fill colIdx and values
    std::vector<int> currentRowPtr(mat.rows, 0);
    for (const auto& entry : entries) {
        int r = std::get<0>(entry);
        int c = std::get<1>(entry);
        double v = std::get<2>(entry);
        int pos = mat.rowPtr[r] + currentRowPtr[r];
        mat.colIdx[pos] = c;
        mat.values[pos] = v;
        currentRowPtr[r]++;
    }

    file.close();
    return true;
}

// Load Matrix Market array format vector (dense column vector)
bool loadVectorMM(const std::string& filename, std::vector<double>& vec) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Cannot open vector file: " << filename << std::endl;
        return false;
    }

    std::string line;
    // Skip header comments
    while (std::getline(file, line)) {
        if (line[0] != '%') break;
    }

    // Parse dimensions (rows cols) - array format has rows elements
    std::istringstream iss(line);
    int rows, cols;
    iss >> rows >> cols;

    vec.clear();
    vec.reserve(rows);

    double val;
    for (int i = 0; i < rows; i++) {
        if (file >> val) {
            vec.push_back(val);
        } else {
            std::cerr << "Warning: Only read " << i << " values, expected " << rows << std::endl;
            break;
        }
    }

    file.close();
    return true;
}

void printDeviceInfo() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "GPU " << i << ": " << prop.name
                  << ", Warp Size: " << prop.warpSize << std::endl;
    }
}

// Calculate bandwidth
double calculateBandwidth(int nnz, int rows, double timeMs) {
    // Data transferred for SpMV:
    // Read: values (nnz * 8 bytes), colIdx (nnz * 4 bytes), rowPtr ((rows+1) * 4 bytes), x (nnz reads * 8 bytes average)
    // Write: y (rows * 8 bytes)
    // Simplified formula: ~4 * nnz * 8 bytes + overhead
    double bytes = (double)nnz * (8.0 + 4.0 + 8.0) + (double)rows * 8.0;  // values + colIdx + x_reads + y_write
    double bandwidthGBs = bytes / (timeMs * 1e-3) / 1e9;
    return bandwidthGBs;
}

// Calculate error metrics
void calculateError(const std::vector<double>& computed, const std::vector<double>& reference,
                    double& maxError, double& avgError, double& relError) {
    maxError = 0.0;
    avgError = 0.0;
    relError = 0.0;

    int n = computed.size();
    if (n != (int)reference.size()) {
        std::cerr << "Error: computed size " << n << " vs reference size " << reference.size() << std::endl;
        return;
    }

    double sumError = 0.0;
    double sumRelError = 0.0;
    int relErrorCount = 0;

    for (int i = 0; i < n; i++) {
        double err = std::abs(computed[i] - reference[i]);
        maxError = std::max(maxError, err);
        sumError += err;

        if (std::abs(reference[i]) > 1e-10) {
            sumRelError += err / std::abs(reference[i]);
            relErrorCount++;
        }
    }

    avgError = sumError / n;
    relError = (relErrorCount > 0) ? sumRelError / relErrorCount : 0.0;
}

int main(int argc, char** argv) {
    std::cout << "===== SpMV Library Comparison Benchmark =====" << std::endl;
    std::cout << std::endl;

    // Print GPU info
    printDeviceInfo();
    std::cout << std::endl;

    // Test cases
    std::vector<std::string> testCases = {
        "pressure_0",
        "pressure_10",
        "pressure_50"
    };

    std::string caseDir = "/home/chenbinxiangc/spmv_comp/cases/shunyun";

    // Check license for Library 2
    spmv_fp64_status_t licenseStatus = spmv_fp64_check_license();
    if (licenseStatus == SPMV_FP64_ERROR_LICENSE_EXPIRED) {
        std::cerr << "Library 2 (spmv_fp64) license expired!" << std::endl;
        return 1;
    }
    std::cout << "Library 2 license valid until: " << spmv_fp64_get_license_expiry() << std::endl;
    std::cout << std::endl;

    // Results table header
    std::cout << "| Test Case | Rows | NNZ | avgNnz | Library | CPU E2E(ms) | Kernel(ms) | CPU BW(GB/s) | Kernel BW(GB/s) | Max Error | Avg Rel Error |" << std::endl;
    std::cout << "|-----------|------|-----|--------|---------|-------------|------------|--------------|-----------------|-----------|---------------|" << std::endl;

    for (const auto& testCase : testCases) {
        std::string matFile = caseDir + "/" + testCase + ".mat";
        std::string rhsFile = caseDir + "/" + testCase + ".rhs";
        std::string solFile = caseDir + "/" + testCase + ".sol";

        // Load matrix
        CSRMatrix mat;
        if (!loadMatrixMarket(matFile, mat)) {
            std::cerr << "Failed to load matrix: " << testCase << std::endl;
            continue;
        }

        // Load vectors (Matrix Market array format)
        std::vector<double> x, refSolution;
        if (!loadVectorMM(rhsFile, x)) {
            std::cerr << "Failed to load RHS: " << testCase << std::endl;
            continue;
        }
        if (!loadVectorMM(solFile, refSolution)) {
            std::cerr << "Failed to load reference solution: " << testCase << std::endl;
            continue;
        }

        std::cout << "Test Case: " << testCase << std::endl;
        std::cout << "  Matrix: " << mat.rows << " x " << mat.cols << ", nnz=" << mat.nnz
                  << ", avgNnz=" << (double)mat.nnz / mat.rows << std::endl;
        std::cout << "  x size: " << x.size() << ", y size: " << refSolution.size() << std::endl;

        // Verify sizes
        if (x.size() != mat.cols) {
            std::cerr << "  Warning: x size mismatch (expected " << mat.cols << ", got " << x.size() << ")" << std::endl;
        }
        if (refSolution.size() != mat.rows) {
            std::cerr << "  Warning: y size mismatch (expected " << mat.rows << ", got " << refSolution.size() << ")" << std::endl;
        }

        // Allocate device memory
        int* d_rowPtr;
        int* d_colIdx;
        double* d_values;
        double* d_x;
        double* d_y1;  // Library 1 output
        double* d_y2;  // Library 2 output

        cudaMalloc(&d_rowPtr, (mat.rows + 1) * sizeof(int));
        cudaMalloc(&d_colIdx, mat.nnz * sizeof(int));
        cudaMalloc(&d_values, mat.nnz * sizeof(double));
        cudaMalloc(&d_x, mat.cols * sizeof(double));
        cudaMalloc(&d_y1, mat.rows * sizeof(double));
        cudaMalloc(&d_y2, mat.rows * sizeof(double));

        // Copy data to device
        cudaMemcpy(d_rowPtr, mat.rowPtr.data(), (mat.rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_colIdx, mat.colIdx.data(), mat.nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_values, mat.values.data(), mat.nnz * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, x.data(), mat.cols * sizeof(double), cudaMemcpyHostToDevice);

        // CUDA Events for timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // ==================== Library 1: nctigpu_spmv ====================
        {
            auto cpuStart = std::chrono::high_resolution_clock::now();

            // Setup descriptors
            ncti::sparse::gpu::nctigpuCsrMatDescr_t<double, int, int> matA;
            matA.rows = mat.rows;
            matA.cols = mat.cols;
            matA.nnz = mat.nnz;
            matA.rowPtr = d_rowPtr;
            matA.colInd = d_colIdx;
            matA.values = d_values;

            ncti::sparse::gpu::nctigpuDnVecDescr_t<const double, int> vecX;
            vecX.size = mat.cols;
            vecX.values = d_x;

            ncti::sparse::gpu::nctigpuDnVecDescr_t<double, int> vecY;
            vecY.size = mat.rows;
            vecY.values = d_y1;

            double alpha = 1.0, beta = 0.0;

            // Kernel timing
            cudaEventRecord(start);
            auto status = ncti::sparse::gpu::nctigpuSpMV<double, int, int>(&alpha, &matA, &vecX, &beta, &vecY);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            auto cpuEnd = std::chrono::high_resolution_clock::now();

            float kernelTimeMs = 0;
            cudaEventElapsedTime(&kernelTimeMs, start, stop);

            double cpuTimeMs = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();

            // Get result back
            std::vector<double> y1(mat.rows);
            cudaMemcpy(y1.data(), d_y1, mat.rows * sizeof(double), cudaMemcpyDeviceToHost);

            // Calculate error
            double maxErr, avgErr, relErr;
            calculateError(y1, refSolution, maxErr, avgErr, relErr);

            // Calculate bandwidth
            double cpuBW = calculateBandwidth(mat.nnz, mat.rows, cpuTimeMs);
            double kernelBW = calculateBandwidth(mat.nnz, mat.rows, kernelTimeMs);

            std::cout << "| " << testCase << " | " << mat.rows << " | " << mat.nnz << " | "
                      << (double)mat.nnz/mat.rows << " | nctigpu | " << cpuTimeMs << " | "
                      << kernelTimeMs << " | " << cpuBW << " | " << kernelBW << " | "
                      << maxErr << " | " << relErr << " |" << std::endl;

            if (status != ncti::sparse::gpu::NCTIGPU_STATUS_SUCCESS) {
                std::cerr << "  Library 1 error status: " << status << std::endl;
            }
        }

        // ==================== Library 2: spmv_fp64 ====================
        {
            auto cpuStart = std::chrono::high_resolution_clock::now();

            // Kernel timing
            cudaEventRecord(start);
            spmv_fp64_status_t status = spmv_fp64_execute_direct(
                mat.rows, mat.nnz, d_rowPtr, d_colIdx, d_values, d_x, d_y2, 0);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            auto cpuEnd = std::chrono::high_resolution_clock::now();

            float kernelTimeMs = 0;
            cudaEventElapsedTime(&kernelTimeMs, start, stop);

            double cpuTimeMs = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();

            // Get result back
            std::vector<double> y2(mat.rows);
            cudaMemcpy(y2.data(), d_y2, mat.rows * sizeof(double), cudaMemcpyDeviceToHost);

            // Calculate error
            double maxErr, avgErr, relErr;
            calculateError(y2, refSolution, maxErr, avgErr, relErr);

            // Calculate bandwidth
            double cpuBW = calculateBandwidth(mat.nnz, mat.rows, cpuTimeMs);
            double kernelBW = calculateBandwidth(mat.nnz, mat.rows, kernelTimeMs);

            std::cout << "| " << testCase << " | " << mat.rows << " | " << mat.nnz << " | "
                      << (double)mat.nnz/mat.rows << " | spmv_fp64 | " << cpuTimeMs << " | "
                      << kernelTimeMs << " | " << cpuBW << " | " << kernelBW << " | "
                      << maxErr << " | " << relErr << " |" << std::endl;

            if (status != SPMV_FP64_SUCCESS) {
                std::cerr << "  Library 2 error: " << spmv_fp64_get_error_string(status) << std::endl;
            }
        }

        // Cleanup
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_rowPtr);
        cudaFree(d_colIdx);
        cudaFree(d_values);
        cudaFree(d_x);
        cudaFree(d_y1);
        cudaFree(d_y2);

        std::cout << std::endl;
    }

    std::cout << "===== Benchmark Complete =====" << std::endl;
    return 0;
}