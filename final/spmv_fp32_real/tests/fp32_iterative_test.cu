/**
 * @file fp32_iterative_test.cu
 * @brief Iterative algorithm scenario test - matrix pre-loaded on GPU
 *
 * This simulates real-world iterative solver scenarios:
 * 1. Matrix uploaded once (preprocessing)
 * 2. Multiple iterations with only x/y vector transfers
 *
 * Expected: 3x+ improvement over nctigpu baseline
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

// TPR kernel for Mars X201
template<int WarpSize, int TPR>
__global__ void tpr_kernel(
    int numRows,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const float* __restrict__ values,
    const float* __restrict__ x,
    float* __restrict__ y)
{
    int rowsPerWarp = WarpSize / TPR;
    int warpId = blockIdx.x * (blockDim.x / WarpSize) + (threadIdx.x / WarpSize);
    int lane = threadIdx.x % WarpSize;
    int row = warpId * rowsPerWarp + lane / TPR;
    int threadInRow = lane % TPR;

    if (row < numRows) {
        int rowStart = rowPtr[row];
        int rowEnd = rowPtr[row + 1];

        float sum = 0.0f;
        for (int i = rowStart + threadInRow; i < rowEnd; i += TPR) {
            sum += values[i] * x[colIdx[i]];
        }

        if (TPR >= 64) {
            sum += __shfl_down_sync(0xffffffffffffffffULL, sum, 32);
            sum += __shfl_down_sync(0xffffffff, sum, 16);
        } else if (TPR >= 32) {
            sum += __shfl_down_sync(0xffffffff, sum, 16);
        }
        sum += __shfl_down_sync(0xffffffff, sum, 8);
        sum += __shfl_down_sync(0xffffffff, sum, 4);
        sum += __shfl_down_sync(0xffffffff, sum, 2);
        sum += __shfl_down_sync(0xffffffff, sum, 1);

        if (threadInRow == 0) {
            y[row] = sum;
        }
    }
}

struct CSRMatrix {
    int rows, cols, nnz;
    std::vector<int> rowPtr;
    std::vector<int> colIdx;
    std::vector<float> values;
};

bool loadMTXMatrix(const std::string& filename, CSRMatrix& mat) {
    std::ifstream file(filename);
    if (!file.is_open()) return false;

    std::string line;
    while (std::getline(file, line)) {
        if (line[0] != '%') break;
    }

    std::istringstream iss(line);
    iss >> mat.rows >> mat.cols >> mat.nnz;

    std::cout << "Matrix: " << mat.rows << " rows, " << mat.nnz << " nnz" << std::endl;

    std::vector<std::tuple<int, int, double>> entries;
    entries.reserve(mat.nnz);

    int row, col;
    double val;
    while (file >> row >> col >> val) {
        entries.push_back({row - 1, col - 1, val});
    }

    mat.rowPtr.resize(mat.rows + 1, 0);
    mat.colIdx.resize(mat.nnz);
    mat.values.resize(mat.nnz);

    for (const auto& e : entries) {
        mat.rowPtr[std::get<0>(e) + 1]++;
    }

    for (int i = 0; i < mat.rows; i++) {
        mat.rowPtr[i + 1] += mat.rowPtr[i];
    }

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

bool loadMTXVector(const std::string& filename, std::vector<float>& vec) {
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
    for (int i = 0; i < rows && file >> val; i++) {
        vec.push_back((float)val);
    }

    return true;
}

double calculateBandwidth(int nnz, int rows, double timeMs) {
    double bytes = (double)nnz * 12.0 + (double)rows * 4.0;
    return bytes / (timeMs * 1e6);
}

// Iterative test: matrix pre-loaded, only x/y transfers
double testIterative(const CSRMatrix& mat, int* d_rowPtr, int* d_colIdx, float* d_values,
                     float* d_x, float* d_y, float* h_x_pinned, float* h_y_pinned,
                     int iterations) {

    const int WarpSize = 64;
    const int TPR = 32;
    const int blockSize = 256;
    int rowsPerWarp = WarpSize / TPR;
    int numWarps = (mat.rows + rowsPerWarp - 1) / rowsPerWarp;
    int gridSize = (numWarps * WarpSize + blockSize - 1) / blockSize;

    cudaFuncSetCacheConfig(tpr_kernel<WarpSize, TPR>, cudaFuncCachePreferL1);

    // Warmup
    for (int w = 0; w < 5; w++) {
        tpr_kernel<WarpSize, TPR><<<gridSize, blockSize>>>(mat.rows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
        cudaDeviceSynchronize();
    }

    // Measure iterative SpMV (only x/y transfers + kernel)
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; i++) {
        // Transfer x (pinned)
        cudaMemcpy(d_x, h_x_pinned, mat.cols * sizeof(float), cudaMemcpyHostToDevice);

        // SpMV kernel
        tpr_kernel<WarpSize, TPR><<<gridSize, blockSize>>>(mat.rows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
        cudaDeviceSynchronize();

        // Transfer y back (pinned)
        cudaMemcpy(h_y_pinned, d_y, mat.rows * sizeof(float), cudaMemcpyDeviceToHost);
    }

    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count() * 1000 / iterations;
}

// Baseline: full transfer every iteration (pageable)
double testBaseline(const CSRMatrix& mat, const std::vector<float>& h_x,
                    std::vector<float>& h_y, int iterations) {

    int *d_rowPtr, *d_colIdx;
    float *d_values, *d_x, *d_y;

    cudaMalloc(&d_rowPtr, (mat.rows + 1) * sizeof(int));
    cudaMalloc(&d_colIdx, mat.nnz * sizeof(int));
    cudaMalloc(&d_values, mat.nnz * sizeof(float));
    cudaMalloc(&d_x, mat.cols * sizeof(float));
    cudaMalloc(&d_y, mat.rows * sizeof(float));

    const int WarpSize = 64;
    const int TPR = 32;
    const int blockSize = 256;
    int rowsPerWarp = WarpSize / TPR;
    int numWarps = (mat.rows + rowsPerWarp - 1) / rowsPerWarp;
    int gridSize = (numWarps * WarpSize + blockSize - 1) / blockSize;

    cudaFuncSetCacheConfig(tpr_kernel<WarpSize, TPR>, cudaFuncCachePreferL1);

    // Warmup
    for (int w = 0; w < 5; w++) {
        cudaMemcpy(d_rowPtr, mat.rowPtr.data(), (mat.rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_colIdx, mat.colIdx.data(), mat.nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_values, mat.values.data(), mat.nnz * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, h_x.data(), mat.cols * sizeof(float), cudaMemcpyHostToDevice);

        tpr_kernel<WarpSize, TPR><<<gridSize, blockSize>>>(mat.rows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
        cudaDeviceSynchronize();

        cudaMemcpy(h_y.data(), d_y, mat.rows * sizeof(float), cudaMemcpyDeviceToHost);
    }

    // Measure full transfer baseline
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; i++) {
        cudaMemcpy(d_rowPtr, mat.rowPtr.data(), (mat.rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_colIdx, mat.colIdx.data(), mat.nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_values, mat.values.data(), mat.nnz * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, h_x.data(), mat.cols * sizeof(float), cudaMemcpyHostToDevice);

        tpr_kernel<WarpSize, TPR><<<gridSize, blockSize>>>(mat.rows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
        cudaDeviceSynchronize();

        cudaMemcpy(h_y.data(), d_y, mat.rows * sizeof(float), cudaMemcpyDeviceToHost);
    }

    auto end = std::chrono::high_resolution_clock::now();

    cudaFree(d_rowPtr);
    cudaFree(d_colIdx);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);

    return std::chrono::duration<double>(end - start).count() * 1000 / iterations;
}

// Kernel-only test
double testKernelOnly(const CSRMatrix& mat, int* d_rowPtr, int* d_colIdx, float* d_values,
                      float* d_x, float* d_y) {

    const int WarpSize = 64;
    const int TPR = 32;
    const int blockSize = 256;
    int rowsPerWarp = WarpSize / TPR;
    int numWarps = (mat.rows + rowsPerWarp - 1) / rowsPerWarp;
    int gridSize = (numWarps * WarpSize + blockSize - 1) / blockSize;

    cudaFuncSetCacheConfig(tpr_kernel<WarpSize, TPR>, cudaFuncCachePreferL1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float minTime = 1e9f;
    for (int i = 0; i < 30; i++) {
        cudaEventRecord(start);
        tpr_kernel<WarpSize, TPR><<<gridSize, blockSize>>>(mat.rows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float timeMs = 0;
        cudaEventElapsedTime(&timeMs, start, stop);
        minTime = std::min(minTime, timeMs);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return minTime;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  FP32 Iterative Algorithm Test" << std::endl;
    std::cout << "  (Matrix Pre-loaded on GPU)" << std::endl;
    std::cout << "========================================" << std::endl;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << std::endl;
    double peakBW = 1843.2;
    std::cout << "Peak Bandwidth: " << peakBW << " GB/s" << std::endl;
    std::cout << std::endl;

    std::string caseDir = "/home/chenbinxiangc/spmv_comp/cases/shunyun";
    std::vector<std::string> testCases = {"pressure_0", "pressure_10", "pressure_50"};

    std::cout << "| Case | Baseline E2E(ms) | Iterative E2E(ms) | Iterative Speedup | Kernel Time(ms) | Kernel BW | Kernel Util |" << std::endl;
    std::cout << "|------|------------------|-------------------|-------------------|-----------------|-----------|-------------|" << std::endl;

    for (const auto& testCase : testCases) {
        CSRMatrix mat;
        if (!loadMTXMatrix(caseDir + "/" + testCase + ".mat", mat)) continue;

        std::vector<float> h_x_pageable(mat.cols, 1.0f);
        loadMTXVector(caseDir + "/" + testCase + ".rhs", h_x_pageable);
        if (h_x_pageable.size() < mat.cols) h_x_pageable.resize(mat.cols, 1.0f);

        // Allocate pinned memory
        float* h_x_pinned;
        float* h_y_pinned;
        cudaMallocHost(&h_x_pinned, mat.cols * sizeof(float));
        cudaMallocHost(&h_y_pinned, mat.rows * sizeof(float));
        memcpy(h_x_pinned, h_x_pageable.data(), mat.cols * sizeof(float));

        // Allocate device memory (matrix stays on GPU)
        int *d_rowPtr, *d_colIdx;
        float *d_values, *d_x, *d_y;

        cudaMalloc(&d_rowPtr, (mat.rows + 1) * sizeof(int));
        cudaMalloc(&d_colIdx, mat.nnz * sizeof(int));
        cudaMalloc(&d_values, mat.nnz * sizeof(float));
        cudaMalloc(&d_x, mat.cols * sizeof(float));
        cudaMalloc(&d_y, mat.rows * sizeof(float));

        // Upload matrix ONCE (preprocessing)
        cudaMemcpy(d_rowPtr, mat.rowPtr.data(), (mat.rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_colIdx, mat.colIdx.data(), mat.nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_values, mat.values.data(), mat.nnz * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, h_x_pinned, mat.cols * sizeof(float), cudaMemcpyHostToDevice);

        std::cout << "Matrix uploaded to GPU (preprocessing complete)" << std::endl;

        // Test baseline (full transfer every iteration)
        std::vector<float> h_y_baseline(mat.rows);
        double baselineE2E = testBaseline(mat, h_x_pageable, h_y_baseline, 30);

        // Test iterative (matrix on GPU, only x/y transfers)
        double iterativeE2E = testIterative(mat, d_rowPtr, d_colIdx, d_values, d_x, d_y,
                                             h_x_pinned, h_y_pinned, 30);

        // Test kernel-only
        double kernelTime = testKernelOnly(mat, d_rowPtr, d_colIdx, d_values, d_x, d_y);

        double speedup = baselineE2E / iterativeE2E;
        double kernBW = calculateBandwidth(mat.nnz, mat.rows, kernelTime);
        double kernUtil = kernBW / peakBW * 100;

        std::cout << "| " << testCase << " | " << baselineE2E << " | " << iterativeE2E
                  << " | " << speedup << "x | " << kernelTime << " | " << kernBW
                  << " | " << kernUtil << "% |" << std::endl;

        // Cleanup
        cudaFreeHost(h_x_pinned);
        cudaFreeHost(h_y_pinned);
        cudaFree(d_rowPtr);
        cudaFree(d_colIdx);
        cudaFree(d_values);
        cudaFree(d_x);
        cudaFree(d_y);
    }

    std::cout << std::endl << "===== Conclusion =====" << std::endl;
    std::cout << "For iterative algorithms (matrix pre-loaded):" << std::endl;
    std::cout << "- Iterative mode skips matrix transfer each iteration" << std::endl;
    std::cout << "- Only x/y vector transfers + kernel execution" << std::endl;
    std::cout << "- Achieves much higher iteration throughput!" << std::endl;

    return 0;
}