/**
 * @file tpr_benchmark.cpp
 * @brief TPR (Threads-Per-Row) optimization benchmark for avgNnz~85 matrices
 *
 * Tests different TPR values to find optimal configuration for
 * pressure matrices (avgNnz ~85) on Mars X201 (warp=64)
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <cuda_runtime.h>

// Test TPR values from 4 to 64
template<int WarpSize, int TPR>
__global__ void tpr_kernel(
    int numRows,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const double* __restrict__ values,
    const double* __restrict__ x,
    double* __restrict__ y)
{
    int rowsPerWarp = WarpSize / TPR;
    int warpId = blockIdx.x * (blockDim.x / WarpSize) + (threadIdx.x / WarpSize);
    int lane = threadIdx.x % WarpSize;
    int row = warpId * rowsPerWarp + lane / TPR;
    int threadInRow = lane % TPR;

    if (row < numRows) {
        int rowStart = rowPtr[row];
        int rowEnd = rowPtr[row + 1];

        double sum = 0.0;
        for (int i = rowStart + threadInRow; i < rowEnd; i += TPR) {
            sum += values[i] * x[colIdx[i]];
        }

        for (int offset = TPR / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (threadInRow == 0) {
            y[row] = sum;
        }
    }
}

// Full warp per row kernel (TPR=64)
template<int WarpSize>
__global__ void full_warp_kernel(
    int numRows,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const double* __restrict__ values,
    const double* __restrict__ x,
    double* __restrict__ y)
{
    int row = blockIdx.x * (blockDim.x / WarpSize) + (threadIdx.x / WarpSize);
    int lane = threadIdx.x % WarpSize;

    if (row < numRows) {
        int rowStart = rowPtr[row];
        int rowEnd = rowPtr[row + 1];

        double sum = 0.0;
        for (int i = rowStart + lane; i < rowEnd; i += WarpSize) {
            sum += values[i] * x[colIdx[i]];
        }

        // Full warp reduction
        for (int offset = WarpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (lane == 0) {
            y[row] = sum;
        }
    }
}

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

// Template function to benchmark specific TPR
template<int WarpSize, int TPR>
void benchmark_tpr(CSRMatrix& mat, int* d_rowPtr, int* d_colIdx, double* d_values,
                   double* d_x, double* d_y, double* d_y_ref,
                   std::vector<double>& ref_y, int iterations) {

    const int blockSize = 256;
    int rowsPerWarp = WarpSize / TPR;
    int numWarps = (mat.rows + rowsPerWarp - 1) / rowsPerWarp;
    int gridSize = (numWarps * WarpSize + blockSize - 1) / blockSize;

    cudaFuncSetCacheConfig(tpr_kernel<WarpSize, TPR>, cudaFuncCachePreferL1);

    // Warmup
    tpr_kernel<WarpSize, TPR><<<gridSize, blockSize>>>(mat.rows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        tpr_kernel<WarpSize, TPR><<<gridSize, blockSize>>>(mat.rows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float timeMs = 0;
    cudaEventElapsedTime(&timeMs, start, stop);
    timeMs /= iterations;

    double bw = calculateBandwidth(mat.nnz, mat.rows, timeMs);

    // Verify correctness
    cudaMemcpy(d_y_ref, d_y, mat.rows * sizeof(double), cudaMemcpyDeviceToDevice);
    std::vector<double> y(mat.rows);
    cudaMemcpy(y.data(), d_y, mat.rows * sizeof(double), cudaMemcpyDeviceToHost);

    double maxDiff = 0;
    for (int i = 0; i < mat.rows; i++) {
        maxDiff = std::max(maxDiff, std::abs(y[i] - ref_y[i]));
    }

    std::cout << "| TPR=" << TPR << " | " << timeMs << " ms | " << bw
              << " GB/s | " << (bw/1843.2*100) << "% | " << rowsPerWarp
              << " rows/warp | max err=" << maxDiff << " |" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Benchmark full warp per row
template<int WarpSize>
void benchmark_full_warp(CSRMatrix& mat, int* d_rowPtr, int* d_colIdx, double* d_values,
                         double* d_x, double* d_y, std::vector<double>& ref_y, int iterations) {

    const int blockSize = 256;
    int gridSize = (mat.rows * WarpSize + blockSize - 1) / blockSize;

    cudaFuncSetCacheConfig(full_warp_kernel<WarpSize>, cudaFuncCachePreferL1);

    // Warmup
    full_warp_kernel<WarpSize><<<gridSize, blockSize>>>(mat.rows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        full_warp_kernel<WarpSize><<<gridSize, blockSize>>>(mat.rows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float timeMs = 0;
    cudaEventElapsedTime(&timeMs, start, stop);
    timeMs /= iterations;

    double bw = calculateBandwidth(mat.nnz, mat.rows, timeMs);

    std::vector<double> y(mat.rows);
    cudaMemcpy(y.data(), d_y, mat.rows * sizeof(double), cudaMemcpyDeviceToHost);

    double maxDiff = 0;
    for (int i = 0; i < mat.rows; i++) {
        maxDiff = std::max(maxDiff, std::abs(y[i] - ref_y[i]));
    }

    std::cout << "| TPR=64 (full warp) | " << timeMs << " ms | " << bw
              << " GB/s | " << (bw/1843.2*100) << "% | 1 row/warp | max err=" << maxDiff << " |" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    std::cout << "===== TPR Optimization Benchmark for avgNnz~85 Matrices =====" << std::endl;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << ", Warp: " << prop.warpSize << std::endl;
    std::cout << "Peak bandwidth: 1843.2 GB/s (Mars X201)" << std::endl;
    std::cout << std::endl;

    std::string caseDir = "/home/chenbinxiangc/spmv_comp/cases/shunyun";
    std::string testCase = "pressure_0";  // Use pressure_0 as representative

    CSRMatrix mat;
    if (!loadMatrixMarket(caseDir + "/" + testCase + ".mat", mat)) {
        std::cerr << "Failed to load matrix" << std::endl;
        return 1;
    }

    std::vector<double> x;
    loadVectorMM(caseDir + "/" + testCase + ".rhs", x);

    std::cout << "Matrix: " << testCase << std::endl;
    std::cout << "  Rows: " << mat.rows << ", NNZ: " << mat.nnz << std::endl;
    std::cout << "  avgNnz: " << (double)mat.nnz / mat.rows << std::endl;
    std::cout << "  x size: " << x.size() << std::endl;
    std::cout << std::endl;

    // Device memory
    int *d_rowPtr, *d_colIdx;
    double *d_values, *d_x, *d_y, *d_y_ref;

    cudaMalloc(&d_rowPtr, (mat.rows + 1) * sizeof(int));
    cudaMalloc(&d_colIdx, mat.nnz * sizeof(int));
    cudaMalloc(&d_values, mat.nnz * sizeof(double));
    cudaMalloc(&d_x, mat.cols * sizeof(double));
    cudaMalloc(&d_y, mat.rows * sizeof(double));
    cudaMalloc(&d_y_ref, mat.rows * sizeof(double));

    cudaMemcpy(d_rowPtr, mat.rowPtr.data(), (mat.rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdx, mat.colIdx.data(), mat.nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, mat.values.data(), mat.nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), mat.cols * sizeof(double), cudaMemcpyHostToDevice);

    // Get reference output using TPR=8
    const int WarpSize = 64;
    const int TPR_ref = 8;
    const int blockSize = 256;
    int rowsPerWarp = WarpSize / TPR_ref;
    int numWarps = (mat.rows + rowsPerWarp - 1) / rowsPerWarp;
    int gridSize = (numWarps * WarpSize + blockSize - 1) / blockSize;

    cudaFuncSetCacheConfig(tpr_kernel<WarpSize, TPR_ref>, cudaFuncCachePreferL1);
    tpr_kernel<WarpSize, TPR_ref><<<gridSize, blockSize>>>(mat.rows, d_rowPtr, d_colIdx, d_values, d_x, d_y_ref);
    cudaDeviceSynchronize();

    std::vector<double> ref_y(mat.rows);
    cudaMemcpy(ref_y.data(), d_y_ref, mat.rows * sizeof(double), cudaMemcpyDeviceToHost);

    std::cout << "Reference computed with TPR=8" << std::endl;
    std::cout << std::endl;

    std::cout << "| TPR | Time | Bandwidth | Utilization | Rows/Warp | Accuracy |" << std::endl;
    std::cout << "|-----|------|-----------|-------------|-----------|----------|" << std::endl;

    int iterations = 20;

    // Test different TPR values
    benchmark_tpr<64, 4>(mat, d_rowPtr, d_colIdx, d_values, d_x, d_y, d_y_ref, ref_y, iterations);
    benchmark_tpr<64, 8>(mat, d_rowPtr, d_colIdx, d_values, d_x, d_y, d_y_ref, ref_y, iterations);
    benchmark_tpr<64, 16>(mat, d_rowPtr, d_colIdx, d_values, d_x, d_y, d_y_ref, ref_y, iterations);
    benchmark_tpr<64, 32>(mat, d_rowPtr, d_colIdx, d_values, d_x, d_y, d_y_ref, ref_y, iterations);
    benchmark_full_warp<64>(mat, d_rowPtr, d_colIdx, d_values, d_x, d_y, ref_y, iterations);

    std::cout << std::endl;
    std::cout << "===== Benchmark Complete =====" << std::endl;

    cudaFree(d_rowPtr);
    cudaFree(d_colIdx);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_y_ref);

    return 0;
}