/**
 * @file fp32_tpr_sweep_test.cu
 * @brief TPR sweep test to find optimal configuration for avgNnz=85 matrices
 *
 * Tests all TPR values (4,8,16,32,64) with different block sizes
 * to find the best configuration for Mars X201
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

// Test kernels with different TPR values
template<int WarpSize, int TPR>
__global__ void tpr_kernel_test(
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

        // Warp reduction
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

// __ldg kernel variant
template<int WarpSize>
__global__ void ldg_kernel_test(
    int numRows,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const float* __restrict__ values,
    const float* __restrict__ x,
    float* __restrict__ y)
{
    int row = blockIdx.x * (blockDim.x / WarpSize) + (threadIdx.x / WarpSize);
    int lane = threadIdx.x % WarpSize;

    if (row < numRows) {
        int rowStart = rowPtr[row];
        int rowEnd = rowPtr[row + 1];

        float sum = 0.0f;
        for (int i = rowStart + lane; i < rowEnd; i += WarpSize) {
            int col = colIdx[i];
            sum += values[i] * x[col];
        }

        // Warp reduction
        for (int offset = WarpSize/2; offset > 0; offset /= 2) {
            if (WarpSize == 64 && offset == 32) {
                sum += __shfl_down_sync(0xffffffffffffffffULL, sum, 32);
            } else {
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            }
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
    std::vector<float> values;
};

bool loadMTXMatrix(const std::string& filename, CSRMatrix& mat) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line[0] != '%') break;
    }

    std::istringstream iss(line);
    iss >> mat.rows >> mat.cols >> mat.nnz;

    std::cout << "Matrix: " << mat.rows << " rows, " << mat.cols << " cols, " << mat.nnz << " nnz" << std::endl;
    std::cout << "avgNnz: " << (double)mat.nnz / mat.rows << std::endl;

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

template<int WarpSize, int TPR>
double testTPR(const CSRMatrix& mat, const std::vector<float>& x,
               std::vector<float>& y, int blockSize) {

    int *d_rowPtr, *d_colIdx;
    float *d_values, *d_x, *d_y;

    cudaMalloc(&d_rowPtr, (mat.rows + 1) * sizeof(int));
    cudaMalloc(&d_colIdx, mat.nnz * sizeof(int));
    cudaMalloc(&d_values, mat.nnz * sizeof(float));
    cudaMalloc(&d_x, mat.cols * sizeof(float));
    cudaMalloc(&d_y, mat.rows * sizeof(float));

    cudaMemcpy(d_rowPtr, mat.rowPtr.data(), (mat.rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdx, mat.colIdx.data(), mat.nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, mat.values.data(), mat.nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), mat.cols * sizeof(float), cudaMemcpyHostToDevice);

    int rowsPerWarp = WarpSize / TPR;
    int numWarps = (mat.rows + rowsPerWarp - 1) / rowsPerWarp;
    int gridSize = (numWarps * WarpSize + blockSize - 1) / blockSize;

    cudaFuncSetCacheConfig(tpr_kernel_test<WarpSize, TPR>, cudaFuncCachePreferL1);

    // Warmup
    for (int w = 0; w < 5; w++) {
        tpr_kernel_test<WarpSize, TPR><<<gridSize, blockSize>>>(mat.rows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
        cudaDeviceSynchronize();
    }

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float minTime = 1e9f;
    for (int i = 0; i < 30; i++) {
        cudaEventRecord(start);
        tpr_kernel_test<WarpSize, TPR><<<gridSize, blockSize>>>(mat.rows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float timeMs = 0;
        cudaEventElapsedTime(&timeMs, start, stop);
        minTime = std::min(minTime, timeMs);
    }

    cudaMemcpy(y.data(), d_y, mat.rows * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_rowPtr);
    cudaFree(d_colIdx);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);

    return minTime;
}

template<int WarpSize>
double testLdg(const CSRMatrix& mat, const std::vector<float>& x,
               std::vector<float>& y, int blockSize) {

    int *d_rowPtr, *d_colIdx;
    float *d_values, *d_x, *d_y;

    cudaMalloc(&d_rowPtr, (mat.rows + 1) * sizeof(int));
    cudaMalloc(&d_colIdx, mat.nnz * sizeof(int));
    cudaMalloc(&d_values, mat.nnz * sizeof(float));
    cudaMalloc(&d_x, mat.cols * sizeof(float));
    cudaMalloc(&d_y, mat.rows * sizeof(float));

    cudaMemcpy(d_rowPtr, mat.rowPtr.data(), (mat.rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdx, mat.colIdx.data(), mat.nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, mat.values.data(), mat.nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), mat.cols * sizeof(float), cudaMemcpyHostToDevice);

    int gridSize = (mat.rows * WarpSize + blockSize - 1) / blockSize;

    cudaFuncSetCacheConfig(ldg_kernel_test<WarpSize>, cudaFuncCachePreferL1);

    // Warmup
    for (int w = 0; w < 5; w++) {
        ldg_kernel_test<WarpSize><<<gridSize, blockSize>>>(mat.rows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
        cudaDeviceSynchronize();
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float minTime = 1e9f;
    for (int i = 0; i < 30; i++) {
        cudaEventRecord(start);
        ldg_kernel_test<WarpSize><<<gridSize, blockSize>>>(mat.rows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float timeMs = 0;
        cudaEventElapsedTime(&timeMs, start, stop);
        minTime = std::min(minTime, timeMs);
    }

    cudaMemcpy(y.data(), d_y, mat.rows * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_rowPtr);
    cudaFree(d_colIdx);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);

    return minTime;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  FP32 TPR Sweep Test for Mars X201" << std::endl;
    std::cout << "========================================" << std::endl;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Warp Size: " << prop.warpSize << std::endl;
    double peakBW = 1843.2;
    std::cout << "Peak Bandwidth: " << peakBW << " GB/s" << std::endl;
    std::cout << std::endl;

    std::string caseDir = "/home/chenbinxiangc/spmv_comp/cases/shunyun";
    std::vector<std::string> testCases = {"pressure_0"};

    const int WarpSize = 64;
    std::vector<int> blockSizes = {128, 256, 512};
    std::vector<int> tprValues = {4, 8, 16, 32, 64};

    std::cout << "| Case | TPR | BlockSize | Time(ms) | BW(GB/s) | Util(%) | Rows/Warp |" << std::endl;
    std::cout << "|------|-----|-----------|----------|----------|---------|-----------|" << std::endl;

    for (const auto& testCase : testCases) {
        CSRMatrix mat;
        std::string matFile = caseDir + "/" + testCase + ".mat";

        if (!loadMTXMatrix(matFile, mat)) continue;

        std::vector<float> x(mat.cols, 1.0f);
        std::string rhsFile = caseDir + "/" + testCase + ".rhs";
        loadMTXVector(rhsFile, x);
        if (x.size() < mat.cols) x.resize(mat.cols, 1.0f);

        std::vector<float> y(mat.rows);

        // Test all TPR configurations
        for (int blockSize : blockSizes) {
            for (int tpr : tprValues) {
                double timeMs;
                switch (tpr) {
                    case 4:  timeMs = testTPR<WarpSize, 4>(mat, x, y, blockSize); break;
                    case 8:  timeMs = testTPR<WarpSize, 8>(mat, x, y, blockSize); break;
                    case 16: timeMs = testTPR<WarpSize, 16>(mat, x, y, blockSize); break;
                    case 32: timeMs = testTPR<WarpSize, 32>(mat, x, y, blockSize); break;
                    case 64: timeMs = testTPR<WarpSize, 64>(mat, x, y, blockSize); break;
                }

                double bw = calculateBandwidth(mat.nnz, mat.rows, timeMs);
                double util = bw / peakBW * 100;
                int rowsPerWarp = WarpSize / tpr;

                std::cout << "| " << testCase << " | " << tpr << " | " << blockSize
                          << " | " << timeMs << " | " << bw << " | " << util << " | "
                          << rowsPerWarp << " |" << std::endl;
            }
        }

        // Test __ldg variant
        std::cout << std::endl << "=== __ldg Kernel Tests ===" << std::endl;
        for (int blockSize : blockSizes) {
            double timeMs = testLdg<WarpSize>(mat, x, y, blockSize);
            double bw = calculateBandwidth(mat.nnz, mat.rows, timeMs);
            double util = bw / peakBW * 100;

            std::cout << "| " << testCase << " | ldg | " << blockSize
                      << " | " << timeMs << " | " << bw << " | " << util << " | 1 |" << std::endl;
        }
    }

    std::cout << std::endl << "===== Test Complete =====" << std::endl;
    return 0;
}