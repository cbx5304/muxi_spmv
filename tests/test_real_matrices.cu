/**
 * @file test_real_matrices.cu
 * @brief Test SpMV kernels with real matrices from /real_cases
 *
 * Matrix Market format loader + end-to-end timing measurement
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <chrono>

#include "formats/sparse_formats.h"

using namespace muxi_spmv;

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

// Parse Matrix Market format
bool loadMatrixMarket(const std::string& filename, CSRMatrix<float>& matrix) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return false;
    }

    std::string line;
    // Skip header
    std::getline(file, line);
    if (line.find("MatrixMarket") == std::string::npos) {
        std::cerr << "Error: Not a Matrix Market file" << std::endl;
        return false;
    }

    // Skip comment lines
    while (std::getline(file, line) && line[0] == '%') {}

    // Parse dimensions
    std::istringstream iss(line);
    int rows, cols, nnz;
    iss >> rows >> cols >> nnz;

    std::cout << "Matrix: " << rows << " x " << cols << ", nnz=" << nnz
              << ", avgNnz=" << (double)nnz / rows << std::endl;

    matrix.numRows = rows;
    matrix.numCols = cols;
    matrix.nnz = nnz;

    // Read COO entries
    std::vector<std::tuple<int, int, float>> entries;
    entries.reserve(nnz);

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '%') continue;
        std::istringstream iss2(line);
        int r, c;
        float v;
        iss2 >> r >> c >> v;
        // Convert from 1-indexed to 0-indexed
        entries.push_back({r - 1, c - 1, v});
    }

    file.close();

    // Sort by row
    std::sort(entries.begin(), entries.end(),
              [](const auto& a, const auto& b) {
                  if (std::get<0>(a) != std::get<0>(b))
                      return std::get<0>(a) < std::get<0>(b);
                  return std::get<1>(a) < std::get<1>(b);
              });

    // Convert to CSR
    matrix.h_rowPtr = new int[rows + 1];
    matrix.h_colIdx = new int[nnz];
    matrix.h_values = new float[nnz];

    matrix.h_rowPtr[0] = 0;
    int currentRow = 0;
    for (int i = 0; i < nnz; i++) {
        int r = std::get<0>(entries[i]);
        int c = std::get<1>(entries[i]);
        float v = std::get<2>(entries[i]);

        while (currentRow < r) {
            currentRow++;
            matrix.h_rowPtr[currentRow] = i;
        }
        matrix.h_colIdx[i] = c;
        matrix.h_values[i] = v;
    }
    while (currentRow < rows) {
        currentRow++;
        matrix.h_rowPtr[currentRow] = nnz;
    }

    return true;
}

// Optimized kernel with prefetch - optimized for Mars X201 (warp=64)
template<typename FloatType, int BLOCK_SIZE, int SMEM_INTS>
__global__ void spmv_optimized_prefetch_kernel(
    int numRows,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y)
{
    __shared__ int sharedRowPtr[SMEM_INTS];

    int warpId = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int globalWarpId = blockIdx.x * (BLOCK_SIZE / WARP_SIZE) + warpId;

    // Adaptive warp: 16 rows per warp, 4 threads per row
    int baseRow = globalWarpId * 16;

    // Load row pointers: each warp needs 17 entries (16 rows + 1)
    int warpOffset = warpId * 17;
    if (lane < 17 && baseRow + lane <= numRows) {
        sharedRowPtr[warpOffset + lane] = rowPtr[baseRow + lane];
    }
    __syncthreads();

    int rowIdx = lane / 4;
    int threadInRow = lane % 4;
    int row = baseRow + rowIdx;

    if (row >= numRows) return;

    FloatType sum = 0;
    int rowStart = sharedRowPtr[warpOffset + rowIdx];
    int rowEnd = sharedRowPtr[warpOffset + rowIdx + 1];

    int idx = rowStart + threadInRow;
    if (idx < rowEnd) {
        FloatType x_val = __ldg(&x[colIdx[idx]]);
        FloatType v_val = values[idx];

        for (idx += 4; idx < rowEnd; idx += 4) {
            FloatType x_next = __ldg(&x[colIdx[idx]]);
            sum += v_val * x_val;
            x_val = x_next;
            v_val = values[idx];
        }
        sum += v_val * x_val;
    }

    // Warp reduction within group of 4 threads
    sum += __shfl_down_sync(0xffffffff, sum, 2, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 1, 4);

    if (threadInRow == 0) {
        y[row] = sum;
    }
}

// Merge-based kernel for comparison
template<typename FloatType, int BLOCK_SIZE>
__global__ void spmv_merge_kernel(
    int numRows,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y)
{
    int lane = threadIdx.x % WARP_SIZE;
    int globalWarpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;

    // Binary search for row boundaries
    int start = globalWarpId * BLOCK_SIZE;
    int end = min(start + BLOCK_SIZE, numRows);

    // Each thread finds its starting row
    int row = 0;
    int lo = 0, hi = numRows;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (rowPtr[mid] <= lane) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    row = lo - 1;

    // Merge-based accumulation
    FloatType sum = 0;
    int rowStart = rowPtr[row];
    int rowEnd = rowPtr[row + 1];

    for (int idx = rowStart + lane; idx < rowEnd; idx += WARP_SIZE) {
        sum += values[idx] * __ldg(&x[colIdx[idx]]);
    }

    // Warp reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) {
        y[row] = sum;
    }
}

void runTest(const std::string& matrixFile, const std::string& xFile, int iterations) {
    std::cout << "\n=== Testing: " << matrixFile << " ===\n";

    CSRMatrix<float> matrix;
    if (!loadMatrixMarket(matrixFile, matrix)) {
        return;
    }

    // Allocate and copy to device
    matrix.allocateDevice();
    matrix.copyToDevice();

    // Load x vector
    float* h_x = new float[matrix.numCols];
    if (!xFile.empty()) {
        std::ifstream xf(xFile);
        std::string line;
        std::getline(xf, line); // Skip header
        for (int i = 0; i < matrix.numCols && std::getline(xf, line); i++) {
            if (line[0] != '%') {
                std::istringstream iss(line);
                iss >> h_x[i];
            }
        }
    } else {
        for (int i = 0; i < matrix.numCols; i++) {
            h_x[i] = rand() / (float)RAND_MAX;
        }
    }

    float* d_x, *d_y;
    cudaMalloc(&d_x, matrix.numCols * sizeof(float));
    cudaMalloc(&d_y, matrix.numRows * sizeof(float));
    cudaMemcpy(d_x, h_x, matrix.numCols * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // Calculate data transfer size and peak bandwidth
    float peakBW = (WARP_SIZE == 64) ? 1843.0f : 1008.0f;
    size_t dataBytes = (matrix.numRows + 1) * sizeof(int) +
                       matrix.nnz * sizeof(int) +
                       matrix.nnz * sizeof(float) * 2 +
                       matrix.numCols * sizeof(float) +
                       matrix.numRows * sizeof(float);

    std::cout << "Data transfer: " << (dataBytes / (1024.0 * 1024.0)) << " MB\n";
    std::cout << "Peak bandwidth: " << peakBW << " GB/s\n";

    GpuTimer timer;

    // Test optimized kernel
    int BLOCK_SIZE = 512;
    int SMEM_INTS = 512;
    int rowsPerBlock = (BLOCK_SIZE / WARP_SIZE) * 16;
    int gridSize = (matrix.numRows + rowsPerBlock - 1) / rowsPerBlock;

    std::cout << "\n--- Optimized Kernel (Block=512, SMEM=512, Prefetch) ---\n";

    // End-to-end timing including data transfer
    float totalTransferTime = 0;
    float totalKernelTime = 0;

    for (int i = 0; i < iterations; i++) {
        // H2D transfer timing
        cudaDeviceSynchronize();
        timer.start();
        cudaMemcpy(d_x, h_x, matrix.numCols * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_y, 0, matrix.numRows * sizeof(float));
        cudaDeviceSynchronize();
        timer.stop();
        totalTransferTime += timer.elapsed_ms();

        // Kernel timing
        cudaDeviceSynchronize();
        timer.start();
        spmv_optimized_prefetch_kernel<float, 512, 512><<<gridSize, 512>>>(
            matrix.numRows, matrix.d_rowPtr, matrix.d_colIdx,
            matrix.d_values, d_x, d_y);
        cudaDeviceSynchronize();
        timer.stop();
        totalKernelTime += timer.elapsed_ms();
    }

    float avgTransferTime = totalTransferTime / iterations;
    float avgKernelTime = totalKernelTime / iterations;
    float avgTotalTime = avgTransferTime + avgKernelTime;

    float kernelBW = (dataBytes / (avgKernelTime * 1e-3)) / (1024 * 1024 * 1024);
    float kernelUtil = kernelBW / peakBW * 100;
    float totalBW = (dataBytes / (avgTotalTime * 1e-3)) / (1024 * 1024 * 1024);
    float totalUtil = totalBW / peakBW * 100;

    std::cout << "Avg transfer time: " << avgTransferTime << " ms\n";
    std::cout << "Avg kernel time: " << avgKernelTime << " ms\n";
    std::cout << "Avg total time: " << avgTotalTime << " ms\n";
    std::cout << "Kernel bandwidth: " << kernelBW << " GB/s (" << kernelUtil << "%)\n";
    std::cout << "End-to-end bandwidth: " << totalBW << " GB/s (" << totalUtil << "%)\n";

    // Verify result
    float* h_y = new float[matrix.numRows];
    cudaMemcpy(h_y, d_y, matrix.numRows * sizeof(float), cudaMemcpyDeviceToHost);

    // Simple verification: compute a few rows
    bool valid = true;
    for (int i = 0; i < 10 && valid; i++) {
        float expected = 0;
        for (int j = matrix.h_rowPtr[i]; j < matrix.h_rowPtr[i + 1]; j++) {
            expected += matrix.h_values[j] * h_x[matrix.h_colIdx[j]];
        }
        if (fabs(h_y[i] - expected) > 1e-3) {
            std::cerr << "Verification failed at row " << i << ": "
                      << h_y[i] << " vs " << expected << std::endl;
            valid = false;
        }
    }
    if (valid) {
        std::cout << "Verification: PASSED\n";
    }

    delete[] h_x;
    delete[] h_y;
    cudaFree(d_x);
    cudaFree(d_y);
}

int main(int argc, char** argv) {
    std::cout << "=== Real Matrix SpMV Test ===\n";
    std::cout << "WARP_SIZE = " << WARP_SIZE << "\n";

    std::string baseDir = argc > 1 ? argv[1] : "./mtx";

    int iterations = argc > 2 ? atoi(argv[2]) : 10;

    // Test all matrices
    for (int i = 0; i < 10; i++) {
        std::string matrixFile = baseDir + "/p" + std::to_string(i) + "_A";
        std::string xFile = baseDir + "/p" + std::to_string(i) + "_x";

        runTest(matrixFile, xFile, iterations);
    }

    return 0;
}