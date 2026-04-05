/**
 * @file test_optimal_thread_config.cu
 * @brief Comprehensive test for optimal thread configuration across all matrices
 *
 * Key finding: Different platforms need different thread configurations!
 * - Mars X201: 4t/row optimal
 * - RTX 4090: 2t/row optimal
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif

struct CSRMatrix {
    int numRows, numCols, nnz;
    int* rowPtr;
    int* colIdx;
    float* values;
    int* d_rowPtr;
    int* d_colIdx;
    float* d_values;
};

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

bool loadMatrixMarket(const std::string& filename, CSRMatrix& matrix) {
    std::ifstream file(filename);
    if (!file.is_open()) return false;

    std::string line;
    std::getline(file, line);
    while (std::getline(file, line) && line[0] == '%') {}

    std::istringstream iss(line);
    int rows, cols, nnz;
    iss >> rows >> cols >> nnz;

    matrix.numRows = rows;
    matrix.numCols = cols;
    matrix.nnz = nnz;

    std::vector<std::tuple<int, int, float>> entries;
    entries.reserve(nnz);

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '%') continue;
        std::istringstream iss2(line);
        int r, c;
        float v;
        iss2 >> r >> c >> v;
        entries.push_back({r - 1, c - 1, v});
    }

    std::sort(entries.begin(), entries.end());

    matrix.rowPtr = new int[rows + 1];
    matrix.colIdx = new int[nnz];
    matrix.values = new float[nnz];

    matrix.rowPtr[0] = 0;
    int currentRow = 0;
    for (int i = 0; i < nnz; i++) {
        int r = std::get<0>(entries[i]);
        while (currentRow < r) { currentRow++; matrix.rowPtr[currentRow] = i; }
        matrix.colIdx[i] = std::get<1>(entries[i]);
        matrix.values[i] = std::get<2>(entries[i]);
    }
    while (currentRow < rows) { currentRow++; matrix.rowPtr[currentRow] = nnz; }

    return true;
}

// 4t/row kernel - optimal for Mars X201
template<int BLOCK_SIZE>
__global__ void spmv_4t(int numRows, const int* __restrict__ rowPtr,
                         const int* __restrict__ colIdx,
                         const float* __restrict__ values,
                         const float* __restrict__ x, float* __restrict__ y) {
    int warpId = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int globalWarpId = blockIdx.x * (BLOCK_SIZE / WARP_SIZE) + warpId;

    int rowsPerWarp = WARP_SIZE / 4;
    int baseRow = globalWarpId * rowsPerWarp;
    int rowIdx = lane / 4;
    int threadInRow = lane % 4;
    int row = baseRow + rowIdx;

    if (row >= numRows) return;

    int rowStart = rowPtr[row];
    int rowEnd = rowPtr[row + 1];

    float sum0 = 0, sum1 = 0;
    int idx = rowStart + threadInRow;
    for (; idx + 4 < rowEnd; idx += 8) {
        sum0 += values[idx] * __ldg(&x[colIdx[idx]]);
        sum1 += values[idx + 4] * __ldg(&x[colIdx[idx + 4]]);
    }
    for (; idx < rowEnd; idx += 4) {
        sum0 += values[idx] * __ldg(&x[colIdx[idx]]);
    }

    float sum = sum0 + sum1;
    sum += __shfl_down_sync(0xffffffff, sum, 2, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 1, 4);

    if (threadInRow == 0) y[row] = sum;
}

// 2t/row kernel - optimal for RTX 4090
template<int BLOCK_SIZE>
__global__ void spmv_2t(int numRows, const int* __restrict__ rowPtr,
                         const int* __restrict__ colIdx,
                         const float* __restrict__ values,
                         const float* __restrict__ x, float* __restrict__ y) {
    int warpId = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int globalWarpId = blockIdx.x * (BLOCK_SIZE / WARP_SIZE) + warpId;

    int rowsPerWarp = WARP_SIZE / 2;
    int baseRow = globalWarpId * rowsPerWarp;
    int rowIdx = lane / 2;
    int threadInRow = lane % 2;
    int row = baseRow + rowIdx;

    if (row >= numRows) return;

    int rowStart = rowPtr[row];
    int rowEnd = rowPtr[row + 1];

    float sum0 = 0, sum1 = 0;
    int idx = rowStart + threadInRow;
    for (; idx + 2 < rowEnd; idx += 4) {
        sum0 += values[idx] * __ldg(&x[colIdx[idx]]);
        sum1 += values[idx + 2] * __ldg(&x[colIdx[idx + 2]]);
    }
    for (; idx < rowEnd; idx += 2) {
        sum0 += values[idx] * __ldg(&x[colIdx[idx]]);
    }

    float sum = sum0 + sum1;
    sum += __shfl_down_sync(0xffffffff, sum, 1, 2);

    if (threadInRow == 0) y[row] = sum;
}

// 8t/row kernel - previous baseline
template<int BLOCK_SIZE>
__global__ void spmv_8t(int numRows, const int* __restrict__ rowPtr,
                         const int* __restrict__ colIdx,
                         const float* __restrict__ values,
                         const float* __restrict__ x, float* __restrict__ y) {
    int warpId = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int globalWarpId = blockIdx.x * (BLOCK_SIZE / WARP_SIZE) + warpId;

    int rowsPerWarp = WARP_SIZE / 8;
    int baseRow = globalWarpId * rowsPerWarp;
    int rowIdx = lane / 8;
    int threadInRow = lane % 8;
    int row = baseRow + rowIdx;

    if (row >= numRows) return;

    int rowStart = rowPtr[row];
    int rowEnd = rowPtr[row + 1];

    float sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
    int idx = rowStart + threadInRow;
    for (; idx + 24 < rowEnd; idx += 32) {
        sum0 += values[idx] * __ldg(&x[colIdx[idx]]);
        sum1 += values[idx + 8] * __ldg(&x[colIdx[idx + 8]]);
        sum2 += values[idx + 16] * __ldg(&x[colIdx[idx + 16]]);
        sum3 += values[idx + 24] * __ldg(&x[colIdx[idx + 24]]);
    }
    for (; idx < rowEnd; idx += 8) {
        sum0 += values[idx] * __ldg(&x[colIdx[idx]]);
    }

    float sum = sum0 + sum1 + sum2 + sum3;
    sum += __shfl_down_sync(0xffffffff, sum, 4, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 2, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 1, 8);

    if (threadInRow == 0) y[row] = sum;
}

template<typename KernelFunc>
float runTest(KernelFunc kernel, int gridSize, int blockSize, int iterations,
              const CSRMatrix& matrix, float* d_x, float* d_y) {
    GpuTimer timer;
    float totalTime = 0;

    for (int i = 0; i < 5; i++) {
        kernel<<<gridSize, blockSize>>>(matrix.numRows, matrix.d_rowPtr,
                                        matrix.d_colIdx, matrix.d_values, d_x, d_y);
        cudaDeviceSynchronize();
    }

    for (int i = 0; i < iterations; i++) {
        cudaMemset(d_y, 0, matrix.numRows * sizeof(float));
        cudaDeviceSynchronize();
        timer.start();
        kernel<<<gridSize, blockSize>>>(matrix.numRows, matrix.d_rowPtr,
                                        matrix.d_colIdx, matrix.d_values, d_x, d_y);
        cudaDeviceSynchronize();
        timer.stop();
        totalTime += timer.elapsed_ms();
    }

    return totalTime / iterations;
}

int main(int argc, char** argv) {
    printf("=== Optimal Thread Configuration Test ===\n");
    printf("WARP_SIZE = %d\n\n", WARP_SIZE);

    std::string baseDir = argc > 1 ? argv[1] : "./real_cases/mtx";
    int iterations = argc > 2 ? atoi(argv[2]) : 20;

    float peakBW = (WARP_SIZE == 64) ? 1843.0f : 1008.0f;
    int blockSize = 512;

    printf("%-10s %10s %10s %10s %10s\n", "Matrix", "8t/row", "4t/row", "2t/row", "Best");
    printf("%-10s %10s %10s %10s %10s\n", "------", "--------", "-------", "-------", "----");

    double total_8t = 0, total_4t = 0, total_2t = 0;
    int count = 0;

    for (int m = 0; m <= 9; m++) {
        std::string matrixFile = baseDir + "/p" + std::to_string(m) + "_A";

        CSRMatrix matrix;
        if (!loadMatrixMarket(matrixFile, matrix)) continue;

        cudaMalloc(&matrix.d_rowPtr, (matrix.numRows + 1) * sizeof(int));
        cudaMalloc(&matrix.d_colIdx, matrix.nnz * sizeof(int));
        cudaMalloc(&matrix.d_values, matrix.nnz * sizeof(float));
        cudaMemcpy(matrix.d_rowPtr, matrix.rowPtr, (matrix.numRows + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(matrix.d_colIdx, matrix.colIdx, matrix.nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(matrix.d_values, matrix.values, matrix.nnz * sizeof(float), cudaMemcpyHostToDevice);

        float* h_x = new float[matrix.numCols];
        for (int i = 0; i < matrix.numCols; i++) h_x[i] = rand() / (float)RAND_MAX;

        float* d_x, *d_y;
        cudaMalloc(&d_x, matrix.numCols * sizeof(float));
        cudaMalloc(&d_y, matrix.numRows * sizeof(float));
        cudaMemcpy(d_x, h_x, matrix.numCols * sizeof(float), cudaMemcpyHostToDevice);

        // Grid sizes for different configurations
        int gs8 = (matrix.numRows + (blockSize / WARP_SIZE) * (WARP_SIZE / 8) - 1) / ((blockSize / WARP_SIZE) * (WARP_SIZE / 8));
        int gs4 = (matrix.numRows + (blockSize / WARP_SIZE) * (WARP_SIZE / 4) - 1) / ((blockSize / WARP_SIZE) * (WARP_SIZE / 4));
        int gs2 = (matrix.numRows + (blockSize / WARP_SIZE) * (WARP_SIZE / 2) - 1) / ((blockSize / WARP_SIZE) * (WARP_SIZE / 2));

        float t8 = runTest(spmv_8t<512>, gs8, blockSize, iterations, matrix, d_x, d_y);
        float t4 = runTest(spmv_4t<512>, gs4, blockSize, iterations, matrix, d_x, d_y);
        float t2 = runTest(spmv_2t<512>, gs2, blockSize, iterations, matrix, d_x, d_y);

        // Calculate utilizations
        size_t dataBytes = (matrix.numRows + 1) * sizeof(int) +
                           matrix.nnz * sizeof(int) +
                           matrix.nnz * sizeof(float) * 2 +
                           matrix.numCols * sizeof(float) +
                           matrix.numRows * sizeof(float);

        float u8 = ((dataBytes / (t8 * 1e-3)) / (1024 * 1024 * 1024)) / peakBW * 100;
        float u4 = ((dataBytes / (t4 * 1e-3)) / (1024 * 1024 * 1024)) / peakBW * 100;
        float u2 = ((dataBytes / (t2 * 1e-3)) / (1024 * 1024 * 1024)) / peakBW * 100;

        const char* best = (u4 >= u8 && u4 >= u2) ? "4t" : ((u2 >= u8) ? "2t" : "8t");
        printf("%-10s %9.2f%% %9.2f%% %9.2f%% %10s\n",
               ("p" + std::to_string(m) + "_A").c_str(), u8, u4, u2, best);

        total_8t += u8;
        total_4t += u4;
        total_2t += u2;
        count++;

        delete[] h_x;
        delete[] matrix.rowPtr;
        delete[] matrix.colIdx;
        delete[] matrix.values;
        cudaFree(matrix.d_rowPtr);
        cudaFree(matrix.d_colIdx);
        cudaFree(matrix.d_values);
        cudaFree(d_x);
        cudaFree(d_y);
    }

    printf("\n=== Summary ===\n");
    printf("Average 8t/row: %.2f%%\n", total_8t / count);
    printf("Average 4t/row: %.2f%%\n", total_4t / count);
    printf("Average 2t/row: %.2f%%\n", total_2t / count);

    if (WARP_SIZE == 64) {
        printf("\n*** Mars X201: 4t/row is optimal ***\n");
    } else {
        printf("\n*** RTX 4090: 2t/row is optimal ***\n");
    }

    return 0;
}