/**
 * @file test_l1_cache_config.cu
 * @brief Test L1 cache configuration for SpMV optimization
 *
 * L1 cache can help with random x-vector access patterns.
 * Test different cache configurations:
 * - PreferL1: More L1, less shared memory
 * - PreferShared: More shared memory, less L1
 * - PreferEqual: Balanced
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

// Optimal kernel for each platform
template<int BLOCK_SIZE, int TPR>
__global__ void spmv_optimized(int numRows, const int* __restrict__ rowPtr,
                                const int* __restrict__ colIdx,
                                const float* __restrict__ values,
                                const float* __restrict__ x, float* __restrict__ y) {
    int warpId = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int globalWarpId = blockIdx.x * (BLOCK_SIZE / WARP_SIZE) + warpId;

    int rowsPerWarp = WARP_SIZE / TPR;
    int baseRow = globalWarpId * rowsPerWarp;
    int rowIdx = lane / TPR;
    int threadInRow = lane % TPR;
    int row = baseRow + rowIdx;

    if (row >= numRows) return;

    int rowStart = rowPtr[row];
    int rowEnd = rowPtr[row + 1];

    float sum0 = 0, sum1 = 0;
    int idx = rowStart + threadInRow;

    // Dual accumulator for better ILP
    for (; idx + TPR < rowEnd; idx += TPR * 2) {
        sum0 += values[idx] * __ldg(&x[colIdx[idx]]);
        sum1 += values[idx + TPR] * __ldg(&x[colIdx[idx + TPR]]);
    }
    for (; idx < rowEnd; idx += TPR) {
        sum0 += values[idx] * __ldg(&x[colIdx[idx]]);
    }

    float sum = sum0 + sum1;

    // Reduce within row
    #pragma unroll
    for (int offset = TPR / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset, TPR);
    }

    if (threadInRow == 0) y[row] = sum;
}

template<int BLOCK_SIZE, int TPR>
float runTestWithCache(const CSRMatrix& matrix, float* d_x, float* d_y,
                       int iterations, cudaFuncCache cacheConfig) {
    GpuTimer timer;
    float totalTime = 0;
    int rowsPerWarp = WARP_SIZE / TPR;
    int gridSize = (matrix.numRows + (BLOCK_SIZE / WARP_SIZE) * rowsPerWarp - 1) /
                   ((BLOCK_SIZE / WARP_SIZE) * rowsPerWarp);

    // Set cache config
    cudaFuncSetCacheConfig(spmv_optimized<BLOCK_SIZE, TPR>, cacheConfig);

    // Warmup
    for (int i = 0; i < 5; i++) {
        spmv_optimized<BLOCK_SIZE, TPR><<<gridSize, BLOCK_SIZE>>>(
            matrix.numRows, matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values, d_x, d_y);
        cudaDeviceSynchronize();
    }

    // Benchmark
    for (int i = 0; i < iterations; i++) {
        cudaMemset(d_y, 0, matrix.numRows * sizeof(float));
        cudaDeviceSynchronize();
        timer.start();
        spmv_optimized<BLOCK_SIZE, TPR><<<gridSize, BLOCK_SIZE>>>(
            matrix.numRows, matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values, d_x, d_y);
        cudaDeviceSynchronize();
        timer.stop();
        totalTime += timer.elapsed_ms();
    }

    return totalTime / iterations;
}

int main(int argc, char** argv) {
    printf("=== L1 Cache Configuration Test ===\n");
    printf("WARP_SIZE = %d\n", WARP_SIZE);

    std::string matrixFile = argc > 1 ? argv[1] : "./real_cases/mtx/p0_A";
    int iterations = argc > 2 ? atoi(argv[2]) : 30;

    CSRMatrix matrix;
    if (!loadMatrixMarket(matrixFile, matrix)) {
        std::cerr << "Failed to load matrix: " << matrixFile << "\n";
        return 1;
    }

    printf("Matrix: %d x %d, nnz=%d, avgNnz=%.2f\n\n",
           matrix.numRows, matrix.numCols, matrix.nnz,
           (double)matrix.nnz / matrix.numRows);

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

    float peakBW = (WARP_SIZE == 64) ? 1843.0f : 1008.0f;
    size_t dataBytes = (matrix.numRows + 1) * sizeof(int) +
                       matrix.nnz * sizeof(int) +
                       matrix.nnz * sizeof(float) * 2 +
                       matrix.numCols * sizeof(float) +
                       matrix.numRows * sizeof(float);

    printf("%-25s %12s %12s\n", "Cache Config", "Time(ms)", "Util(%)");
    printf("%-25s %12s %12s\n", "-------------------------", "--------", "-------");

    auto printResult = [&](const char* name, float t) {
        float bw = (dataBytes / (t * 1e-3)) / (1024 * 1024 * 1024);
        printf("%-25s %12.3f %11.2f%%\n", name, t, bw / peakBW * 100);
    };

    // Use optimal TPR for each platform
    int TPR = (WARP_SIZE == 64) ? 4 : 2;
    int blockSize = 512;

    float t;

    // Test different cache configurations
    t = runTestWithCache<512, 4>(matrix, d_x, d_y, iterations, cudaFuncCachePreferNone);
    printResult("Default (None)", t);

    t = runTestWithCache<512, 4>(matrix, d_x, d_y, iterations, cudaFuncCachePreferShared);
    printResult("PreferShared", t);

    t = runTestWithCache<512, 4>(matrix, d_x, d_y, iterations, cudaFuncCachePreferL1);
    printResult("PreferL1", t);

    t = runTestWithCache<512, 4>(matrix, d_x, d_y, iterations, cudaFuncCachePreferEqual);
    printResult("PreferEqual", t);

    printf("\n=== Analysis ===\n");
    printf("L1 cache can help with random x-vector access:\n");
    printf("- PreferL1: More L1 cache for x-vector caching\n");
    printf("- PreferShared: More shared memory (not useful here)\n");
    printf("- Default: Driver decides automatically\n");

    printf("\n=== Cache Line Size Impact ===\n");
    printf("x-vector size: %.2f MB\n", matrix.numCols * sizeof(float) / (1024.0 * 1024));
    printf("For random access, each load may fetch 128B cache line\n");
    printf("Effective bandwidth depends on cache hit rate\n");

    delete[] h_x;
    delete[] matrix.rowPtr;
    delete[] matrix.colIdx;
    delete[] matrix.values;
    cudaFree(matrix.d_rowPtr);
    cudaFree(matrix.d_colIdx);
    cudaFree(matrix.d_values);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}