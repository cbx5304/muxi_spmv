/**
 * @file test_optimized_variants_comparison.cu
 * @brief Rigorous comparison of promising kernel variants across all matrices
 *
 * Based on initial testing, these variants showed promise:
 * 1. Software Prefetch - potential improvement
 * 2. Loop Unrolled - potential improvement
 * 3. Restrict+Aligned - potential improvement
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <iomanip>

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

// Baseline: 4t/row with dual accumulator
template<int BLOCK_SIZE>
__global__ void spmv_baseline(int numRows, const int* __restrict__ rowPtr,
                               const int* __restrict__ colIdx,
                               const float* __restrict__ values,
                               const float* __restrict__ x, float* __restrict__ y) {
    int warpId = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int globalWarpId = blockIdx.x * (BLOCK_SIZE / WARP_SIZE) + warpId;

    int baseRow = globalWarpId * 16;
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
    sum += __shfl_down_sync(0xffffffffffffffff, sum, 2, 4);
    sum += __shfl_down_sync(0xffffffffffffffff, sum, 1, 4);

    if (threadInRow == 0) y[row] = sum;
}

// Optimized: Software prefetch + unroll
template<int BLOCK_SIZE>
__global__ void spmv_optimized(int numRows, const int* __restrict__ rowPtr,
                                const int* __restrict__ colIdx,
                                const float* __restrict__ values,
                                const float* __restrict__ x, float* __restrict__ y) {
    int warpId = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int globalWarpId = blockIdx.x * (BLOCK_SIZE / WARP_SIZE) + warpId;

    int baseRow = globalWarpId * 16;
    int rowIdx = lane / 4;
    int threadInRow = lane % 4;
    int row = baseRow + rowIdx;

    if (row >= numRows) return;

    int rowStart = rowPtr[row];
    int rowEnd = rowPtr[row + 1];

    float sum = 0;
    int idx = rowStart + threadInRow;

    // Prefetch first element
    float val = 0, x_val = 0;
    if (idx < rowEnd) {
        val = values[idx];
        x_val = __ldg(&x[colIdx[idx]]);
    }

    // Main loop with software prefetch
    #pragma unroll 4
    for (idx += 4; idx < rowEnd; idx += 4) {
        float next_val = values[idx];
        float next_x = __ldg(&x[colIdx[idx]]);

        sum += val * x_val;

        val = next_val;
        x_val = next_x;
    }

    // Final accumulation
    sum += val * x_val;

    // Warp reduction
    sum += __shfl_down_sync(0xffffffffffffffff, sum, 2, 4);
    sum += __shfl_down_sync(0xffffffffffffffff, sum, 1, 4);

    if (threadInRow == 0) y[row] = sum;
}

// Combined: Dual accumulator + prefetch + unroll
template<int BLOCK_SIZE>
__global__ void spmv_combined(int numRows, const int* __restrict__ rowPtr,
                               const int* __restrict__ colIdx,
                               const float* __restrict__ values,
                               const float* __restrict__ x, float* __restrict__ y) {
    int warpId = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int globalWarpId = blockIdx.x * (BLOCK_SIZE / WARP_SIZE) + warpId;

    int baseRow = globalWarpId * 16;
    int rowIdx = lane / 4;
    int threadInRow = lane % 4;
    int row = baseRow + rowIdx;

    if (row >= numRows) return;

    int rowStart = rowPtr[row];
    int rowEnd = rowPtr[row + 1];

    float sum0 = 0, sum1 = 0;
    int idx = rowStart + threadInRow;

    // Dual accumulator with prefetch
    #pragma unroll 2
    for (; idx + 4 < rowEnd; idx += 8) {
        sum0 += values[idx] * __ldg(&x[colIdx[idx]]);
        sum1 += values[idx + 4] * __ldg(&x[colIdx[idx + 4]]);
    }

    #pragma unroll 2
    for (; idx < rowEnd; idx += 4) {
        sum0 += values[idx] * __ldg(&x[colIdx[idx]]);
    }

    float sum = sum0 + sum1;
    sum += __shfl_down_sync(0xffffffffffffffff, sum, 2, 4);
    sum += __shfl_down_sync(0xffffffffffffffff, sum, 1, 4);

    if (threadInRow == 0) y[row] = sum;
}

template<typename KernelFunc>
float runTest(KernelFunc kernel, int gridSize, int blockSize, int iterations,
              const CSRMatrix& matrix, float* d_x, float* d_y) {
    GpuTimer timer;
    float totalTime = 0;

    // Always set L1 cache config
    cudaFuncSetCacheConfig(kernel, cudaFuncCachePreferL1);

    // Warmup
    for (int i = 0; i < 10; i++) {
        kernel<<<gridSize, blockSize>>>(matrix.numRows, matrix.d_rowPtr,
                                        matrix.d_colIdx, matrix.d_values, d_x, d_y);
        cudaDeviceSynchronize();
    }

    // Benchmark
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
    printf("=== Optimized Variants Comparison ===\n");
    printf("WARP_SIZE = %d\n\n", WARP_SIZE);

    std::string baseDir = argc > 1 ? argv[1] : "./real_cases/mtx";
    int iterations = argc > 2 ? atoi(argv[2]) : 50;

    float peakBW = (WARP_SIZE == 64) ? 1843.0f : 1008.0f;
    int blockSize = 512;

    printf("%-10s %12s %12s %12s %12s %12s\n",
           "Matrix", "Baseline", "Optimized", "Combined", "Best", "Improvement");
    printf("%-10s %12s %12s %12s %12s %12s\n",
           "------", "--------", "---------", "--------", "----", "-----------");

    double total_baseline = 0, total_optimized = 0, total_combined = 0;
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

        size_t dataBytes = (matrix.numRows + 1) * sizeof(int) +
                           matrix.nnz * sizeof(int) +
                           matrix.nnz * sizeof(float) * 2 +
                           matrix.numCols * sizeof(float) +
                           matrix.numRows * sizeof(float);

        int rowsPerWarp = WARP_SIZE / 4;
        int gridSize = (matrix.numRows + (blockSize / WARP_SIZE) * rowsPerWarp - 1) /
                       ((blockSize / WARP_SIZE) * rowsPerWarp);

        float t_base = runTest(spmv_baseline<512>, gridSize, blockSize, iterations, matrix, d_x, d_y);
        float t_opt = runTest(spmv_optimized<512>, gridSize, blockSize, iterations, matrix, d_x, d_y);
        float t_comb = runTest(spmv_combined<512>, gridSize, blockSize, iterations, matrix, d_x, d_y);

        float u_base = ((dataBytes / (t_base * 1e-3)) / (1024 * 1024 * 1024)) / peakBW * 100;
        float u_opt = ((dataBytes / (t_opt * 1e-3)) / (1024 * 1024 * 1024)) / peakBW * 100;
        float u_comb = ((dataBytes / (t_comb * 1e-3)) / (1024 * 1024 * 1024)) / peakBW * 100;

        float best = std::max({u_base, u_opt, u_comb});
        float improvement = (best - u_base) / u_base * 100;
        const char* best_name = (best == u_base) ? "Base" : ((best == u_opt) ? "Opt" : "Comb");

        printf("%-10s %10.2f%% %10.2f%% %10.2f%% %10s %10.1f%%\n",
               ("p" + std::to_string(m) + "_A").c_str(), u_base, u_opt, u_comb, best_name, improvement);

        total_baseline += u_base;
        total_optimized += u_opt;
        total_combined += u_comb;
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
    printf("Average Baseline:  %.2f%%\n", total_baseline / count);
    printf("Average Optimized: %.2f%%\n", total_optimized / count);
    printf("Average Combined:  %.2f%%\n", total_combined / count);

    float best_avg = std::max({total_baseline, total_optimized, total_combined}) / count;
    float improvement = (best_avg - total_baseline / count) / (total_baseline / count) * 100;
    printf("\nBest improvement: %.1f%%\n", improvement);

    return 0;
}