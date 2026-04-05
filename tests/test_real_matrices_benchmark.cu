/**
 * @file test_real_matrices_benchmark.cu
 * @brief Comprehensive benchmark across all real matrices
 *
 * Tests all p0_A to p9_A matrices with:
 * 1. Best kernel configuration (Quad Accum + Vectorized)
 * 2. Pinned Memory for end-to-end measurement
 * 3. Both Mars X201 and RTX 4090
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <chrono>

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

// Best kernel: Quad Accumulator + Vectorized + __ldg
template<int BLOCK_SIZE, int THREADS_PER_ROW>
__global__ void spmv_optimized(int numRows, const int* __restrict__ rowPtr,
                                const int* __restrict__ colIdx,
                                const float* __restrict__ values,
                                const float* __restrict__ x, float* __restrict__ y) {
    int warpId = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int globalWarpId = blockIdx.x * (BLOCK_SIZE / WARP_SIZE) + warpId;

    int rowsPerWarp = WARP_SIZE / THREADS_PER_ROW;
    int baseRow = globalWarpId * rowsPerWarp;
    int rowIdx = lane / THREADS_PER_ROW;
    int threadInRow = lane % THREADS_PER_ROW;
    int row = baseRow + rowIdx;

    if (row >= numRows) return;

    int rowStart = rowPtr[row];
    int rowEnd = rowPtr[row + 1];

    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    int idx = rowStart + threadInRow;

    // Vectorized 4x with quad accumulators
    while (idx + THREADS_PER_ROW * 3 < rowEnd) {
        sum0 += values[idx] * __ldg(&x[colIdx[idx]]);
        sum1 += values[idx + THREADS_PER_ROW] * __ldg(&x[colIdx[idx + THREADS_PER_ROW]]);
        sum2 += values[idx + THREADS_PER_ROW * 2] * __ldg(&x[colIdx[idx + THREADS_PER_ROW * 2]]);
        sum3 += values[idx + THREADS_PER_ROW * 3] * __ldg(&x[colIdx[idx + THREADS_PER_ROW * 3]]);
        idx += THREADS_PER_ROW * 4;
    }

    // Remaining
    while (idx < rowEnd) {
        sum0 += values[idx] * __ldg(&x[colIdx[idx]]);
        idx += THREADS_PER_ROW;
    }

    float sum = sum0 + sum1 + sum2 + sum3;

    #pragma unroll
    for (int offset = THREADS_PER_ROW / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset, THREADS_PER_ROW);
    }

    if (threadInRow == 0) y[row] = sum;
}

struct BenchmarkResult {
    std::string matrixName;
    int numRows, numCols, nnz;
    double avgNnz;
    float kernelMs;
    float h2dMs, d2hMs;
    float totalMs;
    float kernelUtil;
    float totalUtil;
};

BenchmarkResult runBenchmark(const std::string& matrixFile, int iterations) {
    BenchmarkResult result;
    result.matrixName = matrixFile.substr(matrixFile.find_last_of("/\\") + 1);

    CSRMatrix matrix;
    if (!loadMatrixMarket(matrixFile, matrix)) {
        std::cerr << "Failed to load: " << matrixFile << "\n";
        return result;
    }

    result.numRows = matrix.numRows;
    result.numCols = matrix.numCols;
    result.nnz = matrix.nnz;
    result.avgNnz = (double)matrix.nnz / matrix.numRows;

    // Allocate pinned memory
    float* h_x;
    cudaMallocHost(&h_x, matrix.numCols * sizeof(float));
    for (int i = 0; i < matrix.numCols; i++) h_x[i] = rand() / (float)RAND_MAX;

    // Device memory
    cudaMalloc(&matrix.d_rowPtr, (matrix.numRows + 1) * sizeof(int));
    cudaMalloc(&matrix.d_colIdx, matrix.nnz * sizeof(int));
    cudaMalloc(&matrix.d_values, matrix.nnz * sizeof(float));
    cudaMemcpy(matrix.d_rowPtr, matrix.rowPtr, (matrix.numRows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(matrix.d_colIdx, matrix.colIdx, matrix.nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(matrix.d_values, matrix.values, matrix.nnz * sizeof(float), cudaMemcpyHostToDevice);

    float* d_x, *d_y;
    cudaMalloc(&d_x, matrix.numCols * sizeof(float));
    cudaMalloc(&d_y, matrix.numRows * sizeof(float));

    int blockSize = (WARP_SIZE == 64) ? 512 : 256;
    int threadsPerRow = (WARP_SIZE == 64) ? 8 : 4;
    int gridSize = (matrix.numRows + (blockSize / WARP_SIZE) * (WARP_SIZE / threadsPerRow) - 1) / ((blockSize / WARP_SIZE) * (WARP_SIZE / threadsPerRow));

    GpuTimer timer;
    float totalH2D = 0, totalKernel = 0, totalD2H = 0;

    // Warmup
    for (int i = 0; i < 3; i++) {
        cudaMemcpy(d_x, h_x, matrix.numCols * sizeof(float), cudaMemcpyHostToDevice);
        spmv_optimized<512, 8><<<gridSize, blockSize>>>(matrix.numRows, matrix.d_rowPtr,
                                                         matrix.d_colIdx, matrix.d_values, d_x, d_y);
        cudaDeviceSynchronize();
    }

    // Benchmark
    for (int i = 0; i < iterations; i++) {
        timer.start();
        cudaMemcpy(d_x, h_x, matrix.numCols * sizeof(float), cudaMemcpyHostToDevice);
        timer.stop();
        totalH2D += timer.elapsed_ms();

        timer.start();
        spmv_optimized<512, 8><<<gridSize, blockSize>>>(matrix.numRows, matrix.d_rowPtr,
                                                         matrix.d_colIdx, matrix.d_values, d_x, d_y);
        cudaDeviceSynchronize();
        timer.stop();
        totalKernel += timer.elapsed_ms();

        timer.start();
        float* h_y;
        cudaMallocHost(&h_y, matrix.numRows * sizeof(float));
        cudaMemcpy(h_y, d_y, matrix.numRows * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFreeHost(h_y);
        timer.stop();
        totalD2H += timer.elapsed_ms();
    }

    result.h2dMs = totalH2D / iterations;
    result.kernelMs = totalKernel / iterations;
    result.d2hMs = totalD2H / iterations;
    result.totalMs = result.h2dMs + result.kernelMs + result.d2hMs;

    // Calculate utilization
    float peakBW = (WARP_SIZE == 64) ? 1843.0f : 1008.0f;
    size_t dataBytes = (matrix.numRows + 1) * sizeof(int) +
                       matrix.nnz * sizeof(int) +
                       matrix.nnz * sizeof(float) * 2 +
                       matrix.numCols * sizeof(float) +
                       matrix.numRows * sizeof(float);

    float kernelBW = (dataBytes / (result.kernelMs * 1e-3)) / (1024 * 1024 * 1024);
    float totalBW = (dataBytes / (result.totalMs * 1e-3)) / (1024 * 1024 * 1024);

    result.kernelUtil = kernelBW / peakBW * 100;
    result.totalUtil = totalBW / peakBW * 100;

    // Cleanup
    cudaFreeHost(h_x);
    delete[] matrix.rowPtr;
    delete[] matrix.colIdx;
    delete[] matrix.values;
    cudaFree(matrix.d_rowPtr);
    cudaFree(matrix.d_colIdx);
    cudaFree(matrix.d_values);
    cudaFree(d_x);
    cudaFree(d_y);

    return result;
}

int main(int argc, char** argv) {
    printf("=== Real Matrices Comprehensive Benchmark ===\n");
    printf("WARP_SIZE = %d\n\n", WARP_SIZE);

    std::string baseDir = argc > 1 ? argv[1] : "./real_cases/mtx";
    int iterations = argc > 2 ? atoi(argv[2]) : 20;

    std::vector<std::string> matrices;
    for (int i = 0; i <= 9; i++) {
        matrices.push_back(baseDir + "/p" + std::to_string(i) + "_A");
    }

    printf("%-10s %10s %10s %12s %8s %8s %8s %8s %8s\n",
           "Matrix", "Rows", "NNZ", "avgNnz", "H2D(ms)", "Kernel(ms)", "D2H(ms)", "Total(ms)", "Util(%)");
    printf("%-10s %10s %10s %12s %8s %8s %8s %8s %8s\n",
           "------", "----", "---", "------", "-------", "---------", "-------", "---------", "-------");

    std::vector<BenchmarkResult> results;

    for (const auto& matrixFile : matrices) {
        BenchmarkResult r = runBenchmark(matrixFile, iterations);
        results.push_back(r);

        printf("%-10s %10d %10d %12.2f %8.3f %8.3f %8.3f %8.3f %7.2f%%\n",
               r.matrixName.c_str(), r.numRows, r.nnz, r.avgNnz,
               r.h2dMs, r.kernelMs, r.d2hMs, r.totalMs, r.kernelUtil);
    }

    // Summary
    printf("\n=== Summary ===\n");
    double avgKernelUtil = 0, avgTotalMs = 0;
    for (const auto& r : results) {
        avgKernelUtil += r.kernelUtil;
        avgTotalMs += r.totalMs;
    }
    avgKernelUtil /= results.size();
    avgTotalMs /= results.size();

    printf("Average kernel utilization: %.2f%%\n", avgKernelUtil);
    printf("Average end-to-end time: %.3f ms\n", avgTotalMs);

    float peakBW = (WARP_SIZE == 64) ? 1843.0f : 1008.0f;
    printf("Theoretical bandwidth: %.0f GB/s\n", peakBW);

    return 0;
}