/**
 * @file test_block_precise.cu
 * @brief Precise block size test with correct template parameters
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

// Precise kernel with BLOCK_SIZE template parameter matching runtime
template<int BLOCK_SIZE>
__global__ void spmv_8t(int numRows, const int* __restrict__ rowPtr,
                        const int* __restrict__ colIdx,
                        const float* __restrict__ values,
                        const float* __restrict__ x, float* __restrict__ y) {
    __shared__ int sharedRowPtr[512];  // Large enough for all cases

    int warpId = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int globalWarpId = blockIdx.x * (BLOCK_SIZE / WARP_SIZE) + warpId;

    int baseRow = globalWarpId * 8;
    int warpOffset = warpId * 9;

    if (lane < 9 && baseRow + lane <= numRows) {
        sharedRowPtr[warpOffset + lane] = rowPtr[baseRow + lane];
    }
    __syncthreads();

    int rowIdx = lane / 8;
    int threadInRow = lane % 8;
    int row = baseRow + rowIdx;

    if (row >= numRows) return;

    float sum = 0;
    int rowStart = sharedRowPtr[warpOffset + rowIdx];
    int rowEnd = sharedRowPtr[warpOffset + rowIdx + 1];

    for (int idx = rowStart + threadInRow; idx < rowEnd; idx += 8) {
        sum += values[idx] * __ldg(&x[colIdx[idx]]);
    }

    sum += __shfl_down_sync(0xffffffff, sum, 4, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 2, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 1, 8);

    if (threadInRow == 0) y[row] = sum;
}

template<int BLOCK_SIZE>
float runTest(int iterations, const CSRMatrix& matrix, float* d_x, float* d_y) {
    GpuTimer timer;
    float totalTime = 0;
    int gridSize = (matrix.numRows + (BLOCK_SIZE / WARP_SIZE) * 8 - 1) / ((BLOCK_SIZE / WARP_SIZE) * 8);

    // Warmup
    for (int i = 0; i < 10; i++) {
        spmv_8t<BLOCK_SIZE><<<gridSize, BLOCK_SIZE>>>(matrix.numRows, matrix.d_rowPtr,
                                                       matrix.d_colIdx, matrix.d_values, d_x, d_y);
        cudaDeviceSynchronize();
    }

    // Timed runs
    for (int i = 0; i < iterations; i++) {
        cudaMemset(d_y, 0, matrix.numRows * sizeof(float));
        cudaDeviceSynchronize();
        timer.start();
        spmv_8t<BLOCK_SIZE><<<gridSize, BLOCK_SIZE>>>(matrix.numRows, matrix.d_rowPtr,
                                                       matrix.d_colIdx, matrix.d_values, d_x, d_y);
        cudaDeviceSynchronize();
        timer.stop();
        totalTime += timer.elapsed_ms();
    }

    return totalTime / iterations;
}

int main(int argc, char** argv) {
    printf("=== Precise Block Size Test ===\n");
    printf("WARP_SIZE = %d\n", WARP_SIZE);

    std::string matrixFile = argc > 1 ? argv[1] : "./real_cases/mtx/p0_A";
    int iterations = argc > 2 ? atoi(argv[2]) : 100;

    CSRMatrix matrix;
    if (!loadMatrixMarket(matrixFile, matrix)) {
        std::cerr << "Failed to load matrix\n";
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

    printf("%-10s %12s %12s %12s\n", "Block", "Time(ms)", "BW(GB/s)", "Util(%)");

    // Test each block size with correct template parameter
    float bestUtil = 0;
    int bestBlock = 0;
    float bestTime = 0;

    float t = runTest<64>(iterations, matrix, d_x, d_y);
    float bw = (dataBytes / (t * 1e-3)) / (1024 * 1024 * 1024);
    float util = bw / peakBW * 100;
    printf("%-10d %12.3f %12.1f %11.2f%%\n", 64, t, bw, util);
    if (util > bestUtil) { bestUtil = util; bestBlock = 64; bestTime = t; }

    t = runTest<128>(iterations, matrix, d_x, d_y);
    bw = (dataBytes / (t * 1e-3)) / (1024 * 1024 * 1024);
    util = bw / peakBW * 100;
    printf("%-10d %12.3f %12.1f %11.2f%%\n", 128, t, bw, util);
    if (util > bestUtil) { bestUtil = util; bestBlock = 128; bestTime = t; }

    t = runTest<256>(iterations, matrix, d_x, d_y);
    bw = (dataBytes / (t * 1e-3)) / (1024 * 1024 * 1024);
    util = bw / peakBW * 100;
    printf("%-10d %12.3f %12.1f %11.2f%%\n", 256, t, bw, util);
    if (util > bestUtil) { bestUtil = util; bestBlock = 256; bestTime = t; }

    t = runTest<512>(iterations, matrix, d_x, d_y);
    bw = (dataBytes / (t * 1e-3)) / (1024 * 1024 * 1024);
    util = bw / peakBW * 100;
    printf("%-10d %12.3f %12.1f %11.2f%%\n", 512, t, bw, util);
    if (util > bestUtil) { bestUtil = util; bestBlock = 512; bestTime = t; }

    t = runTest<1024>(iterations, matrix, d_x, d_y);
    bw = (dataBytes / (t * 1e-3)) / (1024 * 1024 * 1024);
    util = bw / peakBW * 100;
    printf("%-10d %12.3f %12.1f %11.2f%%\n", 1024, t, bw, util);
    if (util > bestUtil) { bestUtil = util; bestBlock = 1024; bestTime = t; }

    printf("\n=== Best Configuration ===\n");
    printf("Block Size: %d, Utilization: %.2f%%, Time: %.3f ms\n", bestBlock, bestUtil, bestTime);

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