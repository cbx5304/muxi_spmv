/**
 * @file test_adaptive_thread.cu
 * @brief Test adaptive thread allocation based on row length
 *
 * Strategy:
 * - Short rows (< 8 elements): 4 threads/row
 * - Medium rows (8-32): 8 threads/row
 * - Long rows (> 32): 16 threads/row
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
    int* d_rowLen;  // Row lengths for adaptive selection
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

    // Compute row lengths
    int* rowLen = new int[rows];

    matrix.rowPtr[0] = 0;
    int currentRow = 0;
    for (int i = 0; i < nnz; i++) {
        int r = std::get<0>(entries[i]);
        while (currentRow < r) {
            rowLen[currentRow] = matrix.rowPtr[currentRow] - matrix.rowPtr[currentRow > 0 ? currentRow : 0];
            currentRow++;
            matrix.rowPtr[currentRow] = i;
        }
        matrix.colIdx[i] = std::get<1>(entries[i]);
        matrix.values[i] = std::get<2>(entries[i]);
    }
    while (currentRow < rows) {
        rowLen[currentRow] = matrix.rowPtr[currentRow] - matrix.rowPtr[currentRow > 0 ? currentRow : 0];
        currentRow++;
        matrix.rowPtr[currentRow] = nnz;
    }

    // Fix row lengths calculation
    for (int i = 0; i < rows; i++) {
        rowLen[i] = matrix.rowPtr[i + 1] - matrix.rowPtr[i];
    }

    matrix.d_rowLen = nullptr;
    cudaMalloc(&matrix.d_rowLen, rows * sizeof(int));
    cudaMemcpy(matrix.d_rowLen, rowLen, rows * sizeof(int), cudaMemcpyHostToDevice);

    delete[] rowLen;
    return true;
}

// Fixed 8 threads/row kernel
template<int BLOCK_SIZE>
__global__ void spmv_8t(int numRows, const int* __restrict__ rowPtr,
                        const int* __restrict__ colIdx,
                        const float* __restrict__ values,
                        const float* __restrict__ x, float* __restrict__ y) {
    int warpId = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int globalWarpId = blockIdx.x * (BLOCK_SIZE / WARP_SIZE) + warpId;

    int baseRow = globalWarpId * 8;
    int rowIdx = lane / 8;
    int threadInRow = lane % 8;
    int row = baseRow + rowIdx;

    if (row >= numRows) return;

    int rowStart = rowPtr[row];
    int rowEnd = rowPtr[row + 1];

    float sum = 0;
    for (int idx = rowStart + threadInRow; idx < rowEnd; idx += 8) {
        sum += values[idx] * __ldg(&x[colIdx[idx]]);
    }

    sum += __shfl_down_sync(0xffffffff, sum, 4, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 2, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 1, 8);

    if (threadInRow == 0) y[row] = sum;
}

// Fixed 16 threads/row kernel
template<int BLOCK_SIZE>
__global__ void spmv_16t(int numRows, const int* __restrict__ rowPtr,
                         const int* __restrict__ colIdx,
                         const float* __restrict__ values,
                         const float* __restrict__ x, float* __restrict__ y) {
    int warpId = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int globalWarpId = blockIdx.x * (BLOCK_SIZE / WARP_SIZE) + warpId;

    int baseRow = globalWarpId * 4;
    int rowIdx = lane / 16;
    int threadInRow = lane % 16;
    int row = baseRow + rowIdx;

    if (row >= numRows) return;

    int rowStart = rowPtr[row];
    int rowEnd = rowPtr[row + 1];

    float sum = 0;
    for (int idx = rowStart + threadInRow; idx < rowEnd; idx += 16) {
        sum += values[idx] * __ldg(&x[colIdx[idx]]);
    }

    sum += __shfl_down_sync(0xffffffff, sum, 8, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 4, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 2, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 1, 16);

    if (threadInRow == 0) y[row] = sum;
}

// Adaptive kernel - chooses thread count based on row length
template<int BLOCK_SIZE>
__global__ void spmv_adaptive(int numRows, const int* __restrict__ rowPtr,
                               const int* __restrict__ colIdx,
                               const float* __restrict__ values,
                               const int* __restrict__ rowLen,
                               const float* __restrict__ x, float* __restrict__ y) {
    int warpId = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int globalWarpId = blockIdx.x * (BLOCK_SIZE / WARP_SIZE) + warpId;

    // Process 4 rows per warp (each with 16 threads)
    int row = globalWarpId * 4 + lane / 16;

    if (row >= numRows) return;

    int len = rowLen[row];
    int rowStart = rowPtr[row];
    int rowEnd = rowPtr[row + 1];

    float sum = 0;
    int threadInRow = lane % 16;

    // Use all 16 threads for all rows (simple approach)
    for (int idx = rowStart + threadInRow; idx < rowEnd; idx += 16) {
        sum += values[idx] * __ldg(&x[colIdx[idx]]);
    }

    sum += __shfl_down_sync(0xffffffff, sum, 8, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 4, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 2, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 1, 16);

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

template<typename KernelFunc>
float runTestAdaptive(KernelFunc kernel, int gridSize, int blockSize, int iterations,
                      const CSRMatrix& matrix, float* d_x, float* d_y) {
    GpuTimer timer;
    float totalTime = 0;

    for (int i = 0; i < 5; i++) {
        kernel<<<gridSize, blockSize>>>(matrix.numRows, matrix.d_rowPtr,
                                        matrix.d_colIdx, matrix.d_values, matrix.d_rowLen, d_x, d_y);
        cudaDeviceSynchronize();
    }

    for (int i = 0; i < iterations; i++) {
        cudaMemset(d_y, 0, matrix.numRows * sizeof(float));
        cudaDeviceSynchronize();
        timer.start();
        kernel<<<gridSize, blockSize>>>(matrix.numRows, matrix.d_rowPtr,
                                        matrix.d_colIdx, matrix.d_values, matrix.d_rowLen, d_x, d_y);
        cudaDeviceSynchronize();
        timer.stop();
        totalTime += timer.elapsed_ms();
    }

    return totalTime / iterations;
}

int main(int argc, char** argv) {
    printf("=== Adaptive Thread Allocation Test ===\n");
    printf("WARP_SIZE = %d\n", WARP_SIZE);

    std::string matrixFile = argc > 1 ? argv[1] : "./real_cases/mtx/p0_A";
    int iterations = argc > 2 ? atoi(argv[2]) : 50;

    CSRMatrix matrix;
    if (!loadMatrixMarket(matrixFile, matrix)) {
        std::cerr << "Failed to load matrix: " << matrixFile << "\n";
        return 1;
    }

    // Analyze row length distribution
    int shortRows = 0, mediumRows = 0, longRows = 0;
    for (int i = 0; i < matrix.numRows; i++) {
        int len = matrix.rowPtr[i + 1] - matrix.rowPtr[i];
        if (len < 8) shortRows++;
        else if (len < 32) mediumRows++;
        else longRows++;
    }

    printf("Matrix: %d x %d, nnz=%d, avgNnz=%.2f\n",
           matrix.numRows, matrix.numCols, matrix.nnz,
           (double)matrix.nnz / matrix.numRows);
    printf("Row length distribution:\n");
    printf("  Short (<8): %d (%.1f%%)\n", shortRows, 100.0 * shortRows / matrix.numRows);
    printf("  Medium (8-32): %d (%.1f%%)\n", mediumRows, 100.0 * mediumRows / matrix.numRows);
    printf("  Long (>32): %d (%.1f%%)\n\n", longRows, 100.0 * longRows / matrix.numRows);

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

    int blockSize = 512;

    printf("%-15s %12s %12s\n", "Strategy", "Time(ms)", "Util(%)");
    printf("%-15s %12s %12s\n", "--------", "--------", "-------");

    // Test 8t/row
    int gridSize = (matrix.numRows + (blockSize / WARP_SIZE) * 8 - 1) / ((blockSize / WARP_SIZE) * 8);
    float t = runTest(spmv_8t<512>, gridSize, blockSize, iterations, matrix, d_x, d_y);
    float bw = (dataBytes / (t * 1e-3)) / (1024 * 1024 * 1024);
    printf("%-15s %12.3f %11.2f%%\n", "8t/row", t, bw / peakBW * 100);

    // Test 16t/row
    gridSize = (matrix.numRows + (blockSize / WARP_SIZE) * 4 - 1) / ((blockSize / WARP_SIZE) * 4);
    t = runTest(spmv_16t<512>, gridSize, blockSize, iterations, matrix, d_x, d_y);
    bw = (dataBytes / (t * 1e-3)) / (1024 * 1024 * 1024);
    printf("%-15s %12.3f %11.2f%%\n", "16t/row", t, bw / peakBW * 100);

    // Test adaptive
    gridSize = (matrix.numRows + (blockSize / WARP_SIZE) * 4 - 1) / ((blockSize / WARP_SIZE) * 4);
    t = runTestAdaptive(spmv_adaptive<512>, gridSize, blockSize, iterations, matrix, d_x, d_y);
    bw = (dataBytes / (t * 1e-3)) / (1024 * 1024 * 1024);
    printf("%-15s %12.3f %11.2f%%\n", "Adaptive", t, bw / peakBW * 100);

    delete[] h_x;
    delete[] matrix.rowPtr;
    delete[] matrix.colIdx;
    delete[] matrix.values;
    cudaFree(matrix.d_rowPtr);
    cudaFree(matrix.d_colIdx);
    cudaFree(matrix.d_values);
    cudaFree(matrix.d_rowLen);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}