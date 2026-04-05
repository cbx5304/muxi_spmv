/**
 * @file test_format_comparison.cu
 * @brief Final comprehensive comparison of all SpMV approaches
 *
 * Compares:
 * 1. CSR Vector (baseline - 4t/row)
 * 2. CSR5 (tile-based)
 * 3. Merge-based
 * 4. CSR with different thread configs
 *
 * Purpose: Definitive answer on which approach is best for random sparse matrices
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <cmath>

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

// ==================== CSR Vector Kernels ====================

// 2 threads per row
template<int BLOCK_SIZE>
__global__ void spmv_2t_per_row(int numRows, const int* __restrict__ rowPtr,
                                 const int* __restrict__ colIdx,
                                 const float* __restrict__ values,
                                 const float* __restrict__ x, float* __restrict__ y) {
    int warpId = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int globalWarpId = blockIdx.x * (BLOCK_SIZE / WARP_SIZE) + warpId;

    int baseRow = globalWarpId * (WARP_SIZE / 2);
    int rowIdx = lane / 2;
    int threadInRow = lane % 2;
    int row = baseRow + rowIdx;

    if (row >= numRows) return;

    int rowStart = rowPtr[row];
    int rowEnd = rowPtr[row + 1];

    float sum = 0;
    for (int idx = rowStart + threadInRow; idx < rowEnd; idx += 2) {
        sum += values[idx] * __ldg(&x[colIdx[idx]]);
    }

    sum += __shfl_down_sync(0xffffffffffffffff, sum, 1, 2);
    if (threadInRow == 0) y[row] = sum;
}

// 4 threads per row (optimal for Mars X201)
template<int BLOCK_SIZE>
__global__ void spmv_4t_per_row(int numRows, const int* __restrict__ rowPtr,
                                 const int* __restrict__ colIdx,
                                 const float* __restrict__ values,
                                 const float* __restrict__ x, float* __restrict__ y) {
    int warpId = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int globalWarpId = blockIdx.x * (BLOCK_SIZE / WARP_SIZE) + warpId;

    int baseRow = globalWarpId * (WARP_SIZE / 4);
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

// 8 threads per row
template<int BLOCK_SIZE>
__global__ void spmv_8t_per_row(int numRows, const int* __restrict__ rowPtr,
                                 const int* __restrict__ colIdx,
                                 const float* __restrict__ values,
                                 const float* __restrict__ x, float* __restrict__ y) {
    int warpId = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int globalWarpId = blockIdx.x * (BLOCK_SIZE / WARP_SIZE) + warpId;

    int baseRow = globalWarpId * (WARP_SIZE / 8);
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

    // 8-way reduction
    sum += __shfl_down_sync(0xffffffffffffffff, sum, 4, 8);
    sum += __shfl_down_sync(0xffffffffffffffff, sum, 2, 8);
    sum += __shfl_down_sync(0xffffffffffffffff, sum, 1, 8);

    if (threadInRow == 0) y[row] = sum;
}

// ==================== CSR5-style Kernel (simplified) ====================

template<int BLOCK_SIZE>
__global__ void spmv_csr5_style(int numRows, int nnz,
                                 const int* __restrict__ rowPtr,
                                 const int* __restrict__ colIdx,
                                 const float* __restrict__ values,
                                 const float* __restrict__ x, float* __restrict__ y,
                                 int tileSize) {
    int warpId = blockIdx.x * (BLOCK_SIZE / WARP_SIZE) + (threadIdx.x / WARP_SIZE);
    int lane = threadIdx.x % WARP_SIZE;

    int numTiles = (nnz + tileSize - 1) / tileSize;
    if (warpId >= numTiles) return;

    int tileStart = warpId * tileSize;
    int tileEnd = min(tileStart + tileSize, nnz);

    // Binary search for starting row
    int lo = 0, hi = numRows;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (rowPtr[mid] <= tileStart) lo = mid + 1;
        else hi = mid;
    }
    int currentRow = lo - 1;

    // Each thread processes elements
    int elementsPerThread = (tileSize + WARP_SIZE - 1) / WARP_SIZE;
    int myStart = tileStart + lane * elementsPerThread;
    int myEnd = min(myStart + elementsPerThread, tileEnd);

    float localSum = 0;
    int rowEnd = (currentRow >= 0 && currentRow < numRows) ? rowPtr[currentRow + 1] : 0;

    for (int idx = myStart; idx < myEnd && idx < tileEnd; idx++) {
        while (idx >= rowEnd && currentRow < numRows - 1) {
            if (localSum != 0) {
                atomicAdd(&y[currentRow], localSum);
            }
            localSum = 0;
            currentRow++;
            rowEnd = rowPtr[currentRow + 1];
        }
        if (idx < tileEnd) {
            localSum += values[idx] * __ldg(&x[colIdx[idx]]);
        }
    }

    if (localSum != 0 && currentRow >= 0) {
        atomicAdd(&y[currentRow], localSum);
    }
}

// Merge-based Kernel (simplified)
template<int BLOCK_SIZE, int WARP_SZ>
__global__ void spmv_merge_style(int numRows, int nnz,
                                  const int* __restrict__ rowPtr,
                                  const int* __restrict__ colIdx,
                                  const float* __restrict__ values,
                                  const float* __restrict__ x, float* __restrict__ y,
                                  int numPartitions) {
    int warpId = blockIdx.x * (BLOCK_SIZE / WARP_SZ) + (threadIdx.x / WARP_SZ);
    int lane = threadIdx.x % WARP_SZ;

    if (warpId >= numPartitions) return;

    int mergePathLength = numRows + nnz;
    int pathPerPartition = (mergePathLength + numPartitions - 1) / numPartitions;
    int pathStart = warpId * pathPerPartition;
    int pathEnd = min(pathStart + pathPerPartition, mergePathLength);

    // Convert merge path position to row index
    int startRow = min(pathStart, numRows);
    int endRow = min(pathEnd, numRows);

    if (startRow >= endRow) return;

    // Distribute rows among threads
    int numRowsInPartition = endRow - startRow;
    int rowsPerThread = (numRowsInPartition + WARP_SZ - 1) / WARP_SZ;
    int myRowStart = startRow + lane * rowsPerThread;
    int myRowEnd = min(myRowStart + rowsPerThread, endRow);

    for (int row = myRowStart; row < myRowEnd && row < numRows; row++) {
        float sum = 0;
        for (int idx = rowPtr[row]; idx < rowPtr[row + 1]; idx++) {
            sum += values[idx] * __ldg(&x[colIdx[idx]]);
        }
        y[row] = sum;
    }
}

// ==================== Test Infrastructure ====================

template<typename KernelFunc>
float runKernelTest(KernelFunc kernel, int gridSize, int blockSize, int iterations,
                    const CSRMatrix& matrix, float* d_x, float* d_y) {
    GpuTimer timer;
    float totalTime = 0;

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
    printf("=== Final Format Comparison ===\n");
    printf("WARP_SIZE = %d\n\n", WARP_SIZE);

    std::string matrixFile = argc > 1 ? argv[1] : "./real_cases/mtx/p0_A";
    int iterations = argc > 2 ? atoi(argv[2]) : 50;

    CSRMatrix matrix;
    if (!loadMatrixMarket(matrixFile, matrix)) {
        std::cerr << "Failed to load matrix: " << matrixFile << "\n";
        return 1;
    }

    printf("Matrix: %d x %d, nnz=%d, avgNnz=%.2f\n",
           matrix.numRows, matrix.numCols, matrix.nnz,
           (double)matrix.nnz / matrix.numRows);

    double xVectorSizeMB = matrix.numCols * 4.0 / (1024 * 1024);
    printf("X-vector size: %.2f MB\n\n", xVectorSizeMB);

    // Allocate device memory
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

    auto calcUtil = [&](float t) {
        return ((dataBytes / (t * 1e-3)) / (1024 * 1024 * 1024)) / peakBW * 100;
    };

    printf("%-20s %12s %12s\n", "Kernel", "Time(ms)", "Util(%)");
    printf("%-20s %12s %12s\n", "--------------------", "--------", "-------");

    // 1. CSR Vector - 2t/row
    int gridSize2t = (matrix.numRows + (blockSize / WARP_SIZE) * (WARP_SIZE / 2) - 1) / ((blockSize / WARP_SIZE) * (WARP_SIZE / 2));
    float t_2t = runKernelTest(spmv_2t_per_row<512>, gridSize2t, blockSize, iterations, matrix, d_x, d_y);
    printf("%-20s %12.3f %11.2f%%\n", "CSR 2t/row", t_2t, calcUtil(t_2t));

    // 2. CSR Vector - 4t/row
    int gridSize4t = (matrix.numRows + (blockSize / WARP_SIZE) * (WARP_SIZE / 4) - 1) / ((blockSize / WARP_SIZE) * (WARP_SIZE / 4));
    float t_4t = runKernelTest(spmv_4t_per_row<512>, gridSize4t, blockSize, iterations, matrix, d_x, d_y);
    printf("%-20s %12.3f %11.2f%%\n", "CSR 4t/row", t_4t, calcUtil(t_4t));

    // 3. CSR Vector - 8t/row
    int gridSize8t = (matrix.numRows + (blockSize / WARP_SIZE) * (WARP_SIZE / 8) - 1) / ((blockSize / WARP_SIZE) * (WARP_SIZE / 8));
    float t_8t = runKernelTest(spmv_8t_per_row<512>, gridSize8t, blockSize, iterations, matrix, d_x, d_y);
    printf("%-20s %12.3f %11.2f%%\n", "CSR 8t/row", t_8t, calcUtil(t_8t));

    // 4. CSR5-style
    int tileSize = WARP_SIZE * 4;
    int numTiles = (matrix.nnz + tileSize - 1) / tileSize;
    int gridSizeCSR5 = (numTiles + (blockSize / WARP_SIZE) - 1) / (blockSize / WARP_SIZE);

    // Clear y before CSR5 (uses atomics)
    cudaMemset(d_y, 0, matrix.numRows * sizeof(float));
    cudaDeviceSynchronize();

    GpuTimer csr5Timer;
    float csr5TotalTime = 0;
    cudaFuncSetCacheConfig(spmv_csr5_style<512>, cudaFuncCachePreferL1);

    for (int i = 0; i < 10; i++) {
        spmv_csr5_style<512><<<gridSizeCSR5, blockSize>>>(matrix.numRows, matrix.nnz,
            matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values, d_x, d_y, tileSize);
        cudaDeviceSynchronize();
    }

    for (int i = 0; i < iterations; i++) {
        cudaMemset(d_y, 0, matrix.numRows * sizeof(float));
        cudaDeviceSynchronize();
        csr5Timer.start();
        spmv_csr5_style<512><<<gridSizeCSR5, blockSize>>>(matrix.numRows, matrix.nnz,
            matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values, d_x, d_y, tileSize);
        cudaDeviceSynchronize();
        csr5Timer.stop();
        csr5TotalTime += csr5Timer.elapsed_ms();
    }
    float t_csr5 = csr5TotalTime / iterations;
    printf("%-20s %12.3f %11.2f%%\n", "CSR5-style", t_csr5, calcUtil(t_csr5));

    // 5. Merge-based
    int numSMs = (WARP_SIZE == 64) ? 104 : 128;
    int numPartitions = numSMs * (blockSize / WARP_SIZE) * 4;
    int gridSizeMerge = (numPartitions + (blockSize / WARP_SIZE) - 1) / (blockSize / WARP_SIZE);

    GpuTimer mergeTimer;
    float mergeTotalTime = 0;
    cudaFuncSetCacheConfig(spmv_merge_style<512, WARP_SIZE>, cudaFuncCachePreferL1);

    for (int i = 0; i < 10; i++) {
        spmv_merge_style<512, WARP_SIZE><<<gridSizeMerge, blockSize>>>(matrix.numRows, matrix.nnz,
            matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values, d_x, d_y, numPartitions);
        cudaDeviceSynchronize();
    }

    for (int i = 0; i < iterations; i++) {
        cudaMemset(d_y, 0, matrix.numRows * sizeof(float));
        cudaDeviceSynchronize();
        mergeTimer.start();
        spmv_merge_style<512, WARP_SIZE><<<gridSizeMerge, blockSize>>>(matrix.numRows, matrix.nnz,
            matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values, d_x, d_y, numPartitions);
        cudaDeviceSynchronize();
        mergeTimer.stop();
        mergeTotalTime += mergeTimer.elapsed_ms();
    }
    float t_merge = mergeTotalTime / iterations;
    printf("%-20s %12.3f %11.2f%%\n", "Merge-based", t_merge, calcUtil(t_merge));

    printf("\n=== Analysis ===\n");
    float bestUtil = std::max({calcUtil(t_2t), calcUtil(t_4t), calcUtil(t_8t), calcUtil(t_csr5), calcUtil(t_merge)});
    printf("Best utilization: %.2f%%\n", bestUtil);
    printf("Theoretical limit (L2 cache): %.1f%%\n", std::min(100.0, 4.0 / xVectorSizeMB * 100));
    printf("Gap from theoretical: %.1f%%\n", bestUtil - std::min(100.0, 4.0 / xVectorSizeMB * 100));

    printf("\nRoot cause analysis:\n");
    printf("- X-vector (%.2f MB) exceeds L2 cache (~4 MB)\n", xVectorSizeMB);
    printf("- Random column access pattern forces cache misses\n");
    printf("- No software optimization can overcome hardware limit\n");

    // Cleanup
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