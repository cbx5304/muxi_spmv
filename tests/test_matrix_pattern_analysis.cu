/**
 * @file test_matrix_pattern_analysis.cu
 * @brief Analyze performance with different matrix patterns
 *
 * Test patterns:
 * 1. Random sparse (baseline)
 * 2. Banded matrix (expected high utilization)
 * 3. Block diagonal
 * 4. Perfectly balanced rows
 *
 * Goal: Verify hardware capabilities and identify bottleneck patterns
 */

#include <iostream>
#include <vector>
#include <random>

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

// Generate banded matrix
void generateBandedMatrix(int numRows, int numCols, int bandwidth, CSRMatrix& matrix) {
    matrix.numRows = numRows;
    matrix.numCols = numCols;

    std::vector<int> rowPtr(numRows + 1);
    std::vector<int> colIdx;
    std::vector<float> values;

    rowPtr[0] = 0;
    for (int i = 0; i < numRows; i++) {
        int rowStart = std::max(0, i - bandwidth / 2);
        int rowEnd = std::min(numCols - 1, i + bandwidth / 2);
        for (int j = rowStart; j <= rowEnd; j++) {
            colIdx.push_back(j);
            values.push_back(1.0f);
        }
        rowPtr[i + 1] = colIdx.size();
    }

    matrix.nnz = colIdx.size();
    matrix.rowPtr = new int[numRows + 1];
    matrix.colIdx = new int[matrix.nnz];
    matrix.values = new float[matrix.nnz];

    std::copy(rowPtr.begin(), rowPtr.end(), matrix.rowPtr);
    std::copy(colIdx.begin(), colIdx.end(), matrix.colIdx);
    std::copy(values.begin(), values.end(), matrix.values);
}

// Generate block diagonal matrix
void generateBlockDiagonalMatrix(int numRows, int numCols, int blockSize, CSRMatrix& matrix) {
    matrix.numRows = numRows;
    matrix.numCols = numCols;

    std::vector<int> rowPtr(numRows + 1);
    std::vector<int> colIdx;
    std::vector<float> values;

    rowPtr[0] = 0;
    for (int i = 0; i < numRows; i++) {
        int blockStart = (i / blockSize) * blockSize;
        for (int j = blockStart; j < blockStart + blockSize && j < numCols; j++) {
            colIdx.push_back(j);
            values.push_back(1.0f);
        }
        rowPtr[i + 1] = colIdx.size();
    }

    matrix.nnz = colIdx.size();
    matrix.rowPtr = new int[numRows + 1];
    matrix.colIdx = new int[matrix.nnz];
    matrix.values = new float[matrix.nnz];

    std::copy(rowPtr.begin(), rowPtr.end(), matrix.rowPtr);
    std::copy(colIdx.begin(), colIdx.end(), matrix.colIdx);
    std::copy(values.begin(), values.end(), matrix.values);
}

// Generate uniform row length matrix
void generateUniformMatrix(int numRows, int numCols, int nnzPerRow, CSRMatrix& matrix) {
    matrix.numRows = numRows;
    matrix.numCols = numCols;
    matrix.nnz = numRows * nnzPerRow;

    matrix.rowPtr = new int[numRows + 1];
    matrix.colIdx = new int[matrix.nnz];
    matrix.values = new float[matrix.nnz];

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> colDist(0, numCols - 1);

    matrix.rowPtr[0] = 0;
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < nnzPerRow; j++) {
            matrix.colIdx[i * nnzPerRow + j] = colDist(rng);
            matrix.values[i * nnzPerRow + j] = 1.0f;
        }
        matrix.rowPtr[i + 1] = (i + 1) * nnzPerRow;
    }
}

// Optimal SpMV kernel
template<int BLOCK_SIZE>
__global__ void spmv_optimal(int numRows, const int* __restrict__ rowPtr,
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
    sum += __shfl_down_sync(0xffffffffffffffff, sum, 2, 4);
    sum += __shfl_down_sync(0xffffffffffffffff, sum, 1, 4);

    if (threadInRow == 0) y[row] = sum;
}

float runTest(CSRMatrix& matrix, int iterations) {
    cudaMalloc(&matrix.d_rowPtr, (matrix.numRows + 1) * sizeof(int));
    cudaMalloc(&matrix.d_colIdx, matrix.nnz * sizeof(int));
    cudaMalloc(&matrix.d_values, matrix.nnz * sizeof(float));
    cudaMemcpy(matrix.d_rowPtr, matrix.rowPtr, (matrix.numRows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(matrix.d_colIdx, matrix.colIdx, matrix.nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(matrix.d_values, matrix.values, matrix.nnz * sizeof(float), cudaMemcpyHostToDevice);

    float* h_x = new float[matrix.numCols];
    for (int i = 0; i < matrix.numCols; i++) h_x[i] = 1.0f;

    float* d_x, *d_y;
    cudaMalloc(&d_x, matrix.numCols * sizeof(float));
    cudaMalloc(&d_y, matrix.numRows * sizeof(float));
    cudaMemcpy(d_x, h_x, matrix.numCols * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 512;
    int gridSize = (matrix.numRows + (blockSize / WARP_SIZE) * (WARP_SIZE / 4) - 1) / ((blockSize / WARP_SIZE) * (WARP_SIZE / 4));

    cudaFuncSetCacheConfig(spmv_optimal<512>, cudaFuncCachePreferL1);

    // Warmup
    for (int i = 0; i < 5; i++) {
        spmv_optimal<512><<<gridSize, blockSize>>>(matrix.numRows, matrix.d_rowPtr,
                                                    matrix.d_colIdx, matrix.d_values, d_x, d_y);
        cudaDeviceSynchronize();
    }

    GpuTimer timer;
    float totalTime = 0;

    for (int i = 0; i < iterations; i++) {
        cudaMemset(d_y, 0, matrix.numRows * sizeof(float));
        cudaDeviceSynchronize();
        timer.start();
        spmv_optimal<512><<<gridSize, blockSize>>>(matrix.numRows, matrix.d_rowPtr,
                                                    matrix.d_colIdx, matrix.d_values, d_x, d_y);
        cudaDeviceSynchronize();
        timer.stop();
        totalTime += timer.elapsed_ms();
    }

    delete[] h_x;
    cudaFree(matrix.d_rowPtr);
    cudaFree(matrix.d_colIdx);
    cudaFree(matrix.d_values);
    cudaFree(d_x);
    cudaFree(d_y);

    return totalTime / iterations;
}

void freeMatrix(CSRMatrix& matrix) {
    delete[] matrix.rowPtr;
    delete[] matrix.colIdx;
    delete[] matrix.values;
}

int main(int argc, char** argv) {
    printf("=== Matrix Pattern Analysis ===\n");
    printf("WARP_SIZE = %d\n\n", WARP_SIZE);

    int numRows = 100000;
    int numCols = 100000;
    int iterations = 30;
    float peakBW = (WARP_SIZE == 64) ? 1843.0f : 1008.0f;

    printf("Matrix size: %d x %d\n\n", numRows, numCols);

    printf("%-20s %12s %12s %12s %12s\n", "Pattern", "NNZ", "Time(ms)", "Util(%)", "GB/s");
    printf("%-20s %12s %12s %12s %12s\n", "--------------------", "--------", "--------", "-------", "-----");

    auto printResult = [&](const char* name, CSRMatrix& matrix, float time) {
        size_t dataBytes = (matrix.numRows + 1) * sizeof(int) +
                           matrix.nnz * sizeof(int) +
                           matrix.nnz * sizeof(float) * 2 +
                           matrix.numCols * sizeof(float) +
                           matrix.numRows * sizeof(float);
        float bw = (dataBytes / (time * 1e-3)) / (1024 * 1024 * 1024);
        float util = bw / peakBW * 100;
        printf("%-20s %12d %12.3f %11.2f%% %12.2f\n", name, matrix.nnz, time, util, bw);
    };

    // Test 1: Banded matrix (should have high utilization)
    {
        CSRMatrix matrix;
        generateBandedMatrix(numRows, numCols, 20, matrix);
        float t = runTest(matrix, iterations);
        printResult("Banded (bw=20)", matrix, t);
        freeMatrix(matrix);
    }

    // Test 2: Banded with larger bandwidth
    {
        CSRMatrix matrix;
        generateBandedMatrix(numRows, numCols, 64, matrix);
        float t = runTest(matrix, iterations);
        printResult("Banded (bw=64)", matrix, t);
        freeMatrix(matrix);
    }

    // Test 3: Block diagonal
    {
        CSRMatrix matrix;
        generateBlockDiagonalMatrix(numRows, numCols, 16, matrix);
        float t = runTest(matrix, iterations);
        printResult("Block Diagonal (16)", matrix, t);
        freeMatrix(matrix);
    }

    // Test 4: Block diagonal larger
    {
        CSRMatrix matrix;
        generateBlockDiagonalMatrix(numRows, numCols, 64, matrix);
        float t = runTest(matrix, iterations);
        printResult("Block Diagonal (64)", matrix, t);
        freeMatrix(matrix);
    }

    // Test 5: Uniform row length (10 nnz/row)
    {
        CSRMatrix matrix;
        generateUniformMatrix(numRows, numCols, 10, matrix);
        float t = runTest(matrix, iterations);
        printResult("Uniform (10 nnz/row)", matrix, t);
        freeMatrix(matrix);
    }

    // Test 6: Uniform row length (32 nnz/row)
    {
        CSRMatrix matrix;
        generateUniformMatrix(numRows, numCols, 32, matrix);
        float t = runTest(matrix, iterations);
        printResult("Uniform (32 nnz/row)", matrix, t);
        freeMatrix(matrix);
    }

    // Test 7: Uniform row length (64 nnz/row)
    {
        CSRMatrix matrix;
        generateUniformMatrix(numRows, numCols, 64, matrix);
        float t = runTest(matrix, iterations);
        printResult("Uniform (64 nnz/row)", matrix, t);
        freeMatrix(matrix);
    }

    printf("\n=== Analysis ===\n");
    printf("Testing different matrix patterns to understand hardware capabilities.\n");
    printf("Banded matrices should show higher utilization due to sequential x access.\n");
    printf("Block diagonal should show good cache locality.\n");

    return 0;
}