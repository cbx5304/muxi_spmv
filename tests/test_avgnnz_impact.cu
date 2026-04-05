/**
 * @file test_avgnnz_impact.cu
 * @brief Test performance impact of different avgNnz patterns (debug version)
 */

#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif

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

// Generate CSR matrix with specific avgNnz
struct CSRMatrix {
    int numRows, numCols, nnz;
    int* rowPtr;
    int* colIdx;
    float* values;
    int* d_rowPtr;
    int* d_colIdx;
    float* d_values;

    void generate(int rows, int cols, int avgNnz, int seed = 42) {
        srand(seed);
        numRows = rows;
        numCols = cols;

        // Allocate maximum possible size
        int maxNnz = rows * (avgNnz + 3);  // Account for variation
        rowPtr = new int[rows + 1];
        colIdx = new int[maxNnz];
        values = new float[maxNnz];

        rowPtr[0] = 0;
        int currentNnz = 0;

        for (int i = 0; i < rows; i++) {
            int rowLen = avgNnz + (rand() % 5) - 2;
            if (rowLen < 1) rowLen = 1;
            if (rowLen > cols) rowLen = cols;

            // Generate unique sorted column indices
            std::vector<int> rowCols;
            for (int j = 0; j < rowLen; j++) {
                rowCols.push_back(rand() % numCols);
            }
            std::sort(rowCols.begin(), rowCols.end());
            rowCols.erase(std::unique(rowCols.begin(), rowCols.end()), rowCols.end());

            for (int c : rowCols) {
                colIdx[currentNnz] = c;
                values[currentNnz] = (float)rand() / (float)RAND_MAX;
                currentNnz++;
            }
            rowPtr[i + 1] = currentNnz;
        }
        nnz = currentNnz;
    }

    void allocateDevice() {
        cudaMalloc(&d_rowPtr, (numRows + 1) * sizeof(int));
        cudaMalloc(&d_colIdx, nnz * sizeof(int));
        cudaMalloc(&d_values, nnz * sizeof(float));
        cudaMemcpy(d_rowPtr, rowPtr, (numRows + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_colIdx, colIdx, nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_values, values, nnz * sizeof(float), cudaMemcpyHostToDevice);
    }

    void freeHost() {
        delete[] rowPtr;
        delete[] colIdx;
        delete[] values;
    }

    void freeDevice() {
        cudaFree(d_rowPtr);
        cudaFree(d_colIdx);
        cudaFree(d_values);
    }
};

// Simple 8 threads/row kernel without shared memory
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

// Simple 4 threads/row kernel
template<int BLOCK_SIZE>
__global__ void spmv_4t(int numRows, const int* __restrict__ rowPtr,
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
    for (int idx = rowStart + threadInRow; idx < rowEnd; idx += 4) {
        sum += values[idx] * __ldg(&x[colIdx[idx]]);
    }

    sum += __shfl_down_sync(0xffffffff, sum, 2, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 1, 4);

    if (threadInRow == 0) y[row] = sum;
}

// Simple 16 threads/row kernel
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

int main(int argc, char** argv) {
    printf("=== avgNnz Impact Test (Debug Version) ===\n");
    printf("WARP_SIZE = %d\n", WARP_SIZE);

    int numRows = 100000;
    int numCols = 100000;
    int iterations = 30;

    float peakBW = (WARP_SIZE == 64) ? 1843.0f : 1008.0f;

    int avgNnzValues[] = {4, 8, 16, 32, 64, 128};
    int numTests = sizeof(avgNnzValues) / sizeof(avgNnzValues[0]);

    printf("\n%-10s %12s %12s %12s %12s\n", "avgNnz", "8t/row", "4t/row", "16t/row", "Best");
    printf("%-10s %12s %12s %12s %12s\n", "", "(%)", "(%)", "(%)", "(%)");
    fflush(stdout);

    for (int t = 0; t < numTests; t++) {
        int avgNnz = avgNnzValues[t];

        // Generate fresh matrix for each test
        CSRMatrix matrix;
        matrix.generate(numRows, numCols, avgNnz);

        printf("Testing avgNnz=%d, nnz=%d... ", avgNnz, matrix.nnz);
        fflush(stdout);

        // Allocate x and y vectors fresh
        float* h_x = new float[numCols];
        for (int i = 0; i < numCols; i++) h_x[i] = (float)i / numCols;

        float* d_x, *d_y;
        cudaMalloc(&d_x, numCols * sizeof(float));
        cudaMalloc(&d_y, numRows * sizeof(float));
        cudaMemcpy(d_x, h_x, numCols * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_y, 0, numRows * sizeof(float));
        cudaDeviceSynchronize();

        // Allocate matrix on device
        matrix.allocateDevice();
        cudaDeviceSynchronize();

        size_t dataBytes = (numRows + 1) * sizeof(int) +
                           matrix.nnz * sizeof(int) +
                           matrix.nnz * sizeof(float) * 2 +
                           numCols * sizeof(float) +
                           numRows * sizeof(float);

        GpuTimer timer;
        float bestUtil = 0;
        float u8 = 0, u4 = 0, u16 = 0;

        // Test 8t/row
        int blockSize = 512;
        int gridSize = (numRows + (blockSize / WARP_SIZE) * 8 - 1) / ((blockSize / WARP_SIZE) * 8);

        // Warmup
        spmv_8t<512><<<gridSize, blockSize>>>(numRows, matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values, d_x, d_y);
        cudaDeviceSynchronize();

        float totalTime = 0;
        for (int i = 0; i < iterations; i++) {
            cudaMemset(d_y, 0, numRows * sizeof(float));
            timer.start();
            spmv_8t<512><<<gridSize, blockSize>>>(numRows, matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values, d_x, d_y);
            cudaDeviceSynchronize();
            timer.stop();
            totalTime += timer.elapsed_ms();
        }
        float t8 = totalTime / iterations;
        u8 = ((dataBytes / (t8 * 1e-3)) / (1024 * 1024 * 1024)) / peakBW * 100;
        if (u8 > bestUtil) bestUtil = u8;

        // Test 4t/row
        gridSize = (numRows + (blockSize / WARP_SIZE) * 16 - 1) / ((blockSize / WARP_SIZE) * 16);
        spmv_4t<512><<<gridSize, blockSize>>>(numRows, matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values, d_x, d_y);
        cudaDeviceSynchronize();

        totalTime = 0;
        for (int i = 0; i < iterations; i++) {
            cudaMemset(d_y, 0, numRows * sizeof(float));
            timer.start();
            spmv_4t<512><<<gridSize, blockSize>>>(numRows, matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values, d_x, d_y);
            cudaDeviceSynchronize();
            timer.stop();
            totalTime += timer.elapsed_ms();
        }
        float t4 = totalTime / iterations;
        u4 = ((dataBytes / (t4 * 1e-3)) / (1024 * 1024 * 1024)) / peakBW * 100;
        if (u4 > bestUtil) bestUtil = u4;

        // Test 16t/row
        gridSize = (numRows + (blockSize / WARP_SIZE) * 4 - 1) / ((blockSize / WARP_SIZE) * 4);
        spmv_16t<512><<<gridSize, blockSize>>>(numRows, matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values, d_x, d_y);
        cudaDeviceSynchronize();

        totalTime = 0;
        for (int i = 0; i < iterations; i++) {
            cudaMemset(d_y, 0, numRows * sizeof(float));
            timer.start();
            spmv_16t<512><<<gridSize, blockSize>>>(numRows, matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values, d_x, d_y);
            cudaDeviceSynchronize();
            timer.stop();
            totalTime += timer.elapsed_ms();
        }
        float t16 = totalTime / iterations;
        u16 = ((dataBytes / (t16 * 1e-3)) / (1024 * 1024 * 1024)) / peakBW * 100;
        if (u16 > bestUtil) bestUtil = u16;

        printf("%11.2f%% %11.2f%% %11.2f%% %11.2f%%\n", u8, u4, u16, bestUtil);
        fflush(stdout);

        // Cleanup
        delete[] h_x;
        matrix.freeHost();
        matrix.freeDevice();
        cudaFree(d_x);
        cudaFree(d_y);
        cudaDeviceSynchronize();
    }

    return 0;
}