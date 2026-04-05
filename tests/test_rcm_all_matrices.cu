/**
 * @file test_rcm_all_matrices.cu
 * @brief Test RCM column reordering on all real matrices
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

void reorderColumnsRCM(const CSRMatrix& matrix, CSRMatrix& reordered) {
    std::vector<std::pair<int, int>> colMinRow(matrix.numCols);
    for (int i = 0; i < matrix.numCols; i++) {
        colMinRow[i] = {i, matrix.numRows};
    }

    for (int row = 0; row < matrix.numRows; row++) {
        for (int idx = matrix.rowPtr[row]; idx < matrix.rowPtr[row + 1]; idx++) {
            int col = matrix.colIdx[idx];
            colMinRow[col].second = std::min(colMinRow[col].second, row);
        }
    }

    std::sort(colMinRow.begin(), colMinRow.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    std::vector<int> invPermutation(matrix.numCols);
    for (int i = 0; i < matrix.numCols; i++) {
        invPermutation[colMinRow[i].first] = i;
    }

    reordered.numRows = matrix.numRows;
    reordered.numCols = matrix.numCols;
    reordered.nnz = matrix.nnz;
    reordered.rowPtr = new int[matrix.numRows + 1];
    reordered.colIdx = new int[matrix.nnz];
    reordered.values = new float[matrix.nnz];

    memcpy(reordered.rowPtr, matrix.rowPtr, (matrix.numRows + 1) * sizeof(int));
    memcpy(reordered.values, matrix.values, matrix.nnz * sizeof(float));

    for (int i = 0; i < matrix.nnz; i++) {
        reordered.colIdx[i] = invPermutation[matrix.colIdx[i]];
    }
}

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

    while (idx + THREADS_PER_ROW * 3 < rowEnd) {
        sum0 += values[idx] * __ldg(&x[colIdx[idx]]);
        sum1 += values[idx + THREADS_PER_ROW] * __ldg(&x[colIdx[idx + THREADS_PER_ROW]]);
        sum2 += values[idx + THREADS_PER_ROW * 2] * __ldg(&x[colIdx[idx + THREADS_PER_ROW * 2]]);
        sum3 += values[idx + THREADS_PER_ROW * 3] * __ldg(&x[colIdx[idx + THREADS_PER_ROW * 3]]);
        idx += THREADS_PER_ROW * 4;
    }

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

float runTest(const CSRMatrix& matrix, float* d_x, float* d_y, int iterations) {
    GpuTimer timer;
    float totalTime = 0;
    int blockSize = (WARP_SIZE == 64) ? 512 : 256;
    int threadsPerRow = (WARP_SIZE == 64) ? 8 : 4;
    int gridSize = (matrix.numRows + (blockSize / WARP_SIZE) * (WARP_SIZE / threadsPerRow) - 1) / ((blockSize / WARP_SIZE) * (WARP_SIZE / threadsPerRow));

    for (int i = 0; i < 5; i++) {
        spmv_optimized<512, 8><<<gridSize, blockSize>>>(
            matrix.numRows, matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values, d_x, d_y);
        cudaDeviceSynchronize();
    }

    for (int i = 0; i < iterations; i++) {
        cudaMemset(d_y, 0, matrix.numRows * sizeof(float));
        cudaDeviceSynchronize();
        timer.start();
        spmv_optimized<512, 8><<<gridSize, blockSize>>>(
            matrix.numRows, matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values, d_x, d_y);
        cudaDeviceSynchronize();
        timer.stop();
        totalTime += timer.elapsed_ms();
    }

    return totalTime / iterations;
}

int main(int argc, char** argv) {
    printf("=== RCM Column Reordering - All Matrices ===\n");
    printf("WARP_SIZE = %d\n\n", WARP_SIZE);

    std::string baseDir = argc > 1 ? argv[1] : "./real_cases/mtx";
    int iterations = argc > 2 ? atoi(argv[2]) : 20;

    float peakBW = (WARP_SIZE == 64) ? 1843.0f : 1008.0f;

    printf("%-10s %10s %10s %10s %10s\n", "Matrix", "Orig(ms)", "RCM(ms)", "Orig(%)", "RCM(%)");
    printf("%-10s %10s %10s %10s %10s\n", "------", "--------", "-------", "-------", "------");

    double totalOrig = 0, totalRCM = 0;
    int count = 0;

    for (int m = 0; m <= 9; m++) {
        std::string matrixFile = baseDir + "/p" + std::to_string(m) + "_A";

        CSRMatrix matrix;
        if (!loadMatrixMarket(matrixFile, matrix)) continue;

        size_t dataBytes = (matrix.numRows + 1) * sizeof(int) +
                           matrix.nnz * sizeof(int) +
                           matrix.nnz * sizeof(float) * 2 +
                           matrix.numCols * sizeof(float) +
                           matrix.numRows * sizeof(float);

        // Allocate device memory for original
        cudaMalloc(&matrix.d_rowPtr, (matrix.numRows + 1) * sizeof(int));
        cudaMalloc(&matrix.d_colIdx, matrix.nnz * sizeof(int));
        cudaMalloc(&matrix.d_values, matrix.nnz * sizeof(float));
        cudaMemcpy(matrix.d_rowPtr, matrix.rowPtr, (matrix.numRows + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(matrix.d_colIdx, matrix.colIdx, matrix.nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(matrix.d_values, matrix.values, matrix.nnz * sizeof(float), cudaMemcpyHostToDevice);

        // RCM reordered
        CSRMatrix rcmMatrix;
        reorderColumnsRCM(matrix, rcmMatrix);
        cudaMalloc(&rcmMatrix.d_rowPtr, (rcmMatrix.numRows + 1) * sizeof(int));
        cudaMalloc(&rcmMatrix.d_colIdx, rcmMatrix.nnz * sizeof(int));
        cudaMalloc(&rcmMatrix.d_values, rcmMatrix.nnz * sizeof(float));
        cudaMemcpy(rcmMatrix.d_rowPtr, rcmMatrix.rowPtr, (rcmMatrix.numRows + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(rcmMatrix.d_colIdx, rcmMatrix.colIdx, rcmMatrix.nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(rcmMatrix.d_values, rcmMatrix.values, rcmMatrix.nnz * sizeof(float), cudaMemcpyHostToDevice);

        float* h_x = new float[matrix.numCols];
        for (int i = 0; i < matrix.numCols; i++) h_x[i] = rand() / (float)RAND_MAX;

        float* d_x, *d_y;
        cudaMalloc(&d_x, matrix.numCols * sizeof(float));
        cudaMalloc(&d_y, matrix.numRows * sizeof(float));
        cudaMemcpy(d_x, h_x, matrix.numCols * sizeof(float), cudaMemcpyHostToDevice);

        float tOrig = runTest(matrix, d_x, d_y, iterations);
        float tRCM = runTest(rcmMatrix, d_x, d_y, iterations);

        float bwOrig = (dataBytes / (tOrig * 1e-3)) / (1024 * 1024 * 1024);
        float bwRCM = (dataBytes / (tRCM * 1e-3)) / (1024 * 1024 * 1024);

        printf("%-10s %10.3f %10.3f %9.2f%% %9.2f%%\n",
               ("p" + std::to_string(m) + "_A").c_str(),
               tOrig, tRCM, bwOrig / peakBW * 100, bwRCM / peakBW * 100);

        totalOrig += bwOrig / peakBW * 100;
        totalRCM += bwRCM / peakBW * 100;
        count++;

        delete[] h_x;
        delete[] matrix.rowPtr;
        delete[] matrix.colIdx;
        delete[] matrix.values;
        delete[] rcmMatrix.rowPtr;
        delete[] rcmMatrix.colIdx;
        delete[] rcmMatrix.values;
        cudaFree(matrix.d_rowPtr);
        cudaFree(matrix.d_colIdx);
        cudaFree(matrix.d_values);
        cudaFree(rcmMatrix.d_rowPtr);
        cudaFree(rcmMatrix.d_colIdx);
        cudaFree(rcmMatrix.d_values);
        cudaFree(d_x);
        cudaFree(d_y);
    }

    printf("\n=== Summary ===\n");
    printf("Average Original: %.2f%%\n", totalOrig / count);
    printf("Average RCM: %.2f%%\n", totalRCM / count);
    printf("Improvement: %.1f%%\n", (totalRCM - totalOrig) / totalOrig * 100);

    return 0;
}