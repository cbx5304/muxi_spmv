/**
 * @file test_column_reordering.cu
 * @brief Test column reordering to improve x-vector access locality
 *
 * Column reordering (permutation) can improve cache locality by:
 * 1. Grouping frequently accessed columns together
 * 2. Reducing the working set size in L2 cache
 * 3. Improving spatial locality for x-vector access
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <map>

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

// Analyze column access frequency
void analyzeColumnAccess(const CSRMatrix& matrix) {
    std::vector<int> colFreq(matrix.numCols, 0);
    for (int i = 0; i < matrix.nnz; i++) {
        colFreq[matrix.colIdx[i]]++;
    }

    // Statistics
    std::sort(colFreq.begin(), colFreq.end(), std::greater<int>());

    printf("Column access frequency analysis:\n");
    printf("  Top 10 columns: ");
    for (int i = 0; i < 10 && i < matrix.numCols; i++) {
        printf("%d ", colFreq[i]);
    }
    printf("\n");

    // Calculate concentration
    int top100sum = 0;
    for (int i = 0; i < 100 && i < matrix.numCols; i++) {
        top100sum += colFreq[i];
    }
    printf("  Top 100 columns cover %.2f%% of accesses\n", 100.0 * top100sum / matrix.nnz);
}

// Reorder columns by access frequency (most accessed first)
void reorderColumnsByFrequency(const CSRMatrix& matrix, CSRMatrix& reordered,
                                std::vector<int>& colPermutation) {
    // Count column frequencies
    std::vector<std::pair<int, int>> colFreq(matrix.numCols);
    for (int i = 0; i < matrix.numCols; i++) {
        colFreq[i] = {i, 0};
    }
    for (int i = 0; i < matrix.nnz; i++) {
        colFreq[matrix.colIdx[i]].second++;
    }

    // Sort by frequency (descending)
    std::sort(colFreq.begin(), colFreq.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    // Create permutation
    colPermutation.resize(matrix.numCols);
    std::vector<int> invPermutation(matrix.numCols);
    for (int i = 0; i < matrix.numCols; i++) {
        colPermutation[i] = colFreq[i].first;
        invPermutation[colFreq[i].first] = i;
    }

    // Apply permutation
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

// Reorder columns by RCM (Reverse Cuthill-McKee) for bandwidth reduction
void reorderColumnsRCM(const CSRMatrix& matrix, CSRMatrix& reordered,
                       std::vector<int>& colPermutation) {
    // Simplified RCM: just sort columns by their minimum row index
    std::vector<std::pair<int, int>> colMinRow(matrix.numCols);
    for (int i = 0; i < matrix.numCols; i++) {
        colMinRow[i] = {i, matrix.numRows};  // Initialize with max
    }

    // Find minimum row for each column
    for (int row = 0; row < matrix.numRows; row++) {
        for (int idx = matrix.rowPtr[row]; idx < matrix.rowPtr[row + 1]; idx++) {
            int col = matrix.colIdx[idx];
            colMinRow[col].second = std::min(colMinRow[col].second, row);
        }
    }

    // Sort by minimum row
    std::sort(colMinRow.begin(), colMinRow.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    // Create permutation
    colPermutation.resize(matrix.numCols);
    std::vector<int> invPermutation(matrix.numCols);
    for (int i = 0; i < matrix.numCols; i++) {
        colPermutation[i] = colMinRow[i].first;
        invPermutation[colMinRow[i].first] = i;
    }

    // Apply permutation
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

template<int BLOCK_SIZE>
float runTest(const CSRMatrix& matrix, float* d_x, float* d_y, int iterations) {
    GpuTimer timer;
    float totalTime = 0;
    int threadsPerRow = (WARP_SIZE == 64) ? 8 : 4;
    int gridSize = (matrix.numRows + (BLOCK_SIZE / WARP_SIZE) * (WARP_SIZE / threadsPerRow) - 1) / ((BLOCK_SIZE / WARP_SIZE) * (WARP_SIZE / threadsPerRow));

    for (int i = 0; i < 5; i++) {
        spmv_optimized<BLOCK_SIZE, 8><<<gridSize, BLOCK_SIZE>>>(
            matrix.numRows, matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values, d_x, d_y);
        cudaDeviceSynchronize();
    }

    for (int i = 0; i < iterations; i++) {
        cudaMemset(d_y, 0, matrix.numRows * sizeof(float));
        cudaDeviceSynchronize();
        timer.start();
        spmv_optimized<BLOCK_SIZE, 8><<<gridSize, BLOCK_SIZE>>>(
            matrix.numRows, matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values, d_x, d_y);
        cudaDeviceSynchronize();
        timer.stop();
        totalTime += timer.elapsed_ms();
    }

    return totalTime / iterations;
}

int main(int argc, char** argv) {
    printf("=== Column Reordering Optimization Test ===\n");
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

    analyzeColumnAccess(matrix);
    printf("\n");

    // Allocate device memory for original
    cudaMalloc(&matrix.d_rowPtr, (matrix.numRows + 1) * sizeof(int));
    cudaMalloc(&matrix.d_colIdx, matrix.nnz * sizeof(int));
    cudaMalloc(&matrix.d_values, matrix.nnz * sizeof(float));
    cudaMemcpy(matrix.d_rowPtr, matrix.rowPtr, (matrix.numRows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(matrix.d_colIdx, matrix.colIdx, matrix.nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(matrix.d_values, matrix.values, matrix.nnz * sizeof(float), cudaMemcpyHostToDevice);

    // Reorder by frequency
    CSRMatrix freqReordered;
    std::vector<int> freqPerm;
    reorderColumnsByFrequency(matrix, freqReordered, freqPerm);
    cudaMalloc(&freqReordered.d_rowPtr, (freqReordered.numRows + 1) * sizeof(int));
    cudaMalloc(&freqReordered.d_colIdx, freqReordered.nnz * sizeof(int));
    cudaMalloc(&freqReordered.d_values, freqReordered.nnz * sizeof(float));
    cudaMemcpy(freqReordered.d_rowPtr, freqReordered.rowPtr, (freqReordered.numRows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(freqReordered.d_colIdx, freqReordered.colIdx, freqReordered.nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(freqReordered.d_values, freqReordered.values, freqReordered.nnz * sizeof(float), cudaMemcpyHostToDevice);

    // Reorder by RCM
    CSRMatrix rcmReordered;
    std::vector<int> rcmPerm;
    reorderColumnsRCM(matrix, rcmReordered, rcmPerm);
    cudaMalloc(&rcmReordered.d_rowPtr, (rcmReordered.numRows + 1) * sizeof(int));
    cudaMalloc(&rcmReordered.d_colIdx, rcmReordered.nnz * sizeof(int));
    cudaMalloc(&rcmReordered.d_values, rcmReordered.nnz * sizeof(float));
    cudaMemcpy(rcmReordered.d_rowPtr, rcmReordered.rowPtr, (rcmReordered.numRows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(rcmReordered.d_colIdx, rcmReordered.colIdx, rcmReordered.nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(rcmReordered.d_values, rcmReordered.values, rcmReordered.nnz * sizeof(float), cudaMemcpyHostToDevice);

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

    printf("%-30s %12s %12s\n", "Configuration", "Time(ms)", "Util(%)");
    printf("%-30s %12s %12s\n", "------------------------------", "--------", "-------");

    auto testKernel = [&](const char* name, const CSRMatrix& mat) {
        float t = runTest<512>(mat, d_x, d_y, iterations);
        float bw = (dataBytes / (t * 1e-3)) / (1024 * 1024 * 1024);
        printf("%-30s %12.3f %11.2f%%\n", name, t, bw / peakBW * 100);
    };

    testKernel("Original", matrix);
    testKernel("Column Reorder (Frequency)", freqReordered);
    testKernel("Column Reorder (RCM)", rcmReordered);

    printf("\n=== Analysis ===\n");
    printf("Column reordering can help when:\n");
    printf("1. Columns have skewed access frequency\n");
    printf("2. Frequently accessed columns fit in cache\n");
    printf("3. Reordering reduces cache conflicts\n");

    delete[] h_x;
    delete[] matrix.rowPtr;
    delete[] matrix.colIdx;
    delete[] matrix.values;
    delete[] freqReordered.rowPtr;
    delete[] freqReordered.colIdx;
    delete[] freqReordered.values;
    delete[] rcmReordered.rowPtr;
    delete[] rcmReordered.colIdx;
    delete[] rcmReordered.values;
    cudaFree(matrix.d_rowPtr);
    cudaFree(matrix.d_colIdx);
    cudaFree(matrix.d_values);
    cudaFree(freqReordered.d_rowPtr);
    cudaFree(freqReordered.d_colIdx);
    cudaFree(freqReordered.d_values);
    cudaFree(rcmReordered.d_rowPtr);
    cudaFree(rcmReordered.d_colIdx);
    cudaFree(rcmReordered.d_values);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}