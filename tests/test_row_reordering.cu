/**
 * @file test_row_reordering.cu
 * @brief Test matrix row reordering for better cache locality
 *
 * Row reordering can improve performance by:
 * 1. Grouping rows with similar nnz together
 * 2. Improving cache hit rate for x-vector access
 * 3. Better memory access patterns
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
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

// Compute row lengths
void computeRowLengths(const CSRMatrix& matrix, std::vector<int>& rowLen) {
    rowLen.resize(matrix.numRows);
    for (int i = 0; i < matrix.numRows; i++) {
        rowLen[i] = matrix.rowPtr[i + 1] - matrix.rowPtr[i];
    }
}

// Reorder matrix by row length (bucket sort)
void reorderMatrixByRowLength(const CSRMatrix& matrix, CSRMatrix& reordered,
                               std::vector<int>& rowMap, std::vector<int>& invRowMap) {
    std::vector<int> rowLen;
    computeRowLengths(matrix, rowLen);

    // Create index array and sort by row length
    std::vector<int> indices(matrix.numRows);
    for (int i = 0; i < matrix.numRows; i++) indices[i] = i;

    std::stable_sort(indices.begin(), indices.end(), [&rowLen](int a, int b) {
        return rowLen[a] < rowLen[b];
    });

    // Create mapping
    rowMap.resize(matrix.numRows);
    invRowMap.resize(matrix.numRows);
    for (int i = 0; i < matrix.numRows; i++) {
        rowMap[i] = indices[i];
        invRowMap[indices[i]] = i;
    }

    // Build reordered matrix
    reordered.numRows = matrix.numRows;
    reordered.numCols = matrix.numCols;
    reordered.nnz = matrix.nnz;
    reordered.rowPtr = new int[matrix.numRows + 1];
    reordered.colIdx = new int[matrix.nnz];
    reordered.values = new float[matrix.nnz];

    reordered.rowPtr[0] = 0;
    int pos = 0;
    for (int i = 0; i < matrix.numRows; i++) {
        int origRow = rowMap[i];
        int start = matrix.rowPtr[origRow];
        int end = matrix.rowPtr[origRow + 1];
        for (int j = start; j < end; j++) {
            reordered.colIdx[pos] = matrix.colIdx[j];
            reordered.values[pos] = matrix.values[j];
            pos++;
        }
        reordered.rowPtr[i + 1] = pos;
    }
}

// Standard kernel
template<int BLOCK_SIZE, int THREADS_PER_ROW>
__global__ void spmv_standard(int numRows, const int* __restrict__ rowPtr,
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

    float sum = 0;
    for (int idx = rowStart + threadInRow; idx < rowEnd; idx += THREADS_PER_ROW) {
        sum += values[idx] * __ldg(&x[colIdx[idx]]);
    }

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
    int gridSize = (matrix.numRows + (BLOCK_SIZE / WARP_SIZE) * 8 - 1) / ((BLOCK_SIZE / WARP_SIZE) * 8);

    for (int i = 0; i < 5; i++) {
        spmv_standard<BLOCK_SIZE, 8><<<gridSize, BLOCK_SIZE>>>(
            matrix.numRows, matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values, d_x, d_y);
        cudaDeviceSynchronize();
    }

    for (int i = 0; i < iterations; i++) {
        cudaMemset(d_y, 0, matrix.numRows * sizeof(float));
        cudaDeviceSynchronize();
        timer.start();
        spmv_standard<BLOCK_SIZE, 8><<<gridSize, BLOCK_SIZE>>>(
            matrix.numRows, matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values, d_x, d_y);
        cudaDeviceSynchronize();
        timer.stop();
        totalTime += timer.elapsed_ms();
    }

    return totalTime / iterations;
}

int main(int argc, char** argv) {
    printf("=== Row Reordering Optimization Test ===\n");
    printf("WARP_SIZE = %d\n", WARP_SIZE);

    std::string matrixFile = argc > 1 ? argv[1] : "./real_cases/mtx/p0_A";
    int iterations = argc > 2 ? atoi(argv[2]) : 30;

    CSRMatrix matrix;
    if (!loadMatrixMarket(matrixFile, matrix)) {
        std::cerr << "Failed to load matrix: " << matrixFile << "\n";
        return 1;
    }

    // Analyze original matrix
    std::vector<int> rowLen;
    computeRowLengths(matrix, rowLen);

    double avgLen = 0, varLen = 0;
    for (int i = 0; i < matrix.numRows; i++) avgLen += rowLen[i];
    avgLen /= matrix.numRows;
    for (int i = 0; i < matrix.numRows; i++) varLen += (rowLen[i] - avgLen) * (rowLen[i] - avgLen);
    varLen /= matrix.numRows;
    double stdLen = sqrt(varLen);

    printf("Original Matrix: %d x %d, nnz=%d, avgNnz=%.2f\n",
           matrix.numRows, matrix.numCols, matrix.nnz, avgLen);
    printf("Row length std dev: %.2f\n\n", stdLen);

    // Reorder matrix
    CSRMatrix reordered;
    std::vector<int> rowMap, invRowMap;
    reorderMatrixByRowLength(matrix, reordered, rowMap, invRowMap);

    printf("Reordered matrix (sorted by row length)\n\n");

    // Allocate device memory for original
    cudaMalloc(&matrix.d_rowPtr, (matrix.numRows + 1) * sizeof(int));
    cudaMalloc(&matrix.d_colIdx, matrix.nnz * sizeof(int));
    cudaMalloc(&matrix.d_values, matrix.nnz * sizeof(float));
    cudaMemcpy(matrix.d_rowPtr, matrix.rowPtr, (matrix.numRows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(matrix.d_colIdx, matrix.colIdx, matrix.nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(matrix.d_values, matrix.values, matrix.nnz * sizeof(float), cudaMemcpyHostToDevice);

    // Allocate device memory for reordered
    cudaMalloc(&reordered.d_rowPtr, (reordered.numRows + 1) * sizeof(int));
    cudaMalloc(&reordered.d_colIdx, reordered.nnz * sizeof(int));
    cudaMalloc(&reordered.d_values, reordered.nnz * sizeof(float));
    cudaMemcpy(reordered.d_rowPtr, reordered.rowPtr, (reordered.numRows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(reordered.d_colIdx, reordered.colIdx, reordered.nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(reordered.d_values, reordered.values, reordered.nnz * sizeof(float), cudaMemcpyHostToDevice);

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

    printf("%-20s %12s %12s\n", "Configuration", "Time(ms)", "Util(%)");
    printf("%-20s %12s %12s\n", "--------------------", "--------", "-------");

    // Test original
    float t = runTest<512>(matrix, d_x, d_y, iterations);
    float bw = (dataBytes / (t * 1e-3)) / (1024 * 1024 * 1024);
    printf("%-20s %12.3f %11.2f%%\n", "Original", t, bw / peakBW * 100);

    // Test reordered
    t = runTest<512>(reordered, d_x, d_y, iterations);
    bw = (dataBytes / (t * 1e-3)) / (1024 * 1024 * 1024);
    printf("%-20s %12.3f %11.2f%%\n", "Reordered", t, bw / peakBW * 100);

    printf("\n=== Analysis ===\n");
    printf("Row reordering groups rows with similar lengths together.\n");
    printf("This can improve:\n");
    printf("- Load balancing across warps\n");
    printf("- Cache locality for row metadata\n");
    printf("- Memory access patterns\n");

    delete[] h_x;
    delete[] matrix.rowPtr;
    delete[] matrix.colIdx;
    delete[] matrix.values;
    delete[] reordered.rowPtr;
    delete[] reordered.colIdx;
    delete[] reordered.values;
    cudaFree(matrix.d_rowPtr);
    cudaFree(matrix.d_colIdx);
    cudaFree(matrix.d_values);
    cudaFree(reordered.d_rowPtr);
    cudaFree(reordered.d_colIdx);
    cudaFree(reordered.d_values);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}