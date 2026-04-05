/**
 * @file test_comprehensive_analysis.cu
 * @brief Comprehensive analysis for understanding Mars X201 performance bottlenecks
 *
 * Analysis dimensions:
 * 1. Different avgNnz patterns (memory access patterns)
 * 2. Different block sizes
 * 3. Different thread configurations
 * 4. Memory coalescing analysis
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
    int numRows;
    int numCols;
    int nnz;
    int* rowPtr;
    int* colIdx;
    float* values;
    int* d_rowPtr;
    int* d_colIdx;
    float* d_values;

    void allocateDevice() {
        cudaMalloc(&d_rowPtr, (numRows + 1) * sizeof(int));
        cudaMalloc(&d_colIdx, nnz * sizeof(int));
        cudaMalloc(&d_values, nnz * sizeof(float));
    }

    void copyToDevice() {
        cudaMemcpy(d_rowPtr, rowPtr, (numRows + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_colIdx, colIdx, nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_values, values, nnz * sizeof(float), cudaMemcpyHostToDevice);
    }

    // Analyze row length distribution
    void analyzeRowLengths() {
        std::vector<int> rowLens(numRows);
        double sum = 0, sumSq = 0;
        int minLen = INT_MAX, maxLen = 0;

        for (int i = 0; i < numRows; i++) {
            int len = rowPtr[i + 1] - rowPtr[i];
            rowLens[i] = len;
            sum += len;
            sumSq += (double)len * len;
            if (len < minLen) minLen = len;
            if (len > maxLen) maxLen = len;
        }

        double avg = sum / numRows;
        double variance = (sumSq / numRows) - (avg * avg);
        double stdDev = sqrt(variance);

        // Calculate coefficient of variation (CV) - measures irregularity
        double cv = (avg > 0) ? stdDev / avg : 0;

        // Count empty rows
        int emptyRows = 0;
        for (int i = 0; i < numRows; i++) {
            if (rowLens[i] == 0) emptyRows++;
        }

        printf("  Row Length Statistics:\n");
        printf("    Avg: %.2f, Min: %d, Max: %d, StdDev: %.2f\n", avg, minLen, maxLen, stdDev);
        printf("    Coefficient of Variation: %.3f (lower = more regular)\n", cv);
        printf("    Empty rows: %d (%.2f%%)\n", emptyRows, 100.0 * emptyRows / numRows);

        // Analyze colIdx distribution for memory access pattern
        std::sort(rowLens.begin(), rowLens.end());
        int p10 = rowLens[(int)(numRows * 0.1)];
        int p50 = rowLens[(int)(numRows * 0.5)];
        int p90 = rowLens[(int)(numRows * 0.9)];
        printf("    Percentiles: P10=%d, P50=%d, P90=%d\n", p10, p50, p90);
    }
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
        while (currentRow < r) {
            currentRow++;
            matrix.rowPtr[currentRow] = i;
        }
        matrix.colIdx[i] = std::get<1>(entries[i]);
        matrix.values[i] = std::get<2>(entries[i]);
    }
    while (currentRow < rows) {
        currentRow++;
        matrix.rowPtr[currentRow] = nnz;
    }

    return true;
}

// ==================== Different Kernel Variants ====================

// Variant 1: Standard 8 threads/row
template<int BLOCK_SIZE, int SMEM_INTS>
__global__ void spmv_8threads(int numRows, const int* __restrict__ rowPtr,
                              const int* __restrict__ colIdx,
                              const float* __restrict__ values,
                              const float* __restrict__ x, float* __restrict__ y) {
    __shared__ int sharedRowPtr[SMEM_INTS + 16];

    int warpId = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int globalWarpId = blockIdx.x * (BLOCK_SIZE / WARP_SIZE) + warpId;

    int baseRow = globalWarpId * 8;
    int warpOffset = warpId * 10;

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

// Variant 2: Vectorized load with float4
template<int BLOCK_SIZE, int SMEM_INTS>
__global__ void spmv_vectorized(int numRows, const int* __restrict__ rowPtr,
                                const int* __restrict__ colIdx,
                                const float* __restrict__ values,
                                const float* __restrict__ x, float* __restrict__ y) {
    __shared__ int sharedRowPtr[SMEM_INTS + 16];

    int warpId = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int globalWarpId = blockIdx.x * (BLOCK_SIZE / WARP_SIZE) + warpId;

    int baseRow = globalWarpId * 8;
    int warpOffset = warpId * 10;

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

    // Vectorized processing
    int idx = rowStart + threadInRow;
    int vecLen = (rowEnd - rowStart) / 8 * 8;

    for (; idx < rowStart + vecLen; idx += 8) {
        float v = values[idx];
        int c = colIdx[idx];
        sum += v * __ldg(&x[c]);
    }

    // Remainder
    for (; idx < rowEnd; idx += 8) {
        sum += values[idx] * __ldg(&x[colIdx[idx]]);
    }

    sum += __shfl_down_sync(0xffffffff, sum, 4, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 2, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 1, 8);

    if (threadInRow == 0) y[row] = sum;
}

// Variant 3: Adaptive threads per row
template<int BLOCK_SIZE, int SMEM_INTS>
__global__ void spmv_adaptive(int numRows, const int* __restrict__ rowPtr,
                              const int* __restrict__ colIdx,
                              const float* __restrict__ values,
                              const float* __restrict__ x, float* __restrict__ y) {
    __shared__ int sharedRowPtr[SMEM_INTS + 16];

    int warpId = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int globalWarpId = blockIdx.x * (BLOCK_SIZE / WARP_SIZE) + warpId;

    // Adaptive: 4 threads/row for short rows, 8 for medium, 16 for long
    // For avgNnz~10, use 8 threads/row
    int threadsPerRow = 8;
    int rowsPerWarp = WARP_SIZE / threadsPerRow;

    int baseRow = globalWarpId * rowsPerWarp;
    int warpOffset = warpId * (rowsPerWarp + 1);

    if (lane <= rowsPerWarp && baseRow + lane <= numRows) {
        sharedRowPtr[warpOffset + lane] = rowPtr[baseRow + lane];
    }
    __syncthreads();

    int rowIdx = lane / threadsPerRow;
    int threadInRow = lane % threadsPerRow;
    int row = baseRow + rowIdx;

    if (row >= numRows) return;

    float sum = 0;
    int rowStart = sharedRowPtr[warpOffset + rowIdx];
    int rowEnd = sharedRowPtr[warpOffset + rowIdx + 1];

    for (int idx = rowStart + threadInRow; idx < rowEnd; idx += threadsPerRow) {
        sum += values[idx] * __ldg(&x[colIdx[idx]]);
    }

    // Reduction based on threadsPerRow
    if (threadsPerRow >= 8) sum += __shfl_down_sync(0xffffffff, sum, 4, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 2, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 1, 8);

    if (threadInRow == 0) y[row] = sum;
}

// Variant 4: Double buffering with prefetch
template<int BLOCK_SIZE, int SMEM_INTS>
__global__ void spmv_double_buffer(int numRows, const int* __restrict__ rowPtr,
                                   const int* __restrict__ colIdx,
                                   const float* __restrict__ values,
                                   const float* __restrict__ x, float* __restrict__ y) {
    __shared__ int sharedRowPtr[SMEM_INTS + 16];

    int warpId = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int globalWarpId = blockIdx.x * (BLOCK_SIZE / WARP_SIZE) + warpId;

    int baseRow = globalWarpId * 8;
    int warpOffset = warpId * 10;

    if (lane < 9 && baseRow + lane <= numRows) {
        sharedRowPtr[warpOffset + lane] = rowPtr[baseRow + lane];
    }
    __syncthreads();

    int rowIdx = lane / 8;
    int threadInRow = lane % 8;
    int row = baseRow + rowIdx;

    if (row >= numRows) return;

    float sum0 = 0, sum1 = 0;
    int rowStart = sharedRowPtr[warpOffset + rowIdx];
    int rowEnd = sharedRowPtr[warpOffset + rowIdx + 1];

    int idx = rowStart + threadInRow;

    // Double buffering: prefetch while computing
    int len = rowEnd - rowStart;
    int halfLen = len / 16 * 8;

    // First half
    for (int i = 0; i < halfLen && idx + i * 8 < rowEnd; i++) {
        sum0 += values[idx + i * 8] * __ldg(&x[colIdx[idx + i * 8]]);
    }

    // Second half (overlapped with first)
    for (int i = halfLen / 8; i < (len + 7) / 8 && idx + i * 8 < rowEnd; i++) {
        sum1 += values[idx + i * 8] * __ldg(&x[colIdx[idx + i * 8]]);
    }

    float sum = sum0 + sum1;

    sum += __shfl_down_sync(0xffffffff, sum, 4, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 2, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 1, 8);

    if (threadInRow == 0) y[row] = sum;
}

template<typename KernelFunc>
float runTest(KernelFunc kernel, int gridSize, int blockSize, int iterations,
              const CSRMatrix& matrix, float* d_x, float* d_y) {
    GpuTimer timer;
    float totalTime = 0;

    // Warmup
    for (int i = 0; i < 5; i++) {
        kernel<<<gridSize, blockSize>>>(matrix.numRows, matrix.d_rowPtr,
                                        matrix.d_colIdx, matrix.d_values, d_x, d_y);
        cudaDeviceSynchronize();
    }

    // Timed runs
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
    printf("=== Comprehensive Performance Analysis ===\n");
    printf("WARP_SIZE = %d\n", WARP_SIZE);

    std::string matrixFile = argc > 1 ? argv[1] : "./real_cases/mtx/p0_A";
    int iterations = argc > 2 ? atoi(argv[2]) : 50;

    CSRMatrix matrix;
    if (!loadMatrixMarket(matrixFile, matrix)) {
        std::cerr << "Failed to load matrix: " << matrixFile << "\n";
        return 1;
    }

    printf("\nMatrix: %s\n", matrixFile.c_str());
    printf("  Dimensions: %d x %d\n", matrix.numRows, matrix.numCols);
    printf("  NNZ: %d, AvgNnzPerRow: %.2f\n", matrix.nnz, (double)matrix.nnz / matrix.numRows);
    printf("  Sparsity: %.6f%%\n", 100.0 * matrix.nnz / ((double)matrix.numRows * matrix.numCols));

    matrix.analyzeRowLengths();

    matrix.allocateDevice();
    matrix.copyToDevice();

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

    printf("\n=== Kernel Variants Comparison ===\n");

    int blockSize = 512;
    int gridSize = (matrix.numRows + (blockSize / WARP_SIZE) * 8 - 1) / ((blockSize / WARP_SIZE) * 8);

    // Test different block sizes for 8threads
    printf("\n--- Block Size Comparison (8threads/row) ---\n");
    int blockSizes[] = {256, 512, 1024};
    for (int bs : blockSizes) {
        int gs = (matrix.numRows + (bs / WARP_SIZE) * 8 - 1) / ((bs / WARP_SIZE) * 8);
        float time = runTest(spmv_8threads<512, 1024>, gs, bs, iterations, matrix, d_x, d_y);
        float bw = (dataBytes / (time * 1e-3)) / (1024 * 1024 * 1024);
        printf("  Block=%4d: %.3f ms, %.1f GB/s, %.2f%%\n", bs, time, bw, bw / peakBW * 100);
    }

    // Test different kernel variants
    printf("\n--- Kernel Variant Comparison (Block=512) ---\n");

    float t1 = runTest(spmv_8threads<512, 1024>, gridSize, blockSize, iterations, matrix, d_x, d_y);
    float bw1 = (dataBytes / (t1 * 1e-3)) / (1024 * 1024 * 1024);
    printf("  Standard 8t/row:  %.3f ms, %.1f GB/s, %.2f%%\n", t1, bw1, bw1 / peakBW * 100);

    float t2 = runTest(spmv_vectorized<512, 1024>, gridSize, blockSize, iterations, matrix, d_x, d_y);
    float bw2 = (dataBytes / (t2 * 1e-3)) / (1024 * 1024 * 1024);
    printf("  Vectorized:       %.3f ms, %.1f GB/s, %.2f%%\n", t2, bw2, bw2 / peakBW * 100);

    float t3 = runTest(spmv_adaptive<512, 1024>, gridSize, blockSize, iterations, matrix, d_x, d_y);
    float bw3 = (dataBytes / (t3 * 1e-3)) / (1024 * 1024 * 1024);
    printf("  Adaptive:         %.3f ms, %.1f GB/s, %.2f%%\n", t3, bw3, bw3 / peakBW * 100);

    float t4 = runTest(spmv_double_buffer<512, 1024>, gridSize, blockSize, iterations, matrix, d_x, d_y);
    float bw4 = (dataBytes / (t4 * 1e-3)) / (1024 * 1024 * 1024);
    printf("  Double Buffer:    %.3f ms, %.1f GB/s, %.2f%%\n", t4, bw4, bw4 / peakBW * 100);

    printf("\n=== Summary ===\n");
    printf("Platform: %s\n", (WARP_SIZE == 64) ? "Mars X201" : "RTX 4090");
    printf("Peak BW: %.0f GB/s\n", peakBW);
    printf("Best Kernel: ");
    if (bw1 >= bw2 && bw1 >= bw3 && bw1 >= bw4) printf("Standard 8t/row");
    else if (bw2 >= bw3 && bw2 >= bw4) printf("Vectorized");
    else if (bw3 >= bw4) printf("Adaptive");
    else printf("Double Buffer");
    printf(" at %.1f GB/s (%.2f%%)\n", std::max({bw1, bw2, bw3, bw4}), std::max({bw1, bw2, bw3, bw4}) / peakBW * 100);

    delete[] h_x;
    delete[] matrix.rowPtr;
    delete[] matrix.colIdx;
    delete[] matrix.values;
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}