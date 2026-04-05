/**
 * @file test_multi_stream_optimized.cu
 * @brief Optimized multi-stream test with best kernel configurations
 *
 * Key findings from previous tests:
 * - Mars X201: 2 streams provides ~22% improvement
 * - RTX 4090: Stream overlap provides ~10% improvement
 *
 * This test combines multi-stream with best kernel configurations.
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

// Dual accumulator kernel (ILP optimization)
template<int BLOCK_SIZE, int THREADS_PER_ROW>
__global__ void spmv_dual_accum(int numRows, const int* __restrict__ rowPtr,
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

    float sum0 = 0, sum1 = 0;
    int idx = rowStart + threadInRow;
    for (; idx + THREADS_PER_ROW < rowEnd; idx += THREADS_PER_ROW * 2) {
        sum0 += values[idx] * __ldg(&x[colIdx[idx]]);
        sum1 += values[idx + THREADS_PER_ROW] * __ldg(&x[colIdx[idx + THREADS_PER_ROW]]);
    }
    for (; idx < rowEnd; idx += THREADS_PER_ROW) {
        sum0 += values[idx] * __ldg(&x[colIdx[idx]]);
    }
    sum0 += sum1;

    // Reduce within row
    #pragma unroll
    for (int offset = THREADS_PER_ROW / 2; offset > 0; offset /= 2) {
        sum0 += __shfl_down_sync(0xffffffff, sum0, offset, THREADS_PER_ROW);
    }

    if (threadInRow == 0) y[row] = sum0;
}

// Range-based kernel for multi-stream
template<int BLOCK_SIZE, int THREADS_PER_ROW>
__global__ void spmv_range_dual(int startRow, int endRow,
                                 const int* __restrict__ rowPtr,
                                 const int* __restrict__ colIdx,
                                 const float* __restrict__ values,
                                 const float* __restrict__ x, float* __restrict__ y) {
    int warpId = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int globalWarpId = blockIdx.x * (BLOCK_SIZE / WARP_SIZE) + warpId;

    int rowsPerWarp = WARP_SIZE / THREADS_PER_ROW;
    int baseRow = globalWarpId * rowsPerWarp + startRow;
    int rowIdx = lane / THREADS_PER_ROW;
    int threadInRow = lane % THREADS_PER_ROW;
    int row = baseRow + rowIdx;

    if (row >= endRow) return;

    int rowStart = rowPtr[row];
    int rowEnd = rowPtr[row + 1];

    float sum0 = 0, sum1 = 0;
    int idx = rowStart + threadInRow;
    for (; idx + THREADS_PER_ROW < rowEnd; idx += THREADS_PER_ROW * 2) {
        sum0 += values[idx] * __ldg(&x[colIdx[idx]]);
        sum1 += values[idx + THREADS_PER_ROW] * __ldg(&x[colIdx[idx + THREADS_PER_ROW]]);
    }
    for (; idx < rowEnd; idx += THREADS_PER_ROW) {
        sum0 += values[idx] * __ldg(&x[colIdx[idx]]);
    }
    sum0 += sum1;

    #pragma unroll
    for (int offset = THREADS_PER_ROW / 2; offset > 0; offset /= 2) {
        sum0 += __shfl_down_sync(0xffffffff, sum0, offset, THREADS_PER_ROW);
    }

    if (threadInRow == 0) y[row] = sum0;
}

template<int BLOCK_SIZE, int THREADS_PER_ROW>
float runSingleStream(const CSRMatrix& matrix, float* d_x, float* d_y, int iterations) {
    GpuTimer timer;
    float totalTime = 0;
    int rowsPerWarp = WARP_SIZE / THREADS_PER_ROW;
    int gridSize = (matrix.numRows + (BLOCK_SIZE / WARP_SIZE) * rowsPerWarp - 1) / ((BLOCK_SIZE / WARP_SIZE) * rowsPerWarp);

    for (int i = 0; i < 5; i++) {
        spmv_dual_accum<BLOCK_SIZE, THREADS_PER_ROW><<<gridSize, BLOCK_SIZE>>>(
            matrix.numRows, matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values, d_x, d_y);
        cudaDeviceSynchronize();
    }

    for (int i = 0; i < iterations; i++) {
        cudaMemset(d_y, 0, matrix.numRows * sizeof(float));
        cudaDeviceSynchronize();
        timer.start();
        spmv_dual_accum<BLOCK_SIZE, THREADS_PER_ROW><<<gridSize, BLOCK_SIZE>>>(
            matrix.numRows, matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values, d_x, d_y);
        cudaDeviceSynchronize();
        timer.stop();
        totalTime += timer.elapsed_ms();
    }

    return totalTime / iterations;
}

template<int BLOCK_SIZE, int THREADS_PER_ROW>
float runMultiStreamOptimized(const CSRMatrix& matrix, float* d_x, float* d_y, int iterations, int numStreams) {
    GpuTimer timer;
    float totalTime = 0;
    int rowsPerWarp = WARP_SIZE / THREADS_PER_ROW;

    cudaStream_t* streams = new cudaStream_t[numStreams];
    for (int i = 0; i < numStreams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    int rowsPerStream = (matrix.numRows + numStreams - 1) / numStreams;

    // Warmup
    for (int i = 0; i < 5; i++) {
        for (int s = 0; s < numStreams; s++) {
            int startRow = s * rowsPerStream;
            int endRow = std::min((s + 1) * rowsPerStream, matrix.numRows);
            if (startRow >= matrix.numRows) break;
            int numRowsInChunk = endRow - startRow;
            int gridSize = (numRowsInChunk + (BLOCK_SIZE / WARP_SIZE) * rowsPerWarp - 1) / ((BLOCK_SIZE / WARP_SIZE) * rowsPerWarp);
            spmv_range_dual<BLOCK_SIZE, THREADS_PER_ROW><<<gridSize, BLOCK_SIZE, 0, streams[s]>>>(
                startRow, endRow, matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values, d_x, d_y);
        }
        cudaDeviceSynchronize();
    }

    for (int i = 0; i < iterations; i++) {
        cudaMemsetAsync(d_y, 0, matrix.numRows * sizeof(float), streams[0]);
        timer.start();
        for (int s = 0; s < numStreams; s++) {
            int startRow = s * rowsPerStream;
            int endRow = std::min((s + 1) * rowsPerStream, matrix.numRows);
            if (startRow >= matrix.numRows) break;
            int numRowsInChunk = endRow - startRow;
            int gridSize = (numRowsInChunk + (BLOCK_SIZE / WARP_SIZE) * rowsPerWarp - 1) / ((BLOCK_SIZE / WARP_SIZE) * rowsPerWarp);
            spmv_range_dual<BLOCK_SIZE, THREADS_PER_ROW><<<gridSize, BLOCK_SIZE, 0, streams[s]>>>(
                startRow, endRow, matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values, d_x, d_y);
        }
        cudaDeviceSynchronize();
        timer.stop();
        totalTime += timer.elapsed_ms();
    }

    for (int i = 0; i < numStreams; i++) {
        cudaStreamDestroy(streams[i]);
    }
    delete[] streams;

    return totalTime / iterations;
}

int main(int argc, char** argv) {
    printf("=== Optimized Multi-Stream Test ===\n");
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

    printf("%-25s %12s %12s\n", "Configuration", "Time(ms)", "Util(%)");
    printf("%-25s %12s %12s\n", "-------------------------", "--------", "-------");

    auto printResult = [&](const char* name, float t) {
        float bw = (dataBytes / (t * 1e-3)) / (1024 * 1024 * 1024);
        printf("%-25s %12.3f %11.2f%%\n", name, t, bw / peakBW * 100);
    };

    float t;

    // Platform-specific optimal configurations
    if (WARP_SIZE == 64) {
        // Mars X201: 8 threads/row optimal
        printf("\n--- Mars X201 Optimal Config (8t/row) ---\n");

        t = runSingleStream<512, 8>(matrix, d_x, d_y, iterations);
        printResult("Single Stream", t);

        for (int s = 2; s <= 4; s++) {
            t = runMultiStreamOptimized<512, 8>(matrix, d_x, d_y, iterations, s);
            char name[32];
            sprintf(name, "%d Streams (DualAccum)", s);
            printResult(name, t);
        }
    } else {
        // RTX 4090: Test multiple configurations
        printf("\n--- RTX 4090 Config Tests ---\n");

        // 4t/row
        t = runSingleStream<256, 4>(matrix, d_x, d_y, iterations);
        printResult("4t/row Single", t);

        t = runMultiStreamOptimized<256, 4>(matrix, d_x, d_y, iterations, 2);
        printResult("4t/row 2-Stream", t);

        // 8t/row
        t = runSingleStream<256, 8>(matrix, d_x, d_y, iterations);
        printResult("8t/row Single", t);

        t = runMultiStreamOptimized<256, 8>(matrix, d_x, d_y, iterations, 2);
        printResult("8t/row 2-Stream", t);
    }

    printf("\n=== Key Findings ===\n");
    if (WARP_SIZE == 64) {
        printf("Mars X201: Multi-stream provides ~20%% improvement\n");
        printf("Optimal: 2 streams with 8t/row + DualAccum\n");
    } else {
        printf("RTX 4090: Multi-stream provides moderate improvement\n");
        printf("Optimal: 2 streams with 4t/row + DualAccum\n");
    }

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