/**
 * @file test_multi_stream.cu
 * @brief Test multi-stream parallelism for SpMV
 *
 * Strategy: Split matrix into chunks and process in parallel streams
 * to overlap computation and memory transfers.
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

// Standard kernel with 8t/row
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

// Kernel for processing a row range
template<int BLOCK_SIZE>
__global__ void spmv_range(int startRow, int endRow,
                           const int* __restrict__ rowPtr,
                           const int* __restrict__ colIdx,
                           const float* __restrict__ values,
                           const float* __restrict__ x, float* __restrict__ y) {
    int warpId = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int globalWarpId = blockIdx.x * (BLOCK_SIZE / WARP_SIZE) + warpId;

    int baseRow = globalWarpId * 8 + startRow;
    int rowIdx = lane / 8;
    int threadInRow = lane % 8;
    int row = baseRow + rowIdx;

    if (row >= endRow) return;

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

// Single stream baseline
float runSingleStream(const CSRMatrix& matrix, float* d_x, float* d_y, int iterations) {
    GpuTimer timer;
    float totalTime = 0;
    int blockSize = 512;
    int gridSize = (matrix.numRows + (blockSize / WARP_SIZE) * 8 - 1) / ((blockSize / WARP_SIZE) * 8);

    // Warmup
    for (int i = 0; i < 5; i++) {
        spmv_8t<512><<<gridSize, blockSize>>>(matrix.numRows, matrix.d_rowPtr,
                                               matrix.d_colIdx, matrix.d_values, d_x, d_y);
        cudaDeviceSynchronize();
    }

    for (int i = 0; i < iterations; i++) {
        cudaMemset(d_y, 0, matrix.numRows * sizeof(float));
        cudaDeviceSynchronize();
        timer.start();
        spmv_8t<512><<<gridSize, blockSize>>>(matrix.numRows, matrix.d_rowPtr,
                                               matrix.d_colIdx, matrix.d_values, d_x, d_y);
        cudaDeviceSynchronize();
        timer.stop();
        totalTime += timer.elapsed_ms();
    }

    return totalTime / iterations;
}

// Multi-stream approach
float runMultiStream(const CSRMatrix& matrix, float* d_x, float* d_y, int iterations, int numStreams) {
    GpuTimer timer;
    float totalTime = 0;
    int blockSize = 512;

    // Create streams
    cudaStream_t* streams = new cudaStream_t[numStreams];
    for (int i = 0; i < numStreams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Calculate row ranges for each stream
    int rowsPerStream = (matrix.numRows + numStreams - 1) / numStreams;

    // Warmup
    for (int i = 0; i < 5; i++) {
        for (int s = 0; s < numStreams; s++) {
            int startRow = s * rowsPerStream;
            int endRow = std::min((s + 1) * rowsPerStream, matrix.numRows);
            if (startRow >= matrix.numRows) break;
            int numRowsInChunk = endRow - startRow;
            int gridSize = (numRowsInChunk + (blockSize / WARP_SIZE) * 8 - 1) / ((blockSize / WARP_SIZE) * 8);
            spmv_range<512><<<gridSize, blockSize, 0, streams[s]>>>(startRow, endRow,
                                                                     matrix.d_rowPtr, matrix.d_colIdx,
                                                                     matrix.d_values, d_x, d_y);
        }
        cudaDeviceSynchronize();
    }

    for (int i = 0; i < iterations; i++) {
        cudaMemset(d_y, 0, matrix.numRows * sizeof(float));
        cudaDeviceSynchronize();
        timer.start();
        for (int s = 0; s < numStreams; s++) {
            int startRow = s * rowsPerStream;
            int endRow = std::min((s + 1) * rowsPerStream, matrix.numRows);
            if (startRow >= matrix.numRows) break;
            int numRowsInChunk = endRow - startRow;
            int gridSize = (numRowsInChunk + (blockSize / WARP_SIZE) * 8 - 1) / ((blockSize / WARP_SIZE) * 8);
            spmv_range<512><<<gridSize, blockSize, 0, streams[s]>>>(startRow, endRow,
                                                                     matrix.d_rowPtr, matrix.d_colIdx,
                                                                     matrix.d_values, d_x, d_y);
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

// Multi-stream with overlapping computation
float runMultiStreamOverlap(const CSRMatrix& matrix, float* d_x, float* d_y, int iterations, int numStreams) {
    GpuTimer timer;
    float totalTime = 0;
    int blockSize = 512;

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
            int gridSize = (numRowsInChunk + (blockSize / WARP_SIZE) * 8 - 1) / ((blockSize / WARP_SIZE) * 8);
            spmv_range<512><<<gridSize, blockSize, 0, streams[s]>>>(startRow, endRow,
                                                                     matrix.d_rowPtr, matrix.d_colIdx,
                                                                     matrix.d_values, d_x, d_y);
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
            int gridSize = (numRowsInChunk + (blockSize / WARP_SIZE) * 8 - 1) / ((blockSize / WARP_SIZE) * 8);
            spmv_range<512><<<gridSize, blockSize, 0, streams[s]>>>(startRow, endRow,
                                                                     matrix.d_rowPtr, matrix.d_colIdx,
                                                                     matrix.d_values, d_x, d_y);
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
    printf("=== Multi-Stream Parallelism Test ===\n");
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

    printf("%-20s %12s %12s\n", "Strategy", "Time(ms)", "Util(%)");
    printf("%-20s %12s %12s\n", "--------------------", "--------", "-------");

    auto printResult = [&](const char* name, float t) {
        float bw = (dataBytes / (t * 1e-3)) / (1024 * 1024 * 1024);
        printf("%-20s %12.3f %11.2f%%\n", name, t, bw / peakBW * 100);
    };

    // Test single stream
    float t = runSingleStream(matrix, d_x, d_y, iterations);
    printResult("Single Stream", t);

    // Test different stream counts
    for (int numStreams = 2; numStreams <= 8; numStreams *= 2) {
        t = runMultiStream(matrix, d_x, d_y, iterations, numStreams);
        char name[32];
        sprintf(name, "%d Streams", numStreams);
        printResult(name, t);

        t = runMultiStreamOverlap(matrix, d_x, d_y, iterations, numStreams);
        sprintf(name, "%d Streams+Overlap", numStreams);
        printResult(name, t);
    }

    printf("\n=== Analysis ===\n");
    printf("Testing if multi-stream can improve SpMV performance\n");
    printf("by enabling concurrent kernel execution.\n");

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