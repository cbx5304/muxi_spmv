/**
 * @file test_block_optimization.cu
 * @brief Find optimal block size for SpMV on both platforms
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

// Parameterized kernel for block size testing
template<int BLOCK_SIZE, int SMEM_INTS, int THREADS_PER_ROW>
__global__ void spmv_tunable(int numRows, const int* __restrict__ rowPtr,
                             const int* __restrict__ colIdx,
                             const float* __restrict__ values,
                             const float* __restrict__ x, float* __restrict__ y) {
    __shared__ int sharedRowPtr[SMEM_INTS + 16];

    int warpId = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int globalWarpId = blockIdx.x * (BLOCK_SIZE / WARP_SIZE) + warpId;

    int rowsPerWarp = WARP_SIZE / THREADS_PER_ROW;
    int baseRow = globalWarpId * rowsPerWarp;
    int warpOffset = warpId * (rowsPerWarp + 1);

    if (lane <= rowsPerWarp && baseRow + lane <= numRows) {
        sharedRowPtr[warpOffset + lane] = rowPtr[baseRow + lane];
    }
    __syncthreads();

    int rowIdx = lane / THREADS_PER_ROW;
    int threadInRow = lane % THREADS_PER_ROW;
    int row = baseRow + rowIdx;

    if (row >= numRows) return;

    float sum = 0;
    int rowStart = sharedRowPtr[warpOffset + rowIdx];
    int rowEnd = sharedRowPtr[warpOffset + rowIdx + 1];

    for (int idx = rowStart + threadInRow; idx < rowEnd; idx += THREADS_PER_ROW) {
        sum += values[idx] * __ldg(&x[colIdx[idx]]);
    }

    // Warp reduction
    if (THREADS_PER_ROW >= 32) {
        sum += __shfl_down_sync(0xffffffff, sum, 16);
    }
    if (THREADS_PER_ROW >= 16) {
        sum += __shfl_down_sync(0xffffffff, sum, 8);
    }
    if (THREADS_PER_ROW >= 8) {
        sum += __shfl_down_sync(0xffffffff, sum, 4, 8);
    }
    sum += __shfl_down_sync(0xffffffff, sum, 2, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 1, 8);

    if (threadInRow == 0) y[row] = sum;
}

int main(int argc, char** argv) {
    printf("=== Block Size Optimization Test ===\n");
    printf("WARP_SIZE = %d\n", WARP_SIZE);

    std::string matrixFile = argc > 1 ? argv[1] : "./real_cases/mtx/p0_A";
    int iterations = argc > 2 ? atoi(argv[2]) : 50;

    CSRMatrix matrix;
    if (!loadMatrixMarket(matrixFile, matrix)) {
        std::cerr << "Failed to load matrix\n";
        return 1;
    }

    printf("Matrix: %d x %d, nnz=%d, avgNnz=%.2f\n\n",
           matrix.numRows, matrix.numCols, matrix.nnz,
           (double)matrix.nnz / matrix.numRows);

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

    printf("Testing combinations of Block Size x Threads/Row:\n\n");

    // Test different block sizes and threads per row
    int blockSizes[] = {64, 128, 192, 256, 320, 384, 448, 512, 640, 768, 896, 1024};
    int threadsPerRow[] = {4, 8, 16, 32};

    float bestBW = 0;
    int bestBlock = 0, bestTpr = 0;

    printf("%-8s", "Block");
    for (int tpr : threadsPerRow) {
        printf("%8dt/row", tpr);
    }
    printf("\n");

    for (int blockSize : blockSizes) {
        printf("%-8d", blockSize);

        for (int tpr : threadsPerRow) {
            if (tpr > WARP_SIZE) {
                printf("      N/A");
                continue;
            }

            int rowsPerWarp = WARP_SIZE / tpr;
            int gridSize = (matrix.numRows + (blockSize / WARP_SIZE) * rowsPerWarp - 1) /
                          ((blockSize / WARP_SIZE) * rowsPerWarp);

            GpuTimer timer;
            float totalTime = 0;

            // Warmup
            for (int i = 0; i < 5; i++) {
                if (tpr == 4) {
                    spmv_tunable<512, 1024, 4><<<gridSize, blockSize>>>(
                        matrix.numRows, matrix.d_rowPtr, matrix.d_colIdx,
                        matrix.d_values, d_x, d_y);
                } else if (tpr == 8) {
                    spmv_tunable<512, 1024, 8><<<gridSize, blockSize>>>(
                        matrix.numRows, matrix.d_rowPtr, matrix.d_colIdx,
                        matrix.d_values, d_x, d_y);
                } else if (tpr == 16) {
                    spmv_tunable<512, 1024, 16><<<gridSize, blockSize>>>(
                        matrix.numRows, matrix.d_rowPtr, matrix.d_colIdx,
                        matrix.d_values, d_x, d_y);
                } else {
                    spmv_tunable<512, 1024, 32><<<gridSize, blockSize>>>(
                        matrix.numRows, matrix.d_rowPtr, matrix.d_colIdx,
                        matrix.d_values, d_x, d_y);
                }
                cudaDeviceSynchronize();
            }

            // Timed runs
            for (int i = 0; i < iterations; i++) {
                cudaMemset(d_y, 0, matrix.numRows * sizeof(float));
                cudaDeviceSynchronize();
                timer.start();

                if (tpr == 4) {
                    spmv_tunable<512, 1024, 4><<<gridSize, blockSize>>>(
                        matrix.numRows, matrix.d_rowPtr, matrix.d_colIdx,
                        matrix.d_values, d_x, d_y);
                } else if (tpr == 8) {
                    spmv_tunable<512, 1024, 8><<<gridSize, blockSize>>>(
                        matrix.numRows, matrix.d_rowPtr, matrix.d_colIdx,
                        matrix.d_values, d_x, d_y);
                } else if (tpr == 16) {
                    spmv_tunable<512, 1024, 16><<<gridSize, blockSize>>>(
                        matrix.numRows, matrix.d_rowPtr, matrix.d_colIdx,
                        matrix.d_values, d_x, d_y);
                } else {
                    spmv_tunable<512, 1024, 32><<<gridSize, blockSize>>>(
                        matrix.numRows, matrix.d_rowPtr, matrix.d_colIdx,
                        matrix.d_values, d_x, d_y);
                }

                cudaDeviceSynchronize();
                timer.stop();
                totalTime += timer.elapsed_ms();
            }

            float avgTime = totalTime / iterations;
            float bw = (dataBytes / (avgTime * 1e-3)) / (1024 * 1024 * 1024);
            float util = bw / peakBW * 100;

            printf("%7.1f%%", util);

            if (bw > bestBW) {
                bestBW = bw;
                bestBlock = blockSize;
                bestTpr = tpr;
            }
        }
        printf("\n");
    }

    printf("\n=== Best Configuration ===\n");
    printf("Block Size: %d, Threads/Row: %d\n", bestBlock, bestTpr);
    printf("Bandwidth: %.1f GB/s (%.2f%%)\n", bestBW, bestBW / peakBW * 100);

    delete[] h_x;
    delete[] matrix.rowPtr;
    delete[] matrix.colIdx;
    delete[] matrix.values;
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}