/**
 * @file test_texture_memory.cu
 * @brief Test texture memory optimization for x-vector access
 *
 * Texture memory can provide cache benefits for random access patterns
 * which is exactly what SpMV needs for x-vector access.
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

// Standard kernel with __ldg
template<int BLOCK_SIZE>
__global__ void spmv_ldg(int numRows, const int* __restrict__ rowPtr,
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

// Kernel with __ldcg (cache streaming)
template<int BLOCK_SIZE>
__global__ void spmv_ldcg(int numRows, const int* __restrict__ rowPtr,
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
        // Use __ldcg for streaming access (bypass L1, use L2)
        sum += values[idx] * __ldcg(&x[colIdx[idx]]);
    }

    sum += __shfl_down_sync(0xffffffff, sum, 4, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 2, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 1, 8);

    if (threadInRow == 0) y[row] = sum;
}

// Kernel with __ldca (cache all levels)
template<int BLOCK_SIZE>
__global__ void spmv_ldca(int numRows, const int* __restrict__ rowPtr,
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
        // Cache at all levels
        sum += values[idx] * __ldca(&x[colIdx[idx]]);
    }

    sum += __shfl_down_sync(0xffffffff, sum, 4, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 2, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 1, 8);

    if (threadInRow == 0) y[row] = sum;
}

// Kernel with __ldcs (cache streaming)
template<int BLOCK_SIZE>
__global__ void spmv_ldcs(int numRows, const int* __restrict__ rowPtr,
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
        // Cache streaming - expected to be read once
        sum += values[idx] * __ldcs(&x[colIdx[idx]]);
    }

    sum += __shfl_down_sync(0xffffffff, sum, 4, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 2, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 1, 8);

    if (threadInRow == 0) y[row] = sum;
}

// Direct load without any cache hints
template<int BLOCK_SIZE>
__global__ void spmv_direct(int numRows, const int* __restrict__ rowPtr,
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
        // Direct load
        sum += values[idx] * x[colIdx[idx]];
    }

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
    printf("=== Cache Hint Optimization Test ===\n");
    printf("WARP_SIZE = %d\n", WARP_SIZE);

    std::string matrixFile = argc > 1 ? argv[1] : "./real_cases/mtx/p0_A";
    int iterations = argc > 2 ? atoi(argv[2]) : 50;

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

    int blockSize = 512;
    int gridSize = (matrix.numRows + (blockSize / WARP_SIZE) * 8 - 1) / ((blockSize / WARP_SIZE) * 8);

    printf("%-15s %12s %12s\n", "Cache Hint", "Time(ms)", "Util(%)");
    printf("%-15s %12s %12s\n", "-----------", "--------", "-------");

    auto testKernel = [&](const char* name, auto kernel) {
        float t = runTest(kernel, gridSize, blockSize, iterations, matrix, d_x, d_y);
        float bw = (dataBytes / (t * 1e-3)) / (1024 * 1024 * 1024);
        printf("%-15s %12.3f %11.2f%%\n", name, t, bw / peakBW * 100);
    };

    testKernel("Direct", spmv_direct<512>);
    testKernel("__ldg (default)", spmv_ldg<512>);
    testKernel("__ldcg (stream)", spmv_ldcg<512>);
    testKernel("__ldca (all cache)", spmv_ldca<512>);
    testKernel("__ldcs (stream cs)", spmv_ldcs<512>);

    printf("\n=== Analysis ===\n");
    printf("Testing different cache access patterns for x-vector:\n");
    printf("- Direct: No cache hints\n");
    printf("- __ldg: Default read-only cache (texture cache on some GPUs)\n");
    printf("- __ldcg: Cache at global level (L2)\n");
    printf("- __ldca: Cache at all levels (L1+L2)\n");
    printf("- __ldcs: Cache streaming (expect read once)\n");

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