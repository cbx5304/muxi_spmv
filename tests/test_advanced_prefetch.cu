/**
 * @file test_advanced_prefetch.cu
 * @brief Test advanced prefetching and warp-level optimizations
 *
 * Techniques tested:
 * 1. Prefetch x-vector elements using __ldg and prefetch instructions
 * 2. Warp-level aggregation to reduce memory transactions
 * 3. Coalesced access patterns
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

// Baseline kernel
template<int BLOCK_SIZE, int THREADS_PER_ROW>
__global__ void spmv_baseline(int numRows, const int* __restrict__ rowPtr,
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

// Prefetch with double buffering
template<int BLOCK_SIZE, int THREADS_PER_ROW>
__global__ void spmv_prefetch(int numRows, const int* __restrict__ rowPtr,
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

    // Double buffering prefetch
    float sum = 0;
    int idx = rowStart + threadInRow;

    if (idx < rowEnd) {
        // Prefetch first element
        int col0 = colIdx[idx];
        float val0 = values[idx];
        float x0 = __ldg(&x[col0]);

        idx += THREADS_PER_ROW;

        while (idx < rowEnd) {
            // Prefetch next while computing current
            int col1 = colIdx[idx];
            float val1 = values[idx];
            float x1 = __ldg(&x[col1]);

            // Compute previous
            sum += val0 * x0;

            // Swap buffers
            val0 = val1;
            x0 = x1;
            idx += THREADS_PER_ROW;
        }

        // Final computation
        sum += val0 * x0;
    }

    #pragma unroll
    for (int offset = THREADS_PER_ROW / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset, THREADS_PER_ROW);
    }

    if (threadInRow == 0) y[row] = sum;
}

// Vectorized load with float4 (for aligned data)
template<int BLOCK_SIZE, int THREADS_PER_ROW>
__global__ void spmv_vectorized(int numRows, const int* __restrict__ rowPtr,
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
    int rowLen = rowEnd - rowStart;

    float sum = 0;
    int idx = rowStart + threadInRow;

    // Process in groups of 4 if aligned
    if (rowLen >= THREADS_PER_ROW * 4) {
        while (idx + THREADS_PER_ROW * 3 < rowEnd) {
            // Load 4 values and 4 indices
            float v0 = values[idx];
            float v1 = values[idx + THREADS_PER_ROW];
            float v2 = values[idx + THREADS_PER_ROW * 2];
            float v3 = values[idx + THREADS_PER_ROW * 3];

            int c0 = colIdx[idx];
            int c1 = colIdx[idx + THREADS_PER_ROW];
            int c2 = colIdx[idx + THREADS_PER_ROW * 2];
            int c3 = colIdx[idx + THREADS_PER_ROW * 3];

            sum += v0 * __ldg(&x[c0]);
            sum += v1 * __ldg(&x[c1]);
            sum += v2 * __ldg(&x[c2]);
            sum += v3 * __ldg(&x[c3]);

            idx += THREADS_PER_ROW * 4;
        }
    }

    // Remaining elements
    for (; idx < rowEnd; idx += THREADS_PER_ROW) {
        sum += values[idx] * __ldg(&x[colIdx[idx]]);
    }

    #pragma unroll
    for (int offset = THREADS_PER_ROW / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset, THREADS_PER_ROW);
    }

    if (threadInRow == 0) y[row] = sum;
}

// Cache streaming with __ldcs (streaming load)
template<int BLOCK_SIZE, int THREADS_PER_ROW>
__global__ void spmv_streaming(int numRows, const int* __restrict__ rowPtr,
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

    // Use streaming load for values (read-once data)
    // Use __ldg for x-vector (may be reused)
    for (int idx = rowStart + threadInRow; idx < rowEnd; idx += THREADS_PER_ROW) {
        float val = __ldcs(&values[idx]);  // Streaming load - bypass L1
        sum += val * __ldg(&x[colIdx[idx]]);  // Cache x-vector in L2
    }

    #pragma unroll
    for (int offset = THREADS_PER_ROW / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset, THREADS_PER_ROW);
    }

    if (threadInRow == 0) y[row] = sum;
}

// Mixed cache hints: __ldca (all caches) for x, __ldcs (streaming) for values
template<int BLOCK_SIZE, int THREADS_PER_ROW>
__global__ void spmv_cache_hints(int numRows, const int* __restrict__ rowPtr,
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
        float val = __ldcs(&values[idx]);      // Streaming - values read once
        float xval = __ldca(&x[colIdx[idx]]);  // Cache all - x may be reused
        sum += val * xval;
    }

    #pragma unroll
    for (int offset = THREADS_PER_ROW / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset, THREADS_PER_ROW);
    }

    if (threadInRow == 0) y[row] = sum;
}

template<typename KernelFunc>
float runTest(KernelFunc kernel, int gridSize, int blockSize, int iterations,
              const CSRMatrix& matrix, float* d_x, float* d_y) {
    GpuTimer timer;
    float totalTime = 0;

    for (int i = 0; i < 5; i++) {
        kernel<<<gridSize, blockSize>>>(matrix.numRows, matrix.d_rowPtr,
                                        matrix.d_colIdx, matrix.d_values, d_x, d_y);
        cudaDeviceSynchronize();
    }

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
    printf("=== Advanced Prefetch & Cache Optimization Test ===\n");
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

    int blockSize = (WARP_SIZE == 64) ? 512 : 256;
    int threadsPerRow = (WARP_SIZE == 64) ? 8 : 4;
    int gridSize = (matrix.numRows + (blockSize / WARP_SIZE) * (WARP_SIZE / threadsPerRow) - 1) / ((blockSize / WARP_SIZE) * (WARP_SIZE / threadsPerRow));

    printf("%-25s %12s %12s\n", "Configuration", "Time(ms)", "Util(%)");
    printf("%-25s %12s %12s\n", "-------------------------", "--------", "-------");

    auto testKernel = [&](const char* name, auto kernel) {
        float t = runTest(kernel, gridSize, blockSize, iterations, matrix, d_x, d_y);
        float bw = (dataBytes / (t * 1e-3)) / (1024 * 1024 * 1024);
        printf("%-25s %12.3f %11.2f%%\n", name, t, bw / peakBW * 100);
    };

    testKernel("Baseline (__ldg)", spmv_baseline<512, 8>);
    testKernel("Prefetch Double-Buffer", spmv_prefetch<512, 8>);
    testKernel("Vectorized (4x)", spmv_vectorized<512, 8>);
    testKernel("Streaming (__ldcs)", spmv_streaming<512, 8>);
    testKernel("Cache Hints Mixed", spmv_cache_hints<512, 8>);

    printf("\n=== Analysis ===\n");
    printf("Prefetch: Double buffering can hide memory latency\n");
    printf("Vectorized: 4x unrolling reduces loop overhead\n");
    printf("Streaming: __ldcs bypasses L1 for read-once data\n");
    printf("Cache Hints: __ldca for reusable data, __ldcs for streaming\n");

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