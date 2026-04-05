/**
 * @file test_advanced_memory_patterns.cu
 * @brief Advanced memory access pattern optimization for Mars X201
 *
 * Explore additional optimization strategies:
 * 1. Coalesced memory access patterns
 * 2. Software prefetching strategies
 * 3. Register blocking
 * 4. Warp-level optimization for warp=64
 * 5. Different reduction strategies
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

// ==================== Kernel Variants ====================

// Baseline: 4t/row with dual accumulator (optimal so far)
template<int BLOCK_SIZE>
__global__ void spmv_baseline(int numRows, const int* __restrict__ rowPtr,
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

    float sum0 = 0, sum1 = 0;
    int idx = rowStart + threadInRow;

    for (; idx + 4 < rowEnd; idx += 8) {
        sum0 += values[idx] * __ldg(&x[colIdx[idx]]);
        sum1 += values[idx + 4] * __ldg(&x[colIdx[idx + 4]]);
    }
    for (; idx < rowEnd; idx += 4) {
        sum0 += values[idx] * __ldg(&x[colIdx[idx]]);
    }

    float sum = sum0 + sum1;
    sum += __shfl_down_sync(0xffffffffffffffff, sum, 2, 4);
    sum += __shfl_down_sync(0xffffffffffffffff, sum, 1, 4);

    if (threadInRow == 0) y[row] = sum;
}

// Variant 1: Quad accumulator (more ILP)
template<int BLOCK_SIZE>
__global__ void spmv_quad_accum(int numRows, const int* __restrict__ rowPtr,
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

    float sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
    int idx = rowStart + threadInRow;

    // Quad accumulator - process 16 elements per iteration
    for (; idx + 12 < rowEnd; idx += 16) {
        sum0 += values[idx] * __ldg(&x[colIdx[idx]]);
        sum1 += values[idx + 4] * __ldg(&x[colIdx[idx + 4]]);
        sum2 += values[idx + 8] * __ldg(&x[colIdx[idx + 8]]);
        sum3 += values[idx + 12] * __ldg(&x[colIdx[idx + 12]]);
    }
    // Handle remaining with dual accumulator
    for (; idx + 4 < rowEnd; idx += 8) {
        sum0 += values[idx] * __ldg(&x[colIdx[idx]]);
        sum1 += values[idx + 4] * __ldg(&x[colIdx[idx + 4]]);
    }
    for (; idx < rowEnd; idx += 4) {
        sum0 += values[idx] * __ldg(&x[colIdx[idx]]);
    }

    float sum = sum0 + sum1 + sum2 + sum3;
    sum += __shfl_down_sync(0xffffffffffffffff, sum, 2, 4);
    sum += __shfl_down_sync(0xffffffffffffffff, sum, 1, 4);

    if (threadInRow == 0) y[row] = sum;
}

// Variant 2: Software prefetching
template<int BLOCK_SIZE>
__global__ void spmv_prefetch(int numRows, const int* __restrict__ rowPtr,
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
    int idx = rowStart + threadInRow;

    // Prefetch first element
    float val = 0, x_val = 0;
    int col = 0;
    if (idx < rowEnd) {
        val = values[idx];
        col = colIdx[idx];
        x_val = __ldg(&x[col]);
    }

    // Main loop with prefetch
    for (idx += 4; idx < rowEnd; idx += 4) {
        float next_val = values[idx];
        int next_col = colIdx[idx];
        float next_x = __ldg(&x[next_col]);

        sum += val * x_val;

        val = next_val;
        x_val = next_x;
    }

    // Final accumulation
    if (idx - 4 >= rowStart && idx - 4 < rowEnd) {
        sum += val * x_val;
    }

    sum += __shfl_down_sync(0xffffffffffffffff, sum, 2, 4);
    sum += __shfl_down_sync(0xffffffffffffffff, sum, 1, 4);

    if (threadInRow == 0) y[row] = sum;
}

// Variant 3: Vectorized load (float4)
template<int BLOCK_SIZE>
__global__ void spmv_vectorized(int numRows, const int* __restrict__ rowPtr,
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
    int idx = rowStart + threadInRow;

    // Try to use vectorized loads when aligned
    if (((unsigned long long)(values + idx) & 0xF) == 0 && idx + 3 < rowEnd) {
        // Aligned, can try vectorized
        for (; idx + 3 < rowEnd; idx += 4) {
            float4 v = *reinterpret_cast<const float4*>(values + idx);
            sum += v.x * __ldg(&x[colIdx[idx]]);
            sum += v.y * __ldg(&x[colIdx[idx + 1]]);
            sum += v.z * __ldg(&x[colIdx[idx + 2]]);
            sum += v.w * __ldg(&x[colIdx[idx + 3]]);
        }
    }

    // Handle remaining elements
    for (; idx < rowEnd; idx += 4) {
        sum += values[idx] * __ldg(&x[colIdx[idx]]);
    }

    sum += __shfl_down_sync(0xffffffffffffffff, sum, 2, 4);
    sum += __shfl_down_sync(0xffffffffffffffff, sum, 1, 4);

    if (threadInRow == 0) y[row] = sum;
}

// Variant 4: Loop unrolling with pragma
template<int BLOCK_SIZE>
__global__ void spmv_unrolled(int numRows, const int* __restrict__ rowPtr,
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
    int idx = rowStart + threadInRow;

    #pragma unroll 8
    for (; idx < rowEnd; idx += 4) {
        sum += values[idx] * __ldg(&x[colIdx[idx]]);
    }

    sum += __shfl_down_sync(0xffffffffffffffff, sum, 2, 4);
    sum += __shfl_down_sync(0xffffffffffffffff, sum, 1, 4);

    if (threadInRow == 0) y[row] = sum;
}

// Variant 5: Restrict pointer optimization
template<int BLOCK_SIZE>
__global__ void __launch_bounds__(512, 2)
spmv_restrict(int numRows,
              const int* __restrict__ __align__(16) rowPtr,
              const int* __restrict__ __align__(16) colIdx,
              const float* __restrict__ __align__(16) values,
              const float* __restrict__ __align__(16) x,
              float* __restrict__ __align__(16) y) {
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

    float sum0 = 0, sum1 = 0;
    int idx = rowStart + threadInRow;

    for (; idx + 4 < rowEnd; idx += 8) {
        sum0 += values[idx] * __ldg(&x[colIdx[idx]]);
        sum1 += values[idx + 4] * __ldg(&x[colIdx[idx + 4]]);
    }
    for (; idx < rowEnd; idx += 4) {
        sum0 += values[idx] * __ldg(&x[colIdx[idx]]);
    }

    float sum = sum0 + sum1;
    sum += __shfl_down_sync(0xffffffffffffffff, sum, 2, 4);
    sum += __shfl_down_sync(0xffffffffffffffff, sum, 1, 4);

    if (threadInRow == 0) y[row] = sum;
}

template<typename KernelFunc>
float runTest(KernelFunc kernel, int gridSize, int blockSize, int iterations,
              const CSRMatrix& matrix, float* d_x, float* d_y, bool setCache = true) {
    GpuTimer timer;
    float totalTime = 0;

    if (setCache) {
        cudaFuncSetCacheConfig(kernel, cudaFuncCachePreferL1);
    }

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
    printf("=== Advanced Memory Pattern Optimization Test ===\n");
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

    int blockSize = 512;
    int rowsPerWarp = WARP_SIZE / 4;
    int gridSize = (matrix.numRows + (blockSize / WARP_SIZE) * rowsPerWarp - 1) /
                   ((blockSize / WARP_SIZE) * rowsPerWarp);

    printf("%-25s %12s %12s %12s\n", "Kernel Variant", "Time(ms)", "Util(%)", "vs Baseline");
    printf("%-25s %12s %12s %12s\n", "-------------------------", "--------", "-------", "-----------");

    auto printResult = [&](const char* name, float t, float baseline) {
        float bw = (dataBytes / (t * 1e-3)) / (1024 * 1024 * 1024);
        float util = bw / peakBW * 100;
        float diff = (baseline - t) / baseline * 100;
        printf("%-25s %12.3f %11.2f%% %11.1f%%\n", name, t, util, diff);
        return t;
    };

    float baseline = runTest(spmv_baseline<512>, gridSize, blockSize, iterations, matrix, d_x, d_y);
    baseline = printResult("Baseline (4t DualAccum)", baseline, baseline);

    float t;
    t = runTest(spmv_quad_accum<512>, gridSize, blockSize, iterations, matrix, d_x, d_y);
    printResult("Quad Accum", t, baseline);

    t = runTest(spmv_prefetch<512>, gridSize, blockSize, iterations, matrix, d_x, d_y);
    printResult("Software Prefetch", t, baseline);

    t = runTest(spmv_vectorized<512>, gridSize, blockSize, iterations, matrix, d_x, d_y);
    printResult("Vectorized Load", t, baseline);

    t = runTest(spmv_unrolled<512>, gridSize, blockSize, iterations, matrix, d_x, d_y);
    printResult("Loop Unrolled", t, baseline);

    t = runTest(spmv_restrict<512>, gridSize, blockSize, iterations, matrix, d_x, d_y);
    printResult("Restrict+Aligned", t, baseline);

    // Test without L1 cache config
    t = runTest(spmv_baseline<512>, gridSize, blockSize, iterations, matrix, d_x, d_y, false);
    printResult("Baseline (No L1 config)", t, baseline);

    printf("\n=== Analysis ===\n");
    printf("Testing advanced memory patterns for Mars X201 optimization.\n");
    printf("All kernels use optimal 4t/row configuration with L1 cache preference.\n");

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