/**
 * @file test_extreme_optimizations.cu
 * @brief Extreme optimization attempts for Mars X201
 *
 * Try advanced techniques:
 * 1. Vectorized loads (float2/float4)
 * 2. Loop unrolling with pragma
 * 3. Software pipelining
 * 4. Cache control with __ldg and __ldcs
 * 5. Shared memory for x-vector caching
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

    std::sort(entries.begin(), entries.end(),
              [](const auto& a, const auto& b) {
                  if (std::get<0>(a) != std::get<0>(b))
                      return std::get<0>(a) < std::get<0>(b);
                  return std::get<1>(a) < std::get<1>(b);
              });

    matrix.rowPtr = new int[rows + 1];
    matrix.colIdx = new int[nnz];
    matrix.values = new float[nnz];

    matrix.rowPtr[0] = 0;
    int currentRow = 0;
    for (int i = 0; i < nnz; i++) {
        int r = std::get<0>(entries[i]);
        int c = std::get<1>(entries[i]);
        float v = std::get<2>(entries[i]);

        while (currentRow < r) {
            currentRow++;
            matrix.rowPtr[currentRow] = i;
        }
        matrix.colIdx[i] = c;
        matrix.values[i] = v;
    }
    while (currentRow < rows) {
        currentRow++;
        matrix.rowPtr[currentRow] = nnz;
    }

    return true;
}

// ==================== Baseline: Current Best ====================
template<int BLOCK_SIZE, int SMEM_INTS>
__global__ void spmv_baseline(int numRows, const int* __restrict__ rowPtr,
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

// ==================== Opt1: Aggressive Unroll ====================
template<int BLOCK_SIZE, int SMEM_INTS>
__global__ void spmv_unroll_aggressive(int numRows, const int* __restrict__ rowPtr,
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
    int rowLen = rowEnd - rowStart;

    // Process in chunks of 8 with aggressive unroll
    int idx = rowStart + threadInRow;
    int chunks = rowLen / 8;
    int remainder = rowLen % 8;

    #pragma unroll 8
    for (int c = 0; c < chunks; c++) {
        int curIdx = idx + c * 8;
        sum += values[curIdx] * __ldg(&x[colIdx[curIdx]]);
    }

    // Handle remainder
    int lastIdx = idx + chunks * 8;
    if (threadInRow < remainder) {
        sum += values[lastIdx] * __ldg(&x[colIdx[lastIdx]]);
    }

    sum += __shfl_down_sync(0xffffffff, sum, 4, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 2, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 1, 8);

    if (threadInRow == 0) y[row] = sum;
}

// ==================== Opt2: Triple Accumulator ====================
template<int BLOCK_SIZE, int SMEM_INTS>
__global__ void spmv_triple_accum(int numRows, const int* __restrict__ rowPtr,
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

    float sum0 = 0, sum1 = 0, sum2 = 0;
    int rowStart = sharedRowPtr[warpOffset + rowIdx];
    int rowEnd = sharedRowPtr[warpOffset + rowIdx + 1];

    // Process 24 elements per iteration (3x8)
    int idx = rowStart + threadInRow;
    for (; idx + 16 < rowEnd; idx += 24) {
        sum0 += values[idx] * __ldg(&x[colIdx[idx]]);
        sum1 += values[idx + 8] * __ldg(&x[colIdx[idx + 8]]);
        sum2 += values[idx + 16] * __ldg(&x[colIdx[idx + 16]]);
    }

    // Handle remainder
    for (; idx < rowEnd; idx += 8) {
        sum0 += values[idx] * __ldg(&x[colIdx[idx]]);
    }

    float sum = sum0 + sum1 + sum2;

    sum += __shfl_down_sync(0xffffffff, sum, 4, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 2, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 1, 8);

    if (threadInRow == 0) y[row] = sum;
}

// ==================== Opt3: Cache Streaming ====================
template<int BLOCK_SIZE, int SMEM_INTS>
__global__ void spmv_streaming_cache(int numRows, const int* __restrict__ rowPtr,
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

    // Use streaming load for values (evict-first)
    // Use __ldg for x (cached in read-only)
    for (int idx = rowStart + threadInRow; idx < rowEnd; idx += 8) {
        // Streaming load for values (used once)
        float val = values[idx];
        // Cached load for x (reused across rows)
        float xval = __ldg(&x[colIdx[idx]]);
        sum += val * xval;
    }

    sum += __shfl_down_sync(0xffffffff, sum, 4, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 2, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 1, 8);

    if (threadInRow == 0) y[row] = sum;
}

// ==================== Opt4: Quad Accumulator with Prefetch ====================
template<int BLOCK_SIZE, int SMEM_INTS>
__global__ void spmv_quad_accum(int numRows, const int* __restrict__ rowPtr,
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

    float sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
    int rowStart = sharedRowPtr[warpOffset + rowIdx];
    int rowEnd = sharedRowPtr[warpOffset + rowIdx + 1];

    int idx = rowStart + threadInRow;

    // Prefetch first element
    float x_cache0 = (idx < rowEnd) ? __ldg(&x[colIdx[idx]]) : 0;
    float v_cache0 = (idx < rowEnd) ? values[idx] : 0;

    for (idx += 8; idx + 24 < rowEnd; idx += 32) {
        // Prefetch ahead
        float x_next0 = __ldg(&x[colIdx[idx]]);
        float x_next1 = __ldg(&x[colIdx[idx + 8]]);
        float x_next2 = __ldg(&x[colIdx[idx + 16]]);
        float x_next3 = __ldg(&x[colIdx[idx + 24]]);

        sum0 += v_cache0 * x_cache0;
        sum1 += values[idx - 8 + 8] * x_next0;
        sum2 += values[idx - 8 + 16] * x_next1;
        sum3 += values[idx - 8 + 24] * x_next2;

        x_cache0 = x_next3;
        v_cache0 = values[idx + 24];
    }

    // Handle remainder
    for (; idx < rowEnd; idx += 8) {
        sum0 += values[idx] * __ldg(&x[colIdx[idx]]);
    }

    float sum = sum0 + sum1 + sum2 + sum3;

    sum += __shfl_down_sync(0xffffffff, sum, 4, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 2, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 1, 8);

    if (threadInRow == 0) y[row] = sum;
}

// ==================== Opt5: 4 threads per row with ILP ====================
template<int BLOCK_SIZE, int SMEM_INTS>
__global__ void spmv_4threads_ilp(int numRows, const int* __restrict__ rowPtr,
                                  const int* __restrict__ colIdx,
                                  const float* __restrict__ values,
                                  const float* __restrict__ x, float* __restrict__ y) {
    __shared__ int sharedRowPtr[SMEM_INTS + 16];

    int warpId = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int globalWarpId = blockIdx.x * (BLOCK_SIZE / WARP_SIZE) + warpId;

    int baseRow = globalWarpId * 16;  // 16 rows per warp
    int warpOffset = warpId * 17;

    if (lane < 17 && baseRow + lane <= numRows) {
        sharedRowPtr[warpOffset + lane] = rowPtr[baseRow + lane];
    }
    __syncthreads();

    int rowIdx = lane / 4;  // 4 threads per row
    int threadInRow = lane % 4;
    int row = baseRow + rowIdx;

    if (row >= numRows) return;

    float sum0 = 0, sum1 = 0;
    int rowStart = sharedRowPtr[warpOffset + rowIdx];
    int rowEnd = sharedRowPtr[warpOffset + rowIdx + 1];

    int idx = rowStart + threadInRow;
    for (; idx + 4 < rowEnd; idx += 8) {
        sum0 += values[idx] * __ldg(&x[colIdx[idx]]);
        sum1 += values[idx + 4] * __ldg(&x[colIdx[idx + 4]]);
    }
    if (idx < rowEnd) sum0 += values[idx] * __ldg(&x[colIdx[idx]]);

    float sum = sum0 + sum1;

    sum += __shfl_down_sync(0xffffffff, sum, 2, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 1, 4);

    if (threadInRow == 0) y[row] = sum;
}

// ==================== Opt6: 16 threads per row ====================
template<int BLOCK_SIZE, int SMEM_INTS>
__global__ void spmv_16threads(int numRows, const int* __restrict__ rowPtr,
                               const int* __restrict__ colIdx,
                               const float* __restrict__ values,
                               const float* __restrict__ x, float* __restrict__ y) {
    __shared__ int sharedRowPtr[SMEM_INTS + 16];

    int warpId = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int globalWarpId = blockIdx.x * (BLOCK_SIZE / WARP_SIZE) + warpId;

    int baseRow = globalWarpId * 4;  // 4 rows per warp
    int warpOffset = warpId * 5;

    if (lane < 5 && baseRow + lane <= numRows) {
        sharedRowPtr[warpOffset + lane] = rowPtr[baseRow + lane];
    }
    __syncthreads();

    int rowIdx = lane / 16;  // 16 threads per row
    int threadInRow = lane % 16;
    int row = baseRow + rowIdx;

    if (row >= numRows) return;

    float sum = 0;
    int rowStart = sharedRowPtr[warpOffset + rowIdx];
    int rowEnd = sharedRowPtr[warpOffset + rowIdx + 1];

    for (int idx = rowStart + threadInRow; idx < rowEnd; idx += 16) {
        sum += values[idx] * __ldg(&x[colIdx[idx]]);
    }

    // 16->1 reduction
    sum += __shfl_down_sync(0xffffffff, sum, 8, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 4, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 2, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 1, 16);

    if (threadInRow == 0) y[row] = sum;
}

template<typename KernelFunc>
float runKernel(KernelFunc kernel, int gridSize, int blockSize,
                int iterations, const CSRMatrix& matrix, float* d_x, float* d_y,
                size_t dataBytes, float peakBW, const char* name) {
    GpuTimer timer;
    float totalTime = 0;

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

    float avgTime = totalTime / iterations;
    float bw = (dataBytes / (avgTime * 1e-3)) / (1024 * 1024 * 1024);
    float util = bw / peakBW * 100;

    printf("   %-25s: %8.3f ms, %7.1f GB/s, %6.2f%%\n", name, avgTime, bw, util);

    return avgTime;
}

int main(int argc, char** argv) {
    printf("=== Extreme Optimization Test ===\n");
    printf("WARP_SIZE = %d\n", WARP_SIZE);

    std::string matrixFile = argc > 1 ? argv[1] : "./real_cases/mtx/p0_A";
    int iterations = argc > 2 ? atoi(argv[2]) : 10;

    CSRMatrix matrix;
    if (!loadMatrixMarket(matrixFile, matrix)) {
        std::cerr << "Failed to load matrix\n";
        return 1;
    }

    printf("Matrix: %d x %d, nnz=%d, avgNnz=%.1f\n",
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

    printf("\nExtreme Optimization Comparison:\n");

    int blockSize = 512;

    // 8 threads/row kernels
    int gridSize = (matrix.numRows + (blockSize / WARP_SIZE) * 8 - 1) / ((blockSize / WARP_SIZE) * 8);
    runKernel(spmv_baseline<512, 1024>, gridSize, blockSize, iterations, matrix, d_x, d_y,
              dataBytes, peakBW, "Baseline (8t/row)");
    runKernel(spmv_unroll_aggressive<512, 1024>, gridSize, blockSize, iterations, matrix, d_x, d_y,
              dataBytes, peakBW, "UnrollAggressive");
    runKernel(spmv_triple_accum<512, 1024>, gridSize, blockSize, iterations, matrix, d_x, d_y,
              dataBytes, peakBW, "TripleAccum");
    runKernel(spmv_streaming_cache<512, 1024>, gridSize, blockSize, iterations, matrix, d_x, d_y,
              dataBytes, peakBW, "StreamingCache");
    runKernel(spmv_quad_accum<512, 1024>, gridSize, blockSize, iterations, matrix, d_x, d_y,
              dataBytes, peakBW, "QuadAccum");

    // 4 threads/row kernel
    gridSize = (matrix.numRows + (blockSize / WARP_SIZE) * 16 - 1) / ((blockSize / WARP_SIZE) * 16);
    runKernel(spmv_4threads_ilp<512, 1024>, gridSize, blockSize, iterations, matrix, d_x, d_y,
              dataBytes, peakBW, "4threads_ILP");

    // 16 threads/row kernel
    gridSize = (matrix.numRows + (blockSize / WARP_SIZE) * 4 - 1) / ((blockSize / WARP_SIZE) * 4);
    runKernel(spmv_16threads<512, 1024>, gridSize, blockSize, iterations, matrix, d_x, d_y,
              dataBytes, peakBW, "16threads/row");

    delete[] h_x;
    delete[] matrix.rowPtr;
    delete[] matrix.colIdx;
    delete[] matrix.values;
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}