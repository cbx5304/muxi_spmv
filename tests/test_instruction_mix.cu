/**
 * @file test_instruction_mix.cu
 * @brief Test instruction-level optimizations: register usage, ILP, warp occupancy
 *
 * Key hypotheses:
 * 1. More registers per thread -> better latency hiding
 * 2. Independent operations -> ILP (Instruction Level Parallelism)
 * 3. Warp occupancy analysis
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

// 2-accumulator ILP (dual accumulators)
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

    // Dual accumulators for ILP
    float sum0 = 0.0f, sum1 = 0.0f;
    int idx = rowStart + threadInRow;

    // Process pairs
    while (idx + THREADS_PER_ROW < rowEnd) {
        sum0 += values[idx] * __ldg(&x[colIdx[idx]]);
        sum1 += values[idx + THREADS_PER_ROW] * __ldg(&x[colIdx[idx + THREADS_PER_ROW]]);
        idx += THREADS_PER_ROW * 2;
    }

    // Remaining
    if (idx < rowEnd) {
        sum0 += values[idx] * __ldg(&x[colIdx[idx]]);
    }

    float sum = sum0 + sum1;

    #pragma unroll
    for (int offset = THREADS_PER_ROW / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset, THREADS_PER_ROW);
    }

    if (threadInRow == 0) y[row] = sum;
}

// 4-accumulator ILP (quad accumulators)
template<int BLOCK_SIZE, int THREADS_PER_ROW>
__global__ void spmv_quad_accum(int numRows, const int* __restrict__ rowPtr,
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

    // Quad accumulators for ILP
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    int idx = rowStart + threadInRow;

    // Process quads
    while (idx + THREADS_PER_ROW * 3 < rowEnd) {
        sum0 += values[idx] * __ldg(&x[colIdx[idx]]);
        sum1 += values[idx + THREADS_PER_ROW] * __ldg(&x[colIdx[idx + THREADS_PER_ROW]]);
        sum2 += values[idx + THREADS_PER_ROW * 2] * __ldg(&x[colIdx[idx + THREADS_PER_ROW * 2]]);
        sum3 += values[idx + THREADS_PER_ROW * 3] * __ldg(&x[colIdx[idx + THREADS_PER_ROW * 3]]);
        idx += THREADS_PER_ROW * 4;
    }

    // Remaining elements
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

// 8-accumulator ILP (octo accumulators)
template<int BLOCK_SIZE, int THREADS_PER_ROW>
__global__ void spmv_octo_accum(int numRows, const int* __restrict__ rowPtr,
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

    // Octo accumulators for maximum ILP
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    float sum4 = 0.0f, sum5 = 0.0f, sum6 = 0.0f, sum7 = 0.0f;
    int idx = rowStart + threadInRow;

    // Process octets
    while (idx + THREADS_PER_ROW * 7 < rowEnd) {
        sum0 += values[idx] * __ldg(&x[colIdx[idx]]);
        sum1 += values[idx + THREADS_PER_ROW] * __ldg(&x[colIdx[idx + THREADS_PER_ROW]]);
        sum2 += values[idx + THREADS_PER_ROW * 2] * __ldg(&x[colIdx[idx + THREADS_PER_ROW * 2]]);
        sum3 += values[idx + THREADS_PER_ROW * 3] * __ldg(&x[colIdx[idx + THREADS_PER_ROW * 3]]);
        sum4 += values[idx + THREADS_PER_ROW * 4] * __ldg(&x[colIdx[idx + THREADS_PER_ROW * 4]]);
        sum5 += values[idx + THREADS_PER_ROW * 5] * __ldg(&x[colIdx[idx + THREADS_PER_ROW * 5]]);
        sum6 += values[idx + THREADS_PER_ROW * 6] * __ldg(&x[colIdx[idx + THREADS_PER_ROW * 6]]);
        sum7 += values[idx + THREADS_PER_ROW * 7] * __ldg(&x[colIdx[idx + THREADS_PER_ROW * 7]]);
        idx += THREADS_PER_ROW * 8;
    }

    // Remaining
    while (idx < rowEnd) {
        sum0 += values[idx] * __ldg(&x[colIdx[idx]]);
        idx += THREADS_PER_ROW;
    }

    float sum = (sum0 + sum1) + (sum2 + sum3) + (sum4 + sum5) + (sum6 + sum7);

    #pragma unroll
    for (int offset = THREADS_PER_ROW / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset, THREADS_PER_ROW);
    }

    if (threadInRow == 0) y[row] = sum;
}

// Aggressive unrolling with #pragma unroll
template<int BLOCK_SIZE, int THREADS_PER_ROW>
__global__ void spmv_aggressive_unroll(int numRows, const int* __restrict__ rowPtr,
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

    float sum = 0.0f;

    #pragma unroll 8
    for (int idx = rowStart + threadInRow; idx < rowEnd; idx += THREADS_PER_ROW) {
        sum += values[idx] * __ldg(&x[colIdx[idx]]);
    }

    #pragma unroll
    for (int offset = THREADS_PER_ROW / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset, THREADS_PER_ROW);
    }

    if (threadInRow == 0) y[row] = sum;
}

// Launch bounds to hint occupancy
template<int BLOCK_SIZE, int THREADS_PER_ROW>
__launch_bounds__(BLOCK_SIZE, 2)  // 2 blocks per SM
__global__ void spmv_launch_bounds(int numRows, const int* __restrict__ rowPtr,
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

    float sum = 0.0f;

    #pragma unroll 4
    for (int idx = rowStart + threadInRow; idx < rowEnd; idx += THREADS_PER_ROW) {
        sum += values[idx] * __ldg(&x[colIdx[idx]]);
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
    printf("=== Instruction Mix & ILP Optimization Test ===\n");
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

    testKernel("Dual Accum (2x ILP)", spmv_dual_accum<512, 8>);
    testKernel("Quad Accum (4x ILP)", spmv_quad_accum<512, 8>);
    testKernel("Octo Accum (8x ILP)", spmv_octo_accum<512, 8>);
    testKernel("Aggressive Unroll", spmv_aggressive_unroll<512, 8>);
    testKernel("Launch Bounds", spmv_launch_bounds<512, 8>);

    printf("\n=== Analysis ===\n");
    printf("ILP (Instruction Level Parallelism):\n");
    printf("- Dual accum: 2 independent FMA operations\n");
    printf("- Quad accum: 4 independent FMA operations\n");
    printf("- Octo accum: 8 independent FMA operations\n");
    printf("\nRegister usage increases with more accumulators.\n");
    printf("This can hide memory latency but may reduce occupancy.\n");

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