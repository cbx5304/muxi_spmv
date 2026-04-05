/**
 * @file test_low_density_optimized.cu
 * @brief Specialized kernel for matrices with avgNnz < 10
 *
 * For very sparse matrices (avgNnz < 10), each row has very few elements.
 * Traditional approaches waste resources. This test explores:
 * 1. 4 threads per row (instead of 8) for better packing
 * 2. L1 cache configuration
 * 3. Launch bounds optimization
 * 4. Shared memory for rowPtr caching
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

// 4 threads per row - better for avgNnz < 10
template<int BLOCK_SIZE>
__launch_bounds__(512, 2)
__global__ void spmv_4t_row(int numRows, const int* __restrict__ rowPtr,
                             const int* __restrict__ colIdx,
                             const float* __restrict__ values,
                             const float* __restrict__ x, float* __restrict__ y) {
    int warpId = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int globalWarpId = blockIdx.x * (BLOCK_SIZE / WARP_SIZE) + warpId;

    int baseRow = globalWarpId * 16;  // 64/4 = 16 rows per warp
    int rowIdx = lane / 4;
    int threadInRow = lane % 4;
    int row = baseRow + rowIdx;

    if (row >= numRows) return;

    int rowStart = rowPtr[row];
    int rowEnd = rowPtr[row + 1];

    float sum = 0;
    for (int idx = rowStart + threadInRow; idx < rowEnd; idx += 4) {
        sum += values[idx] * __ldg(&x[colIdx[idx]]);
    }

    sum += __shfl_down_sync(0xffffffff, sum, 2, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 1, 4);

    if (threadInRow == 0) y[row] = sum;
}

// 2 threads per row - for extremely sparse rows
template<int BLOCK_SIZE>
__launch_bounds__(512, 2)
__global__ void spmv_2t_row(int numRows, const int* __restrict__ rowPtr,
                             const int* __restrict__ colIdx,
                             const float* __restrict__ values,
                             const float* __restrict__ x, float* __restrict__ y) {
    int warpId = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int globalWarpId = blockIdx.x * (BLOCK_SIZE / WARP_SIZE) + warpId;

    int baseRow = globalWarpId * 32;  // 64/2 = 32 rows per warp
    int rowIdx = lane / 2;
    int threadInRow = lane % 2;
    int row = baseRow + rowIdx;

    if (row >= numRows) return;

    int rowStart = rowPtr[row];
    int rowEnd = rowPtr[row + 1];

    float sum = 0;
    for (int idx = rowStart + threadInRow; idx < rowEnd; idx += 2) {
        sum += values[idx] * __ldg(&x[colIdx[idx]]);
    }

    sum += __shfl_down_sync(0xffffffff, sum, 1, 2);

    if (threadInRow == 0) y[row] = sum;
}

// 1 thread per row - for extremely sparse
template<int BLOCK_SIZE>
__launch_bounds__(512, 2)
__global__ void spmv_1t_row(int numRows, const int* __restrict__ rowPtr,
                             const int* __restrict__ colIdx,
                             const float* __restrict__ values,
                             const float* __restrict__ x, float* __restrict__ y) {
    int row = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (row >= numRows) return;

    int rowStart = rowPtr[row];
    int rowEnd = rowPtr[row + 1];

    float sum = 0;
    for (int idx = rowStart; idx < rowEnd; idx++) {
        sum += values[idx] * __ldg(&x[colIdx[idx]]);
    }

    y[row] = sum;
}

// Vectorized 2 threads with double accumulator
template<int BLOCK_SIZE>
__launch_bounds__(512, 2)
__global__ void spmv_2t_dual(int numRows, const int* __restrict__ rowPtr,
                              const int* __restrict__ colIdx,
                              const float* __restrict__ values,
                              const float* __restrict__ x, float* __restrict__ y) {
    int warpId = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int globalWarpId = blockIdx.x * (BLOCK_SIZE / WARP_SIZE) + warpId;

    int baseRow = globalWarpId * 32;
    int rowIdx = lane / 2;
    int threadInRow = lane % 2;
    int row = baseRow + rowIdx;

    if (row >= numRows) return;

    int rowStart = rowPtr[row];
    int rowEnd = rowPtr[row + 1];

    float sum0 = 0, sum1 = 0;
    int idx = rowStart + threadInRow;
    for (; idx + 2 < rowEnd; idx += 4) {
        sum0 += values[idx] * __ldg(&x[colIdx[idx]]);
        sum1 += values[idx + 2] * __ldg(&x[colIdx[idx + 2]]);
    }
    for (; idx < rowEnd; idx += 2) {
        sum0 += values[idx] * __ldg(&x[colIdx[idx]]);
    }

    float sum = sum0 + sum1;
    sum += __shfl_down_sync(0xffffffff, sum, 1, 2);

    if (threadInRow == 0) y[row] = sum;
}

// L1 cache prefer L1
template<int BLOCK_SIZE>
__launch_bounds__(512, 2)
__global__ void spmv_l1_prefer(int numRows, const int* __restrict__ rowPtr,
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
    printf("=== Low Density Matrix Optimization Test ===\n");
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

    printf("%-20s %12s %12s\n", "Configuration", "Time(ms)", "Util(%)");
    printf("%-20s %12s %12s\n", "--------------------", "--------", "-------");

    auto printResult = [&](const char* name, float t) {
        float bw = (dataBytes / (t * 1e-3)) / (1024 * 1024 * 1024);
        printf("%-20s %12.3f %11.2f%%\n", name, t, bw / peakBW * 100);
    };

    float t;
    int blockSize = 512;

    // Test different thread configurations
    // 8t/row (baseline)
    int gs8 = (matrix.numRows + (blockSize / WARP_SIZE) * 8 - 1) / ((blockSize / WARP_SIZE) * 8);
    t = runTest(spmv_l1_prefer<512>, gs8, blockSize, iterations, matrix, d_x, d_y);
    printResult("8t/row (baseline)", t);

    // 4t/row
    int gs4 = (matrix.numRows + (blockSize / WARP_SIZE) * 16 - 1) / ((blockSize / WARP_SIZE) * 16);
    t = runTest(spmv_4t_row<512>, gs4, blockSize, iterations, matrix, d_x, d_y);
    printResult("4t/row", t);

    // 2t/row
    int gs2 = (matrix.numRows + (blockSize / WARP_SIZE) * 32 - 1) / ((blockSize / WARP_SIZE) * 32);
    t = runTest(spmv_2t_row<512>, gs2, blockSize, iterations, matrix, d_x, d_y);
    printResult("2t/row", t);

    // 2t/row with dual accumulator
    t = runTest(spmv_2t_dual<512>, gs2, blockSize, iterations, matrix, d_x, d_y);
    printResult("2t/row + DualAccum", t);

    // 1t/row
    int gs1 = (matrix.numRows + blockSize - 1) / blockSize;
    t = runTest(spmv_1t_row<512>, gs1, blockSize, iterations, matrix, d_x, d_y);
    printResult("1t/row", t);

    // Test L1 cache preference
    cudaFuncSetCacheConfig(spmv_l1_prefer<512>, cudaFuncCachePreferL1);
    t = runTest(spmv_l1_prefer<512>, gs8, blockSize, iterations, matrix, d_x, d_y);
    printResult("8t/row + PreferL1", t);

    // Test shared memory cache config
    cudaFuncSetCacheConfig(spmv_4t_row<512>, cudaFuncCachePreferShared);
    t = runTest(spmv_4t_row<512>, gs4, blockSize, iterations, matrix, d_x, d_y);
    printResult("4t/row + PreferShared", t);

    printf("\n=== Analysis ===\n");
    printf("For matrices with avgNnz < 10:\n");
    printf("- Fewer threads/row may improve packing efficiency\n");
    printf("- L1 cache preference might help for random access\n");
    printf("- Launch bounds can improve register usage\n");

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