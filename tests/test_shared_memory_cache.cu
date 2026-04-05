/**
 * @file test_shared_memory_cache.cu
 * @brief Test shared memory caching for x-vector access
 *
 * Shared memory can cache frequently accessed x-vector elements,
 * which is beneficial when the same column indices are accessed multiple times.
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif

#define SHARED_CACHE_SIZE 4096  // Number of float elements in shared memory

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

// Standard kernel without shared memory
template<int BLOCK_SIZE, int THREADS_PER_ROW>
__global__ void spmv_standard(int numRows, const int* __restrict__ rowPtr,
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

// Kernel with shared memory cache for x-vector
template<int BLOCK_SIZE, int THREADS_PER_ROW, int CACHE_SIZE>
__global__ void spmv_shared_cache(int numRows, const int* __restrict__ rowPtr,
                                   const int* __restrict__ colIdx,
                                   const float* __restrict__ values,
                                   const float* __restrict__ x, float* __restrict__ y) {
    __shared__ float x_cache[CACHE_SIZE];

    // Cooperative loading of x-vector cache
    // Each thread loads multiple elements
    for (int i = threadIdx.x; i < CACHE_SIZE; i += BLOCK_SIZE) {
        x_cache[i] = x[i];
    }
    __syncthreads();

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
        int col = colIdx[idx];
        // Use cache if within range, otherwise global memory
        float x_val = (col < CACHE_SIZE) ? x_cache[col] : __ldg(&x[col]);
        sum += values[idx] * x_val;
    }

    #pragma unroll
    for (int offset = THREADS_PER_ROW / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset, THREADS_PER_ROW);
    }

    if (threadInRow == 0) y[row] = sum;
}

// Kernel with software-managed cache using __ldg
template<int BLOCK_SIZE, int THREADS_PER_ROW>
__global__ void spmv_ldg_cache(int numRows, const int* __restrict__ rowPtr,
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

    // Prefetch first few elements
    float sum = 0;
    int idx = rowStart + threadInRow;

    // Unrolled loop with prefetch
    #pragma unroll 4
    for (; idx + THREADS_PER_ROW * 4 < rowEnd; idx += THREADS_PER_ROW * 4) {
        float v0 = values[idx];
        float v1 = values[idx + THREADS_PER_ROW];
        float v2 = values[idx + THREADS_PER_ROW * 2];
        float v3 = values[idx + THREADS_PER_ROW * 3];

        sum += v0 * __ldg(&x[colIdx[idx]]);
        sum += v1 * __ldg(&x[colIdx[idx + THREADS_PER_ROW]]);
        sum += v2 * __ldg(&x[colIdx[idx + THREADS_PER_ROW * 2]]);
        sum += v3 * __ldg(&x[colIdx[idx + THREADS_PER_ROW * 3]]);
    }

    for (; idx < rowEnd; idx += THREADS_PER_ROW) {
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
    printf("=== Shared Memory Cache Optimization Test ===\n");
    printf("WARP_SIZE = %d\n", WARP_SIZE);

    std::string matrixFile = argc > 1 ? argv[1] : "./real_cases/mtx/p0_A";
    int iterations = argc > 2 ? atoi(argv[2]) : 30;

    CSRMatrix matrix;
    if (!loadMatrixMarket(matrixFile, matrix)) {
        std::cerr << "Failed to load matrix: " << matrixFile << "\n";
        return 1;
    }

    printf("Matrix: %d x %d, nnz=%d, avgNnz=%.2f\n",
           matrix.numRows, matrix.numCols, matrix.nnz,
           (double)matrix.nnz / matrix.numRows);

    // Analyze column index range
    int minCol = matrix.numCols, maxCol = 0;
    for (int i = 0; i < matrix.nnz; i++) {
        minCol = std::min(minCol, matrix.colIdx[i]);
        maxCol = std::max(maxCol, matrix.colIdx[i]);
    }
    printf("Column index range: %d to %d\n\n", minCol, maxCol);

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

    printf("%-25s %12s %12s\n", "Configuration", "Time(ms)", "Util(%)");
    printf("%-25s %12s %12s\n", "-------------------------", "--------", "-------");

    auto testKernel = [&](const char* name, auto kernel) {
        float t = runTest(kernel, gridSize, blockSize, iterations, matrix, d_x, d_y);
        float bw = (dataBytes / (t * 1e-3)) / (1024 * 1024 * 1024);
        printf("%-25s %12.3f %11.2f%%\n", name, t, bw / peakBW * 100);
    };

    testKernel("Standard", spmv_standard<512, 8>);
    testKernel("__ldg + Unroll", spmv_ldg_cache<512, 8>);

    // Test different shared memory cache sizes
    if (matrix.numCols >= 1024) {
        testKernel("Shared Cache 1K", spmv_shared_cache<512, 8, 1024>);
    }
    if (matrix.numCols >= 4096) {
        testKernel("Shared Cache 4K", spmv_shared_cache<512, 8, 4096>);
    }

    printf("\n=== Analysis ===\n");
    printf("Shared memory cache can help when:\n");
    printf("1. Column indices have high locality\n");
    printf("2. Same x-vector elements are accessed multiple times\n");
    printf("3. Matrix columns fit in shared memory\n");
    printf("\nLimitations:\n");
    printf("- Shared memory is limited (~48KB per SM)\n");
    printf("- Large matrices exceed cache capacity\n");
    printf("- Random access patterns have low locality\n");

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