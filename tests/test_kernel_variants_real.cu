/**
 * @file test_kernel_variants_real.cu
 * @brief Test multiple kernel variants on real matrices
 *
 * Compare:
 * 1. Scalar (1 thread/row)
 * 2. Vector (1 warp/row)
 * 3. Adaptive Warp (16 rows/warp, 4 threads/row)
 * 4. Virtual Warp (simulated smaller warp)
 * 5. CSR5-style (fixed NNZ per warp)
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <cmath>

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

// 1. Scalar kernel
__global__ void spmv_scalar(int numRows, const int* __restrict__ rowPtr,
                            const int* __restrict__ colIdx,
                            const float* __restrict__ values,
                            const float* __restrict__ x, float* __restrict__ y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= numRows) return;

    float sum = 0;
    for (int i = rowPtr[row]; i < rowPtr[row + 1]; i++) {
        sum += values[i] * x[colIdx[i]];
    }
    y[row] = sum;
}

// 2. Vector kernel (1 warp per row)
__global__ void spmv_vector(int numRows, const int* __restrict__ rowPtr,
                            const int* __restrict__ colIdx,
                            const float* __restrict__ values,
                            const float* __restrict__ x, float* __restrict__ y) {
    int row = blockIdx.x * blockDim.x / WARP_SIZE + threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    if (row >= numRows) return;

    float sum = 0;
    for (int i = rowPtr[row] + lane; i < rowPtr[row + 1]; i += WARP_SIZE) {
        sum += values[i] * __ldg(&x[colIdx[i]]);
    }

    // Warp reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) y[row] = sum;
}

// 3. Adaptive Warp (16 rows per warp, 4 threads per row)
template<int BLOCK_SIZE, int SMEM_INTS>
__global__ void spmv_adaptive(int numRows, const int* __restrict__ rowPtr,
                              const int* __restrict__ colIdx,
                              const float* __restrict__ values,
                              const float* __restrict__ x, float* __restrict__ y) {
    __shared__ int sharedRowPtr[SMEM_INTS];

    int warpId = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int globalWarpId = blockIdx.x * (BLOCK_SIZE / WARP_SIZE) + warpId;

    int baseRow = globalWarpId * 16;
    int warpOffset = warpId * 17;

    if (lane < 17 && baseRow + lane <= numRows) {
        sharedRowPtr[warpOffset + lane] = rowPtr[baseRow + lane];
    }
    __syncthreads();

    int rowIdx = lane / 4;
    int threadInRow = lane % 4;
    int row = baseRow + rowIdx;

    if (row >= numRows) return;

    float sum = 0;
    int rowStart = sharedRowPtr[warpOffset + rowIdx];
    int rowEnd = sharedRowPtr[warpOffset + rowIdx + 1];

    int idx = rowStart + threadInRow;
    if (idx < rowEnd) {
        float x_val = __ldg(&x[colIdx[idx]]);
        float v_val = values[idx];

        for (idx += 4; idx < rowEnd; idx += 4) {
            float x_next = __ldg(&x[colIdx[idx]]);
            sum += v_val * x_val;
            x_val = x_next;
            v_val = values[idx];
        }
        sum += v_val * x_val;
    }

    sum += __shfl_down_sync(0xffffffff, sum, 2, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 1, 4);

    if (threadInRow == 0) y[row] = sum;
}

// 4. Virtual Warp (8 threads per row)
template<int VIRTUAL_WARP>
__global__ void spmv_virtual_warp(int numRows, const int* __restrict__ rowPtr,
                                  const int* __restrict__ colIdx,
                                  const float* __restrict__ values,
                                  const float* __restrict__ x, float* __restrict__ y) {
    int rowsPerWarp = WARP_SIZE / VIRTUAL_WARP;
    int virtualWarpId = threadIdx.x / VIRTUAL_WARP;
    int lane = threadIdx.x % VIRTUAL_WARP;
    int globalVirtualWarpId = blockIdx.x * (blockDim.x / VIRTUAL_WARP) + virtualWarpId;

    int row = globalVirtualWarpId;
    if (row >= numRows) return;

    float sum = 0;
    for (int i = rowPtr[row] + lane; i < rowPtr[row + 1]; i += VIRTUAL_WARP) {
        sum += values[i] * __ldg(&x[colIdx[i]]);
    }

    // Reduce within virtual warp
    for (int offset = VIRTUAL_WARP / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) y[row] = sum;
}

// 5. CSR5-style (fixed NNZ per warp)
template<int TILE_SIZE>
__global__ void spmv_csr5_style(int numRows, int nnz, const int* __restrict__ rowPtr,
                                const int* __restrict__ colIdx,
                                const float* __restrict__ values,
                                const float* __restrict__ x, float* __restrict__ y,
                                int* tileRowPtr) {
    int warpId = blockIdx.x * blockDim.x / WARP_SIZE + threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    int tileStart = warpId * TILE_SIZE;
    if (tileStart >= nnz) return;

    int tileEnd = min(tileStart + TILE_SIZE, nnz);

    // Find starting row (binary search would be better, but use linear for simplicity)
    int row = 0;
    if (tileRowPtr) {
        row = tileRowPtr[warpId];
    } else {
        // Linear search
        while (row < numRows && rowPtr[row + 1] <= tileStart) row++;
    }

    float sum = 0;
    for (int i = tileStart + lane; i < tileEnd; i += WARP_SIZE) {
        // Find which row this element belongs to
        int curRow = row;
        while (curRow < numRows && rowPtr[curRow + 1] <= i) curRow++;

        // This is simplified - real CSR5 needs atomics for cross-row tiles
        sum += values[i] * __ldg(&x[colIdx[i]]);
    }

    // Warp reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Write result (simplified - real CSR5 needs row tracking)
    if (lane == 0) {
        // Write to multiple rows if tile spans multiple rows
        // This is a simplification
    }
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

    printf("   %-20s: %8.3f ms, %7.1f GB/s, %6.2f%%\n", name, avgTime, bw, util);

    return avgTime;
}

void runTest(const std::string& matrixFile, int iterations) {
    std::cout << "\n=== Testing: " << matrixFile << " ===\n";

    CSRMatrix matrix;
    if (!loadMatrixMarket(matrixFile, matrix)) return;

    std::cout << "Matrix: " << matrix.numRows << " x " << matrix.numCols
              << ", nnz=" << matrix.nnz << ", avgNnz=" << (double)matrix.nnz / matrix.numRows << "\n";

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

    std::cout << "\nKernel Comparison:\n";

    // 1. Scalar
    int blockSize = 256;
    int gridSize = (matrix.numRows + blockSize - 1) / blockSize;
    runKernel(spmv_scalar, gridSize, blockSize, iterations, matrix, d_x, d_y,
              dataBytes, peakBW, "Scalar");

    // 2. Vector
    blockSize = 256;
    gridSize = (matrix.numRows + blockSize / WARP_SIZE - 1) / (blockSize / WARP_SIZE);
    runKernel(spmv_vector, gridSize, blockSize, iterations, matrix, d_x, d_y,
              dataBytes, peakBW, "Vector");

    // 3. Adaptive Warp
    blockSize = 512;
    gridSize = (matrix.numRows + (blockSize / WARP_SIZE) * 16 - 1) / ((blockSize / WARP_SIZE) * 16);
    runKernel(spmv_adaptive<512, 512>, gridSize, blockSize, iterations, matrix, d_x, d_y,
              dataBytes, peakBW, "Adaptive(B512)");

    // 4. Virtual Warp 8
    blockSize = 256;
    gridSize = (matrix.numRows + blockSize / 8 - 1) / (blockSize / 8);
    runKernel(spmv_virtual_warp<8>, gridSize, blockSize, iterations, matrix, d_x, d_y,
              dataBytes, peakBW, "VirtualWarp8");

    // 5. Virtual Warp 16
    gridSize = (matrix.numRows + blockSize / 16 - 1) / (blockSize / 16);
    runKernel(spmv_virtual_warp<16>, gridSize, blockSize, iterations, matrix, d_x, d_y,
              dataBytes, peakBW, "VirtualWarp16");

    // 6. Virtual Warp 32
    gridSize = (matrix.numRows + blockSize / 32 - 1) / (blockSize / 32);
    runKernel(spmv_virtual_warp<32>, gridSize, blockSize, iterations, matrix, d_x, d_y,
              dataBytes, peakBW, "VirtualWarp32");

    delete[] h_x;
    delete[] matrix.rowPtr;
    delete[] matrix.colIdx;
    delete[] matrix.values;
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(matrix.d_rowPtr);
    cudaFree(matrix.d_colIdx);
    cudaFree(matrix.d_values);
}

int main(int argc, char** argv) {
    std::cout << "=== Kernel Variants Comparison ===\n";
    std::cout << "WARP_SIZE = " << WARP_SIZE << "\n";

    std::string baseDir = argc > 1 ? argv[1] : "./mtx";
    int iterations = argc > 2 ? atoi(argv[2]) : 5;

    for (int i = 0; i < 3; i++) {
        std::string matrixFile = baseDir + "/p" + std::to_string(i) + "_A";
        runTest(matrixFile, iterations);
    }

    return 0;
}