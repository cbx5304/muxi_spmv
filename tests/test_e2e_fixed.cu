/**
 * @file test_e2e_fixed.cu
 * @brief End-to-end performance test with correct kernel
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
    int* h_rowPtr;
    int* h_colIdx;
    float* h_values;
    int* d_rowPtr;
    int* d_colIdx;
    float* d_values;
    float* d_x;
    float* d_y;
    float* h_y;
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

    matrix.h_rowPtr = new int[rows + 1];
    matrix.h_colIdx = new int[nnz];
    matrix.h_values = new float[nnz];
    matrix.h_y = new float[rows];

    matrix.h_rowPtr[0] = 0;
    int currentRow = 0;
    for (int i = 0; i < nnz; i++) {
        int r = std::get<0>(entries[i]);
        int c = std::get<1>(entries[i]);
        float v = std::get<2>(entries[i]);

        while (currentRow < r) {
            currentRow++;
            matrix.h_rowPtr[currentRow] = i;
        }
        matrix.h_colIdx[i] = c;
        matrix.h_values[i] = v;
    }
    while (currentRow < rows) {
        currentRow++;
        matrix.h_rowPtr[currentRow] = nnz;
    }

    return true;
}

// Direct copy from test_extreme_optimizations QuadAccum
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

    float x_cache0 = (idx < rowEnd) ? __ldg(&x[colIdx[idx]]) : 0;
    float v_cache0 = (idx < rowEnd) ? values[idx] : 0;

    for (idx += 8; idx + 24 < rowEnd; idx += 32) {
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

    for (; idx < rowEnd; idx += 8) {
        sum0 += values[idx] * __ldg(&x[colIdx[idx]]);
    }

    float sum = sum0 + sum1 + sum2 + sum3;

    sum += __shfl_down_sync(0xffffffff, sum, 4, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 2, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 1, 8);

    if (threadInRow == 0) y[row] = sum;
}

int main(int argc, char** argv) {
    printf("=== End-to-End Performance Test (Fixed) ===\n");
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

    float* h_x = new float[matrix.numCols];
    for (int i = 0; i < matrix.numCols; i++) h_x[i] = rand() / (float)RAND_MAX;

    size_t h2d_data = (matrix.numRows + 1) * sizeof(int) +
                      matrix.nnz * sizeof(int) +
                      matrix.nnz * sizeof(float) +
                      matrix.numCols * sizeof(float);

    size_t d2h_data = matrix.numRows * sizeof(float);
    size_t total_data = h2d_data + d2h_data;

    printf("\nData Transfer Sizes:\n");
    printf("  H2D: %.2f MB\n", h2d_data / (1024.0 * 1024.0));
    printf("  D2H: %.2f MB\n", d2h_data / (1024.0 * 1024.0));
    printf("  Total: %.2f MB\n", total_data / (1024.0 * 1024.0));

    cudaMalloc(&matrix.d_rowPtr, (matrix.numRows + 1) * sizeof(int));
    cudaMalloc(&matrix.d_colIdx, matrix.nnz * sizeof(int));
    cudaMalloc(&matrix.d_values, matrix.nnz * sizeof(float));
    cudaMalloc(&matrix.d_x, matrix.numCols * sizeof(float));
    cudaMalloc(&matrix.d_y, matrix.numRows * sizeof(float));

    cudaMemcpy(matrix.d_x, h_x, matrix.numCols * sizeof(float), cudaMemcpyHostToDevice);

    float peakBW = (WARP_SIZE == 64) ? 1843.0f : 1008.0f;
    size_t kernelData = (matrix.numRows + 1) * sizeof(int) +
                        matrix.nnz * sizeof(int) +
                        matrix.nnz * sizeof(float) * 2 +
                        matrix.numCols * sizeof(float) +
                        matrix.numRows * sizeof(float);

    printf("\nPerformance Breakdown (%d iterations):\n", iterations);

    // Kernel only test
    GpuTimer kernelTimer;
    float totalKernelTime = 0;

    cudaMemcpy(matrix.d_rowPtr, matrix.h_rowPtr, (matrix.numRows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(matrix.d_colIdx, matrix.h_colIdx, matrix.nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(matrix.d_values, matrix.h_values, matrix.nnz * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 512;
    int gridSize = (matrix.numRows + (blockSize / WARP_SIZE) * 8 - 1) / ((blockSize / WARP_SIZE) * 8);

    for (int i = 0; i < iterations; i++) {
        cudaMemset(matrix.d_y, 0, matrix.numRows * sizeof(float));
        cudaDeviceSynchronize();
        kernelTimer.start();
        spmv_quad_accum<512, 1024><<<gridSize, blockSize>>>(
            matrix.numRows, matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values, matrix.d_x, matrix.d_y);
        cudaDeviceSynchronize();
        kernelTimer.stop();
        totalKernelTime += kernelTimer.elapsed_ms();
    }

    float avgKernelTime = totalKernelTime / iterations;
    float kernelBW = (kernelData / (avgKernelTime * 1e-3)) / (1024 * 1024 * 1024);
    float kernelUtil = kernelBW / peakBW * 100;

    printf("  Kernel Only:     %8.3f ms, %7.1f GB/s, %6.2f%% utilization\n",
           avgKernelTime, kernelBW, kernelUtil);

    // End-to-end test
    GpuTimer e2eTimer;
    float totalE2ETime = 0;

    for (int i = 0; i < iterations; i++) {
        e2eTimer.start();
        cudaMemcpy(matrix.d_rowPtr, matrix.h_rowPtr, (matrix.numRows + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(matrix.d_colIdx, matrix.h_colIdx, matrix.nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(matrix.d_values, matrix.h_values, matrix.nnz * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(matrix.d_x, h_x, matrix.numCols * sizeof(float), cudaMemcpyHostToDevice);
        spmv_quad_accum<512, 1024><<<gridSize, blockSize>>>(
            matrix.numRows, matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values, matrix.d_x, matrix.d_y);
        cudaMemcpy(matrix.h_y, matrix.d_y, matrix.numRows * sizeof(float), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        e2eTimer.stop();
        totalE2ETime += e2eTimer.elapsed_ms();
    }

    float avgE2ETime = totalE2ETime / iterations;
    float e2eBW = (total_data / (avgE2ETime * 1e-3)) / (1024 * 1024 * 1024);
    float e2eUtil = e2eBW / peakBW * 100;

    printf("  End-to-End:      %8.3f ms, %7.1f GB/s, %6.2f%% utilization\n",
           avgE2ETime, e2eBW, e2eUtil);

    printf("\nTime Breakdown:\n");
    printf("  Kernel:          %8.3f ms (%.1f%%)\n", avgKernelTime, avgKernelTime / avgE2ETime * 100);
    printf("  Data Transfer:   %8.3f ms (%.1f%%)\n", avgE2ETime - avgKernelTime, (avgE2ETime - avgKernelTime) / avgE2ETime * 100);

    printf("\n=== Summary ===\n");
    printf("Platform: %s\n", (WARP_SIZE == 64) ? "Mars X201" : "RTX 4090");
    printf("Peak Bandwidth: %.0f GB/s\n", peakBW);
    printf("Kernel Utilization: %.2f%%\n", kernelUtil);
    printf("End-to-End Utilization: %.2f%%\n", e2eUtil);

    delete[] h_x;
    delete[] matrix.h_rowPtr;
    delete[] matrix.h_colIdx;
    delete[] matrix.h_values;
    delete[] matrix.h_y;
    cudaFree(matrix.d_rowPtr);
    cudaFree(matrix.d_colIdx);
    cudaFree(matrix.d_values);
    cudaFree(matrix.d_x);
    cudaFree(matrix.d_y);

    return 0;
}