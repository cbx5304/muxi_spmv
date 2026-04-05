/**
 * @file test_merge_vs_bankaware.cu
 * @brief Compare merge-based kernel with BankAware kernel for Mars X201
 *
 * This test evaluates whether merge-based SpMV can improve performance
 * beyond the current best (BankAware ~26.83%)
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

// ==================== BankAware Kernel (Current Best) ====================
template<int BLOCK_SIZE, int SMEM_INTS>
__global__ void spmv_bank_aware(int numRows, const int* __restrict__ rowPtr,
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

// ==================== Merge Path Search ====================
__device__ __forceinline__ void merge_path_search(
    const int* __restrict__ rowPtr,
    int numRows,
    int nnz,
    int k,
    int& rowIdx,
    int& nnzIdx)
{
    int lo = max(0, k - nnz);
    int hi = min(k, numRows);

    while (lo < hi) {
        int mid = (lo + hi) / 2;
        int nnz_at_mid = k - mid;
        if (mid < numRows && nnz_at_mid > rowPtr[mid + 1]) {
            lo = mid + 1;
        } else if (mid > 0 && nnz_at_mid < rowPtr[mid]) {
            hi = mid;
        } else {
            lo = mid;
            hi = mid;
        }
    }

    rowIdx = lo;
    nnzIdx = k - lo;

    if (rowIdx > numRows) rowIdx = numRows;
    if (nnzIdx > nnz) nnzIdx = nnz;
}

// ==================== Warp Reduce ====================
template<int WarpSize>
__device__ __forceinline__ float warp_reduce_sum(float val) {
    if (WarpSize >= 64) val += __shfl_down_sync(0xffffffff, val, 32);
    if (WarpSize >= 32) val += __shfl_down_sync(0xffffffff, val, 16);
    if (WarpSize >= 16) val += __shfl_down_sync(0xffffffff, val, 8);
    if (WarpSize >= 8) val += __shfl_down_sync(0xffffffff, val, 4);
    if (WarpSize >= 4) val += __shfl_down_sync(0xffffffff, val, 2);
    if (WarpSize >= 2) val += __shfl_down_sync(0xffffffff, val, 1);
    return val;
}

// ==================== Merge-Based Kernel ====================
template<int BLOCK_SIZE, int WarpSize>
__global__ void spmv_merge_based(
    int numRows,
    int numCols,
    int nnz,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const float* __restrict__ values,
    const float* __restrict__ x,
    float* __restrict__ y,
    const int* __restrict__ mergePathPos,
    int numPartitions)
{
    int warpId = blockIdx.x * (BLOCK_SIZE / WarpSize) + (threadIdx.x / WarpSize);
    int lane = threadIdx.x % WarpSize;

    if (warpId >= numPartitions) return;

    int pathStart = mergePathPos[warpId];
    int pathEnd = mergePathPos[warpId + 1];

    int startRow, startNnz, endRow, endNnz;

    if (lane == 0) {
        merge_path_search(rowPtr, numRows, nnz, pathStart, startRow, startNnz);
        merge_path_search(rowPtr, numRows, nnz, pathEnd, endRow, endNnz);
    }

    startRow = __shfl_sync(0xffffffff, startRow, 0);
    startNnz = __shfl_sync(0xffffffff, startNnz, 0);
    endRow = __shfl_sync(0xffffffff, endRow, 0);
    endNnz = __shfl_sync(0xffffffff, endNnz, 0);

    int numRowsInPartition = endRow - startRow;
    if (numRowsInPartition <= 0) return;

    int rowsPerThread = (numRowsInPartition + WarpSize - 1) / WarpSize;
    int myRowStart = startRow + lane * rowsPerThread;
    int myRowEnd = min(myRowStart + rowsPerThread, endRow);

    // Handle partial first row
    if (lane == 0 && startNnz > rowPtr[startRow]) {
        float partialSum = 0;
        int rowEndNnz = (startRow + 1 <= endRow) ? rowPtr[startRow + 1] : endNnz;
        for (int idx = startNnz; idx < rowEndNnz; idx++) {
            partialSum += values[idx] * __ldg(&x[colIdx[idx]]);
        }
        atomicAdd(&y[startRow], partialSum);
        myRowStart = startRow + 1;
    }

    // Handle partial last row
    if (lane == WarpSize - 1 && endRow < numRows && endNnz < rowPtr[endRow + 1] && endNnz > rowPtr[endRow]) {
        float partialSum = 0;
        for (int idx = endNnz; idx < rowPtr[endRow + 1]; idx++) {
            partialSum += values[idx] * __ldg(&x[colIdx[idx]]);
        }
        atomicAdd(&y[endRow], partialSum);
        myRowEnd = endRow;
    }

    // Process complete rows
    for (int row = myRowStart; row < myRowEnd && row < numRows; row++) {
        float sum = 0;
        for (int idx = rowPtr[row]; idx < rowPtr[row + 1]; idx++) {
            sum += values[idx] * __ldg(&x[colIdx[idx]]);
        }
        y[row] = sum;
    }
}

// ==================== Compute Merge Partitions ====================
__global__ void compute_merge_partitions(
    int numRows,
    int nnz,
    int mergePathLength,
    int numPartitions,
    int* __restrict__ mergePathPos)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > numPartitions) return;

    int pathPerPartition = (mergePathLength + numPartitions - 1) / numPartitions;
    mergePathPos[tid] = min(tid * pathPerPartition, mergePathLength);
}

// ==================== Test Runner ====================
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

float runMergeBasedKernel(int iterations, const CSRMatrix& matrix, float* d_x, float* d_y,
                          size_t dataBytes, float peakBW, const char* name, int elementsPerPartition) {
    int blockSize = 256;
    int warpsPerBlock = blockSize / WARP_SIZE;
    int mergePathLength = matrix.numRows + matrix.nnz;

    int numSMs = (WARP_SIZE == 64) ? 104 : 128;
    int targetPartitions = mergePathLength / elementsPerPartition;
    int maxPartitions = numSMs * warpsPerBlock * 8;
    int numPartitions = max(1, min(targetPartitions, maxPartitions));

    int* d_mergePathPos;
    cudaMalloc(&d_mergePathPos, (numPartitions + 1) * sizeof(int));

    int gridSize = (numPartitions + 2 + 255) / 256;
    compute_merge_partitions<<<gridSize, 256>>>(matrix.numRows, matrix.nnz, mergePathLength, numPartitions, d_mergePathPos);

    gridSize = (numPartitions + warpsPerBlock - 1) / warpsPerBlock;

    GpuTimer timer;
    float totalTime = 0;

    for (int i = 0; i < iterations; i++) {
        cudaMemset(d_y, 0, matrix.numRows * sizeof(float));
        cudaDeviceSynchronize();
        timer.start();
        spmv_merge_based<256, WARP_SIZE><<<gridSize, blockSize>>>(
            matrix.numRows, matrix.numCols, matrix.nnz,
            matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values, d_x, d_y,
            d_mergePathPos, numPartitions);
        cudaDeviceSynchronize();
        timer.stop();
        totalTime += timer.elapsed_ms();
    }

    cudaFree(d_mergePathPos);

    float avgTime = totalTime / iterations;
    float bw = (dataBytes / (avgTime * 1e-3)) / (1024 * 1024 * 1024);
    float util = bw / peakBW * 100;

    printf("   %-25s: %8.3f ms, %7.1f GB/s, %6.2f%% (partitions=%d)\n",
           name, avgTime, bw, util, numPartitions);

    return avgTime;
}

int main(int argc, char** argv) {
    printf("=== Merge-Based vs BankAware Comparison ===\n");
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

    printf("\nKernel Comparison:\n");

    // BankAware (current best)
    int blockSize = 512;
    int gridSize = (matrix.numRows + (blockSize / WARP_SIZE) * 8 - 1) / ((blockSize / WARP_SIZE) * 8);
    runKernel(spmv_bank_aware<512, 1024>, gridSize, blockSize, iterations, matrix, d_x, d_y,
              dataBytes, peakBW, "BankAware (current best)");

    // Merge-based with different partition sizes
    printf("\nMerge-Based with different partition sizes:\n");
    runMergeBasedKernel(iterations, matrix, d_x, d_y, dataBytes, peakBW, "Merge(elems=64)", 64);
    runMergeBasedKernel(iterations, matrix, d_x, d_y, dataBytes, peakBW, "Merge(elems=32)", 32);
    runMergeBasedKernel(iterations, matrix, d_x, d_y, dataBytes, peakBW, "Merge(elems=16)", 16);
    runMergeBasedKernel(iterations, matrix, d_x, d_y, dataBytes, peakBW, "Merge(elems=8)", 8);
    runMergeBasedKernel(iterations, matrix, d_x, d_y, dataBytes, peakBW, "Merge(elems=4)", 4);

    delete[] h_x;
    delete[] matrix.rowPtr;
    delete[] matrix.colIdx;
    delete[] matrix.values;
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}