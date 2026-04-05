/**
 * @file test_standalone_comparison.cu
 * @brief Standalone test for comparing Mars X201 and RTX 4090
 * No external dependencies
 */

#include <iostream>
#include <cmath>
#include <cstdlib>

#define WARP_SIZE 32  // RTX 4090

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

class GpuTimer {
public:
    GpuTimer() { CUDA_CHECK(cudaEventCreate(&start_)); CUDA_CHECK(cudaEventCreate(&stop_)); }
    ~GpuTimer() { cudaEventDestroy(start_); cudaEventDestroy(stop_); }
    void start() { CUDA_CHECK(cudaEventRecord(start_, 0)); }
    void stop() { CUDA_CHECK(cudaEventRecord(stop_, 0)); CUDA_CHECK(cudaEventSynchronize(stop_)); }
    float elapsed_ms() { float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_)); return ms; }
private:
    cudaEvent_t start_, stop_;
};

// Simple CSR matrix structure
struct CSRMatrix {
    int numRows, numCols, nnz;
    int* d_rowPtr;
    int* d_colIdx;
    float* d_values;
};

// Generate random matrix
void generateRandomMatrix(int rows, int cols, int nnz, CSRMatrix& matrix) {
    matrix.numRows = rows;
    matrix.numCols = cols;
    matrix.nnz = nnz;

    // Allocate host memory
    int* h_rowPtr = new int[rows + 1];
    int* h_colIdx = new int[nnz];
    float* h_values = new float[nnz];

    // Generate random matrix with approximately avgNnz non-zeros per row
    int avgNnz = nnz / rows;
    int currentNnz = 0;
    h_rowPtr[0] = 0;

    for (int i = 0; i < rows; i++) {
        int rowNnz = avgNnz;
        // Add some variation
        if (rand() % 3 == 0) rowNnz = std::max(1, rowNnz - 1);
        if (rand() % 3 == 0) rowNnz = std::min(avgNnz * 2, rowNnz + 1);

        for (int j = 0; j < rowNnz && currentNnz < nnz; j++) {
            h_colIdx[currentNnz] = rand() % cols;
            h_values[currentNnz] = static_cast<float>(rand()) / RAND_MAX;
            currentNnz++;
        }
        h_rowPtr[i + 1] = currentNnz;
    }

    matrix.nnz = currentNnz;

    // Allocate and copy to device
    CUDA_CHECK(cudaMalloc(&matrix.d_rowPtr, (rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&matrix.d_colIdx, matrix.nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&matrix.d_values, matrix.nnz * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(matrix.d_rowPtr, h_rowPtr, (rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(matrix.d_colIdx, h_colIdx, matrix.nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(matrix.d_values, h_values, matrix.nnz * sizeof(float), cudaMemcpyHostToDevice));

    delete[] h_rowPtr;
    delete[] h_colIdx;
    delete[] h_values;
}

// Scalar kernel (1 thread per row)
template<typename FloatType, int BLOCK_SIZE>
__global__ void spmv_scalar_kernel(
    int numRows,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= numRows) return;

    FloatType sum = 0;
    for (int idx = rowPtr[row]; idx < rowPtr[row + 1]; idx++) {
        sum += values[idx] * x[colIdx[idx]];
    }
    y[row] = sum;
}

// Adaptive Warp kernel for RTX 4090 (warp=32)
template<typename FloatType, int BLOCK_SIZE>
__global__ void spmv_adaptive_kernel(
    int numRows,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y)
{
    __shared__ int sharedRowPtr[BLOCK_SIZE + 1];

    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int globalWarpId = blockIdx.x * (BLOCK_SIZE / WARP_SIZE) + warpId;

    // Each warp handles 8 rows (32/4 = 8 rows, 4 threads each)
    int baseRow = globalWarpId * 8;

    if (lane < 9 && baseRow + lane <= numRows) {
        sharedRowPtr[threadIdx.x / WARP_SIZE * 9 + lane] = rowPtr[baseRow + lane];
    }
    __syncthreads();

    int rowIdx = lane / 4;
    int threadInRow = lane % 4;
    int row = baseRow + rowIdx;

    if (row >= numRows) return;

    FloatType sum = 0;
    int rowStart = sharedRowPtr[threadIdx.x / WARP_SIZE * 9 + rowIdx];
    int rowEnd = sharedRowPtr[threadIdx.x / WARP_SIZE * 9 + rowIdx + 1];

    for (int idx = rowStart + threadInRow; idx < rowEnd; idx += 4) {
        sum += values[idx] * __ldg(&x[colIdx[idx]]);
    }

    sum += __shfl_down_sync(0xffffffff, sum, 2, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 1, 4);

    if (threadInRow == 0) {
        y[row] = sum;
    }
}

void runTest(int rows, int cols, int avgNnz, int iterations) {
    std::cout << "\n=== Test: rows=" << rows << ", avgNnz=" << avgNnz << " ===\n";

    CSRMatrix matrix;
    int nnz = rows * avgNnz;
    generateRandomMatrix(rows, cols, nnz, matrix);
    CUDA_CHECK(cudaDeviceSynchronize());

    float* h_x = new float[cols];
    for (int i = 0; i < cols; i++) h_x[i] = rand() / (float)RAND_MAX;

    float* d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, rows * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, cols * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    // RTX 4090 bandwidth: 1008 GB/s
    float peakBW = 1008.0f;
    size_t dataBytes = rows * sizeof(int) * 2 + matrix.nnz * sizeof(int) +
                       matrix.nnz * sizeof(float) * 2 + rows * sizeof(float);

    GpuTimer timer;
    int blockSize = 256;
    int gridSize = (rows + blockSize - 1) / blockSize;

    // Scalar kernel
    std::cout << "Scalar: ";
    float totalTime = 0;
    for (int i = 0; i < iterations; i++) {
        CUDA_CHECK(cudaMemset(d_y, 0, rows * sizeof(float)));
        CUDA_CHECK(cudaDeviceSynchronize());
        timer.start();
        spmv_scalar_kernel<float, 256><<<gridSize, blockSize>>>(
            rows, matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values, d_x, d_y);
        timer.stop();
        totalTime += timer.elapsed_ms();
    }
    float avgTime = totalTime / iterations;
    float bw = (dataBytes / (avgTime * 1e-3)) / (1024 * 1024 * 1024);
    float util = bw / peakBW * 100;
    std::cout << avgTime << " ms, " << bw << " GB/s, " << util << "%\n";

    // Adaptive Warp kernel
    std::cout << "Adaptive: ";
    int rowsPerBlock = (blockSize / WARP_SIZE) * 8;
    int adaptiveGridSize = (rows + rowsPerBlock - 1) / rowsPerBlock;
    totalTime = 0;
    for (int i = 0; i < iterations; i++) {
        CUDA_CHECK(cudaMemset(d_y, 0, rows * sizeof(float)));
        CUDA_CHECK(cudaDeviceSynchronize());
        timer.start();
        spmv_adaptive_kernel<float, 256><<<adaptiveGridSize, blockSize>>>(
            rows, matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values, d_x, d_y);
        timer.stop();
        totalTime += timer.elapsed_ms();
    }
    avgTime = totalTime / iterations;
    bw = (dataBytes / (avgTime * 1e-3)) / (1024 * 1024 * 1024);
    util = bw / peakBW * 100;
    std::cout << avgTime << " ms, " << bw << " GB/s, " << util << "%\n";

    delete[] h_x;
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(matrix.d_rowPtr));
    CUDA_CHECK(cudaFree(matrix.d_colIdx));
    CUDA_CHECK(cudaFree(matrix.d_values));
}

int main(int argc, char** argv) {
    std::cout << "=== RTX 4090 SpMV Test ===\n";
    std::cout << "WARP_SIZE = " << WARP_SIZE << "\n";

    int iterations = 100;  // More iterations for accurate timing

    runTest(100000, 1000, 4, iterations);
    runTest(500000, 1000, 4, iterations);
    runTest(1000000, 1000, 4, iterations);

    return 0;
}