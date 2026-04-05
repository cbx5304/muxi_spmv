/**
 * @file test_pinned_memory.cu
 * @brief Test pinned (page-locked) memory for faster data transfers
 *
 * Pinned memory can significantly improve PCIe transfer speeds by:
 * 1. Avoiding page faults during transfer
 * 2. Enabling true async transfers
 * 3. Allowing higher PCIe bandwidth utilization
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

template<int BLOCK_SIZE, int THREADS_PER_ROW>
__global__ void spmv_full(int numRows, const int* __restrict__ rowPtr,
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

struct TimingResult {
    float totalTime;
    float transferH2D;
    float kernelTime;
    float transferD2H;
};

// Test with pageable memory (standard malloc)
TimingResult testPageableMemory(const CSRMatrix& matrix, int iterations) {
    // Standard pageable memory
    float* h_x = new float[matrix.numCols];
    float* h_y = new float[matrix.numRows];
    for (int i = 0; i < matrix.numCols; i++) h_x[i] = rand() / (float)RAND_MAX;

    float* d_x, *d_y;
    cudaMalloc(&d_x, matrix.numCols * sizeof(float));
    cudaMalloc(&d_y, matrix.numRows * sizeof(float));

    int blockSize = 512;
    int rowsPerWarp = WARP_SIZE / 8;
    int gridSize = (matrix.numRows + (blockSize / WARP_SIZE) * rowsPerWarp - 1) / ((blockSize / WARP_SIZE) * rowsPerWarp);

    GpuTimer timer;
    float totalH2D = 0, totalKernel = 0, totalD2H = 0;

    // Warmup
    for (int i = 0; i < 3; i++) {
        cudaMemcpy(d_x, h_x, matrix.numCols * sizeof(float), cudaMemcpyHostToDevice);
        spmv_full<512, 8><<<gridSize, blockSize>>>(matrix.numRows, matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values, d_x, d_y);
        cudaMemcpy(h_y, d_y, matrix.numRows * sizeof(float), cudaMemcpyDeviceToHost);
    }
    cudaDeviceSynchronize();

    for (int i = 0; i < iterations; i++) {
        timer.start();
        cudaMemcpy(d_x, h_x, matrix.numCols * sizeof(float), cudaMemcpyHostToDevice);
        timer.stop();
        totalH2D += timer.elapsed_ms();

        timer.start();
        spmv_full<512, 8><<<gridSize, blockSize>>>(matrix.numRows, matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values, d_x, d_y);
        cudaDeviceSynchronize();
        timer.stop();
        totalKernel += timer.elapsed_ms();

        timer.start();
        cudaMemcpy(h_y, d_y, matrix.numRows * sizeof(float), cudaMemcpyDeviceToHost);
        timer.stop();
        totalD2H += timer.elapsed_ms();
    }

    delete[] h_x;
    delete[] h_y;
    cudaFree(d_x);
    cudaFree(d_y);

    return {
        (totalH2D + totalKernel + totalD2H) / iterations,
        totalH2D / iterations,
        totalKernel / iterations,
        totalD2H / iterations
    };
}

// Test with pinned memory
TimingResult testPinnedMemory(const CSRMatrix& matrix, int iterations) {
    // Pinned (page-locked) memory
    float* h_x;
    float* h_y;
    cudaMallocHost(&h_x, matrix.numCols * sizeof(float));
    cudaMallocHost(&h_y, matrix.numRows * sizeof(float));
    for (int i = 0; i < matrix.numCols; i++) h_x[i] = rand() / (float)RAND_MAX;

    float* d_x, *d_y;
    cudaMalloc(&d_x, matrix.numCols * sizeof(float));
    cudaMalloc(&d_y, matrix.numRows * sizeof(float));

    int blockSize = 512;
    int rowsPerWarp = WARP_SIZE / 8;
    int gridSize = (matrix.numRows + (blockSize / WARP_SIZE) * rowsPerWarp - 1) / ((blockSize / WARP_SIZE) * rowsPerWarp);

    GpuTimer timer;
    float totalH2D = 0, totalKernel = 0, totalD2H = 0;

    // Warmup
    for (int i = 0; i < 3; i++) {
        cudaMemcpy(d_x, h_x, matrix.numCols * sizeof(float), cudaMemcpyHostToDevice);
        spmv_full<512, 8><<<gridSize, blockSize>>>(matrix.numRows, matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values, d_x, d_y);
        cudaMemcpy(h_y, d_y, matrix.numRows * sizeof(float), cudaMemcpyDeviceToHost);
    }
    cudaDeviceSynchronize();

    for (int i = 0; i < iterations; i++) {
        timer.start();
        cudaMemcpy(d_x, h_x, matrix.numCols * sizeof(float), cudaMemcpyHostToDevice);
        timer.stop();
        totalH2D += timer.elapsed_ms();

        timer.start();
        spmv_full<512, 8><<<gridSize, blockSize>>>(matrix.numRows, matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values, d_x, d_y);
        cudaDeviceSynchronize();
        timer.stop();
        totalKernel += timer.elapsed_ms();

        timer.start();
        cudaMemcpy(h_y, d_y, matrix.numRows * sizeof(float), cudaMemcpyDeviceToHost);
        timer.stop();
        totalD2H += timer.elapsed_ms();
    }

    cudaFreeHost(h_x);
    cudaFreeHost(h_y);
    cudaFree(d_x);
    cudaFree(d_y);

    return {
        (totalH2D + totalKernel + totalD2H) / iterations,
        totalH2D / iterations,
        totalKernel / iterations,
        totalD2H / iterations
    };
}

// Test with pinned memory + async transfers
TimingResult testPinnedAsync(const CSRMatrix& matrix, int iterations) {
    float* h_x;
    float* h_y;
    cudaMallocHost(&h_x, matrix.numCols * sizeof(float));
    cudaMallocHost(&h_y, matrix.numRows * sizeof(float));
    for (int i = 0; i < matrix.numCols; i++) h_x[i] = rand() / (float)RAND_MAX;

    float* d_x, *d_y;
    cudaMalloc(&d_x, matrix.numCols * sizeof(float));
    cudaMalloc(&d_y, matrix.numRows * sizeof(float));

    int blockSize = 512;
    int rowsPerWarp = WARP_SIZE / 8;
    int gridSize = (matrix.numRows + (blockSize / WARP_SIZE) * rowsPerWarp - 1) / ((blockSize / WARP_SIZE) * rowsPerWarp);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    GpuTimer timer;
    float totalH2D = 0, totalKernel = 0, totalD2H = 0;

    // Warmup
    for (int i = 0; i < 3; i++) {
        cudaMemcpyAsync(d_x, h_x, matrix.numCols * sizeof(float), cudaMemcpyHostToDevice, stream);
        spmv_full<512, 8><<<gridSize, blockSize, 0, stream>>>(matrix.numRows, matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values, d_x, d_y);
        cudaMemcpyAsync(h_y, d_y, matrix.numRows * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }

    for (int i = 0; i < iterations; i++) {
        timer.start();
        cudaMemcpyAsync(d_x, h_x, matrix.numCols * sizeof(float), cudaMemcpyHostToDevice, stream);
        timer.stop();
        totalH2D += timer.elapsed_ms();

        timer.start();
        spmv_full<512, 8><<<gridSize, blockSize, 0, stream>>>(matrix.numRows, matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values, d_x, d_y);
        cudaStreamSynchronize(stream);
        timer.stop();
        totalKernel += timer.elapsed_ms();

        timer.start();
        cudaMemcpyAsync(h_y, d_y, matrix.numRows * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        timer.stop();
        totalD2H += timer.elapsed_ms();
    }

    cudaStreamDestroy(stream);
    cudaFreeHost(h_x);
    cudaFreeHost(h_y);
    cudaFree(d_x);
    cudaFree(d_y);

    return {
        (totalH2D + totalKernel + totalD2H) / iterations,
        totalH2D / iterations,
        totalKernel / iterations,
        totalD2H / iterations
    };
}

// Test with pinned + multi-stream
TimingResult testPinnedMultiStream(const CSRMatrix& matrix, int iterations, int numStreams) {
    float* h_x;
    float* h_y;
    cudaMallocHost(&h_x, matrix.numCols * sizeof(float));
    cudaMallocHost(&h_y, matrix.numRows * sizeof(float));
    for (int i = 0; i < matrix.numCols; i++) h_x[i] = rand() / (float)RAND_MAX;

    float* d_x, *d_y;
    cudaMalloc(&d_x, matrix.numCols * sizeof(float));
    cudaMalloc(&d_y, matrix.numRows * sizeof(float));

    int blockSize = 512;
    int rowsPerWarp = WARP_SIZE / 8;
    int rowsPerStream = (matrix.numRows + numStreams - 1) / numStreams;

    cudaStream_t* streams = new cudaStream_t[numStreams];
    for (int i = 0; i < numStreams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    GpuTimer timer;
    float totalH2D = 0, totalKernel = 0, totalD2H = 0;

    // Warmup
    for (int i = 0; i < 3; i++) {
        cudaMemcpyAsync(d_x, h_x, matrix.numCols * sizeof(float), cudaMemcpyHostToDevice, streams[0]);
        for (int s = 0; s < numStreams; s++) {
            int startRow = s * rowsPerStream;
            int endRow = std::min((s + 1) * rowsPerStream, matrix.numRows);
            if (startRow >= matrix.numRows) break;
            int gridSize = (endRow - startRow + (blockSize / WARP_SIZE) * rowsPerWarp - 1) / ((blockSize / WARP_SIZE) * rowsPerWarp);
            spmv_full<512, 8><<<gridSize, blockSize, 0, streams[s]>>>(endRow - startRow, matrix.d_rowPtr + startRow, matrix.d_colIdx, matrix.d_values, d_x, d_y + startRow);
        }
        cudaMemcpyAsync(h_y, d_y, matrix.numRows * sizeof(float), cudaMemcpyDeviceToHost, streams[0]);
        cudaDeviceSynchronize();
    }

    for (int i = 0; i < iterations; i++) {
        timer.start();
        cudaMemcpyAsync(d_x, h_x, matrix.numCols * sizeof(float), cudaMemcpyHostToDevice, streams[0]);
        timer.stop();
        totalH2D += timer.elapsed_ms();

        timer.start();
        for (int s = 0; s < numStreams; s++) {
            int startRow = s * rowsPerStream;
            int endRow = std::min((s + 1) * rowsPerStream, matrix.numRows);
            if (startRow >= matrix.numRows) break;
            int gridSize = (endRow - startRow + (blockSize / WARP_SIZE) * rowsPerWarp - 1) / ((blockSize / WARP_SIZE) * rowsPerWarp);
            spmv_full<512, 8><<<gridSize, blockSize, 0, streams[s]>>>(endRow - startRow, matrix.d_rowPtr + startRow, matrix.d_colIdx, matrix.d_values, d_x, d_y + startRow);
        }
        cudaDeviceSynchronize();
        timer.stop();
        totalKernel += timer.elapsed_ms();

        timer.start();
        cudaMemcpyAsync(h_y, d_y, matrix.numRows * sizeof(float), cudaMemcpyDeviceToHost, streams[0]);
        cudaStreamSynchronize(streams[0]);
        timer.stop();
        totalD2H += timer.elapsed_ms();
    }

    for (int i = 0; i < numStreams; i++) {
        cudaStreamDestroy(streams[i]);
    }
    delete[] streams;
    cudaFreeHost(h_x);
    cudaFreeHost(h_y);
    cudaFree(d_x);
    cudaFree(d_y);

    return {
        (totalH2D + totalKernel + totalD2H) / iterations,
        totalH2D / iterations,
        totalKernel / iterations,
        totalD2H / iterations
    };
}

int main(int argc, char** argv) {
    printf("=== Pinned Memory Optimization Test ===\n");
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

    // Allocate CSR data on device (one-time)
    cudaMalloc(&matrix.d_rowPtr, (matrix.numRows + 1) * sizeof(int));
    cudaMalloc(&matrix.d_colIdx, matrix.nnz * sizeof(int));
    cudaMalloc(&matrix.d_values, matrix.nnz * sizeof(float));
    cudaMemcpy(matrix.d_rowPtr, matrix.rowPtr, (matrix.numRows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(matrix.d_colIdx, matrix.colIdx, matrix.nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(matrix.d_values, matrix.values, matrix.nnz * sizeof(float), cudaMemcpyHostToDevice);

    printf("%-35s %10s %10s %10s %10s\n", "Configuration", "Total(ms)", "H2D(ms)", "Kernel(ms)", "D2H(ms)");
    printf("%-35s %10s %10s %10s %10s\n", "-----------------------------------", "----------", "----------", "----------", "----------");

    auto printResult = [](const char* name, const TimingResult& r) {
        printf("%-35s %10.3f %10.3f %10.3f %10.3f\n", name, r.totalTime, r.transferH2D, r.kernelTime, r.transferD2H);
    };

    // Test pageable memory
    TimingResult r = testPageableMemory(matrix, iterations);
    printResult("Pageable Memory", r);

    // Test pinned memory
    r = testPinnedMemory(matrix, iterations);
    printResult("Pinned Memory", r);

    // Test pinned + async
    r = testPinnedAsync(matrix, iterations);
    printResult("Pinned + Async", r);

    // Test pinned + multi-stream
    r = testPinnedMultiStream(matrix, iterations, 2);
    printResult("Pinned + 2 Streams", r);

    printf("\n=== Analysis ===\n");
    printf("Pinned memory can improve transfer speeds by:\n");
    printf("1. Avoiding page faults during transfer\n");
    printf("2. Enabling true async transfers\n");
    printf("3. Higher PCIe bandwidth utilization\n");

    delete[] matrix.rowPtr;
    delete[] matrix.colIdx;
    delete[] matrix.values;
    cudaFree(matrix.d_rowPtr);
    cudaFree(matrix.d_colIdx);
    cudaFree(matrix.d_values);

    return 0;
}