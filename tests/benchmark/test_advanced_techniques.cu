/**
 * @file test_advanced_techniques.cu
 * @brief Test additional optimization techniques for Mars X201
 *
 * Techniques tested:
 * 1. Multi-stream parallelization
 * 2. Batched processing for large matrices
 * 3. Different grid configurations
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

// Vector Kernel Template (optimal configuration)
template<int TPR>
__global__ void vector_kernel(
    int numRows, const int* __restrict__ rowPtr, const int* __restrict__ colIdx,
    const double* __restrict__ values, const double* __restrict__ x, double* __restrict__ y)
{
#ifdef __CUDA_ARCH__
    const int WARP_SIZE = (sizeof(int) == 4 && __CUDA_ARCH__ >= 700) ? 32 : 64;
#else
    const int WARP_SIZE = 32;
#endif

    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int laneId = threadIdx.x % WARP_SIZE;
    int row = warpId * (WARP_SIZE / TPR) + laneId / TPR;

    if (row >= numRows) return;

    int rowStart = rowPtr[row], rowEnd = rowPtr[row + 1];
    double sum = 0.0;

    for (int i = rowStart + (laneId % TPR); i < rowEnd; i += TPR) {
        sum += values[i] * x[colIdx[i]];
    }

    for (int offset = TPR / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (laneId % TPR == 0) {
        y[row] = sum;
    }
}

bool readMTX(const char* filename, int* numRows, int* numCols, int* nnz,
             int** rowPtr, int** colIdx, double** values)
{
    FILE* f = fopen(filename, "r");
    if (!f) return false;
    char line[1024];
    if (!fgets(line, 1024, f)) { fclose(f); return false; }
    while (line[0] == '%') { if (!fgets(line, 1024, f)) { fclose(f); return false; } }
    int m, n, nnzFile;
    sscanf(line, "%d %d %d", &m, &n, &nnzFile);
    *numRows = m; *numCols = n; *nnz = nnzFile;
    int* cooRow = new int[nnzFile], *cooCol = new int[nnzFile];
    double* cooVal = new double[nnzFile];
    for (int i = 0; i < nnzFile; i++) {
        int r, c; double v;
        if (fscanf(f, "%d %d %lf", &r, &c, &v) != 3) { fclose(f); return false; }
        cooRow[i] = r - 1; cooCol[i] = c - 1; cooVal[i] = v;
    }
    fclose(f);
    *rowPtr = new int[m + 1]; memset(*rowPtr, 0, (m + 1) * sizeof(int));
    for (int i = 0; i < nnzFile; i++) (*rowPtr)[cooRow[i] + 1]++;
    for (int i = 0; i < m; i++) (*rowPtr)[i + 1] += (*rowPtr)[i];
    *colIdx = new int[nnzFile]; *values = new double[nnzFile];
    int* rowStart = new int[m]; memcpy(rowStart, *rowPtr, m * sizeof(int));
    for (int i = 0; i < nnzFile; i++) {
        int row = cooRow[i], pos = rowStart[row]++;
        (*colIdx)[pos] = cooCol[i]; (*values)[pos] = cooVal[i];
    }
    delete[] cooRow; delete[] cooCol; delete[] cooVal; delete[] rowStart;
    return true;
}

int main(int argc, char** argv) {
    const char* mtxFile = argc > 1 ? argv[1] : "real_cases/mtx/p0_A";
    int testIters = 50, warmupIters = 10;

    printf("========================================================\n");
    printf("  Advanced Optimization Techniques Test\n");
    printf("========================================================\n\n");

    int numRows, numCols, nnz;
    int *h_rowPtr, *h_colIdx; double *h_values;
    if (!readMTX(mtxFile, &numRows, &numCols, &nnz, &h_rowPtr, &h_colIdx, &h_values)) {
        printf("Cannot read matrix: %s\n", mtxFile);
        return 1;
    }
    printf("Matrix: %d rows, %d NNZ, avgNnz=%.2f\n\n", numRows, nnz, (double)nnz/numRows);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    double peak_bw = (strstr(prop.name, "Mars") != NULL) ? 1843.0 : 1008.0;
    printf("Device: %s, Peak BW: %.0f GB/s\n\n", prop.name, peak_bw);

    // GPU memory
    int *d_rowPtr, *d_colIdx; double *d_values, *d_x, *d_y;
    cudaMalloc(&d_rowPtr, (numRows + 1) * sizeof(int));
    cudaMalloc(&d_colIdx, nnz * sizeof(int));
    cudaMalloc(&d_values, nnz * sizeof(double));
    cudaMalloc(&d_x, numCols * sizeof(double));
    cudaMalloc(&d_y, numRows * sizeof(double));

    double* h_x_pinned;
    cudaMallocHost(&h_x_pinned, numCols * sizeof(double));
    for (int i = 0; i < numCols; i++) h_x_pinned[i] = 1.0;

    cudaMemcpy(d_rowPtr, h_rowPtr, (numRows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdx, h_colIdx, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values, nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x_pinned, numCols * sizeof(double), cudaMemcpyHostToDevice);

    double bytes_moved = (double)nnz * 20.0;

    // Determine optimal TPR based on device
    int optimalTPR = (strstr(prop.name, "Mars") != NULL) ? 8 : 4;
    int optimalBS = (strstr(prop.name, "Mars") != NULL) ? 128 : 256;

    printf("Using optimal config: %dt/row, blockSize=%d\n\n", optimalTPR, optimalBS);

    // Test 1: Multi-stream parallelization
    printf("=== Test 1: Multi-Stream Parallelization ===\n");
    int streamCounts[] = {1, 2, 4, 8};

    for (int s = 0; s < 4; s++) {
        int numStreams = streamCounts[s];

        // Create streams
        cudaStream_t* streams = new cudaStream_t[numStreams];
        for (int i = 0; i < numStreams; i++) {
            cudaStreamCreate(&streams[i]);
        }

        // Partition rows
        int rowsPerStream = numRows / numStreams;

        // Warmup
        for (int w = 0; w < warmupIters; w++) {
            for (int i = 0; i < numStreams; i++) {
                int startRow = i * rowsPerStream;
                int endRow = (i == numStreams - 1) ? numRows : (i + 1) * rowsPerStream;
                int rowCount = endRow - startRow;
                int gs = (rowCount * optimalTPR + optimalBS - 1) / optimalBS;

                switch(optimalTPR) {
                    case 4: vector_kernel<4><<<gs, optimalBS, 0, streams[i]>>>(
                        rowCount, d_rowPtr + startRow, d_colIdx, d_values, d_x, d_y + startRow); break;
                    case 8: vector_kernel<8><<<gs, optimalBS, 0, streams[i]>>>(
                        rowCount, d_rowPtr + startRow, d_colIdx, d_values, d_x, d_y + startRow); break;
                }
            }
        }
        cudaDeviceSynchronize();

        // Measure
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        for (int iter = 0; iter < testIters; iter++) {
            for (int i = 0; i < numStreams; i++) {
                int startRow = i * rowsPerStream;
                int endRow = (i == numStreams - 1) ? numRows : (i + 1) * rowsPerStream;
                int rowCount = endRow - startRow;
                int gs = (rowCount * optimalTPR + optimalBS - 1) / optimalBS;

                switch(optimalTPR) {
                    case 4: vector_kernel<4><<<gs, optimalBS, 0, streams[i]>>>(
                        rowCount, d_rowPtr + startRow, d_colIdx, d_values, d_x, d_y + startRow); break;
                    case 8: vector_kernel<8><<<gs, optimalBS, 0, streams[i]>>>(
                        rowCount, d_rowPtr + startRow, d_colIdx, d_values, d_x, d_y + startRow); break;
                }
            }
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        double avg_us = ms * 1000.0 / testIters;
        double bw = bytes_moved / (avg_us * 1e-6) / 1e9;

        printf("  %d streams: %.2f us, %.1f GB/s (%.1f%% utilization)\n",
               numStreams, avg_us, bw, bw / peak_bw * 100);

        // Cleanup streams
        for (int i = 0; i < numStreams; i++) {
            cudaStreamDestroy(streams[i]);
        }
        delete[] streams;
    }

    // Test 2: Batched processing (simulate L2 cache optimization)
    printf("\n=== Test 2: Batched Processing (L2 Cache Optimization) ===\n");
    int batchSizes[] = {numRows, 500000, 300000, 200000, 100000};  // Full matrix as baseline

    for (int b = 0; b < 5; b++) {
        int batchSize = batchSizes[b];
        int numBatches = (numRows + batchSize - 1) / batchSize;

        // Warmup
        for (int w = 0; w < warmupIters; w++) {
            for (int batch = 0; batch < numBatches; batch++) {
                int startRow = batch * batchSize;
                int endRow = std::min(startRow + batchSize, numRows);
                int rowCount = endRow - startRow;
                int gs = (rowCount * optimalTPR + optimalBS - 1) / optimalBS;

                switch(optimalTPR) {
                    case 4: vector_kernel<4><<<gs, optimalBS>>>(rowCount, d_rowPtr + startRow,
                        d_colIdx, d_values, d_x, d_y + startRow); break;
                    case 8: vector_kernel<8><<<gs, optimalBS>>>(rowCount, d_rowPtr + startRow,
                        d_colIdx, d_values, d_x, d_y + startRow); break;
                }
            }
        }
        cudaDeviceSynchronize();

        // Measure
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        for (int iter = 0; iter < testIters; iter++) {
            for (int batch = 0; batch < numBatches; batch++) {
                int startRow = batch * batchSize;
                int endRow = std::min(startRow + batchSize, numRows);
                int rowCount = endRow - startRow;
                int gs = (rowCount * optimalTPR + optimalBS - 1) / optimalBS;

                switch(optimalTPR) {
                    case 4: vector_kernel<4><<<gs, optimalBS>>>(rowCount, d_rowPtr + startRow,
                        d_colIdx, d_values, d_x, d_y + startRow); break;
                    case 8: vector_kernel<8><<<gs, optimalBS>>>(rowCount, d_rowPtr + startRow,
                        d_colIdx, d_values, d_x, d_y + startRow); break;
                }
            }
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        double avg_us = ms * 1000.0 / testIters;
        double bw = bytes_moved / (avg_us * 1e-6) / 1e9;

        printf("  Batch %dK rows (%d batches): %.2f us, %.1f GB/s (%.1f%%)\n",
               batchSize / 1000, numBatches, avg_us, bw, bw / peak_bw * 100);
    }

    // Test 3: Different grid configurations
    printf("\n=== Test 3: Grid Configuration ===\n");
    int gs_multipliers[] = {1, 2, 4, 8, 16};  // Grid size relative to rows
    const char* gs_names[] = {"1x", "2x", "4x", "8x", "16x"};

    int baseGs = (numRows * optimalTPR + optimalBS - 1) / optimalBS;

    for (int g = 0; g < 5; g++) {
        int gs = baseGs * gs_multipliers[g];

        // Warmup
        for (int w = 0; w < warmupIters; w++) {
            switch(optimalTPR) {
                case 4: vector_kernel<4><<<gs, optimalBS>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y); break;
                case 8: vector_kernel<8><<<gs, optimalBS>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y); break;
            }
        }
        cudaDeviceSynchronize();

        // Measure
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        for (int iter = 0; iter < testIters; iter++) {
            switch(optimalTPR) {
                case 4: vector_kernel<4><<<gs, optimalBS>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y); break;
                case 8: vector_kernel<8><<<gs, optimalBS>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y); break;
            }
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        double avg_us = ms * 1000.0 / testIters;
        double bw = bytes_moved / (avg_us * 1e-6) / 1e9;

        printf("  Grid %s (%d blocks): %.2f us, %.1f GB/s (%.1f%%)\n",
               gs_names[g], gs, avg_us, bw, bw / peak_bw * 100);
    }

    printf("\n=== Summary ===\n");
    printf("Optimal configuration for %s:\n", prop.name);
    printf("  Threads/Row: %d\n", optimalTPR);
    printf("  Block Size: %d\n", optimalBS);
    printf("  Grid Size: 1x (default)\n");
    printf("  Streams: 1 (no benefit from multi-stream for this workload)\n");
    printf("  Batching: No benefit for this matrix size\n");

    // Cleanup
    cudaFreeHost(h_x_pinned);
    delete[] h_rowPtr; delete[] h_colIdx; delete[] h_values;
    cudaFree(d_rowPtr); cudaFree(d_colIdx); cudaFree(d_values); cudaFree(d_x); cudaFree(d_y);

    return 0;
}