/**
 * @file test_comprehensive_optimization.cu
 * @brief Comprehensive optimization test exploring all possible optimizations
 *
 * Tests:
 * 1. Different threads per row (2, 4, 8, 16, 32)
 * 2. Different block sizes (64, 128, 256, 512)
 * 3. Different cache configurations (PreferShared, PreferL1, PreferEqual)
 * 4. Pinned Memory for end-to-end performance
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

// Simple Scalar Kernel
__global__ void simple_scalar_kernel(
    int numRows, const int* __restrict__ rowPtr, const int* __restrict__ colIdx,
    const double* __restrict__ values, const double* __restrict__ x, double* __restrict__ y)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numRows) return;
    double sum = 0.0;
    for (int i = rowPtr[tid]; i < rowPtr[tid + 1]; i++) {
        sum += values[i] * x[colIdx[i]];
    }
    y[tid] = sum;
}

// Vector Kernel Template
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

double testKernel(void (*kernel)(), int gs, int bs, int iterations, cudaStream_t stream = 0) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, stream);
    for (int i = 0; i < iterations; i++) {
        kernel<<<gs, bs, 0, stream>>>();
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms * 1000.0 / iterations;  // microseconds
}

int main(int argc, char** argv) {
    const char* mtxFile = argc > 1 ? argv[1] : "real_cases/mtx/p0_A";
    int testIters = 50, warmupIters = 10;

    printf("========================================================\n");
    printf("  Comprehensive Optimization Test\n");
    printf("  FP64, Testing Multiple Configurations\n");
    printf("========================================================\n\n");

    int numRows, numCols, nnz;
    int *h_rowPtr, *h_colIdx; double *h_values;
    if (!readMTX(mtxFile, &numRows, &numCols, &nnz, &h_rowPtr, &h_colIdx, &h_values)) {
        printf("Cannot read matrix: %s\n", mtxFile);
        return 1;
    }
    printf("Matrix: %d rows, %d NNZ, avgNnz=%.2f\n\n", numRows, nnz, (double)nnz/numRows);

    // Get device info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    double peak_bw = 1008.0;  // RTX 4090 default
    if (strstr(prop.name, "Mars") != NULL) {
        peak_bw = 1843.0;  // Mars X201
    }
    printf("Device: %s\n", prop.name);
    printf("Peak Bandwidth: %.0f GB/s\n\n", peak_bw);

    // GPU memory
    int *d_rowPtr, *d_colIdx; double *d_values, *d_x, *d_y;
    cudaMalloc(&d_rowPtr, (numRows + 1) * sizeof(int));
    cudaMalloc(&d_colIdx, nnz * sizeof(int));
    cudaMalloc(&d_values, nnz * sizeof(double));
    cudaMalloc(&d_x, numCols * sizeof(double));
    cudaMalloc(&d_y, numRows * sizeof(double));

    // Pinned memory for x vector
    double* h_x_pinned;
    cudaMallocHost(&h_x_pinned, numCols * sizeof(double));
    for (int i = 0; i < numCols; i++) h_x_pinned[i] = 1.0;

    double* h_x = new double[numCols];
    for (int i = 0; i < numCols; i++) h_x[i] = 1.0;

    cudaMemcpy(d_rowPtr, h_rowPtr, (numRows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdx, h_colIdx, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values, nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x_pinned, numCols * sizeof(double), cudaMemcpyHostToDevice);

    double bytes_moved = (double)nnz * 20.0;  // Approximate bytes

    printf("%-15s %-8s %-15s %-12s %-12s\n", "Kernel", "BS", "Cache", "Time(us)", "BW(GB/s)");
    printf("---------------------------------------------------------------\n");

    // Test 1: Simple Scalar with different block sizes and cache configs
    printf("\n=== Simple Scalar Kernel ===\n");
    int blockSizes[] = {64, 128, 256, 512};
    const char* cacheNames[] = {"PreferShared", "PreferL1", "PreferEqual"};
    cudaFuncCache cacheConfigs[] = {cudaFuncCachePreferShared, cudaFuncCachePreferL1, cudaFuncCachePreferEqual};

    for (int c = 0; c < 3; c++) {
        cudaFuncSetCacheConfig(simple_scalar_kernel, cacheConfigs[c]);
        for (int b = 0; b < 4; b++) {
            int bs = blockSizes[b];
            int gs = (numRows + bs - 1) / bs;

            // Warmup
            for (int w = 0; w < warmupIters; w++) {
                simple_scalar_kernel<<<gs, bs>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
            }
            cudaDeviceSynchronize();

            double avg_us = testKernel((void(*)())simple_scalar_kernel, gs, bs, testIters);
            double bw = bytes_moved / (avg_us * 1e-6) / 1e9;

            printf("%-15s %-8d %-15s %-12.2f %-12.1f\n", "Scalar", bs, cacheNames[c], avg_us, bw);
        }
    }

    // Test 2: Vector Kernel with different TPR, block sizes, and cache configs
    printf("\n=== Vector Kernel ===\n");
    int tprValues[] = {2, 4, 8, 16, 32};

    auto testVector = [&](int tpr, int bs, cudaFuncCache cache, const char* cacheName) {
        int gs = (numRows * tpr + bs - 1) / bs;

        // Warmup
        for (int w = 0; w < warmupIters; w++) {
            switch(tpr) {
                case 2: vector_kernel<2><<<gs, bs>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y); break;
                case 4: vector_kernel<4><<<gs, bs>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y); break;
                case 8: vector_kernel<8><<<gs, bs>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y); break;
                case 16: vector_kernel<16><<<gs, bs>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y); break;
                case 32: vector_kernel<32><<<gs, bs>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y); break;
            }
        }
        cudaDeviceSynchronize();

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        for (int i = 0; i < testIters; i++) {
            switch(tpr) {
                case 2: vector_kernel<2><<<gs, bs>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y); break;
                case 4: vector_kernel<4><<<gs, bs>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y); break;
                case 8: vector_kernel<8><<<gs, bs>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y); break;
                case 16: vector_kernel<16><<<gs, bs>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y); break;
                case 32: vector_kernel<32><<<gs, bs>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y); break;
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

        char kernelName[32];
        sprintf(kernelName, "Vector %dt/row", tpr);
        printf("%-15s %-8d %-15s %-12.2f %-12.1f\n", kernelName, bs, cacheName, avg_us, bw);

        return avg_us;
    };

    double bestTime = 1e10;
    int bestTpr = 4, bestBs = 128, bestCache = 0;

    for (int t = 0; t < 5; t++) {
        int tpr = tprValues[t];
        for (int c = 0; c < 3; c++) {
            for (int b = 0; b < 4; b++) {
                int bs = blockSizes[b];
                switch(tpr) {
                    case 2: cudaFuncSetCacheConfig(vector_kernel<2>, cacheConfigs[c]); break;
                    case 4: cudaFuncSetCacheConfig(vector_kernel<4>, cacheConfigs[c]); break;
                    case 8: cudaFuncSetCacheConfig(vector_kernel<8>, cacheConfigs[c]); break;
                    case 16: cudaFuncSetCacheConfig(vector_kernel<16>, cacheConfigs[c]); break;
                    case 32: cudaFuncSetCacheConfig(vector_kernel<32>, cacheConfigs[c]); break;
                }
                double avg_us = testVector(tpr, bs, cacheConfigs[c], cacheNames[c]);
                if (avg_us < bestTime) {
                    bestTime = avg_us;
                    bestTpr = tpr;
                    bestBs = bs;
                    bestCache = c;
                }
            }
        }
    }

    printf("\n=== Best Configuration ===\n");
    printf("Vector %dt/row, blockSize=%d, %s: %.2f us, %.1f GB/s (%.1f%% utilization)\n",
           bestTpr, bestBs, cacheNames[bestCache], bestTime,
           bytes_moved / (bestTime * 1e-6) / 1e9,
           bytes_moved / (bestTime * 1e-6) / 1e9 / peak_bw * 100);

    // Test 3: End-to-end with Pinned Memory
    printf("\n=== End-to-End Performance (Pinned Memory) ===\n");

    // Warmup
    cudaMemcpy(d_x, h_x_pinned, numCols * sizeof(double), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // E2E test
    cudaEventRecord(start);
    cudaMemcpy(d_x, h_x_pinned, numCols * sizeof(double), cudaMemcpyHostToDevice);
    switch(bestTpr) {
        case 2: vector_kernel<2><<<(numRows * bestTpr + bestBs - 1) / bestBs, bestBs>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y); break;
        case 4: vector_kernel<4><<<(numRows * bestTpr + bestBs - 1) / bestBs, bestBs>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y); break;
        case 8: vector_kernel<8><<<(numRows * bestTpr + bestBs - 1) / bestBs, bestBs>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y); break;
        case 16: vector_kernel<16><<<(numRows * bestTpr + bestBs - 1) / bestBs, bestBs>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y); break;
        case 32: vector_kernel<32><<<(numRows * bestTpr + bestBs - 1) / bestBs, bestBs>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y); break;
    }
    cudaMemcpy(h_x, d_y, numRows * sizeof(double), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float e2e_ms;
    cudaEventElapsedTime(&e2e_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("End-to-end time: %.3f ms\n", e2e_ms);
    printf("End-to-end bandwidth: %.1f GB/s\n", bytes_moved / (e2e_ms * 1e-3) / 1e9);

    // Cleanup
    cudaFreeHost(h_x_pinned);
    delete[] h_rowPtr; delete[] h_colIdx; delete[] h_values; delete[] h_x;
    cudaFree(d_rowPtr); cudaFree(d_colIdx); cudaFree(d_values); cudaFree(d_x); cudaFree(d_y);

    return 0;
}