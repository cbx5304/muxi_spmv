/**
 * @file test_vector_vs_scalar_timed.cu
 * @brief 验证Vector kernel的正确性并与Simple Scalar对比性能 (带精确计时)
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

// Vector 4t/row Kernel
__global__ void vector_4t_kernel(
    int numRows, const int* __restrict__ rowPtr, const int* __restrict__ colIdx,
    const double* __restrict__ values, const double* __restrict__ x, double* __restrict__ y)
{
#ifdef __CUDA_ARCH__
    const int WARP_SIZE = (sizeof(int) == 4 && __CUDA_ARCH__ >= 700) ? 32 : 64;  // Auto-detect
#else
    const int WARP_SIZE = 32;  // Default for NVIDIA
#endif
    const int THREADS_PER_ROW = 4;

    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int laneId = threadIdx.x % WARP_SIZE;
    int row = warpId * (WARP_SIZE / THREADS_PER_ROW) + laneId / THREADS_PER_ROW;

    if (row >= numRows) return;

    int rowStart = rowPtr[row], rowEnd = rowPtr[row + 1];
    double sum = 0.0;

    for (int i = rowStart + (laneId % THREADS_PER_ROW); i < rowEnd; i += THREADS_PER_ROW) {
        sum += values[i] * x[colIdx[i]];
    }

    for (int offset = THREADS_PER_ROW / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (laneId % THREADS_PER_ROW == 0) {
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
    int warmupIters = 10, testIters = 50;

    printf("========================================\n");
    printf("  Vector vs Scalar Performance Test\n");
    printf("  FP64, blockSize=128, PreferShared\n");
    printf("========================================\n\n");

    int numRows, numCols, nnz;
    int *h_rowPtr, *h_colIdx; double *h_values;
    if (!readMTX(mtxFile, &numRows, &numCols, &nnz, &h_rowPtr, &h_colIdx, &h_values)) {
        printf("Cannot read matrix: %s\n", mtxFile);
        return 1;
    }
    printf("Matrix: %d rows, %d NNZ, avgNnz=%.2f\n\n", numRows, nnz, (double)nnz/numRows);

    // GPU memory
    int *d_rowPtr, *d_colIdx; double *d_values, *d_x, *d_y_scalar, *d_y_vector;
    cudaMalloc(&d_rowPtr, (numRows + 1) * sizeof(int));
    cudaMalloc(&d_colIdx, nnz * sizeof(int));
    cudaMalloc(&d_values, nnz * sizeof(double));
    cudaMalloc(&d_x, numCols * sizeof(double));
    cudaMalloc(&d_y_scalar, numRows * sizeof(double));
    cudaMalloc(&d_y_vector, numRows * sizeof(double));

    double* h_x = new double[numCols];
    for (int i = 0; i < numCols; i++) h_x[i] = 1.0;
    cudaMemcpy(d_rowPtr, h_rowPtr, (numRows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdx, h_colIdx, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values, nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, numCols * sizeof(double), cudaMemcpyHostToDevice);

    cudaFuncSetCacheConfig(simple_scalar_kernel, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(vector_4t_kernel, cudaFuncCachePreferShared);

    int bs = 128;
    int gs_scalar = (numRows + bs - 1) / bs;
    int gs_vector = (numRows * 4 + bs - 1) / bs;

    // Create events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    for (int i = 0; i < warmupIters; i++) {
        simple_scalar_kernel<<<gs_scalar, bs>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y_scalar);
        vector_4t_kernel<<<gs_vector, bs>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y_vector);
    }
    cudaDeviceSynchronize();

    // Correctness check
    printf("Correctness verification:\n");
    simple_scalar_kernel<<<gs_scalar, bs>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y_scalar);
    vector_4t_kernel<<<gs_vector, bs>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y_vector);
    cudaDeviceSynchronize();

    double* h_y_scalar = new double[numRows];
    double* h_y_vector = new double[numRows];
    cudaMemcpy(h_y_scalar, d_y_scalar, numRows * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y_vector, d_y_vector, numRows * sizeof(double), cudaMemcpyDeviceToHost);

    double maxDiff = 0.0;
    for (int i = 0; i < numRows; i++) {
        double diff = fabs(h_y_scalar[i] - h_y_vector[i]);
        if (diff > maxDiff) maxDiff = diff;
    }
    printf("  Max difference: %.2e %s\n", maxDiff, maxDiff < 1e-10 ? "(PASSED)" : "(FAILED)");

    // Performance: Simple Scalar
    printf("\nPerformance test (50 iterations):\n");
    cudaEventRecord(start);
    for (int i = 0; i < testIters; i++) {
        simple_scalar_kernel<<<gs_scalar, bs>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y_scalar);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float scalar_ms;
    cudaEventElapsedTime(&scalar_ms, start, stop);
    float scalar_avg = scalar_ms * 1000 / testIters;  // microseconds

    // Performance: Vector 4t/row
    cudaEventRecord(start);
    for (int i = 0; i < testIters; i++) {
        vector_4t_kernel<<<gs_vector, bs>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y_vector);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float vector_ms;
    cudaEventElapsedTime(&vector_ms, start, stop);
    float vector_avg = vector_ms * 1000 / testIters;  // microseconds

    // Results
    printf("\n");
    printf("%-20s %12s %12s %12s\n", "Kernel", "Avg (us)", "BW (GB/s)", "Utilization");
    printf("------------------------------------------------------------\n");

    double nnz_d = nnz;
    double bytes_moved = nnz_d * 20;  // approximate

    // Get device name and peak bandwidth
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    // Use approximate peak bandwidth for RTX 4090
    double peak_bw = 1008.0;  // RTX 4090 has ~1008 GB/s

    // For other devices, estimate
    if (strstr(prop.name, "Mars") != NULL) {
        peak_bw = 1843.0;  // Mars X201
    }

    printf("\nDevice: %s\n", prop.name);
    printf("Peak Bandwidth: %.0f GB/s\n\n", peak_bw);

    double scalar_bw = bytes_moved / (scalar_avg * 1e-6) / 1e9;
    double vector_bw = bytes_moved / (vector_avg * 1e-6) / 1e9;

    printf("%-20s %12.2f %12.1f %11.1f%%\n", "Simple Scalar", scalar_avg, scalar_bw, scalar_bw/peak_bw*100);
    printf("%-20s %12.2f %12.1f %11.1f%%\n", "Vector 4t/row", vector_avg, vector_bw, vector_bw/peak_bw*100);

    printf("\nSpeedup: %.2fx\n", scalar_avg / vector_avg);
    printf("Utilization improvement: %.1f%% -> %.1f%%\n", scalar_bw/peak_bw*100, vector_bw/peak_bw*100);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    delete[] h_rowPtr; delete[] h_colIdx; delete[] h_values;
    delete[] h_x; delete[] h_y_scalar; delete[] h_y_vector;
    cudaFree(d_rowPtr); cudaFree(d_colIdx); cudaFree(d_values);
    cudaFree(d_x); cudaFree(d_y_scalar); cudaFree(d_y_vector);

    return 0;
}