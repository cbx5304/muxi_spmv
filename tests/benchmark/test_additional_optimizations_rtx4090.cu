/**
 * @file test_additional_optimizations_rtx4090.cu
 * @brief 额外优化技术测试 - RTX 4090专用版本
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

// ========== Kernels ==========

__global__ void scalar_spmv_kernel(
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

__global__ void vectorized_spmv_kernel(
    int numRows, const int* __restrict__ rowPtr, const int* __restrict__ colIdx,
    const double* __restrict__ values, const double* __restrict__ x, double* __restrict__ y)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numRows) return;
    int rowStart = rowPtr[tid], rowEnd = rowPtr[tid + 1];
    double sum = 0.0;
    int i = rowStart;
    for (; i + 1 < rowEnd; i += 2) {
        double2 vals = *reinterpret_cast<const double2*>(&values[i]);
        int2 cols = *reinterpret_cast<const int2*>(&colIdx[i]);
        sum += vals.x * x[cols.x] + vals.y * x[cols.y];
    }
    for (; i < rowEnd; i++) sum += values[i] * x[colIdx[i]];
    y[tid] = sum;
}

__global__ void ilp2_spmv_kernel(
    int numRows, const int* __restrict__ rowPtr, const int* __restrict__ colIdx,
    const double* __restrict__ values, const double* __restrict__ x, double* __restrict__ y)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numRows) return;
    int rowStart = rowPtr[tid], rowEnd = rowPtr[tid + 1];
    double sum0 = 0.0, sum1 = 0.0;
    int i = rowStart;
    for (; i + 1 < rowEnd; i += 2) {
        sum0 += values[i] * x[colIdx[i]];
        sum1 += values[i+1] * x[colIdx[i+1]];
    }
    double sum = sum0 + sum1;
    for (; i < rowEnd; i++) sum += values[i] * x[colIdx[i]];
    y[tid] = sum;
}

__global__ void ldg_spmv_kernel(
    int numRows, const int* __restrict__ rowPtr, const int* __restrict__ colIdx,
    const double* __restrict__ values, const double* __restrict__ x, double* __restrict__ y)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numRows) return;
    double sum = 0.0;
    for (int i = rowPtr[tid]; i < rowPtr[tid + 1]; i++) {
        sum += values[i] * __ldg(&x[colIdx[i]]);
    }
    y[tid] = sum;
}

// ========== MTX Reader ==========
bool readMTX(const char* filename, int* numRows, int* numCols, int* nnz,
             int** rowPtr, int** colIdx, double** values)
{
    FILE* f = fopen(filename, "r");
    if (!f) return false;
    char line[1024];
    if (!fgets(line, 1024, f)) { fclose(f); return false; }
    while (line[0] == '%') {
        if (!fgets(line, 1024, f)) { fclose(f); return false; }
    }
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

    printf("========================================\n");
    printf("  Additional Optimizations (RTX 4090)\n");
    printf("  FP64, cudaEvent Timing\n");
    printf("========================================\n\n");

    int numRows, numCols, nnz;
    int *h_rowPtr, *h_colIdx; double *h_values;
    if (!readMTX(mtxFile, &numRows, &numCols, &nnz, &h_rowPtr, &h_colIdx, &h_values)) {
        printf("Cannot read matrix: %s\n", mtxFile);
        return 1;
    }
    printf("Matrix: %d rows, %d NNZ, avgNnz=%.2f\n\n", numRows, nnz, (double)nnz/numRows);

    int *d_rowPtr, *d_colIdx; double *d_values, *d_x, *d_y;
    cudaMalloc(&d_rowPtr, (numRows + 1) * sizeof(int));
    cudaMalloc(&d_colIdx, nnz * sizeof(int));
    cudaMalloc(&d_values, nnz * sizeof(double));
    cudaMalloc(&d_x, numCols * sizeof(double));
    cudaMalloc(&d_y, numRows * sizeof(double));

    double* h_x = new double[numCols];
    for (int i = 0; i < numCols; i++) h_x[i] = 1.0;
    cudaMemcpy(d_rowPtr, h_rowPtr, (numRows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdx, h_colIdx, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values, nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, numCols * sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (numRows + blockSize - 1) / blockSize;

    struct Result { std::string name; float us; float bw; float util; };
    std::vector<Result> results;

    auto runTest = [&](const char* name, cudaFuncCache cache) {
        cudaFuncSetCacheConfig(scalar_spmv_kernel, cache);
        // Warmup
        for (int w = 0; w < warmupIters; w++) {
            scalar_spmv_kernel<<<gridSize, blockSize>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
        }
        cudaDeviceSynchronize();
        // Timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        cudaEventRecord(start);
        for (int i = 0; i < testIters; i++) {
            scalar_spmv_kernel<<<gridSize, blockSize>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        cudaEventDestroy(start); cudaEventDestroy(stop);
        float us = ms / testIters * 1000;
        float bw = (nnz * 20 + numRows * 8) / (us * 1e-6) / 1e9;
        float util = bw / 1008 * 100;
        results.push_back({name, us, bw, util});
        printf("%-30s: %8.1f us, %7.1f GB/s, %5.1f%%\n", name, us, bw, util);
    };

    printf("Testing Scalar with different cache configs...\n");
    runTest("1. Scalar (PreferL1)", cudaFuncCachePreferL1);
    runTest("2. Scalar (PreferShared)", cudaFuncCachePreferShared);
    runTest("3. Scalar (PreferEqual)", cudaFuncCachePreferEqual);
    runTest("4. Scalar (PreferNone)", cudaFuncCachePreferNone);

    // Test ILP2
    printf("\nTesting ILP2...\n");
    cudaFuncSetCacheConfig(ilp2_spmv_kernel, cudaFuncCachePreferL1);
    for (int w = 0; w < warmupIters; w++) {
        ilp2_spmv_kernel<<<gridSize, blockSize>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    }
    cudaDeviceSynchronize();
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < testIters; i++) {
        ilp2_spmv_kernel<<<gridSize, blockSize>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    float us = ms / testIters * 1000;
    float bw = (nnz * 20 + numRows * 8) / (us * 1e-6) / 1e9;
    float util = bw / 1008 * 100;
    results.push_back({"5. ILP2 (PreferL1)", us, bw, util});
    printf("5. ILP2 (PreferL1)            : %8.1f us, %7.1f GB/s, %5.1f%%\n", us, bw, util);

    // Test Vectorized
    printf("\nTesting Vectorized...\n");
    cudaFuncSetCacheConfig(vectorized_spmv_kernel, cudaFuncCachePreferL1);
    for (int w = 0; w < warmupIters; w++) {
        vectorized_spmv_kernel<<<gridSize, blockSize>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    }
    cudaDeviceSynchronize();
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < testIters; i++) {
        vectorized_spmv_kernel<<<gridSize, blockSize>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    us = ms / testIters * 1000;
    bw = (nnz * 20 + numRows * 8) / (us * 1e-6) / 1e9;
    util = bw / 1008 * 100;
    results.push_back({"6. Vectorized (PreferL1)", us, bw, util});
    printf("6. Vectorized (PreferL1)      : %8.1f us, %7.1f GB/s, %5.1f%%\n", us, bw, util);

    // Test __ldg
    printf("\nTesting __ldg...\n");
    cudaFuncSetCacheConfig(ldg_spmv_kernel, cudaFuncCachePreferL1);
    for (int w = 0; w < warmupIters; w++) {
        ldg_spmv_kernel<<<gridSize, blockSize>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    }
    cudaDeviceSynchronize();
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < testIters; i++) {
        ldg_spmv_kernel<<<gridSize, blockSize>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    us = ms / testIters * 1000;
    bw = (nnz * 20 + numRows * 8) / (us * 1e-6) / 1e9;
    util = bw / 1008 * 100;
    results.push_back({"7. __ldg (PreferL1)", us, bw, util});
    printf("7. __ldg (PreferL1)           : %8.1f us, %7.1f GB/s, %5.1f%%\n", us, bw, util);

    printf("\n========================================\n");
    printf("Sorted by performance:\n");
    printf("========================================\n");
    std::sort(results.begin(), results.end(), [](const Result& a, const Result& b) { return a.us < b.us; });
    float baseline = results[0].us;
    for (const auto& r : results) {
        printf("%-30s: %8.1f us, %7.1f GB/s, %5.1f%%, %.3fx\n", r.name.c_str(), r.us, r.bw, r.util, r.us / baseline);
    }

    delete[] h_rowPtr; delete[] h_colIdx; delete[] h_values; delete[] h_x;
    cudaFree(d_rowPtr); cudaFree(d_colIdx); cudaFree(d_values); cudaFree(d_x); cudaFree(d_y);
    return 0;
}