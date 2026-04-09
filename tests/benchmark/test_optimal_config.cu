/**
 * @file test_optimal_config.cu
 * @brief 最优配置测试 - blockSize=128 + 不同cache配置
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

__global__ void scalar_kernel(
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

__global__ void aligned_kernel(
    int numRows, const int* __restrict__ rowPtr, const int* __restrict__ colIdx,
    const double* __restrict__ values, const double* __restrict__ x, double* __restrict__ y)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numRows) return;
    int rowStart = rowPtr[tid], rowEnd = rowPtr[tid + 1];
    double sum = 0.0;
    int i = rowStart;
    int align = (16 - (i & 15)) & 15;
    for (int j = 0; j < align && i < rowEnd; j++, i++) {
        sum += values[i] * x[colIdx[i]];
    }
    #pragma unroll 4
    for (; i + 15 < rowEnd; i += 16) {
        for (int j = 0; j < 16; j++) sum += values[i+j] * x[colIdx[i+j]];
    }
    for (; i < rowEnd; i++) sum += values[i] * x[colIdx[i]];
    y[tid] = sum;
}

template<int THREADS_PER_ROW>
__global__ void vector_kernel(
    int numRows, const int* __restrict__ rowPtr, const int* __restrict__ colIdx,
    const double* __restrict__ values, const double* __restrict__ x, double* __restrict__ y)
{
    const int WARP_SIZE = 64;
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
    if (laneId % THREADS_PER_ROW == 0) y[row] = sum;
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

    printf("========================================\n");
    printf("  Optimal Config Tests\n");
    printf("  blockSize=128, FP64\n");
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

    int bs = 128;
    int gs = (numRows + bs - 1) / bs;

    // Test: Scalar + PreferEqual
    printf("Test 1: Scalar + PreferEqual (blockSize=128)\n");
    cudaFuncSetCacheConfig(scalar_kernel, cudaFuncCachePreferEqual);
    for (int w = 0; w < warmupIters; w++) scalar_kernel<<<gs, bs>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    cudaDeviceSynchronize();
    for (int i = 0; i < testIters; i++) scalar_kernel<<<gs, bs>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    cudaDeviceSynchronize();
    printf("  Completed\n\n");

    // Test: Scalar + PreferShared
    printf("Test 2: Scalar + PreferShared (blockSize=128)\n");
    cudaFuncSetCacheConfig(scalar_kernel, cudaFuncCachePreferShared);
    for (int w = 0; w < warmupIters; w++) scalar_kernel<<<gs, bs>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    cudaDeviceSynchronize();
    for (int i = 0; i < testIters; i++) scalar_kernel<<<gs, bs>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    cudaDeviceSynchronize();
    printf("  Completed\n\n");

    // Test: Aligned + PreferEqual
    printf("Test 3: Aligned + PreferEqual (blockSize=128)\n");
    cudaFuncSetCacheConfig(aligned_kernel, cudaFuncCachePreferEqual);
    for (int w = 0; w < warmupIters; w++) aligned_kernel<<<gs, bs>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    cudaDeviceSynchronize();
    for (int i = 0; i < testIters; i++) aligned_kernel<<<gs, bs>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    cudaDeviceSynchronize();
    printf("  Completed\n\n");

    // Test: Aligned + PreferShared
    printf("Test 4: Aligned + PreferShared (blockSize=128)\n");
    cudaFuncSetCacheConfig(aligned_kernel, cudaFuncCachePreferShared);
    for (int w = 0; w < warmupIters; w++) aligned_kernel<<<gs, bs>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    cudaDeviceSynchronize();
    for (int i = 0; i < testIters; i++) aligned_kernel<<<gs, bs>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    cudaDeviceSynchronize();
    printf("  Completed\n\n");

    // Test: Vector 32t/row + PreferEqual
    printf("Test 5: Vector 32t/row + PreferEqual (blockSize=128)\n");
    int gs_v = (numRows * 32 + bs - 1) / bs;
    cudaFuncSetCacheConfig(vector_kernel<32>, cudaFuncCachePreferEqual);
    for (int w = 0; w < warmupIters; w++) vector_kernel<32><<<gs_v, bs>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    cudaDeviceSynchronize();
    for (int i = 0; i < testIters; i++) vector_kernel<32><<<gs_v, bs>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    cudaDeviceSynchronize();
    printf("  Completed\n\n");

    // Test: Vector 32t/row + PreferShared
    printf("Test 6: Vector 32t/row + PreferShared (blockSize=128)\n");
    cudaFuncSetCacheConfig(vector_kernel<32>, cudaFuncCachePreferShared);
    for (int w = 0; w < warmupIters; w++) vector_kernel<32><<<gs_v, bs>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    cudaDeviceSynchronize();
    for (int i = 0; i < testIters; i++) vector_kernel<32><<<gs_v, bs>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    cudaDeviceSynchronize();
    printf("  Completed\n\n");

    // Additional: blockSize=256 for comparison
    printf("Test 7: Scalar + PreferEqual (blockSize=256)\n");
    gs = (numRows + 255) / 256;
    cudaFuncSetCacheConfig(scalar_kernel, cudaFuncCachePreferEqual);
    for (int w = 0; w < warmupIters; w++) scalar_kernel<<<gs, 256>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    cudaDeviceSynchronize();
    for (int i = 0; i < testIters; i++) scalar_kernel<<<gs, 256>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    cudaDeviceSynchronize();
    printf("  Completed\n\n");

    printf("========================================\n");
    printf("Use hcTracer for timing:\n");
    printf("  hcTracer --hctx ./test_optimal_config\n");
    printf("========================================\n");

    delete[] h_rowPtr; delete[] h_colIdx; delete[] h_values; delete[] h_x;
    cudaFree(d_rowPtr); cudaFree(d_colIdx); cudaFree(d_values); cudaFree(d_x); cudaFree(d_y);
    return 0;
}