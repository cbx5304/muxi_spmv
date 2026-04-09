/**
 * @file test_direct_kernel_comparison.cu
 * @brief Direct kernel comparison under identical conditions
 *
 * Compares:
 * 1. Simple Scalar Kernel
 * 2. Scalar with Manual Unroll
 * 3. ILP4 Kernel
 * 4. Vector Kernel (various threads per row)
 *
 * All with same matrix, same blockSize=128, PreferShared cache
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// Scalar with Manual Unroll
__global__ void scalar_unroll_kernel(
    int numRows, const int* __restrict__ rowPtr, const int* __restrict__ colIdx,
    const double* __restrict__ values, const double* __restrict__ x, double* __restrict__ y)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numRows) return;
    int rowStart = rowPtr[tid], rowEnd = rowPtr[tid + 1];
    double sum = 0.0;
    int i = rowStart;
    int len4 = (rowEnd - rowStart) - ((rowEnd - rowStart) % 4);
    for (; i < rowStart + len4; i += 4) {
        sum += values[i] * x[colIdx[i]];
        sum += values[i+1] * x[colIdx[i+1]];
        sum += values[i+2] * x[colIdx[i+2]];
        sum += values[i+3] * x[colIdx[i+3]];
    }
    for (; i < rowEnd; i++) sum += values[i] * x[colIdx[i]];
    y[tid] = sum;
}

// ILP4 Kernel
__global__ void ilp4_kernel(
    int numRows, const int* __restrict__ rowPtr, const int* __restrict__ colIdx,
    const double* __restrict__ values, const double* __restrict__ x, double* __restrict__ y)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numRows) return;
    int rowStart = rowPtr[tid], rowEnd = rowPtr[tid + 1];
    double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
    int i = rowStart;
    for (; i + 3 < rowEnd; i += 4) {
        sum0 += values[i] * x[colIdx[i]];
        sum1 += values[i+1] * x[colIdx[i+1]];
        sum2 += values[i+2] * x[colIdx[i+2]];
        sum3 += values[i+3] * x[colIdx[i+3]];
    }
    double sum = sum0 + sum1 + sum2 + sum3;
    for (; i < rowEnd; i++) sum += values[i] * x[colIdx[i]];
    y[tid] = sum;
}

template<int TPR>
__global__ void vector_kernel(
    int numRows, const int* __restrict__ rowPtr, const int* __restrict__ colIdx,
    const double* __restrict__ values, const double* __restrict__ x, double* __restrict__ y)
{
    const int WARP_SIZE = 64;
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
    if (laneId % TPR == 0) y[row] = sum;
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
    printf("  Direct Kernel Comparison\n");
    printf("  FP64, blockSize=128, PreferShared\n");
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
    int gs_v = (numRows * 4 + bs - 1) / bs;  // For 4 threads per row

    // Set cache config for all kernels
    cudaFuncSetCacheConfig(simple_scalar_kernel, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(scalar_unroll_kernel, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(ilp4_kernel, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(vector_kernel<4>, cudaFuncCachePreferShared);

    // Test 1: Simple Scalar
    printf("Test 1: Simple Scalar Kernel\n");
    for (int w = 0; w < warmupIters; w++) simple_scalar_kernel<<<gs, bs>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    cudaDeviceSynchronize();
    for (int i = 0; i < testIters; i++) simple_scalar_kernel<<<gs, bs>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    cudaDeviceSynchronize();
    printf("  Completed\n\n");

    // Test 2: Scalar with Unroll
    printf("Test 2: Scalar with Manual Unroll\n");
    for (int w = 0; w < warmupIters; w++) scalar_unroll_kernel<<<gs, bs>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    cudaDeviceSynchronize();
    for (int i = 0; i < testIters; i++) scalar_unroll_kernel<<<gs, bs>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    cudaDeviceSynchronize();
    printf("  Completed\n\n");

    // Test 3: ILP4
    printf("Test 3: ILP4 Kernel (4 accumulators)\n");
    for (int w = 0; w < warmupIters; w++) ilp4_kernel<<<gs, bs>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    cudaDeviceSynchronize();
    for (int i = 0; i < testIters; i++) ilp4_kernel<<<gs, bs>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    cudaDeviceSynchronize();
    printf("  Completed\n\n");

    // Test 4: Vector 4t/row
    printf("Test 4: Vector Kernel (4 threads/row)\n");
    for (int w = 0; w < warmupIters; w++) vector_kernel<4><<<gs_v, bs>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    cudaDeviceSynchronize();
    for (int i = 0; i < testIters; i++) vector_kernel<4><<<gs_v, bs>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    cudaDeviceSynchronize();
    printf("  Completed\n\n");

    printf("========================================\n");
    printf("Use hcTracer for timing:\n");
    printf("  hcTracer --hctx ./test_direct_comparison\n");
    printf("========================================\n");

    delete[] h_rowPtr; delete[] h_colIdx; delete[] h_values; delete[] h_x;
    cudaFree(d_rowPtr); cudaFree(d_colIdx); cudaFree(d_values); cudaFree(d_x); cudaFree(d_y);
    return 0;
}