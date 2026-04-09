/**
 * @file test_additional_optimizations.cu
 * @brief 额外优化技术测试 (不含Texture Memory - API不兼容)
 *
 * 测试:
 * 1. Vectorized Load - 合并内存访问
 * 2. Grid-Stride Loop - 线程利用率优化
 * 3. Warp-Centric - Warp级优化
 * 4. L1/L2 Cache配置对比
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

// ========== 1. Vectorized Load Kernel ==========
__global__ void vectorized_spmv_kernel(
    int numRows,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const double* __restrict__ values,
    const double* __restrict__ x,
    double* __restrict__ y)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numRows) return;

    int rowStart = rowPtr[tid];
    int rowEnd = rowPtr[tid + 1];
    double sum = 0.0;

    // 尝试向量化加载
    int i = rowStart;

    // Process pairs for potential vectorization
    for (; i + 1 < rowEnd; i += 2) {
        // Load pairs
        double2 vals = *reinterpret_cast<const double2*>(&values[i]);
        int2 cols = *reinterpret_cast<const int2*>(&colIdx[i]);

        sum += vals.x * x[cols.x];
        sum += vals.y * x[cols.y];
    }

    // Handle remainder
    for (; i < rowEnd; i++) {
        sum += values[i] * x[colIdx[i]];
    }
    y[tid] = sum;
}

// ========== 2. Grid-Stride Loop Kernel ==========
__global__ void grid_stride_spmv_kernel(
    int numRows,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const double* __restrict__ values,
    const double* __restrict__ x,
    double* __restrict__ y)
{
    for (int row = blockIdx.x * blockDim.x + threadIdx.x;
         row < numRows;
         row += blockDim.x * gridDim.x) {

        int rowStart = rowPtr[row];
        int rowEnd = rowPtr[row + 1];
        double sum = 0.0;

        for (int i = rowStart; i < rowEnd; i++) {
            sum += values[i] * x[colIdx[i]];
        }
        y[row] = sum;
    }
}

// ========== 3. Warp-Centric Kernel ==========
template<int WARP_SIZE>
__global__ void warp_centric_spmv_kernel(
    int numRows,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const double* __restrict__ values,
    const double* __restrict__ x,
    double* __restrict__ y,
    int threadsPerRow)
{
    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int laneId = threadIdx.x % WARP_SIZE;

    int row = warpId * (WARP_SIZE / threadsPerRow) + laneId / threadsPerRow;

    if (row >= numRows) return;

    int rowStart = rowPtr[row];
    int rowEnd = rowPtr[row + 1];
    double sum = 0.0;

    // Each thread processes its portion
    for (int i = rowStart + (laneId % threadsPerRow); i < rowEnd; i += threadsPerRow) {
        sum += values[i] * x[colIdx[i]];
    }

    // Warp reduction
    for (int offset = threadsPerRow / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (laneId % threadsPerRow == 0) {
        y[row] = sum;
    }
}

// ========== 4. ILP Double Accumulator Kernel ==========
__global__ void ilp2_spmv_kernel(
    int numRows,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const double* __restrict__ values,
    const double* __restrict__ x,
    double* __restrict__ y)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numRows) return;

    int rowStart = rowPtr[tid];
    int rowEnd = rowPtr[tid + 1];
    double sum0 = 0.0, sum1 = 0.0;

    int i = rowStart;
    for (; i + 1 < rowEnd; i += 2) {
        sum0 += values[i] * x[colIdx[i]];
        sum1 += values[i+1] * x[colIdx[i+1]];
    }

    double sum = sum0 + sum1;
    for (; i < rowEnd; i++) {
        sum += values[i] * x[colIdx[i]];
    }
    y[tid] = sum;
}

// ========== 5. __ldg Prefetch Kernel ==========
__global__ void ldg_spmv_kernel(
    int numRows,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const double* __restrict__ values,
    const double* __restrict__ x,
    double* __restrict__ y)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numRows) return;

    int rowStart = rowPtr[tid];
    int rowEnd = rowPtr[tid + 1];
    double sum = 0.0;

    for (int i = rowStart; i < rowEnd; i++) {
        sum += values[i] * __ldg(&x[colIdx[i]]);
    }
    y[tid] = sum;
}

// ========== 6. Scalar Baseline Kernel ==========
__global__ void scalar_spmv_kernel(
    int numRows,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const double* __restrict__ values,
    const double* __restrict__ x,
    double* __restrict__ y)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numRows) return;

    int rowStart = rowPtr[tid];
    int rowEnd = rowPtr[tid + 1];
    double sum = 0.0;

    for (int i = rowStart; i < rowEnd; i++) {
        sum += values[i] * x[colIdx[i]];
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
    do { fgets(line, 1024, f); } while (line[0] == '%');

    int m, n, nnzFile;
    sscanf(line, "%d %d %d", &m, &n, &nnzFile);
    *numRows = m; *numCols = n; *nnz = nnzFile;

    int* cooRow = new int[nnzFile];
    int* cooCol = new int[nnzFile];
    double* cooVal = new double[nnzFile];

    for (int i = 0; i < nnzFile; i++) {
        int r, c; double v;
        fscanf(f, "%d %d %lf", &r, &c, &v);
        cooRow[i] = r - 1;
        cooCol[i] = c - 1;
        cooVal[i] = v;
    }
    fclose(f);

    *rowPtr = new int[m + 1];
    memset(*rowPtr, 0, (m + 1) * sizeof(int));
    for (int i = 0; i < nnzFile; i++) (*rowPtr)[cooRow[i] + 1]++;
    for (int i = 0; i < m; i++) (*rowPtr)[i + 1] += (*rowPtr)[i];

    *colIdx = new int[nnzFile];
    *values = new double[nnzFile];

    int* rowStart = new int[m];
    memcpy(rowStart, *rowPtr, m * sizeof(int));

    for (int i = 0; i < nnzFile; i++) {
        int row = cooRow[i];
        int pos = rowStart[row]++;
        (*colIdx)[pos] = cooCol[i];
        (*values)[pos] = cooVal[i];
    }

    delete[] cooRow; delete[] cooCol; delete[] cooVal; delete[] rowStart;
    return true;
}

// ========== Main Test ==========
int main(int argc, char** argv) {
    const char* mtxFile = argc > 1 ? argv[1] : "real_cases/mtx/p0_A";
    int testIters = 50;

    printf("========================================\n");
    printf("  Additional Optimization Techniques\n");
    printf("  (FP64, no Texture Memory)\n");
    printf("========================================\n\n");

    int numRows, numCols, nnz;
    int *h_rowPtr, *h_colIdx;
    double *h_values;

    if (!readMTX(mtxFile, &numRows, &numCols, &nnz, &h_rowPtr, &h_colIdx, &h_values)) {
        printf("Cannot read matrix: %s\n", mtxFile);
        return 1;
    }

    printf("Matrix: %d rows, %d NNZ, avgNnz=%.2f\n\n", numRows, nnz, (double)nnz/numRows);

    // Allocate GPU memory
    int *d_rowPtr, *d_colIdx;
    double *d_values, *d_x, *d_y;
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

    // Test 1: Scalar Baseline
    printf("Test 1: Scalar Baseline\n");
    cudaFuncSetCacheConfig(scalar_spmv_kernel, cudaFuncCachePreferL1);

    // Warmup
    for (int w = 0; w < 10; w++) {
        scalar_spmv_kernel<<<gridSize, blockSize>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    }
    cudaDeviceSynchronize();

    // Test
    for (int i = 0; i < testIters; i++) {
        scalar_spmv_kernel<<<gridSize, blockSize>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    }
    cudaDeviceSynchronize();
    printf("  Completed\n\n");

    // Test 2: Vectorized Load
    printf("Test 2: Vectorized Load\n");
    cudaFuncSetCacheConfig(vectorized_spmv_kernel, cudaFuncCachePreferL1);

    for (int w = 0; w < 10; w++) {
        vectorized_spmv_kernel<<<gridSize, blockSize>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    }
    cudaDeviceSynchronize();

    for (int i = 0; i < testIters; i++) {
        vectorized_spmv_kernel<<<gridSize, blockSize>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    }
    cudaDeviceSynchronize();
    printf("  Completed\n\n");

    // Test 3: Grid-Stride Loop
    printf("Test 3: Grid-Stride Loop\n");
    cudaFuncSetCacheConfig(grid_stride_spmv_kernel, cudaFuncCachePreferL1);

    for (int w = 0; w < 10; w++) {
        grid_stride_spmv_kernel<<<gridSize, blockSize>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    }
    cudaDeviceSynchronize();

    for (int i = 0; i < testIters; i++) {
        grid_stride_spmv_kernel<<<gridSize, blockSize>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    }
    cudaDeviceSynchronize();
    printf("  Completed\n\n");

    // Test 4: ILP Double Accumulator
    printf("Test 4: ILP Double Accumulator\n");
    cudaFuncSetCacheConfig(ilp2_spmv_kernel, cudaFuncCachePreferL1);

    for (int w = 0; w < 10; w++) {
        ilp2_spmv_kernel<<<gridSize, blockSize>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    }
    cudaDeviceSynchronize();

    for (int i = 0; i < testIters; i++) {
        ilp2_spmv_kernel<<<gridSize, blockSize>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    }
    cudaDeviceSynchronize();
    printf("  Completed\n\n");

    // Test 5: __ldg Prefetch
    printf("Test 5: __ldg Prefetch\n");
    cudaFuncSetCacheConfig(ldg_spmv_kernel, cudaFuncCachePreferL1);

    for (int w = 0; w < 10; w++) {
        ldg_spmv_kernel<<<gridSize, blockSize>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    }
    cudaDeviceSynchronize();

    for (int i = 0; i < testIters; i++) {
        ldg_spmv_kernel<<<gridSize, blockSize>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    }
    cudaDeviceSynchronize();
    printf("  Completed\n\n");

    // Test 6: Warp-Centric (different threads per row)
    printf("Test 6: Warp-Centric (4t/row)\n");
    cudaFuncSetCacheConfig(warp_centric_spmv_kernel<64>, cudaFuncCachePreferL1);

    for (int w = 0; w < 10; w++) {
        warp_centric_spmv_kernel<64><<<gridSize, blockSize>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y, 4);
    }
    cudaDeviceSynchronize();

    for (int i = 0; i < testIters; i++) {
        warp_centric_spmv_kernel<64><<<gridSize, blockSize>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y, 4);
    }
    cudaDeviceSynchronize();
    printf("  Completed\n\n");

    // Test 7: Cache Config Comparison (L1 vs Shared)
    printf("Test 7: Cache Config - PreferShared\n");
    cudaFuncSetCacheConfig(scalar_spmv_kernel, cudaFuncCachePreferShared);

    for (int w = 0; w < 10; w++) {
        scalar_spmv_kernel<<<gridSize, blockSize>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    }
    cudaDeviceSynchronize();

    for (int i = 0; i < testIters; i++) {
        scalar_spmv_kernel<<<gridSize, blockSize>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    }
    cudaDeviceSynchronize();
    printf("  Completed\n\n");

    // Test 8: Cache Config - PreferEqual
    printf("Test 8: Cache Config - PreferEqual\n");
    cudaFuncSetCacheConfig(scalar_spmv_kernel, cudaFuncCachePreferEqual);

    for (int w = 0; w < 10; w++) {
        scalar_spmv_kernel<<<gridSize, blockSize>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    }
    cudaDeviceSynchronize();

    for (int i = 0; i < testIters; i++) {
        scalar_spmv_kernel<<<gridSize, blockSize>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    }
    cudaDeviceSynchronize();
    printf("  Completed\n\n");

    printf("========================================\n");
    printf("Use hcTracer for actual timing:\n");
    printf("  hcTracer --hctx ./test_additional_optimizations\n");
    printf("========================================\n");

    // Cleanup
    delete[] h_rowPtr; delete[] h_colIdx; delete[] h_values; delete[] h_x;
    cudaFree(d_rowPtr); cudaFree(d_colIdx); cudaFree(d_values); cudaFree(d_x); cudaFree(d_y);

    return 0;
}