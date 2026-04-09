/**
 * @file test_exhaustive_optimizations.cu
 * @brief 穷尽性优化测试 - Mars X201专用
 *
 * 测试内容:
 * 1. 不同blockSize (128, 256, 512)
 * 2. 不同threadsPerRow (2, 4, 8, 16)
 * 3. Loop unrolling
 * 4. 多流并行
 * 5. 内存对齐
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

// ========== Scalar Kernel with Loop Unroll ==========
__global__ void scalar_unroll_kernel(
    int numRows, const int* __restrict__ rowPtr, const int* __restrict__ colIdx,
    const double* __restrict__ values, const double* __restrict__ x, double* __restrict__ y)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numRows) return;

    int rowStart = rowPtr[tid];
    int rowEnd = rowPtr[tid + 1];
    double sum = 0.0;

    // Manual unroll with 4x
    int i = rowStart;
    int len = rowEnd - rowStart;
    int len4 = len - (len % 4);

    for (; i < rowStart + len4; i += 4) {
        sum += values[i] * x[colIdx[i]];
        sum += values[i+1] * x[colIdx[i+1]];
        sum += values[i+2] * x[colIdx[i+2]];
        sum += values[i+3] * x[colIdx[i+3]];
    }
    for (; i < rowEnd; i++) {
        sum += values[i] * x[colIdx[i]];
    }
    y[tid] = sum;
}

// ========== Scalar with #pragma unroll ==========
__global__ void scalar_pragma_unroll_kernel(
    int numRows, const int* __restrict__ rowPtr, const int* __restrict__ colIdx,
    const double* __restrict__ values, const double* __restrict__ x, double* __restrict__ y)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numRows) return;

    int rowStart = rowPtr[tid];
    int rowEnd = rowPtr[tid + 1];
    double sum = 0.0;

    #pragma unroll 4
    for (int i = rowStart; i < rowEnd; i++) {
        sum += values[i] * x[colIdx[i]];
    }
    y[tid] = sum;
}

// ========== Vector Kernel with different threads per row ==========
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

    int rowStart = rowPtr[row];
    int rowEnd = rowPtr[row + 1];
    double sum = 0.0;

    // Each thread processes stride elements
    for (int i = rowStart + (laneId % THREADS_PER_ROW); i < rowEnd; i += THREADS_PER_ROW) {
        sum += values[i] * x[colIdx[i]];
    }

    // Warp reduction
    for (int offset = THREADS_PER_ROW / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (laneId % THREADS_PER_ROW == 0) {
        y[row] = sum;
    }
}

// ========== ILP4 Kernel ==========
__global__ void ilp4_kernel(
    int numRows, const int* __restrict__ rowPtr, const int* __restrict__ colIdx,
    const double* __restrict__ values, const double* __restrict__ x, double* __restrict__ y)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numRows) return;

    int rowStart = rowPtr[tid];
    int rowEnd = rowPtr[tid + 1];
    double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;

    int i = rowStart;
    for (; i + 3 < rowEnd; i += 4) {
        sum0 += values[i] * x[colIdx[i]];
        sum1 += values[i+1] * x[colIdx[i+1]];
        sum2 += values[i+2] * x[colIdx[i+2]];
        sum3 += values[i+3] * x[colIdx[i+3]];
    }

    double sum = sum0 + sum1 + sum2 + sum3;
    for (; i < rowEnd; i++) {
        sum += values[i] * x[colIdx[i]];
    }
    y[tid] = sum;
}

// ========== Memory-aligned load kernel ==========
__global__ void aligned_kernel(
    int numRows, const int* __restrict__ rowPtr, const int* __restrict__ colIdx,
    const double* __restrict__ values, const double* __restrict__ x, double* __restrict__ y)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numRows) return;

    int rowStart = rowPtr[tid];
    int rowEnd = rowPtr[tid + 1];
    double sum = 0.0;

    // Try aligned loads
    int i = rowStart;
    // Align to 128-byte boundary (16 doubles)
    int align = (16 - (i & 15)) & 15;
    for (int j = 0; j < align && i < rowEnd; j++, i++) {
        sum += values[i] * x[colIdx[i]];
    }

    // Vectorized aligned loads
    for (; i + 15 < rowEnd; i += 16) {
        // Process 16 elements at once
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            sum += values[i+j] * x[colIdx[i+j]];
        }
    }

    for (; i < rowEnd; i++) {
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

// ========== Main ==========
int main(int argc, char** argv) {
    const char* mtxFile = argc > 1 ? argv[1] : "real_cases/mtx/p0_A";
    int testIters = 50, warmupIters = 10;

    printf("========================================\n");
    printf("  Exhaustive Optimization Tests\n");
    printf("  Mars X201 FP64\n");
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

    // ========== Test 1: Different Block Sizes ==========
    printf("Test 1: Block Size Comparison (Scalar, PreferEqual)\n");
    int blockSizes[] = {64, 128, 256, 512, 1024};
    for (int bs : blockSizes) {
        int gs = (numRows + bs - 1) / bs;
        cudaFuncSetCacheConfig(scalar_unroll_kernel, cudaFuncCachePreferEqual);
        for (int w = 0; w < warmupIters; w++) {
            scalar_unroll_kernel<<<gs, bs>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
        }
        cudaDeviceSynchronize();
        for (int i = 0; i < testIters; i++) {
            scalar_unroll_kernel<<<gs, bs>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
        }
        cudaDeviceSynchronize();
        printf("  blockSize=%4d completed\n", bs);
    }
    printf("\n");

    // ========== Test 2: Different threadsPerRow ==========
    printf("Test 2: Threads Per Row (Vector Kernel, PreferEqual)\n");
    int threadsPerRow[] = {2, 4, 8, 16, 32};
    int bs = 256;
    for (int tpr : threadsPerRow) {
        int gs = (numRows * tpr + bs - 1) / bs;
        switch(tpr) {
            case 2: cudaFuncSetCacheConfig(vector_kernel<2>, cudaFuncCachePreferEqual); break;
            case 4: cudaFuncSetCacheConfig(vector_kernel<4>, cudaFuncCachePreferEqual); break;
            case 8: cudaFuncSetCacheConfig(vector_kernel<8>, cudaFuncCachePreferEqual); break;
            case 16: cudaFuncSetCacheConfig(vector_kernel<16>, cudaFuncCachePreferEqual); break;
            case 32: cudaFuncSetCacheConfig(vector_kernel<32>, cudaFuncCachePreferEqual); break;
        }
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
        for (int i = 0; i < testIters; i++) {
            switch(tpr) {
                case 2: vector_kernel<2><<<gs, bs>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y); break;
                case 4: vector_kernel<4><<<gs, bs>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y); break;
                case 8: vector_kernel<8><<<gs, bs>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y); break;
                case 16: vector_kernel<16><<<gs, bs>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y); break;
                case 32: vector_kernel<32><<<gs, bs>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y); break;
            }
        }
        cudaDeviceSynchronize();
        printf("  threadsPerRow=%2d completed\n", tpr);
    }
    printf("\n");

    // ========== Test 3: Loop Unroll Variants ==========
    printf("Test 3: Loop Unroll Variants (blockSize=256, PreferEqual)\n");
    int gs = (numRows + 255) / 256;

    // Manual unroll
    cudaFuncSetCacheConfig(scalar_unroll_kernel, cudaFuncCachePreferEqual);
    for (int w = 0; w < warmupIters; w++) scalar_unroll_kernel<<<gs, 256>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    cudaDeviceSynchronize();
    for (int i = 0; i < testIters; i++) scalar_unroll_kernel<<<gs, 256>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    cudaDeviceSynchronize();
    printf("  Manual Unroll 4x completed\n");

    // Pragma unroll
    cudaFuncSetCacheConfig(scalar_pragma_unroll_kernel, cudaFuncCachePreferEqual);
    for (int w = 0; w < warmupIters; w++) scalar_pragma_unroll_kernel<<<gs, 256>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    cudaDeviceSynchronize();
    for (int i = 0; i < testIters; i++) scalar_pragma_unroll_kernel<<<gs, 256>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    cudaDeviceSynchronize();
    printf("  #pragma unroll 4 completed\n");

    // ILP4
    cudaFuncSetCacheConfig(ilp4_kernel, cudaFuncCachePreferEqual);
    for (int w = 0; w < warmupIters; w++) ilp4_kernel<<<gs, 256>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    cudaDeviceSynchronize();
    for (int i = 0; i < testIters; i++) ilp4_kernel<<<gs, 256>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    cudaDeviceSynchronize();
    printf("  ILP4 completed\n");

    printf("\n");

    // ========== Test 4: Aligned Load ==========
    printf("Test 4: Aligned Load (blockSize=256, PreferEqual)\n");
    cudaFuncSetCacheConfig(aligned_kernel, cudaFuncCachePreferEqual);
    for (int w = 0; w < warmupIters; w++) aligned_kernel<<<gs, 256>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    cudaDeviceSynchronize();
    for (int i = 0; i < testIters; i++) aligned_kernel<<<gs, 256>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    cudaDeviceSynchronize();
    printf("  Aligned Load completed\n\n");

    printf("========================================\n");
    printf("Use hcTracer for timing:\n");
    printf("  hcTracer --hctx ./test_exhaustive_optimizations\n");
    printf("========================================\n");

    delete[] h_rowPtr; delete[] h_colIdx; delete[] h_values; delete[] h_x;
    cudaFree(d_rowPtr); cudaFree(d_colIdx); cudaFree(d_values); cudaFree(d_x); cudaFree(d_y);
    return 0;
}