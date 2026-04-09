/**
 * @file test_multi_stream.cu
 * @brief 多流并行测试
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
    printf("  Multi-Stream Parallelism Test\n");
    printf("  FP64, blockSize=128, PreferShared\n");
    printf("========================================\n\n");

    int numRows, numCols, nnz;
    int *h_rowPtr, *h_colIdx; double *h_values;
    if (!readMTX(mtxFile, &numRows, &numCols, &nnz, &h_rowPtr, &h_colIdx, &h_values)) {
        printf("Cannot read matrix: %s\n", mtxFile);
        return 1;
    }
    printf("Matrix: %d rows, %d NNZ, avgNnz=%.2f\n\n", numRows, nnz, (double)nnz/numRows);

    // Allocate GPU memory
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

    cudaFuncSetCacheConfig(scalar_kernel, cudaFuncCachePreferShared);

    // Test 1: Single stream (baseline)
    printf("Test 1: Single Stream (baseline)\n");
    int bs = 128;
    int gs = (numRows + bs - 1) / bs;
    for (int w = 0; w < warmupIters; w++) {
        scalar_kernel<<<gs, bs>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    }
    cudaDeviceSynchronize();
    for (int i = 0; i < testIters; i++) {
        scalar_kernel<<<gs, bs>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    }
    cudaDeviceSynchronize();
    printf("  Completed\n\n");

    // Test 2: 2 streams
    printf("Test 2: 2 Streams\n");
    cudaStream_t stream0, stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);
    int halfRows = numRows / 2;
    int gs0 = (halfRows + bs - 1) / bs;
    int gs1 = (numRows - halfRows + bs - 1) / bs;

    for (int w = 0; w < warmupIters; w++) {
        scalar_kernel<<<gs0, bs, 0, stream0>>>(halfRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
        scalar_kernel<<<gs1, bs, 0, stream1>>>(numRows - halfRows, d_rowPtr + halfRows, d_colIdx, d_values, d_x, d_y + halfRows);
    }
    cudaDeviceSynchronize();
    for (int i = 0; i < testIters; i++) {
        scalar_kernel<<<gs0, bs, 0, stream0>>>(halfRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
        scalar_kernel<<<gs1, bs, 0, stream1>>>(numRows - halfRows, d_rowPtr + halfRows, d_colIdx, d_values, d_x, d_y + halfRows);
    }
    cudaDeviceSynchronize();
    printf("  Completed\n\n");

    // Test 3: 4 streams
    printf("Test 3: 4 Streams\n");
    cudaStream_t streams[4];
    for (int s = 0; s < 4; s++) cudaStreamCreate(&streams[s]);
    int rowsPerStream = numRows / 4;

    for (int w = 0; w < warmupIters; w++) {
        for (int s = 0; s < 4; s++) {
            int startRow = s * rowsPerStream;
            int numStreamRows = (s == 3) ? (numRows - startRow) : rowsPerStream;
            int gs_s = (numStreamRows + bs - 1) / bs;
            scalar_kernel<<<gs_s, bs, 0, streams[s]>>>(numStreamRows, d_rowPtr + startRow, d_colIdx, d_values, d_x, d_y + startRow);
        }
    }
    cudaDeviceSynchronize();
    for (int i = 0; i < testIters; i++) {
        for (int s = 0; s < 4; s++) {
            int startRow = s * rowsPerStream;
            int numStreamRows = (s == 3) ? (numRows - startRow) : rowsPerStream;
            int gs_s = (numStreamRows + bs - 1) / bs;
            scalar_kernel<<<gs_s, bs, 0, streams[s]>>>(numStreamRows, d_rowPtr + startRow, d_colIdx, d_values, d_x, d_y + startRow);
        }
    }
    cudaDeviceSynchronize();
    printf("  Completed\n\n");

    // Test 4: 8 streams
    printf("Test 4: 8 Streams\n");
    cudaStream_t streams8[8];
    for (int s = 0; s < 8; s++) cudaStreamCreate(&streams8[s]);
    rowsPerStream = numRows / 8;

    for (int w = 0; w < warmupIters; w++) {
        for (int s = 0; s < 8; s++) {
            int startRow = s * rowsPerStream;
            int numStreamRows = (s == 7) ? (numRows - startRow) : rowsPerStream;
            int gs_s = (numStreamRows + bs - 1) / bs;
            scalar_kernel<<<gs_s, bs, 0, streams8[s]>>>(numStreamRows, d_rowPtr + startRow, d_colIdx, d_values, d_x, d_y + startRow);
        }
    }
    cudaDeviceSynchronize();
    for (int i = 0; i < testIters; i++) {
        for (int s = 0; s < 8; s++) {
            int startRow = s * rowsPerStream;
            int numStreamRows = (s == 7) ? (numRows - startRow) : rowsPerStream;
            int gs_s = (numStreamRows + bs - 1) / bs;
            scalar_kernel<<<gs_s, bs, 0, streams8[s]>>>(numStreamRows, d_rowPtr + startRow, d_colIdx, d_values, d_x, d_y + startRow);
        }
    }
    cudaDeviceSynchronize();
    printf("  Completed\n\n");

    printf("========================================\n");
    printf("Use hcTracer for timing:\n");
    printf("  hcTracer --hctx ./test_multi_stream\n");
    printf("========================================\n");

    // Cleanup
    cudaStreamDestroy(stream0); cudaStreamDestroy(stream1);
    for (int s = 0; s < 4; s++) cudaStreamDestroy(streams[s]);
    for (int s = 0; s < 8; s++) cudaStreamDestroy(streams8[s]);
    delete[] h_rowPtr; delete[] h_colIdx; delete[] h_values; delete[] h_x;
    cudaFree(d_rowPtr); cudaFree(d_colIdx); cudaFree(d_values); cudaFree(d_x); cudaFree(d_y);
    return 0;
}