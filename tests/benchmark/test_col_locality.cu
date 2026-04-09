/**
 * @file test_col_locality_analysis.cu
 * @brief 分析矩阵列访问局部性，探索优化空间
 *
 * 核心问题：SpMV性能瓶颈是x[colIdx[i]]的随机访问
 * 分析：
 * 1. 列访问分布（同一列被访问次数）
 * 2. 列访问空间局部性（相邻列访问概率）
 * 3. 潜在优化策略：列重排序
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

// ========== CSR SpMV Kernel ==========
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

// ========== Column Locality Analysis ==========
void analyzeColLocality(int numCols, int nnz, const int* colIdx) {
    printf("\n=== Column Access Locality Analysis ===\n");

    // 1. Count column accesses
    std::vector<int> colAccessCount(numCols, 0);
    for (int i = 0; i < nnz; i++) {
        colAccessCount[colIdx[i]]++;
    }

    // Sort by access count
    std::vector<std::pair<int, int>> sortedCols;  // (count, col)
    for (int c = 0; c < numCols; c++) {
        if (colAccessCount[c] > 0) {
            sortedCols.push_back({colAccessCount[c], c});
        }
    }
    std::sort(sortedCols.begin(), sortedCols.end(), std::greater<std::pair<int,int>>());

    // Statistics
    int maxAccess = sortedCols[0].first;
    int minAccess = sortedCols.back().first;
    double avgAccess = (double)nnz / numCols;

    printf("Column access statistics:\n");
    printf("  Total accesses: %d\n", nnz);
    printf("  Unique columns: %zu / %d\n", sortedCols.size(), numCols);
    printf("  Max accesses per column: %d\n", maxAccess);
    printf("  Min accesses per column: %d\n", minAccess);
    printf("  Avg accesses per column: %.2f\n", avgAccess);

    // Top 10 columns
    printf("\nTop 10 most accessed columns:\n");
    for (int i = 0; i < 10 && i < (int)sortedCols.size(); i++) {
        printf("  Col %d: %d accesses (%.2f%%)\n",
               sortedCols[i].second, sortedCols[i].first,
               100.0 * sortedCols[i].first / nnz);
    }

    // Access distribution histogram
    printf("\nAccess count distribution:\n");
    std::vector<int> histBuckets = {1, 2, 5, 10, 20, 50, 100, 500, 1000};
    std::vector<int> histCounts(histBuckets.size() + 1, 0);

    for (auto& p : sortedCols) {
        int count = p.first;
        int bucket = 0;
        for (int b = 0; b < (int)histBuckets.size(); b++) {
            if (count <= histBuckets[b]) {
                bucket = b;
                break;
            }
            bucket = histBuckets.size();
        }
        histCounts[bucket]++;
    }

    printf("  1 access: %d columns\n", histCounts[0]);
    for (int b = 0; b < (int)histBuckets.size() - 1; b++) {
        printf("  %d-%d accesses: %d columns\n",
               histBuckets[b] + 1, histBuckets[b + 1], histCounts[b + 1]);
    }
    printf("  > %d accesses: %d columns\n", histBuckets.back(), histCounts[histBuckets.size()]);

    // 2. Spatial locality analysis
    printf("\n=== Spatial Locality Analysis ===\n");

    // Count adjacent column access patterns in each row
    long long adjacentPairs = 0;
    long long totalPairs = 0;

    for (int row = 0; row < numCols; row++) {  // Assuming square matrix
        // Sort columns within each row
        std::vector<int> rowCols;
        // This is inefficient for CSR, skip for now
    }

    // Instead, check global column adjacency
    std::vector<int> sortedColIdx(nnz);
    memcpy(sortedColIdx.data(), colIdx, nnz * sizeof(int));
    std::sort(sortedColIdx.begin(), sortedColIdx.end());

    int adjacentCount = 0;
    for (int i = 1; i < nnz; i++) {
        if (sortedColIdx[i] == sortedColIdx[i-1] || sortedColIdx[i] == sortedColIdx[i-1] + 1) {
            adjacentCount++;
        }
    }
    printf("Adjacent/same column access ratio: %.2f%%\n", 100.0 * adjacentCount / (nnz - 1));
}

// ========== Test Column Reordering ==========
void testColReordering(const char* mtxFile) {
    int numRows, numCols, nnz;
    int *h_rowPtr, *h_colIdx;
    double *h_values;

    if (!readMTX(mtxFile, &numRows, &numCols, &nnz, &h_rowPtr, &h_colIdx, &h_values)) {
        printf("Cannot read: %s\n", mtxFile);
        return;
    }

    printf("Matrix: %s\n", mtxFile);
    printf("  Rows: %d, Cols: %d, NNZ: %d, avgNnz=%.2f\n",
           numRows, numCols, nnz, (double)nnz/numRows);

    // Analyze column locality
    analyzeColLocality(numCols, nnz, h_colIdx);

    // Test 1: Original matrix
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
    int testIters = 50;
    cudaFuncSetCacheConfig(scalar_spmv_kernel, cudaFuncCachePreferL1);

    // Warmup
    for (int w = 0; w < 10; w++) {
        scalar_spmv_kernel<<<gridSize, blockSize>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    }
    cudaDeviceSynchronize();

    // Test original
    printf("\n=== Testing Original Matrix ===\n");
    for (int i = 0; i < testIters; i++) {
        scalar_spmv_kernel<<<gridSize, blockSize>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    }
    cudaDeviceSynchronize();

    // Cleanup
    delete[] h_rowPtr; delete[] h_colIdx; delete[] h_values; delete[] h_x;
    cudaFree(d_rowPtr); cudaFree(d_colIdx); cudaFree(d_values); cudaFree(d_x); cudaFree(d_y);
}

int main(int argc, char** argv) {
    printf("========================================\n");
    printf("  Column Locality Analysis\n");
    printf("========================================\n\n");

    const char* mtxFile = argc > 1 ? argv[1] : "real_cases/mtx/p0_A";
    testColReordering(mtxFile);

    printf("\n========================================\n");
    printf("Use hcTracer for actual timing\n");
    printf("========================================\n");

    return 0;
}