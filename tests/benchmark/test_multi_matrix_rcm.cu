/**
 * @file test_multi_matrix_comparison.cu
 * @brief 多矩阵对比测试：原始 vs RCM重排序
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <queue>
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

// ========== RCM Ordering ==========
void buildAdjacencyList(int numRows, int nnz, const int* rowPtr, const int* colIdx,
                        std::vector<std::vector<int>>& adj) {
    adj.resize(numRows);
    for (int row = 0; row < numRows; row++) {
        for (int i = rowPtr[row]; i < rowPtr[row + 1]; i++) {
            int col = colIdx[i];
            if (col != row) adj[row].push_back(col);
        }
    }
    for (int i = 0; i < numRows; i++) {
        std::sort(adj[i].begin(), adj[i].end());
        adj[i].erase(std::unique(adj[i].begin(), adj[i].end()), adj[i].end());
    }
}

std::vector<int> rcmOrdering(int numRows, const std::vector<std::vector<int>>& adj) {
    std::vector<int> order;
    std::vector<bool> visited(numRows, false);
    std::queue<int> q;

    int minDegreeNode = 0;
    int minDegree = adj[0].size();
    for (int i = 1; i < numRows; i++) {
        if (adj[i].size() < minDegree) {
            minDegree = adj[i].size();
            minDegreeNode = i;
        }
    }

    q.push(minDegreeNode);
    visited[minDegreeNode] = true;

    while (!q.empty()) {
        int current = q.front();
        q.pop();
        order.push_back(current);

        std::vector<std::pair<int, int>> neighbors;
        for (int neighbor : adj[current]) {
            if (!visited[neighbor]) {
                neighbors.push_back({adj[neighbor].size(), neighbor});
            }
        }
        std::sort(neighbors.begin(), neighbors.end());

        for (auto& p : neighbors) {
            visited[p.second] = true;
            q.push(p.second);
        }
    }

    for (int i = 0; i < numRows; i++) {
        if (!visited[i]) order.push_back(i);
    }

    std::reverse(order.begin(), order.end());
    return order;
}

void applyOrdering(int numRows, int numCols, int nnz,
                   const int* origRowPtr, const int* origColIdx, const double* origValues,
                   const std::vector<int>& order,
                   int** newRowPtr, int** newColIdx, double** newValues,
                   std::vector<int>& reverseOrder) {

    reverseOrder.resize(numRows);
    for (int i = 0; i < numRows; i++) reverseOrder[order[i]] = i;

    *newRowPtr = new int[numRows + 1];
    *newColIdx = new int[nnz];
    *newValues = new double[nnz];

    (*newRowPtr)[0] = 0;
    int pos = 0;

    for (int newRow = 0; newRow < numRows; newRow++) {
        int origRow = order[newRow];
        for (int i = origRowPtr[origRow]; i < origRowPtr[origRow + 1]; i++) {
            (*newColIdx)[pos] = reverseOrder[origColIdx[i]];
            (*newValues)[pos] = origValues[i];
            pos++;
        }
        (*newRowPtr)[newRow + 1] = pos;
    }
}

double calculateAvgBandwidth(int numRows, const int* rowPtr, const int* colIdx) {
    long long total = 0;
    int count = 0;
    for (int row = 0; row < numRows; row++) {
        for (int i = rowPtr[row]; i < rowPtr[row + 1]; i++) {
            total += std::abs(colIdx[i] - row);
            count++;
        }
    }
    return count > 0 ? (double)total / count : 0;
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

// ========== Test Single Matrix ==========
void testMatrix(const char* mtxFile, int testIters) {
    int numRows, numCols, nnz;
    int *h_rowPtr, *h_colIdx;
    double *h_values;

    if (!readMTX(mtxFile, &numRows, &numCols, &nnz, &h_rowPtr, &h_colIdx, &h_values)) {
        printf("  Cannot read: %s\n", mtxFile);
        return;
    }

    double origAvgBW = calculateAvgBandwidth(numRows, h_rowPtr, h_colIdx);

    // RCM ordering
    std::vector<std::vector<int>> adj;
    buildAdjacencyList(numRows, nnz, h_rowPtr, h_colIdx, adj);
    std::vector<int> order = rcmOrdering(numRows, adj);

    int *h_newRowPtr, *h_newColIdx;
    double *h_newValues;
    std::vector<int> reverseOrder;
    applyOrdering(numRows, numCols, nnz, h_rowPtr, h_colIdx, h_values, order,
                  &h_newRowPtr, &h_newColIdx, &h_newValues, reverseOrder);

    double rcmAvgBW = calculateAvgBandwidth(numRows, h_newRowPtr, h_newColIdx);

    printf("Matrix: %s\n", mtxFile);
    printf("  Size: %d rows, %d NNZ, avgNnz=%.2f\n", numRows, nnz, (double)nnz/numRows);
    printf("  Bandwidth: orig=%.1f, rcm=%.1f (%.2fx reduction)\n",
           origAvgBW, rcmAvgBW, origAvgBW/rcmAvgBW);

    // GPU test
    int *d_rowPtr_orig, *d_colIdx_orig;
    double *d_values_orig, *d_x_orig, *d_y_orig;
    int *d_rowPtr_rcm, *d_colIdx_rcm;
    double *d_values_rcm, *d_x_rcm, *d_y_rcm;

    cudaMalloc(&d_rowPtr_orig, (numRows + 1) * sizeof(int));
    cudaMalloc(&d_colIdx_orig, nnz * sizeof(int));
    cudaMalloc(&d_values_orig, nnz * sizeof(double));
    cudaMalloc(&d_x_orig, numCols * sizeof(double));
    cudaMalloc(&d_y_orig, numRows * sizeof(double));

    cudaMalloc(&d_rowPtr_rcm, (numRows + 1) * sizeof(int));
    cudaMalloc(&d_colIdx_rcm, nnz * sizeof(int));
    cudaMalloc(&d_values_rcm, nnz * sizeof(double));
    cudaMalloc(&d_x_rcm, numCols * sizeof(double));
    cudaMalloc(&d_y_rcm, numRows * sizeof(double));

    double* h_x = new double[numCols];
    for (int i = 0; i < numCols; i++) h_x[i] = 1.0;

    double* h_x_rcm = new double[numCols];
    for (int i = 0; i < numCols; i++) h_x_rcm[i] = h_x[order[i]];

    cudaMemcpy(d_rowPtr_orig, h_rowPtr, (numRows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdx_orig, h_colIdx, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values_orig, h_values, nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x_orig, h_x, numCols * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_rowPtr_rcm, h_newRowPtr, (numRows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdx_rcm, h_newColIdx, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values_rcm, h_newValues, nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x_rcm, h_x_rcm, numCols * sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (numRows + blockSize - 1) / blockSize;
    cudaFuncSetCacheConfig(scalar_spmv_kernel, cudaFuncCachePreferL1);

    // Warmup
    for (int w = 0; w < 10; w++) {
        scalar_spmv_kernel<<<gridSize, blockSize>>>(numRows, d_rowPtr_orig, d_colIdx_orig,
                                                     d_values_orig, d_x_orig, d_y_orig);
        scalar_spmv_kernel<<<gridSize, blockSize>>>(numRows, d_rowPtr_rcm, d_colIdx_rcm,
                                                     d_values_rcm, d_x_rcm, d_y_rcm);
    }
    cudaDeviceSynchronize();

    // Test original
    for (int i = 0; i < testIters; i++) {
        scalar_spmv_kernel<<<gridSize, blockSize>>>(numRows, d_rowPtr_orig, d_colIdx_orig,
                                                     d_values_orig, d_x_orig, d_y_orig);
    }
    cudaDeviceSynchronize();

    // Test RCM
    for (int i = 0; i < testIters; i++) {
        scalar_spmv_kernel<<<gridSize, blockSize>>>(numRows, d_rowPtr_rcm, d_colIdx_rcm,
                                                     d_values_rcm, d_x_rcm, d_y_rcm);
    }
    cudaDeviceSynchronize();

    // Cleanup
    delete[] h_rowPtr; delete[] h_colIdx; delete[] h_values; delete[] h_x;
    delete[] h_newRowPtr; delete[] h_newColIdx; delete[] h_newValues; delete[] h_x_rcm;

    cudaFree(d_rowPtr_orig); cudaFree(d_colIdx_orig); cudaFree(d_values_orig);
    cudaFree(d_x_orig); cudaFree(d_y_orig);
    cudaFree(d_rowPtr_rcm); cudaFree(d_colIdx_rcm); cudaFree(d_values_rcm);
    cudaFree(d_x_rcm); cudaFree(d_y_rcm);

    printf("\n");
}

int main(int argc, char** argv) {
    printf("========================================\n");
    printf("  Multi-Matrix RCM Comparison Test\n");
    printf("========================================\n\n");

    const char* matrices[] = {
        "real_cases/mtx/p0_A",
        "real_cases/mtx/p1_A",
        "real_cases/mtx/p2_A",
        "real_cases/mtx/p3_A",
        "real_cases/mtx/p4_A"
    };

    int testIters = 50;

    for (int i = 0; i < 5; i++) {
        testMatrix(matrices[i], testIters);
    }

    printf("========================================\n");
    printf("Use hcTracer for actual timing\n");
    printf("========================================\n");

    return 0;
}