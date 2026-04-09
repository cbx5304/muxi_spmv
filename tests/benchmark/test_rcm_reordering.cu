/**
 * @file test_rcm_reordering.cu
 * @brief RCM (Reverse Cuthill-McKee) 矩阵重排序优化测试
 *
 * 目标：通过RCM重排序减少矩阵带宽，提高缓存局部性
 * 运行: hcTracer --hctx ./test_rcm_reordering <mtx_file>
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <queue>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

// ========== CSR SpMV Kernel (SCALAR策略) ==========
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

// ========== RCM Ordering Implementation ==========
void buildAdjacencyList(int numRows, int nnz, const int* rowPtr, const int* colIdx,
                        std::vector<std::vector<int>>& adj) {
    adj.resize(numRows);
    for (int row = 0; row < numRows; row++) {
        for (int i = rowPtr[row]; i < rowPtr[row + 1]; i++) {
            int col = colIdx[i];
            if (col != row) {
                adj[row].push_back(col);
            }
        }
    }
    for (int i = 0; i < numRows; i++) {
        std::sort(adj[i].begin(), adj[i].end());
        adj[i].erase(std::unique(adj[i].begin(), adj[i].end()), adj[i].end());
    }
}

int getDegree(int node, const std::vector<std::vector<int>>& adj) {
    return adj[node].size();
}

std::vector<int> rcmOrdering(int numRows, const std::vector<std::vector<int>>& adj) {
    std::vector<int> order;
    std::vector<bool> visited(numRows, false);
    std::queue<int> q;

    // 找最小度节点作为起点
    int minDegreeNode = 0;
    int minDegree = getDegree(0, adj);
    for (int i = 1; i < numRows; i++) {
        int deg = getDegree(i, adj);
        if (deg < minDegree) {
            minDegree = deg;
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
                neighbors.push_back({getDegree(neighbor, adj), neighbor});
            }
        }

        std::sort(neighbors.begin(), neighbors.end());

        for (auto& p : neighbors) {
            visited[p.second] = true;
            q.push(p.second);
        }
    }

    for (int i = 0; i < numRows; i++) {
        if (!visited[i]) {
            order.push_back(i);
        }
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
    for (int i = 0; i < numRows; i++) {
        reverseOrder[order[i]] = i;
    }

    *newRowPtr = new int[numRows + 1];
    *newColIdx = new int[nnz];
    *newValues = new double[nnz];

    (*newRowPtr)[0] = 0;
    int pos = 0;

    for (int newRow = 0; newRow < numRows; newRow++) {
        int origRow = order[newRow];

        for (int i = origRowPtr[origRow]; i < origRowPtr[origRow + 1]; i++) {
            int origCol = origColIdx[i];
            int newCol = reverseOrder[origCol];
            (*newColIdx)[pos] = newCol;
            (*newValues)[pos] = origValues[i];
            pos++;
        }
        (*newRowPtr)[newRow + 1] = pos;
    }
}

int calculateBandwidth(int numRows, const int* rowPtr, const int* colIdx) {
    int maxBandwidth = 0;
    for (int row = 0; row < numRows; row++) {
        for (int i = rowPtr[row]; i < rowPtr[row + 1]; i++) {
            int bandwidth = std::abs(colIdx[i] - row);
            maxBandwidth = std::max(maxBandwidth, bandwidth);
        }
    }
    return maxBandwidth;
}

double calculateAvgBandwidth(int numRows, const int* rowPtr, const int* colIdx) {
    long long totalBandwidth = 0;
    int count = 0;
    for (int row = 0; row < numRows; row++) {
        for (int i = rowPtr[row]; i < rowPtr[row + 1]; i++) {
            totalBandwidth += std::abs(colIdx[i] - row);
            count++;
        }
    }
    return (double)totalBandwidth / count;
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
    int testIters = 100;

    printf("========================================\n");
    printf("  RCM Matrix Reordering Optimization\n");
    printf("========================================\n\n");

    // 读取原始矩阵
    int numRows, numCols, nnz;
    int *h_rowPtr, *h_colIdx;
    double *h_values;

    if (!readMTX(mtxFile, &numRows, &numCols, &nnz, &h_rowPtr, &h_colIdx, &h_values)) {
        printf("Cannot read matrix: %s\n", mtxFile);
        return 1;
    }

    printf("Matrix: %s\n", mtxFile);
    printf("  Rows: %d, Cols: %d, NNZ: %d, avgNnz=%.2f\n",
           numRows, numCols, nnz, (double)nnz/numRows);

    // 计算原始带宽
    int origMaxBW = calculateBandwidth(numRows, h_rowPtr, h_colIdx);
    double origAvgBW = calculateAvgBandwidth(numRows, h_rowPtr, h_colIdx);
    printf("  Original bandwidth: max=%d, avg=%.2f\n", origMaxBW, origAvgBW);

    // 构建邻接表并计算RCM排序
    printf("\nComputing RCM ordering...\n");
    std::vector<std::vector<int>> adj;
    buildAdjacencyList(numRows, nnz, h_rowPtr, h_colIdx, adj);

    std::vector<int> order = rcmOrdering(numRows, adj);

    // 应用重排序
    int *h_newRowPtr, *h_newColIdx;
    double *h_newValues;
    std::vector<int> reverseOrder;
    applyOrdering(numRows, numCols, nnz, h_rowPtr, h_colIdx, h_values, order,
                  &h_newRowPtr, &h_newColIdx, &h_newValues, reverseOrder);

    // 计算新带宽
    int newMaxBW = calculateBandwidth(numRows, h_newRowPtr, h_newColIdx);
    double newAvgBW = calculateAvgBandwidth(numRows, h_newRowPtr, h_newColIdx);
    printf("  RCM bandwidth: max=%d, avg=%.2f\n", newMaxBW, newAvgBW);
    printf("  Bandwidth reduction: max %.1fx, avg %.1fx\n",
           (double)origMaxBW/newMaxBW, origAvgBW/newAvgBW);

    // 分配GPU内存
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

    // 初始化x向量
    double* h_x = new double[numCols];
    for (int i = 0; i < numCols; i++) h_x[i] = 1.0;

    double* h_x_rcm = new double[numCols];
    for (int i = 0; i < numCols; i++) {
        h_x_rcm[i] = h_x[order[i]];
    }

    // 上传数据
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

    // 设置L1缓存配置
    cudaFuncSetCacheConfig(scalar_spmv_kernel, cudaFuncCachePreferL1);

    printf("\n========================================\n");
    printf("  Kernel Performance Test\n");
    printf("========================================\n");
    printf("Running %d iterations for each matrix...\n\n", testIters);

    // Warmup
    for (int w = 0; w < 20; w++) {
        scalar_spmv_kernel<<<gridSize, blockSize>>>(numRows, d_rowPtr_orig, d_colIdx_orig,
                                                     d_values_orig, d_x_orig, d_y_orig);
        scalar_spmv_kernel<<<gridSize, blockSize>>>(numRows, d_rowPtr_rcm, d_colIdx_rcm,
                                                     d_values_rcm, d_x_rcm, d_y_rcm);
    }
    cudaDeviceSynchronize();

    // 测试原始矩阵
    printf("Testing ORIGINAL matrix...\n");
    for (int i = 0; i < testIters; i++) {
        scalar_spmv_kernel<<<gridSize, blockSize>>>(numRows, d_rowPtr_orig, d_colIdx_orig,
                                                     d_values_orig, d_x_orig, d_y_orig);
    }
    cudaDeviceSynchronize();

    // 测试RCM重排矩阵
    printf("Testing RCM matrix...\n");
    for (int i = 0; i < testIters; i++) {
        scalar_spmv_kernel<<<gridSize, blockSize>>>(numRows, d_rowPtr_rcm, d_colIdx_rcm,
                                                     d_values_rcm, d_x_rcm, d_y_rcm);
    }
    cudaDeviceSynchronize();

    printf("\n========================================\n");
    printf("Use hcTracer to get actual timing:\n");
    printf("  hcTracer --hctx ./test_rcm_reordering %s\n", mtxFile);
    printf("========================================\n");

    // 清理
    delete[] h_rowPtr; delete[] h_colIdx; delete[] h_values; delete[] h_x;
    delete[] h_newRowPtr; delete[] h_newColIdx; delete[] h_newValues; delete[] h_x_rcm;

    cudaFree(d_rowPtr_orig); cudaFree(d_colIdx_orig); cudaFree(d_values_orig);
    cudaFree(d_x_orig); cudaFree(d_y_orig);
    cudaFree(d_rowPtr_rcm); cudaFree(d_colIdx_rcm); cudaFree(d_values_rcm);
    cudaFree(d_x_rcm); cudaFree(d_y_rcm);

    return 0;
}