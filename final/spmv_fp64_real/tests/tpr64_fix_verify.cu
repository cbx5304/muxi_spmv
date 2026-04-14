/**
 * @file tpr64_fix_verify.cu
 * @brief Verify TPR=64 fix works correctly
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

#include "spmv_fp64.h"

// Test kernel for TPR=64 (full warp per row)
template<int WarpSize, int TPR>
__global__ void test_tpr_kernel(
    int numRows,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const double* __restrict__ values,
    const double* __restrict__ x,
    double* __restrict__ y)
{
    int rowsPerWarp = WarpSize / TPR;
    int warpId = blockIdx.x * (blockDim.x / WarpSize) + (threadIdx.x / WarpSize);
    int lane = threadIdx.x % WarpSize;
    int row = warpId * rowsPerWarp + lane / TPR;
    int threadInRow = lane % TPR;

    if (row < numRows) {
        int rowStart = rowPtr[row];
        int rowEnd = rowPtr[row + 1];

        double sum = 0.0;
        for (int i = rowStart + threadInRow; i < rowEnd; i += TPR) {
            sum += values[i] * x[colIdx[i]];
        }

        // Correct reduction with appropriate mask
        if (TPR >= 64) {
            sum += __shfl_down_sync(0xffffffffffffffffULL, sum, 32);
            sum += __shfl_down_sync(0xffffffff, sum, 16);
        } else if (TPR >= 32) {
            sum += __shfl_down_sync(0xffffffff, sum, 16);
        }
        sum += __shfl_down_sync(0xffffffff, sum, 8);
        sum += __shfl_down_sync(0xffffffff, sum, 4);
        sum += __shfl_down_sync(0xffffffff, sum, 2);
        sum += __shfl_down_sync(0xffffffff, sum, 1);

        if (threadInRow == 0) {
            y[row] = sum;
        }
    }
}

// Generate dense matrix (avgNnz=150) to trigger TPR=64
void generateDenseMatrix(int rows, int avgNnz,
                         std::vector<int>& rowPtr,
                         std::vector<int>& colIdx,
                         std::vector<double>& values) {
    int nnz = rows * avgNnz;
    rowPtr.resize(rows + 1);
    colIdx.resize(nnz);
    values.resize(nnz);

    rowPtr[0] = 0;
    for (int i = 0; i < rows; i++) {
        rowPtr[i + 1] = rowPtr[i] + avgNnz;
        for (int j = 0; j < avgNnz; j++) {
            colIdx[rowPtr[i] + j] = j;  // Dense pattern: first avgNnz columns
            values[rowPtr[i] + j] = 1.0;
        }
    }
}

double cpuReference(const std::vector<int>& rowPtr,
                    const std::vector<int>& colIdx,
                    const std::vector<double>& values,
                    const std::vector<double>& x,
                    int row) {
    double sum = 0.0;
    for (int i = rowPtr[row]; i < rowPtr[row + 1]; i++) {
        sum += values[i] * x[colIdx[i]];
    }
    return sum;
}

int main() {
    std::cout << "===== TPR=64 Fix Verification =====" << std::endl;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << ", Warp Size: " << prop.warpSize << std::endl;
    std::cout << std::endl;

    int rows = 1000;
    int avgNnz = 150;  // Dense matrix to test TPR=64
    int nnz = rows * avgNnz;

    std::cout << "Test matrix: " << rows << " rows, avgNnz=" << avgNnz << std::endl;
    std::cout << "Expected TPR selection: TPR=64 (avgNnz >= 128)" << std::endl;
    std::cout << std::endl;

    std::vector<int> rowPtr, colIdx;
    std::vector<double> values;
    generateDenseMatrix(rows, avgNnz, rowPtr, colIdx, values);

    std::vector<double> x(avgNnz, 1.0);  // x vector
    std::vector<double> ref_y(rows);

    // CPU reference
    for (int i = 0; i < rows; i++) {
        ref_y[i] = cpuReference(rowPtr, colIdx, values, x, i);
    }

    // Device memory
    int *d_rowPtr, *d_colIdx;
    double *d_values, *d_x, *d_y;

    cudaMalloc(&d_rowPtr, (rows + 1) * sizeof(int));
    cudaMalloc(&d_colIdx, nnz * sizeof(int));
    cudaMalloc(&d_values, nnz * sizeof(double));
    cudaMalloc(&d_x, avgNnz * sizeof(double));
    cudaMalloc(&d_y, rows * sizeof(double));

    cudaMemcpy(d_rowPtr, rowPtr.data(), (rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdx, colIdx.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values.data(), nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), avgNnz * sizeof(double), cudaMemcpyHostToDevice);

    const int WarpSize = 64;
    const int TPR = 64;
    const int blockSize = 256;
    int rowsPerWarp = WarpSize / TPR;
    int numWarps = (rows + rowsPerWarp - 1) / rowsPerWarp;
    int gridSize = (numWarps * WarpSize + blockSize - 1) / blockSize;

    // Test fixed kernel
    test_tpr_kernel<64, 64><<<gridSize, blockSize>>>(rows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    cudaDeviceSynchronize();

    std::vector<double> gpu_y(rows);
    cudaMemcpy(gpu_y.data(), d_y, rows * sizeof(double), cudaMemcpyDeviceToHost);

    // Verify correctness
    double maxErr = 0.0;
    int errCount = 0;
    for (int i = 0; i < rows; i++) {
        double err = std::abs(gpu_y[i] - ref_y[i]);
        if (err > maxErr) maxErr = err;
        if (err > 1e-10) errCount++;
    }

    std::cout << "TPR=64 Fix Verification Results:" << std::endl;
    std::cout << "| Metric | Value |" << std::endl;
    std::cout << "|--------|-------|" << std::endl;
    std::cout << "| Max Error | " << maxErr << " |" << std::endl;
    std::cout << "| Error Count | " << errCount << " / " << rows << " |" << std::endl;
    std::cout << "| Status | " << (maxErr < 1e-10 ? "PASSED" : "FAILED") << " |" << std::endl;
    std::cout << std::endl;

    // Also test optimized library
    if (spmv_fp64_check_license() == SPMV_FP64_SUCCESS) {
        std::cout << "Testing optimized library (adaptive TPR):" << std::endl;
        spmv_fp64_execute_direct(rows, nnz, d_rowPtr, d_colIdx, d_values, d_x, d_y, 0);
        cudaDeviceSynchronize();

        cudaMemcpy(gpu_y.data(), d_y, rows * sizeof(double), cudaMemcpyDeviceToHost);

        maxErr = 0.0;
        for (int i = 0; i < rows; i++) {
            maxErr = std::max(maxErr, std::abs(gpu_y[i] - ref_y[i]));
        }

        std::cout << "Library Max Error: " << maxErr << std::endl;
        std::cout << "Library Status: " << (maxErr < 1e-10 ? "PASSED" : "FAILED") << std::endl;
    }

    cudaFree(d_rowPtr);
    cudaFree(d_colIdx);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);

    std::cout << std::endl;
    std::cout << "===== Verification Complete =====" << std::endl;

    return 0;
}