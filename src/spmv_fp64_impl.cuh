/**
 * @file spmv_fp64_impl.cuh
 * @brief Internal kernel implementations for FP64 SpMV
 */

#ifndef SPMV_FP64_IMPL_CUH_
#define SPMV_FP64_IMPL_CUH_

#include <cuda_runtime.h>

namespace spmv_fp64_impl {

// ==================== Warp Reduction ====================

template<int WarpSize>
__device__ __forceinline__ double warpReduceSum(double val) {
    if (WarpSize >= 64) {
        val += __shfl_down_sync(0xffffffffffffffffULL, val, 32);
    }
    if (WarpSize >= 32) {
        val += __shfl_down_sync(0xffffffff, val, 16);
    }
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    return val;
}

// ==================== TPR Kernel (Mars X201) ====================

template<int WarpSize, int TPR>
__global__ void tpr_kernel(
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

        for (int offset = TPR / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (threadInRow == 0) {
            y[row] = sum;
        }
    }
}

// ==================== __ldg Kernel (NVIDIA) ====================

template<int WarpSize>
__global__ void ldg_kernel(
    int numRows,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const double* __restrict__ values,
    const double* __restrict__ x,
    double* __restrict__ y)
{
    int row = blockIdx.x * (blockDim.x / WarpSize) + (threadIdx.x / WarpSize);
    int lane = threadIdx.x % WarpSize;

    if (row < numRows) {
        int rowStart = __ldg(&rowPtr[row]);
        int rowEnd = __ldg(&rowPtr[row + 1]);

        double sum = 0.0;
        for (int i = rowStart + lane; i < rowEnd; i += WarpSize) {
            int col = __ldg(&colIdx[i]);
            sum += __ldg(&values[i]) * __ldg(&x[col]);
        }

        sum = warpReduceSum<WarpSize>(sum);

        if (lane == 0) {
            y[row] = sum;
        }
    }
}

// ==================== Launch Functions ====================

inline void launch_mars_optimal(
    int numRows,
    const int* d_rowPtr,
    const int* d_colIdx,
    const double* d_values,
    const double* d_x,
    double* d_y,
    cudaStream_t stream = 0)
{
    const int WarpSize = 64;
    const int TPR = 8;
    const int blockSize = 256;

    int rowsPerWarp = WarpSize / TPR;
    int numWarps = (numRows + rowsPerWarp - 1) / rowsPerWarp;
    int gridSize = (numWarps * WarpSize + blockSize - 1) / blockSize;

    cudaFuncSetCacheConfig(tpr_kernel<WarpSize, TPR>, cudaFuncCachePreferL1);

    tpr_kernel<WarpSize, TPR><<<gridSize, blockSize, 0, stream>>>(
        numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
}

inline void launch_nvidia_optimal(
    int numRows,
    const int* d_rowPtr,
    const int* d_colIdx,
    const double* d_values,
    const double* d_x,
    double* d_y,
    cudaStream_t stream = 0)
{
    const int WarpSize = 32;
    const int blockSize = 256;

    int numWarps = numRows;
    int gridSize = (numWarps * WarpSize + blockSize - 1) / blockSize;

    cudaFuncSetCacheConfig(ldg_kernel<WarpSize>, cudaFuncCachePreferL1);

    ldg_kernel<WarpSize><<<gridSize, blockSize, 0, stream>>>(
        numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
}

// ==================== Bandwidth Calculation ====================

inline double calculate_bandwidth(int nnz, int numRows, double time_ms) {
    // FP64 SpMV: values(8B) + colIdx(4B) + x(col)(8B) per nnz
    // Plus y output: numRows * 8B
    double bytes = (double)nnz * 20.0 + (double)numRows * 8.0;
    return bytes / (time_ms * 1e6);  // GB/s
}

} // namespace spmv_fp64_impl

#endif /* SPMV_FP64_IMPL_CUH_ */