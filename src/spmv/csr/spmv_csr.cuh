/**
 * @file spmv_csr.cuh
 * @brief CSR format SpMV kernel declarations
 *
 * Supports multiple floating-point types and warp sizes (32/64)
 */

#ifndef SPMV_CSR_CUH_
#define SPMV_CSR_CUH_

#include "utils/common.h"
#include "formats/sparse_formats.h"

namespace muxi_spmv {

// Forward declarations - device functions must be declared before use
template<typename FloatType, int WarpSize>
__device__ __forceinline__ FloatType warpReduceSum(FloatType val);

__device__ __forceinline__ int binarySearchRow(
    const int* __restrict__ rowPtr, int numRows, int nnzIdx);

/**
 * @brief CSR SpMV kernel configurations
 */
struct CSRKernelConfig {
    int blockSize;
    int rowsPerBlock;
    int itemsPerThread;
    bool useSharedMem;
    int sharedMemSize;
};

CSRKernelConfig getOptimalConfig(int numRows, int nnz, int avgNnzPerRow, int warpSize);

// ==================== Device Functions ====================

template<typename FloatType, int WarpSize>
__device__ __forceinline__ FloatType warpReduceSum(FloatType val) {
    if (WarpSize >= 64) {
        val += __shfl_down_sync(0xffffffff, val, 32);
    }
    if (WarpSize >= 32) {
        val += __shfl_down_sync(0xffffffff, val, 16);
    }
    if (WarpSize >= 16) {
        val += __shfl_down_sync(0xffffffff, val, 8);
    }
    if (WarpSize >= 8) {
        val += __shfl_down_sync(0xffffffff, val, 4);
    }
    if (WarpSize >= 4) {
        val += __shfl_down_sync(0xffffffff, val, 2);
    }
    if (WarpSize >= 2) {
        val += __shfl_down_sync(0xffffffff, val, 1);
    }
    return val;
}

__device__ __forceinline__ int binarySearchRow(
    const int* __restrict__ rowPtr, int numRows, int nnzIdx)
{
    int lo = 0;
    int hi = numRows;

    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (rowPtr[mid] <= nnzIdx) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo - 1;
}

// ==================== Kernel Functions ====================

/**
 * @brief CSR SpMV kernel - Scalar approach (one thread per row)
 */
template<typename FloatType, bool TRANSPOSE = false>
__global__ void spmv_csr_scalar_kernel(
    int numRows,
    int numCols,
    int nnz,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numRows) {
        FloatType sum = static_cast<FloatType>(0);

        int rowStart = rowPtr[row];
        int rowEnd = rowPtr[row + 1];

        for (int i = rowStart; i < rowEnd; i++) {
            int col = colIdx[i];
            if (TRANSPOSE) {
                atomicAdd(&y[col], values[i] * x[row]);
            } else {
                sum += values[i] * x[col];
            }
        }

        if (!TRANSPOSE) {
            y[row] = sum;
        }
    }
}

/**
 * @brief CSR SpMV kernel - Vector approach (one warp per row)
 */
template<typename FloatType, int WarpSize, bool TRANSPOSE = false>
__global__ void spmv_csr_vector_kernel(
    int numRows,
    int numCols,
    int nnz,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y)
{
    int row = blockIdx.x * (blockDim.x / WarpSize) + (threadIdx.x / WarpSize);
    int lane = threadIdx.x % WarpSize;

    if (row < numRows) {
        int rowStart = rowPtr[row];
        int rowEnd = rowPtr[row + 1];

        FloatType sum = static_cast<FloatType>(0);

        for (int i = rowStart + lane; i < rowEnd; i += WarpSize) {
            int col = colIdx[i];
            sum += values[i] * x[col];
        }

        sum = warpReduceSum<FloatType, WarpSize>(sum);

        if (lane == 0) {
            y[row] = sum;
        }
    }
}

/**
 * @brief Merge-based CSR SpMV kernel
 */
template<typename FloatType, int WarpSize>
__global__ void spmv_csr_merge_kernel(
    int numRows,
    int numCols,
    int nnz,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int idx = tid; idx < nnz; idx += stride) {
        int row = binarySearchRow(rowPtr, numRows, idx);
        FloatType val = values[idx];
        int col = colIdx[idx];
        atomicAdd(&y[row], val * x[col]);
    }
}

// ==================== Host API ====================

template<typename FloatType>
spmv_status_t spmv_csr(
    const CSRMatrix<FloatType>& matrix,
    const FloatType* x,
    FloatType* y,
    FloatType alpha,
    FloatType beta,
    const spmv_opts_t& opts);

template<typename FloatType>
spmv_status_t spmv_csr_transpose(
    const CSRMatrix<FloatType>& matrix,
    const FloatType* x,
    FloatType* y,
    FloatType alpha,
    FloatType beta,
    const spmv_opts_t& opts);

} // namespace muxi_spmv

#endif // SPMV_CSR_CUH_