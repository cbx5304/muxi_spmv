/**
 * @file spmv_fp32_impl.cuh
 * @brief Internal kernel implementations for FP32 SpMV
 *
 * This file contains:
 * 1. Warp reduction utilities
 * 2. Full SpMV kernels (for host API)
 * 3. Device-level SpMV functions (for user kernel integration)
 */

#ifndef SPMV_FP32_IMPL_CUH_
#define SPMV_FP32_IMPL_CUH_

#include <cuda_runtime.h>

namespace spmv_fp32_impl {

// ==================== Warp Reduction ====================

template<int WarpSize>
__device__ __forceinline__ float warpReduceSum(float val) {
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

// ==================== Device-Level SpMV Functions ====================
// For use in user kernels - exposes optimized computation logic

// ==================== NVIDIA Optimized (__ldg) ====================

/**
 * @brief Compute single row SpMV using __ldg optimization
 *
 * Use this when: WarpSize=32, want __ldg cache hints for random x access
 *
 * @param row        Row index to compute
 * @param rowPtr     Row pointer array
 * @param colIdx     Column index array
 * @param values     Value array
 * @param x          Input vector
 * @return Partial sum for this row (needs warp reduction if used by multiple threads)
 */
__device__ float compute_row_ldg_partial(
    int row,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const float* __restrict__ values,
    const float* __restrict__ x)
{
    int rowStart = __ldg(&rowPtr[row]);
    int rowEnd = __ldg(&rowPtr[row + 1]);

    float sum = 0.0f;
    // Single thread processes entire row
    for (int i = rowStart; i < rowEnd; i++) {
        int col = __ldg(&colIdx[i]);
        sum += __ldg(&values[i]) * __ldg(&x[col]);
    }
    return sum;
}

/**
 * @brief Compute single row SpMV with warp cooperation (NVIDIA style)
 *
 * Template parameters:
 * - WarpSize: 32 for NVIDIA
 *
 * @param row        Row index to compute
 * @param lane       Thread lane within warp (threadIdx.x % WarpSize)
 * @param rowPtr     Row pointer array
 * @param colIdx     Column index array
 * @param values     Value array
 * @param x          Input vector
 * @return Full row sum (only valid for lane==0, others return 0)
 */
template<int WarpSize = 32>
__device__ float compute_row_ldg_cooperative(
    int row,
    int lane,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const float* __restrict__ values,
    const float* __restrict__ x)
{
    int rowStart = __ldg(&rowPtr[row]);
    int rowEnd = __ldg(&rowPtr[row + 1]);

    float sum = 0.0f;
    for (int i = rowStart + lane; i < rowEnd; i += WarpSize) {
        int col = __ldg(&colIdx[i]);
        sum += __ldg(&values[i]) * __ldg(&x[col]);
    }

    sum = warpReduceSum<WarpSize>(sum);

    return sum;  // All lanes have same value after reduction
}

/**
 * @brief Compute row with alpha/beta support (NVIDIA __ldg)
 */
template<int WarpSize = 32>
__device__ float compute_row_ldg_general(
    int row,
    int lane,
    float alpha,
    float beta,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const float* __restrict__ values,
    const float* __restrict__ x,
    float y_old)
{
    float spmv_result = compute_row_ldg_cooperative<WarpSize>(
        row, lane, rowPtr, colIdx, values, x);
    return alpha * spmv_result + beta * y_old;
}

// ==================== Mars Optimized (TPR=8) ====================

/**
 * @brief Compute single row SpMV using TPR strategy
 *
 * Use this when: WarpSize=64, want TPR optimization for better thread utilization
 *
 * Template parameters:
 * - WarpSize: 64 for Mars X201
 * - TPR: Threads per row (4,8,16,32,64)
 *
 * @param row            Row index to compute
 * @param threadInRow    Thread index within row group (lane % TPR)
 * @param rowPtr         Row pointer array
 * @param colIdx         Column index array
 * @param values         Value array
 * @param x              Input vector
 * @return Full row sum (only valid for threadInRow==0, others return partial)
 */
template<int WarpSize = 64, int TPR = 8>
__device__ float compute_row_tpr(
    int row,
    int threadInRow,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const float* __restrict__ values,
    const float* __restrict__ x)
{
    int rowStart = rowPtr[row];
    int rowEnd = rowPtr[row + 1];

    float sum = 0.0f;
    for (int i = rowStart + threadInRow; i < rowEnd; i += TPR) {
        sum += values[i] * x[colIdx[i]];
    }

    // Correct TPR reduction with appropriate mask for WarpSize=64
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

    return sum;  // All TPR threads have same value after reduction
}

/**
 * @brief Compute row with alpha/beta support (Mars TPR)
 */
template<int WarpSize = 64, int TPR = 8>
__device__ float compute_row_tpr_general(
    int row,
    int threadInRow,
    float alpha,
    float beta,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const float* __restrict__ values,
    const float* __restrict__ x,
    float y_old)
{
    float spmv_result = compute_row_tpr<WarpSize, TPR>(
        row, threadInRow, rowPtr, colIdx, values, x);
    return alpha * spmv_result + beta * y_old;
}

// ==================== Convenience: Auto-select based on WarpSize ====================

/**
 * @brief Auto-select optimal compute function based on WarpSize
 *
 * Usage: Call in your kernel with compile-time WarpSize constant
 * - WarpSize=32: Uses __ldg optimization
 * - WarpSize=64: Uses TPR=8 optimization
 */
template<int WarpSize>
__device__ float compute_row_auto(
    int row,
    int lane,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const float* __restrict__ values,
    const float* __restrict__ x)
{
    if (WarpSize == 32) {
        return compute_row_ldg_cooperative<32>(row, lane, rowPtr, colIdx, values, x);
    } else {
        constexpr int TPR = 8;
        int threadInRow = lane % TPR;
        return compute_row_tpr<WarpSize, TPR>(row, threadInRow, rowPtr, colIdx, values, x);
    }
}

template<int WarpSize>
__device__ float compute_row_auto_general(
    int row,
    int lane,
    float alpha,
    float beta,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const float* __restrict__ values,
    const float* __restrict__ x,
    float y_old)
{
    if (WarpSize == 32) {
        return compute_row_ldg_general<32>(row, lane, alpha, beta,
            rowPtr, colIdx, values, x, y_old);
    } else {
        constexpr int TPR = 8;
        int threadInRow = lane % TPR;
        return compute_row_tpr_general<WarpSize, TPR>(row, threadInRow, alpha, beta,
            rowPtr, colIdx, values, x, y_old);
    }
}

// ==================== TPR Kernel (Mars X201) ====================
// Fixed version: Correct 64-bit mask for TPR>=32

template<int WarpSize, int TPR>
__global__ void tpr_kernel(
    int numRows,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const float* __restrict__ values,
    const float* __restrict__ x,
    float* __restrict__ y)
{
    int rowsPerWarp = WarpSize / TPR;
    int warpId = blockIdx.x * (blockDim.x / WarpSize) + (threadIdx.x / WarpSize);
    int lane = threadIdx.x % WarpSize;
    int row = warpId * rowsPerWarp + lane / TPR;
    int threadInRow = lane % TPR;

    if (row < numRows) {
        int rowStart = rowPtr[row];
        int rowEnd = rowPtr[row + 1];

        float sum = 0.0f;
        for (int i = rowStart + threadInRow; i < rowEnd; i += TPR) {
            sum += values[i] * x[colIdx[i]];
        }

        // Correct reduction with appropriate mask for WarpSize=64
        // TPR=64 needs 64-bit mask (0xffffffffffffffffULL) for offset=32
        // TPR<=32 uses 32-bit mask (0xffffffff) which is sufficient
        if (TPR >= 64) {
            // Full warp reduction for TPR=64
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

// ==================== __ldg Kernel (NVIDIA) ====================

template<int WarpSize>
__global__ void ldg_kernel(
    int numRows,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const float* __restrict__ values,
    const float* __restrict__ x,
    float* __restrict__ y)
{
    int row = blockIdx.x * (blockDim.x / WarpSize) + (threadIdx.x / WarpSize);
    int lane = threadIdx.x % WarpSize;

    if (row < numRows) {
        int rowStart = __ldg(&rowPtr[row]);
        int rowEnd = __ldg(&rowPtr[row + 1]);

        float sum = 0.0f;
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

// Adaptive TPR kernel launcher for Mars X201
// Automatically selects optimal TPR based on avgNnz:
// - avgNnz < 16: TPR=4 (16 rows/warp) - very sparse
// - avgNnz < 40: TPR=8 (8 rows/warp) - sparse
// - avgNnz < 80: TPR=16 (4 rows/warp) - moderately dense (optimal for avgNnz~62)
// - avgNnz < 128: TPR=32 (2 rows/warp) - dense
// - avgNnz >= 128: TPR=64 (1 row/warp) - very dense

inline void launch_mars_adaptive(
    int numRows,
    int nnz,  // Used to calculate avgNnz
    const int* d_rowPtr,
    const int* d_colIdx,
    const float* d_values,
    const float* d_x,
    float* d_y,
    cudaStream_t stream = 0)
{
    const int WarpSize = 64;
    const int blockSize = 256;

    // Calculate avgNnz and select optimal TPR
    double avgNnz = (double)nnz / numRows;
    int TPR;

    if (avgNnz >= 128) {
        TPR = 64;  // Very dense: 1 row/warp, full warp per row
    } else if (avgNnz >= 80) {
        TPR = 32;  // Dense matrices: 2 rows/warp, ~83% utilization
    } else if (avgNnz >= 40) {
        TPR = 16;  // Moderately dense: 4 rows/warp (optimal for avgNnz~62)
    } else if (avgNnz >= 16) {
        TPR = 8;   // Sparse: 8 rows/warp, ~42% utilization
    } else {
        TPR = 4;   // Very sparse: 16 rows/warp
    }

    int rowsPerWarp = WarpSize / TPR;
    int numWarps = (numRows + rowsPerWarp - 1) / rowsPerWarp;
    int gridSize = (numWarps * WarpSize + blockSize - 1) / blockSize;

    // Set cache config for all possible kernels
    cudaFuncSetCacheConfig(tpr_kernel<64, 4>, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(tpr_kernel<64, 8>, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(tpr_kernel<64, 16>, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(tpr_kernel<64, 32>, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(tpr_kernel<64, 64>, cudaFuncCachePreferL1);

    // Launch appropriate kernel
    switch (TPR) {
        case 4:
            tpr_kernel<64, 4><<<gridSize, blockSize, 0, stream>>>(
                numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
            break;
        case 8:
            tpr_kernel<64, 8><<<gridSize, blockSize, 0, stream>>>(
                numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
            break;
        case 16:
            tpr_kernel<64, 16><<<gridSize, blockSize, 0, stream>>>(
                numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
            break;
        case 32:
            tpr_kernel<64, 32><<<gridSize, blockSize, 0, stream>>>(
                numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
            break;
        case 64:
            tpr_kernel<64, 64><<<gridSize, blockSize, 0, stream>>>(
                numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
            break;
    }
}

// Legacy function for backward compatibility (TPR=8)
inline void launch_mars_optimal(
    int numRows,
    const int* d_rowPtr,
    const int* d_colIdx,
    const float* d_values,
    const float* d_x,
    float* d_y,
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
    const float* d_values,
    const float* d_x,
    float* d_y,
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
    // FP32 SpMV: values(4B) + colIdx(4B) + x(col)(4B) per nnz
    // Plus y output: numRows * 4B
    double bytes = (double)nnz * 12.0 + (double)numRows * 4.0;
    return bytes / (time_ms * 1e6);  // GB/s
}

// ==================== Vector Operations ====================

// Scale vector: y[i] = alpha * y[i]
__global__ void scale_kernel(
    int n,
    float alpha,
    float* __restrict__ y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = alpha * y[i];
    }
}

// Blend vectors: y[i] = alpha * temp[i] + beta * y[i]
__global__ void blend_kernel(
    int n,
    float alpha,
    float beta,
    const float* __restrict__ temp,
    float* __restrict__ y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = alpha * temp[i] + beta * y[i];
    }
}

inline void launch_scale(
    int n,
    float alpha,
    float* d_y,
    cudaStream_t stream = 0)
{
    const int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    scale_kernel<<<gridSize, blockSize, 0, stream>>>(n, alpha, d_y);
}

inline void launch_blend(
    int n,
    float alpha,
    float beta,
    const float* d_temp,
    float* d_y,
    cudaStream_t stream = 0)
{
    const int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    blend_kernel<<<gridSize, blockSize, 0, stream>>>(n, alpha, beta, d_temp, d_y);
}

} // namespace spmv_fp32_impl

#endif /* SPMV_FP32_IMPL_CUH_ */