/**
 * @file spmv_sparse_optimized.cu
 * @brief Optimized SpMV kernels for very sparse matrices (avgNnz < 10)
 *
 * Key insight: For Mars X201 with warp size=64, when avgNnz=4:
 * - Each partition processes ~4 rows (with elementsPerPartition=16)
 * - Only 4 threads work, 60 threads idle -> low utilization
 *
 * Solution: Use virtual warp size (e.g., 16) to better match row count
 */

#include "utils/common.h"
#include "formats/sparse_formats.h"
#include "spmv/csr5/spmv_csr5.cuh"
#include <cuda_runtime.h>

namespace muxi_spmv {

// ==================== Device Functions ====================

__device__ int binarySearchRow(const int* rowPtr, int numRows, int nnzIdx) {
    int lo = 0, hi = numRows;
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

// ==================== Light SpMV Kernel ====================

/**
 * @brief Lightweight SpMV for very sparse matrices
 *
 * Each thread processes multiple rows independently.
 * No merge-path partitioning overhead.
 * Best for avgNnz <= 4.
 */
template<typename FloatType, int BLOCK_SIZE>
__global__ void spmv_light_kernel(
    int numRows,
    int numCols,
    int nnz,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y,
    int rowsPerThread)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread processes multiple consecutive rows
    int myRowStart = tid * rowsPerThread;
    int myRowEnd = min(myRowStart + rowsPerThread, numRows);

    for (int row = myRowStart; row < myRowEnd; row++) {
        FloatType sum = static_cast<FloatType>(0);
        int rowStartNnz = rowPtr[row];
        int rowEndNnz = rowPtr[row + 1];

        for (int idx = rowStartNnz; idx < rowEndNnz; idx++) {
            int col = colIdx[idx];
            sum += values[idx] * x[col];
        }

        y[row] = sum;
    }
}

// ==================== Host Functions ====================

/**
 * @brief Select optimal kernel based on avgNnz
 */
template<typename FloatType>
spmv_status_t spmv_sparse_optimized(
    const CSRMatrix<FloatType>& matrix,
    const FloatType* x,
    FloatType* y,
    cudaStream_t stream)
{
    if (matrix.nnz == 0) {
        cudaMemsetAsync(y, 0, matrix.numRows * sizeof(FloatType), stream);
        return SPMV_SUCCESS;
    }

    int avgNnz = matrix.nnz / max(matrix.numRows, 1);
    int blockSize = 256;

    // Select strategy based on avgNnz and warp size
    if (WARP_SIZE == 64) {
        // Mars X201
        if (avgNnz <= 4) {
            // Very sparse: use light kernel with each thread processing multiple rows
            int numSMs = 104;
            int targetThreads = matrix.numRows;  // One thread per row ideally
            int rowsPerThread = max(1, (int)(matrix.numRows / (numSMs * blockSize * 4)));

            int gridSize = (matrix.numRows + rowsPerThread * blockSize - 1) / (rowsPerThread * blockSize);

            cudaMemsetAsync(y, 0, matrix.numRows * sizeof(FloatType), stream);
            spmv_light_kernel<FloatType, 256><<<gridSize, blockSize, 0, stream>>>(
                matrix.numRows, matrix.numCols, matrix.nnz,
                matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values,
                x, y, rowsPerThread);
        } else {
            // Normal sparsity: use standard merge-based
            return spmv_merge_based<FloatType>(matrix, x, y, stream);
        }
    } else {
        // RTX 4090 (warp=32): standard merge-based is already good
        return spmv_merge_based<FloatType>(matrix, x, y, stream);
    }

    return SPMV_SUCCESS;
}

// ==================== Explicit Template Instantiation ====================

template spmv_status_t spmv_sparse_optimized<float>(
    const CSRMatrix<float>&, const float*, float*, cudaStream_t);
template spmv_status_t spmv_sparse_optimized<double>(
    const CSRMatrix<double>&, const double*, double*, cudaStream_t);

} // namespace muxi_spmv