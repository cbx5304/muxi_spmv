/**
 * @file spmv_ellpack.cuh
 * @brief ELLPACK format SpMV kernels
 *
 * ELLPACK stores sparse matrices in a fixed-width format:
 * - Each row has the same number of elements (padded with zeros)
 * - Column indices and values are stored in column-major order
 * - This enables coalesced memory access patterns
 *
 * Best for: Matrices where row lengths are similar (low variance)
 * Not ideal for: Matrices with highly variable row lengths (wastes memory)
 */

#ifndef SPMV_ELLPACK_CUH
#define SPMV_ELLPACK_CUH

#include <cuda_runtime.h>
#include "utils/common.h"
#include "formats/sparse_formats.h"

namespace muxi_spmv {

/**
 * @brief ELLPACK format matrix storage
 */
template<typename FloatType>
struct ELLPACKMatrix {
    int numRows;        // Number of rows
    int numCols;        // Number of columns
    int numElements;    // Number of elements per row (max row length)
    int nnz;            // Actual number of non-zeros (excluding padding)

    // Device pointers
    int* d_colIdx;      // Column indices [numRows * numElements]
    FloatType* d_values; // Values [numRows * numElements]

    ELLPACKMatrix() : numRows(0), numCols(0), numElements(0), nnz(0),
                      d_colIdx(nullptr), d_values(nullptr) {}
};

/**
 * @brief Convert CSR to ELLPACK format
 *
 * @param csr CSR matrix
 * @param ellpack Output ELLPACK matrix
 * @param maxElements Maximum elements per row (0 = auto-detect from max row length)
 * @return Status code
 */
template<typename FloatType>
spmv_status_t csr_to_ellpack(
    const CSRMatrix<FloatType>& csr,
    ELLPACKMatrix<FloatType>& ellpack,
    int maxElements = 0);

/**
 * @brief Execute ELLPACK SpMV
 *
 * @param ellpack ELLPACK matrix
 * @param d_x Input vector (device)
 * @param d_y Output vector (device)
 * @param stream CUDA stream (0 for default)
 * @return Status code
 */
template<typename FloatType>
spmv_status_t spmv_ellpack(
    const ELLPACKMatrix<FloatType>& ellpack,
    const FloatType* d_x,
    FloatType* d_y,
    cudaStream_t stream);

/**
 * @brief ELLPACK SpMV kernel
 *
 * Memory access pattern:
 * - colIdx and values accessed in column-major order (coalesced)
 * - x accessed randomly based on column indices
 *
 * @param numRows Number of rows
 * @param numElements Elements per row
 * @param colIdx Column indices (column-major)
 * @param values Values (column-major)
 * @param x Input vector
 * @param y Output vector
 */
template<typename FloatType, int BLOCK_SIZE>
__global__ void spmv_ellpack_kernel(
    int numRows,
    int numElements,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numRows) {
        FloatType sum = static_cast<FloatType>(0);

        // Access column-major: element i of row r is at position r + i * numRows
        for (int i = 0; i < numElements; i++) {
            int idx = row + i * numRows;
            int col = colIdx[idx];
            // col == -1 indicates padding element
            if (col >= 0) {
                sum += values[idx] * x[col];
            }
        }

        y[row] = sum;
    }
}

/**
 * @brief ELLPACK SpMV with shared memory caching for x
 */
template<typename FloatType, int BLOCK_SIZE, int X_CACHE_SIZE>
__global__ void spmv_ellpack_cached_kernel(
    int numRows,
    int numElements,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y,
    int numCols)
{
    __shared__ FloatType s_x[X_CACHE_SIZE];

    // Load x into shared memory
    for (int i = threadIdx.x; i < min(numCols, X_CACHE_SIZE); i += BLOCK_SIZE) {
        s_x[i] = x[i];
    }
    __syncthreads();

    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numRows) {
        FloatType sum = static_cast<FloatType>(0);

        for (int i = 0; i < numElements; i++) {
            int idx = row + i * numRows;
            int col = colIdx[idx];

            if (col >= 0) {
                if (col < X_CACHE_SIZE) {
                    sum += values[idx] * s_x[col];
                } else {
                    sum += values[idx] * x[col];
                }
            }
        }

        y[row] = sum;
    }
}

} // namespace muxi_spmv

#endif // SPMV_ELLPACK_CUH