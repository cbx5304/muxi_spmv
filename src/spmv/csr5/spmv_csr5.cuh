/**
 * @file spmv_csr5.cuh
 * @brief CSR5 format SpMV kernel declarations
 *
 * CSR5 divides NNZ into fixed-size tiles, each processed by one warp.
 * This provides load balancing for sparse matrices with varying row lengths.
 */

#ifndef SPMV_CSR5_CUH_
#define SPMV_CSR5_CUH_

#include "utils/common.h"
#include "formats/sparse_formats.h"

namespace muxi_spmv {

// ==================== CSR5 Configuration ====================

/**
 * @brief CSR5 kernel configuration
 */
struct CSR5KernelConfig {
    int blockSize;
    int warpSize;
    int sigma;              // Tile size (NNZ per tile)
    int warpsPerBlock;
    int sharedMemSize;
};

/**
 * @brief Get optimal CSR5 configuration for given GPU
 */
CSR5KernelConfig getCSR5Config(int warpSize, int nnz, int avgNnzPerRow);

// ==================== Device Functions ====================

/**
 * @brief Binary search for row index given NNZ offset
 * @param rowPtr CSR row pointers
 * @param numRows Number of rows
 * @param nnzIdx NNZ index to find row for
 * @return Row index containing the NNZ element
 */
__device__ __forceinline__ int csr5_binary_search_row(
    const int* __restrict__ rowPtr,
    int numRows,
    int nnzIdx)
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

/**
 * @brief Warp reduce sum for CSR5
 */
template<typename FloatType, int WarpSize>
__device__ __forceinline__ FloatType csr5_warp_reduce_sum(FloatType val) {
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

// ==================== CSR5 Preprocessing Kernels ====================

/**
 * @brief Kernel: Compute tile boundaries for CSR5
 *
 * Each thread computes metadata for one tile:
 * - Which row the tile starts at
 * - Offset within that row
 *
 * @param rowPtr CSR row pointers
 * @param numRows Number of rows
 * @param nnz Total NNZ count
 * @param sigma Tile size
 * @param tile_row_ptr Output: row index for each tile start
 * @param tile_nnz_offset Output: NNZ offset within start row
 * @param numTiles Number of tiles
 */
template<int BLOCK_SIZE>
__global__ void csr5_compute_tile_boundaries_kernel(
    const int* __restrict__ rowPtr,
    int numRows,
    int nnz,
    int sigma,
    int* __restrict__ tile_row_ptr,
    int* __restrict__ tile_nnz_offset,
    int numTiles);

// ==================== CSR5 SpMV Kernels ====================

/**
 * @brief CSR5 SpMV kernel - Basic tile-based approach
 *
 * Each warp processes one tile of sigma NNZ elements.
 * Threads cooperate to process all elements in the tile,
 * handling row boundaries with atomic accumulation.
 *
 * @param numRows Matrix rows
 * @param numCols Matrix columns
 * @param nnz Total NNZ
 * @param rowPtr Row pointers
 * @param colIdx Column indices
 * @param values Matrix values
 * @param x Input vector
 * @param y Output vector
 * @param tile_row_ptr Precomputed row index for each tile
 * @param tile_nnz_offset Offset within start row
 * @param sigma Tile size
 */
template<typename FloatType, int BLOCK_SIZE, int WarpSize, int Sigma>
__global__ void spmv_csr5_basic_kernel(
    int numRows,
    int numCols,
    int nnz,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y,
    const int* __restrict__ tile_row_ptr,
    const int* __restrict__ tile_nnz_offset,
    int sigma);

/**
 * @brief CSR5 SpMV kernel - Optimized for warp=64 (Mars X201)
 *
 * Specialized for 64-thread warp architecture.
 * Uses larger tile sizes and optimized memory access patterns.
 */
template<typename FloatType, int BLOCK_SIZE, int Sigma>
__global__ void spmv_csr5_warp64_kernel(
    int numRows,
    int numCols,
    int nnz,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y,
    const int* __restrict__ tile_row_ptr,
    const int* __restrict__ tile_nnz_offset,
    int sigma);

/**
 * @brief CSR5 SpMV kernel - Optimized for warp=32 (NVIDIA)
 */
template<typename FloatType, int BLOCK_SIZE, int Sigma>
__global__ void spmv_csr5_warp32_kernel(
    int numRows,
    int numCols,
    int nnz,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y,
    const int* __restrict__ tile_row_ptr,
    const int* __restrict__ tile_nnz_offset,
    int sigma);

/**
 * @brief CSR5 SpMV kernel - Optimized with warp-level aggregation
 *
 * Each warp processes one tile of sigma NNZ elements.
 * Warp-level aggregation reduces atomic operations.
 *
 * @param numRows Matrix rows
 * @param numCols Matrix columns
 * @param nnz Total NNZ
 * @param rowPtr Row pointers
 * @param colIdx Column indices
 * @param values Matrix values
 * @param x Input vector
 * @param y Output vector
 * @param tile_row_ptr Precomputed row index for each tile
 * @param tile_nnz_offset Offset within start row
 * @param sigma Tile size
 */
template<typename FloatType, int BLOCK_SIZE, int WarpSize, int Sigma>
__global__ void spmv_csr5_optimized_kernel(
    int numRows,
    int numCols,
    int nnz,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y,
    const int* __restrict__ tile_row_ptr,
    const int* __restrict__ tile_nnz_offset,
    int sigma);

/**
 * @brief Merge-based SpMV kernel
 *
 * Uses merge-path algorithm for load balancing.
 * Each warp processes a segment of the merge path.
 * Atomics only needed for partial rows at partition boundaries.
 *
 * @param numRows Matrix rows
 * @param numCols Matrix columns
 * @param nnz Total NNZ
 * @param rowPtr Row pointers
 * @param colIdx Column indices
 * @param values Matrix values
 * @param x Input vector
 * @param y Output vector
 * @param mergePathPos Merge path positions for partitions
 * @param numPartitions Number of partitions
 */
template<typename FloatType, int BLOCK_SIZE, int WarpSize>
__global__ void spmv_merge_based_kernel(
    int numRows,
    int numCols,
    int nnz,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y,
    const int* __restrict__ mergePathPos,
    int numPartitions);

// ==================== Host API ====================

/**
 * @brief Execute CSR5 SpMV
 *
 * @param matrix CSR5 matrix (must have tile metadata computed)
 * @param x Input vector (device pointer)
 * @param y Output vector (device pointer)
 * @param alpha Scaling factor for A*x
 * @param beta Scaling factor for y
 * @param opts Execution options
 * @return Status code
 */
template<typename FloatType>
spmv_status_t spmv_csr5(
    const CSR5Matrix<FloatType>& matrix,
    const FloatType* x,
    FloatType* y,
    FloatType alpha,
    FloatType beta,
    const spmv_opts_t& opts);

/**
 * @brief Execute optimized CSR5 SpMV with warp aggregation
 */
template<typename FloatType>
spmv_status_t spmv_csr5_optimized(
    const CSR5Matrix<FloatType>& matrix,
    const FloatType* x,
    FloatType* y,
    FloatType alpha,
    FloatType beta,
    const spmv_opts_t& opts);

/**
 * @brief Execute Merge-based SpMV
 *
 * @param matrix CSR matrix
 * @param x Input vector (device pointer)
 * @param y Output vector (device pointer)
 * @param stream CUDA stream
 * @return Status code
 */
template<typename FloatType>
spmv_status_t spmv_merge_based(
    const CSRMatrix<FloatType>& matrix,
    const FloatType* x,
    FloatType* y,
    cudaStream_t stream);

/**
 * @brief CSR5 preprocessing - compute tile metadata
 *
 * Must be called before spmv_csr5.
 * Computes tile_row_ptr and tile_nnz_offset arrays.
 *
 * @param csr Input CSR matrix
 * @param csr5 Output CSR5 matrix (tile metadata will be allocated)
 * @param sigma Tile size (0 = auto-select)
 * @param stream CUDA stream
 * @return Status code
 */
template<typename FloatType>
spmv_status_t csr5_preprocess(
    const CSRMatrix<FloatType>& csr,
    CSR5Matrix<FloatType>& csr5,
    int sigma,
    cudaStream_t stream);

} // namespace muxi_spmv

#endif // SPMV_CSR5_CUH_