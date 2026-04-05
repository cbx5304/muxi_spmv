/**
 * @file spmv_sparse_optimized.cuh
 * @brief Optimized SpMV kernels for very sparse matrices (avgNnz < 10)
 */

#ifndef SPMV_SPARSE_OPTIMIZED_CUH_
#define SPMV_SPARSE_OPTIMIZED_CUH_

#include "utils/common.h"
#include "formats/sparse_formats.h"

namespace muxi_spmv {

/**
 * @brief Select optimal SpMV kernel based on matrix sparsity
 *
 * Strategy:
 * - avgNnz <= 4: Light kernel (each thread processes multiple rows)
 * - avgNnz <= 8: Virtual warp kernel (warp size = 32 for Mars X201)
 * - avgNnz > 8: Standard merge-based kernel
 *
 * @param matrix CSR matrix
 * @param x Input vector
 * @param y Output vector
 * @param stream CUDA stream
 * @return Status code
 */
template<typename FloatType>
spmv_status_t spmv_sparse_optimized(
    const CSRMatrix<FloatType>& matrix,
    const FloatType* x,
    FloatType* y,
    cudaStream_t stream);

} // namespace muxi_spmv

#endif // SPMV_SPARSE_OPTIMIZED_CUH_