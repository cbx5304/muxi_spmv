/**
 * @file spmv_api.h
 * @brief Public API for SpMV library
 */

#ifndef SPMV_API_H_
#define SPMV_API_H_

#include "utils/common.h"
#include "formats/sparse_formats.h"
#include "spmv/csr/spmv_csr.cuh"

namespace muxi_spmv {

/**
 * @brief SpMV handle for stateful operations
 */
typedef struct spmv_handle_t {
    int deviceId;
    cudaStream_t stream;
    spmv_opts_t opts;

    // Cached device properties
    int warpSize;
    int smCount;
    int maxThreadsPerBlock;
    size_t sharedMemPerSM;
} spmv_handle_t;

/**
 * @brief Create SpMV handle
 */
spmv_status_t spmv_create_handle(spmv_handle_t** handle);

/**
 * @brief Destroy SpMV handle
 */
spmv_status_t spmv_destroy_handle(spmv_handle_t* handle);

/**
 * @brief Set CUDA stream for handle
 */
spmv_status_t spmv_set_stream(spmv_handle_t* handle, cudaStream_t stream);

/**
 * @brief Execute SpMV operation y = alpha * A * x + beta * y
 *
 * @tparam FloatType Floating point type (float, double)
 * @param handle SpMV handle
 * @param format Matrix format (CSR, COO, etc.)
 * @param matrix Pointer to matrix structure
 * @param x Input vector
 * @param y Output vector (also input if beta != 0)
 * @param alpha Scaling factor for A*x
 * @param beta Scaling factor for y
 */
template<typename FloatType>
spmv_status_t spmv_exec(
    spmv_handle_t* handle,
    spmv_format_t format,
    const void* matrix,
    const FloatType* x,
    FloatType* y,
    FloatType alpha = 1.0,
    FloatType beta = 0.0);

/**
 * @brief Execute SpMV with transpose: y = alpha * A^T * x + beta * y
 */
template<typename FloatType>
spmv_status_t spmv_exec_transpose(
    spmv_handle_t* handle,
    spmv_format_t format,
    const void* matrix,
    const FloatType* x,
    FloatType* y,
    FloatType alpha = 1.0,
    FloatType beta = 0.0);

// Template function API for direct use (CSR format)

template<typename FloatType>
inline spmv_status_t spmv_csr_exec(
    const CSRMatrix<FloatType>& A,
    const FloatType* x,
    FloatType* y,
    FloatType alpha = 1.0,
    FloatType beta = 0.0,
    const spmv_opts_t& opts = spmv_default_opts())
{
    return spmv_csr<FloatType>(A, x, y, alpha, beta, opts);
}

} // namespace muxi_spmv

#endif // SPMV_API_H_