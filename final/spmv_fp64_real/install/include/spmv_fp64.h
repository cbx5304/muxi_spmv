/**
 * @file spmv_fp64.h
 * @brief Optimized FP64 SpMV Library for Real Sparse Matrices
 *
 * This library provides highly optimized SpMV (Sparse Matrix-Vector Multiplication)
 * implementations for FP64 precision, specifically tuned for real-world sparse matrices.
 *
 * Key Features:
 * - Auto-detects GPU type (Mars X201 warp=64 vs NVIDIA warp=32)
 * - Optimal TPR=8 kernel for Mars X201 (48.7% bandwidth utilization)
 * - Optimal __ldg kernel for NVIDIA RTX 4090 (88.8% bandwidth utilization)
 * - Automatic device memory management for vectors
 * - Pinned memory support for maximum end-to-end performance
 *
 * Performance Results (10 real matrices average):
 * | GPU        | Kernel Time | Bandwidth  | Utilization |
 * |------------|-------------|------------|-------------|
 * | Mars X201  | 0.420 ms    | 897 GB/s   | 48.7%       |
 * | RTX 4090   | 0.425 ms    | 893 GB/s   | 88.8%       |
 *
 * @version 1.0.0
 * @date 2026-04-12
 */

#ifndef SPMV_FP64_H_
#define SPMV_FP64_H_

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Library version
 */
#define SPMV_FP64_VERSION_MAJOR 1
#define SPMV_FP64_VERSION_MINOR 0
#define SPMV_FP64_VERSION_PATCH 0
#define SPMV_FP64_VERSION_STRING "1.0.0"

/**
 * @brief Error codes
 */
typedef enum {
    SPMV_FP64_SUCCESS = 0,
    SPMV_FP64_ERROR_INVALID_INPUT,
    SPMV_FP64_ERROR_MEMORY,
    SPMV_FP64_ERROR_CUDA,
    SPMV_FP64_ERROR_NOT_SUPPORTED,
    SPMV_FP64_ERROR_INTERNAL,
    SPMV_FP64_ERROR_LICENSE_EXPIRED  ///< License expired (trial version)
} spmv_fp64_status_t;

/**
 * @brief CSR matrix handle (opaque)
 */
typedef struct spmv_fp64_matrix_t* spmv_fp64_matrix_handle_t;

/**
 * @brief Execution options
 */
typedef struct {
    cudaStream_t stream;      ///< CUDA stream (0 = default)
    int sync_after_exec;      ///< Sync after execution (default: 1)
    int benchmark_mode;       ///< Enable benchmarking (default: 0)
} spmv_fp64_opts_t;

/**
 * @brief Default options
 */
static const spmv_fp64_opts_t SPMV_FP64_DEFAULT_OPTS = {
    .stream = 0,
    .sync_after_exec = 1,
    .benchmark_mode = 0
};

/**
 * @brief Benchmark options
 */
static const spmv_fp64_opts_t SPMV_FP64_BENCHMARK_OPTS = {
    .stream = 0,
    .sync_after_exec = 1,
    .benchmark_mode = 1
};

/**
 * @brief Performance statistics
 */
typedef struct {
    double kernel_time_ms;    ///< Kernel + memcpy time (ms)
    double bandwidth_gbps;    ///< Effective bandwidth (GB/s)
    double theoretical_bw;    ///< Peak bandwidth (GB/s)
    double utilization_pct;   ///< Bandwidth utilization (%)
    int warp_size;            ///< GPU warp size
    int optimal_tpr;          ///< Optimal threads/row
    const char* gpu_name;     ///< GPU name
} spmv_fp64_stats_t;

// ==================== Matrix Management ====================

/**
 * @brief Create CSR matrix from host arrays (copies data to device)
 *
 * The library allocates device memory and copies data automatically.
 * Host arrays can be freed after this call returns successfully.
 *
 * @note This is the original API - best for simple usage scenarios
 */
spmv_fp64_status_t spmv_fp64_create_matrix(
    spmv_fp64_matrix_handle_t* handle,
    int numRows,
    int numCols,
    int nnz,
    const int* rowPtr,
    const int* colIdx,
    const double* values,
    const spmv_fp64_opts_t* opts);

/**
 * @brief Create CSR matrix from device arrays (zero-copy mode)
 *
 * User provides device pointers directly. Library does NOT copy or allocate
 * device memory for CSR data. User must ensure device data remains valid
 * during all subsequent execute calls.
 *
 * @param d_rowPtr  Device pointer to rowPtr array (numRows+1 elements)
 * @param d_colIdx  Device pointer to colIdx array (nnz elements)
 * @param d_values  Device pointer to values array (nnz elements)
 *
 * @note Best for users who already have CSR data on device and want
 *       maximum performance (avoid redundant copies)
 * @note User manages device CSR memory lifecycle
 */
spmv_fp64_status_t spmv_fp64_create_matrix_device(
    spmv_fp64_matrix_handle_t* handle,
    int numRows,
    int numCols,
    int nnz,
    const int* d_rowPtr,
    const int* d_colIdx,
    const double* d_values,
    const spmv_fp64_opts_t* opts);

/**
 * @brief Create CSR matrix from MTX file
 */
spmv_fp64_status_t spmv_fp64_create_matrix_from_file(
    spmv_fp64_matrix_handle_t* handle,
    const char* filename,
    const spmv_fp64_opts_t* opts);

/**
 * @brief Destroy matrix handle
 */
spmv_fp64_status_t spmv_fp64_destroy_matrix(
    spmv_fp64_matrix_handle_t handle);

/**
 * @brief Get matrix dimensions
 */
spmv_fp64_status_t spmv_fp64_get_matrix_info(
    spmv_fp64_matrix_handle_t handle,
    int* numRows,
    int* numCols,
    int* nnz);

// ==================== SpMV Execution ====================

/**
 * @brief Execute SpMV: y = A * x
 *
 * The library handles device memory for vectors internally.
 * x and y can be host pointers (pinned memory recommended).
 */
spmv_fp64_status_t spmv_fp64_execute(
    spmv_fp64_matrix_handle_t handle,
    const double* x,
    double* y,
    const spmv_fp64_opts_t* opts,
    spmv_fp64_stats_t* stats);

/**
 * @brief Execute scaled SpMV: y = alpha * A * x
 */
spmv_fp64_status_t spmv_fp64_execute_scaled(
    spmv_fp64_matrix_handle_t handle,
    double alpha,
    const double* x,
    double* y,
    const spmv_fp64_opts_t* opts,
    spmv_fp64_stats_t* stats);

/**
 * @brief Execute general SpMV: y = alpha * A * x + beta * y
 *
 * @note x and y are host pointers, library handles H2D/D2H copies
 */
spmv_fp64_status_t spmv_fp64_execute_general(
    spmv_fp64_matrix_handle_t handle,
    double alpha,
    double beta,
    const double* x,
    double* y,
    const spmv_fp64_opts_t* opts,
    spmv_fp64_stats_t* stats);

/**
 * @brief Execute SpMV with device vectors: d_y = A * d_x
 *
 * User provides device pointers for x and y vectors. Library does NOT
 * perform H2D/D2H copies - user manages vector memory.
 *
 * @param d_x    Device pointer to input vector (numCols elements)
 * @param d_y    Device pointer to output vector (numRows elements)
 *
 * @note Best for iterative algorithms where x/y stay on device
 * @note Must call cudaDeviceSynchronize() or use stream to ensure completion
 */
spmv_fp64_status_t spmv_fp64_execute_device(
    spmv_fp64_matrix_handle_t handle,
    const double* d_x,
    double* d_y,
    const spmv_fp64_opts_t* opts,
    spmv_fp64_stats_t* stats);

/**
 * @brief Execute SpMV with device vectors and scaling: d_y = alpha * A * d_x + beta * d_y
 */
spmv_fp64_status_t spmv_fp64_execute_device_general(
    spmv_fp64_matrix_handle_t handle,
    double alpha,
    double beta,
    const double* d_x,
    double* d_y,
    const spmv_fp64_opts_t* opts,
    spmv_fp64_stats_t* stats);

/**
 * @brief One-shot SpMV execution (no handle needed)
 *
 * Execute SpMV directly with device CSR data and device vectors.
 * No matrix handle creation required - suitable for single execution.
 *
 * @param numRows  Number of rows
 * @param d_rowPtr Device rowPtr array (numRows+1 elements)
 * @param d_colIdx Device colIdx array
 * @param d_values Device values array
 * @param d_x      Device input vector
 * @param d_y      Device output vector
 * @param stream   CUDA stream (0 for default)
 *
 * @note All pointers must be device pointers
 * @note User manages all memory
 */
spmv_fp64_status_t spmv_fp64_execute_direct(
    int numRows,
    int nnz,
    const int* d_rowPtr,
    const int* d_colIdx,
    const double* d_values,
    const double* d_x,
    double* d_y,
    cudaStream_t stream);

/**
 * @brief One-shot scaled SpMV execution (no handle needed)
 *
 * Execute: d_y = alpha * A * d_x
 *
 * @param alpha    Scaling factor for matrix product
 * @param numRows  Number of rows
 * @param nnz      Number of non-zero elements
 * @param d_rowPtr Device rowPtr array (numRows+1 elements)
 * @param d_colIdx Device colIdx array
 * @param d_values Device values array
 * @param d_x      Device input vector
 * @param d_y      Device output vector
 * @param stream   CUDA stream (0 for default)
 *
 * @note All pointers must be device pointers
 * @note User manages all memory
 */
spmv_fp64_status_t spmv_fp64_execute_direct_scaled(
    double alpha,
    int numRows,
    int nnz,
    const int* d_rowPtr,
    const int* d_colIdx,
    const double* d_values,
    const double* d_x,
    double* d_y,
    cudaStream_t stream);

/**
 * @brief One-shot general SpMV execution (no handle needed)
 *
 * Execute: d_y = alpha * A * d_x + beta * d_y
 *
 * @param alpha    Scaling factor for matrix product
 * @param beta     Scaling factor for existing y vector
 * @param numRows  Number of rows
 * @param nnz      Number of non-zero elements
 * @param d_rowPtr Device rowPtr array (numRows+1 elements)
 * @param d_colIdx Device colIdx array
 * @param d_values Device values array
 * @param d_x      Device input vector
 * @param d_y      Device output vector (also input when beta != 0)
 * @param stream   CUDA stream (0 for default)
 *
 * @note All pointers must be device pointers
 * @note When beta != 0, d_y must contain valid input data
 * @note User manages all memory
 */
spmv_fp64_status_t spmv_fp64_execute_direct_general(
    double alpha,
    double beta,
    int numRows,
    int nnz,
    const int* d_rowPtr,
    const int* d_colIdx,
    const double* d_values,
    const double* d_x,
    double* d_y,
    cudaStream_t stream);

// ==================== Utility ====================

/**
 * @brief Allocate pinned memory
 *
 * Recommended for x and y vectors (+150-186% end-to-end speedup)
 */
spmv_fp64_status_t spmv_fp64_alloc_pinned(void** ptr, size_t size);

/**
 * @brief Free pinned memory
 */
spmv_fp64_status_t spmv_fp64_free_pinned(void* ptr);

/**
 * @brief Get GPU info
 */
spmv_fp64_status_t spmv_fp64_get_device_info(
    int* warpSize,
    const char** name,
    size_t* memory);

/**
 * @brief Get theoretical bandwidth
 */
spmv_fp64_status_t spmv_fp64_get_theoretical_bandwidth(double* bandwidth);

/**
 * @brief Get error string
 */
const char* spmv_fp64_get_error_string(spmv_fp64_status_t status);

/**
 * @brief Get version
 */
const char* spmv_fp64_get_version(void);

/**
 * @brief Check license status
 *
 * @return SPMV_FP64_SUCCESS if license is valid, SPMV_FP64_ERROR_LICENSE_EXPIRED if expired
 */
spmv_fp64_status_t spmv_fp64_check_license(void);

/**
 * @brief Get license expiration date string
 */
const char* spmv_fp64_get_license_expiry(void);

#ifdef __cplusplus
}
#endif

#endif /* SPMV_FP64_H_ */