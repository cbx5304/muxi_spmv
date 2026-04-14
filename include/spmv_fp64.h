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
    SPMV_FP64_ERROR_INTERNAL
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
 * @brief Create CSR matrix from host arrays
 *
 * The library allocates device memory and copies data automatically.
 * Host arrays can be freed after this call returns successfully.
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
 */
spmv_fp64_status_t spmv_fp64_execute_general(
    spmv_fp64_matrix_handle_t handle,
    double alpha,
    double beta,
    const double* x,
    double* y,
    const spmv_fp64_opts_t* opts,
    spmv_fp64_stats_t* stats);

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

#ifdef __cplusplus
}
#endif

#endif /* SPMV_FP64_H_ */