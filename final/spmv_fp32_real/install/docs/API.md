# SPMV FP32 Library API Reference

## Overview

The SPMV FP32 library provides optimized Sparse Matrix-Vector Multiplication (SpMV) for FP32 precision sparse matrices.

## Error Codes

```c
typedef enum {
    SPMV_FP32_SUCCESS = 0,              // Success
    SPMV_FP32_ERROR_INVALID_INPUT,      // Invalid input parameters
    SPMV_FP32_ERROR_MEMORY,             // Memory allocation error
    SPMV_FP32_ERROR_CUDA,               // CUDA runtime error
    SPMV_FP32_ERROR_NOT_SUPPORTED,      // Feature not supported
    SPMV_FP32_ERROR_INTERNAL,           // Internal library error
    SPMV_FP32_ERROR_LICENSE_EXPIRED     // License expired
} spmv_fp32_status_t;
```

## Matrix Management

### spmv_fp32_create_matrix

Create CSR matrix from host arrays.

```c
spmv_fp32_status_t spmv_fp32_create_matrix(
    spmv_fp32_matrix_handle_t* handle,
    int numRows,
    int numCols,
    int nnz,
    const int* rowPtr,
    const int* colIdx,
    const float* values,
    const spmv_fp32_opts_t* opts);
```

**Parameters:**
- `handle`: Output matrix handle
- `numRows`: Number of rows
- `numCols`: Number of columns
- `nnz`: Number of non-zero elements
- `rowPtr`: Row pointer array (numRows+1 elements)
- `colIdx`: Column index array (nnz elements)
- `values`: Value array (nnz elements)
- `opts`: Execution options (NULL for defaults)

### spmv_fp32_create_matrix_device

Create CSR matrix from device arrays (zero-copy mode).

```c
spmv_fp32_status_t spmv_fp32_create_matrix_device(
    spmv_fp32_matrix_handle_t* handle,
    int numRows,
    int numCols,
    int nnz,
    const int* d_rowPtr,
    const int* d_colIdx,
    const float* d_values,
    const spmv_fp32_opts_t* opts);
```

### spmv_fp32_destroy_matrix

Destroy matrix handle and free resources.

```c
spmv_fp32_status_t spmv_fp32_destroy_matrix(spmv_fp32_matrix_handle_t handle);
```

## SpMV Execution

### spmv_fp32_execute

Execute SpMV: y = A * x

```c
spmv_fp32_status_t spmv_fp32_execute(
    spmv_fp32_matrix_handle_t handle,
    const float* x,
    float* y,
    const spmv_fp32_opts_t* opts,
    spmv_fp32_stats_t* stats);
```

### spmv_fp32_execute_scaled

Execute scaled SpMV: y = alpha * A * x

```c
spmv_fp32_status_t spmv_fp32_execute_scaled(
    spmv_fp32_matrix_handle_t handle,
    float alpha,
    const float* x,
    float* y,
    const spmv_fp32_opts_t* opts,
    spmv_fp32_stats_t* stats);
```

### spmv_fp32_execute_general

Execute general SpMV: y = alpha * A * x + beta * y

```c
spmv_fp32_status_t spmv_fp32_execute_general(
    spmv_fp32_matrix_handle_t handle,
    float alpha,
    float beta,
    const float* x,
    float* y,
    const spmv_fp32_opts_t* opts,
    spmv_fp32_stats_t* stats);
```

### spmv_fp32_execute_device

Execute SpMV with device vectors: d_y = A * d_x

```c
spmv_fp32_status_t spmv_fp32_execute_device(
    spmv_fp32_matrix_handle_t handle,
    const float* d_x,
    float* d_y,
    const spmv_fp32_opts_t* opts,
    spmv_fp32_stats_t* stats);
```

### spmv_fp32_execute_direct

One-shot SpMV execution (no handle needed).

```c
spmv_fp32_status_t spmv_fp32_execute_direct(
    int numRows,
    int nnz,
    const int* d_rowPtr,
    const int* d_colIdx,
    const float* d_values,
    const float* d_x,
    float* d_y,
    cudaStream_t stream);
```

**Note:** All pointers must be device pointers. User manages all memory.

### spmv_fp32_execute_direct_scaled

One-shot scaled SpMV execution.

```c
spmv_fp32_status_t spmv_fp32_execute_direct_scaled(
    float alpha,
    int numRows,
    int nnz,
    const int* d_rowPtr,
    const int* d_colIdx,
    const float* d_values,
    const float* d_x,
    float* d_y,
    cudaStream_t stream);
```

### spmv_fp32_execute_direct_general

One-shot general SpMV execution: d_y = alpha * A * d_x + beta * d_y

```c
spmv_fp32_status_t spmv_fp32_execute_direct_general(
    float alpha,
    float beta,
    int numRows,
    int nnz,
    const int* d_rowPtr,
    const int* d_colIdx,
    const float* d_values,
    const float* d_x,
    float* d_y,
    cudaStream_t stream);
```

## Utility Functions

### spmv_fp32_alloc_pinned

Allocate pinned memory for vectors (recommended for best performance).

```c
spmv_fp32_status_t spmv_fp32_alloc_pinned(void** ptr, size_t size);
```

### spmv_fp32_free_pinned

Free pinned memory.

```c
spmv_fp32_status_t spmv_fp32_free_pinned(void* ptr);
```

### spmv_fp32_get_device_info

Get GPU information.

```c
spmv_fp32_status_t spmv_fp32_get_device_info(
    int* warpSize,
    const char** name,
    size_t* memory);
```

### spmv_fp32_get_theoretical_bandwidth

Get theoretical memory bandwidth.

```c
spmv_fp32_status_t spmv_fp32_get_theoretical_bandwidth(double* bandwidth);
```

### License Functions

```c
spmv_fp32_status_t spmv_fp32_check_license(void);
const char* spmv_fp32_get_license_expiry(void);
```

## Execution Options

```c
typedef struct {
    cudaStream_t stream;      // CUDA stream (0 = default)
    int sync_after_exec;      // Sync after execution (default: 1)
    int benchmark_mode;       // Enable benchmarking (default: 0)
} spmv_fp32_opts_t;

static const spmv_fp32_opts_t SPMV_FP32_DEFAULT_OPTS = {
    .stream = 0,
    .sync_after_exec = 1,
    .benchmark_mode = 0
};

static const spmv_fp32_opts_t SPMV_FP32_BENCHMARK_OPTS = {
    .stream = 0,
    .sync_after_exec = 1,
    .benchmark_mode = 1
};
```

## Performance Statistics

```c
typedef struct {
    double kernel_time_ms;    // Kernel + memcpy time (ms)
    double bandwidth_gbps;    // Effective bandwidth (GB/s)
    double theoretical_bw;    // Peak bandwidth (GB/s)
    double utilization_pct;   // Bandwidth utilization (%)
    int warp_size;            // GPU warp size
    int optimal_tpr;          // Optimal threads/row
    const char* gpu_name;     // GPU name
} spmv_fp32_stats_t;
```

## Bandwidth Calculation

For FP32 SpMV:
- Bytes per nnz = 4 (value) + 4 (colIdx) + 4 (x[col]) = 12 bytes
- Bytes for y output = numRows * 4 bytes

```
Bandwidth (GB/s) = (nnz * 12 + numRows * 4) / (time_ms * 1e6)
```

---
Version: 1.0.0
Last updated: 2026-04-13