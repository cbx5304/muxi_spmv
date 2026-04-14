# API Documentation

## Overview

The SPMV FP64 library provides a simple C-style API for optimized FP64 sparse matrix-vector multiplication.

## Types

### `spmv_fp64_matrix_handle_t`

Opaque handle to a CSR matrix. Created by `spmv_fp64_create_matrix()` and destroyed by `spmv_fp64_destroy_matrix()`.

### `spmv_fp64_status_t`

Error codes:

| Value | Description |
|-------|-------------|
| `SPMV_FP64_SUCCESS` | Operation successful |
| `SPMV_FP64_ERROR_INVALID_INPUT` | Invalid input parameters |
| `SPMV_FP64_ERROR_MEMORY` | Memory allocation error |
| `SPMV_FP64_ERROR_CUDA` | CUDA runtime error |
| `SPMV_FP64_ERROR_NOT_SUPPORTED` | Feature not supported |
| `SPMV_FP64_ERROR_INTERNAL` | Internal library error |

### `spmv_fp64_opts_t`

Execution options:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `stream` | `cudaStream_t` | 0 | CUDA stream for async execution |
| `use_pinned_memory` | `int` | 1 | Use pinned/host-locked memory |
| `sync_after_exec` | `int` | 1 | Synchronize after execution |
| `benchmark_mode` | `int` | 0 | Enable performance benchmarking |

### `spmv_fp64_stats_t`

Performance statistics (only filled when `benchmark_mode = 1`):

| Field | Type | Description |
|-------|------|-------------|
| `kernel_time_ms` | `double` | Kernel execution time (ms) |
| `bandwidth_gbps` | `double` | Effective bandwidth (GB/s) |
| `utilization_pct` | `double` | Bandwidth utilization (%) |
| `theoretical_bw` | `double` | Theoretical peak bandwidth (GB/s) |
| `warp_size` | `int` | GPU warp size (32 or 64) |
| `optimal_tpr` | `int` | Optimal threads-per-row |
| `gpu_name` | `const char*` | GPU device name |

## Matrix Management

### `spmv_fp64_create_matrix()`

```c
spmv_fp64_status_t spmv_fp64_create_matrix(
    spmv_fp64_matrix_handle_t* handle,
    int numRows,
    int numCols,
    int nnz,
    const int* rowPtr,
    const int* colIdx,
    const double* values,
    const spmv_fp64_opts_t* opts);
```

Create CSR matrix handle from host data.

**Parameters:**
- `handle`: Output matrix handle
- `numRows`: Number of rows
- `numCols`: Number of columns
- `nnz`: Number of non-zero elements
- `rowPtr`: Row pointer array (size: numRows + 1)
- `colIdx`: Column index array (size: nnz)
- `values`: Value array (size: nnz)
- `opts`: Execution options (NULL for defaults)

**Returns:** Status code

### `spmv_fp64_create_matrix_from_file()`

```c
spmv_fp64_status_t spmv_fp64_create_matrix_from_file(
    spmv_fp64_matrix_handle_t* handle,
    const char* filename,
    const spmv_fp64_opts_t* opts);
```

Load CSR matrix from Matrix Market (.mtx) file.

### `spmv_fp64_destroy_matrix()`

```c
spmv_fp64_status_t spmv_fp64_destroy_matrix(
    spmv_fp64_matrix_handle_t handle);
```

Destroy matrix handle and release all memory.

### `spmv_fp64_get_matrix_info()`

```c
spmv_fp64_status_t spmv_fp64_get_matrix_info(
    spmv_fp64_matrix_handle_t handle,
    int* numRows,
    int* numCols,
    int* nnz);
```

Get matrix dimensions.

## SpMV Execution

### `spmv_fp64_execute()`

```c
spmv_fp64_status_t spmv_fp64_execute(
    spmv_fp64_matrix_handle_t handle,
    const double* x,
    double* y,
    const spmv_fp64_opts_t* opts,
    spmv_fp64_stats_t* stats);
```

Execute SpMV: y = A * x

### `spmv_fp64_execute_scaled()`

```c
spmv_fp64_status_t spmv_fp64_execute_scaled(
    spmv_fp64_matrix_handle_t handle,
    double alpha,
    const double* x,
    double* y,
    const spmv_fp64_opts_t* opts,
    spmv_fp64_stats_t* stats);
```

Execute scaled SpMV: y = alpha * A * x

### `spmv_fp64_execute_general()`

```c
spmv_fp64_status_t spmv_fp64_execute_general(
    spmv_fp64_matrix_handle_t handle,
    double alpha,
    double beta,
    const double* x,
    double* y,
    const spmv_fp64_opts_t* opts,
    spmv_fp64_stats_t* stats);
```

Execute general SpMV: y = alpha * A * x + beta * y

## Utility Functions

### `spmv_fp64_alloc_pinned()`

```c
spmv_fp64_status_t spmv_fp64_alloc_pinned(void** ptr, size_t size);
```

Allocate pinned (page-locked) host memory.

**CRITICAL for performance!** Pinned memory provides:
- Mars X201: +186% end-to-end improvement
- RTX 4090: +152% end-to-end improvement

Use `cudaFreeHost()` or `spmv_fp64_free_pinned()` to free.

### `spmv_fp64_free_pinned()`

```c
spmv_fp64_status_t spmv_fp64_free_pinned(void* ptr);
```

Free pinned memory.

### `spmv_fp64_get_device_info()`

```c
spmv_fp64_status_t spmv_fp64_get_device_info(
    int* warpSize,
    const char** name,
    size_t* memory);
```

Get GPU device information.

### `spmv_fp64_get_theoretical_bandwidth()`

```c
spmv_fp64_status_t spmv_fp64_get_theoretical_bandwidth(double* bandwidth);
```

Get theoretical peak memory bandwidth.

### `spmv_fp64_get_error_string()`

```c
const char* spmv_fp64_get_error_string(spmv_fp64_status_t status);
```

Get human-readable error description.

### `spmv_fp64_get_version()`

```c
const char* spmv_fp64_get_version(void);
```

Get library version string.

## Default Options

```c
static const spmv_fp64_opts_t SPMV_FP64_DEFAULT_OPTS = {
    .stream = 0,
    .use_pinned_memory = 1,   // CRITICAL!
    .sync_after_exec = 1,
    .benchmark_mode = 0
};

static const spmv_fp64_opts_t SPMV_FP64_BENCHMARK_OPTS = {
    .stream = 0,
    .use_pinned_memory = 1,
    .sync_after_exec = 1,
    .benchmark_mode = 1       // Enable benchmarking
};
```

## Complete Example

```c
#include "spmv_fp64.h"
#include <stdio.h>

int main(int argc, char** argv) {
    // Initialize options
    spmv_fp64_opts_t opts = SPMV_FP64_BENCHMARK_OPTS;
    
    // Load matrix
    spmv_fp64_matrix_handle_t matrix;
    spmv_fp64_status_t status = spmv_fp64_create_matrix_from_file(
        &matrix, argv[1], &opts);
    if (status != SPMV_FP64_SUCCESS) {
        fprintf(stderr, "Error: %s\n", spmv_fp64_get_error_string(status));
        return 1;
    }
    
    // Get dimensions
    int numRows, numCols, nnz;
    spmv_fp64_get_matrix_info(matrix, &numRows, &numCols, &nnz);
    
    // Allocate pinned memory for vectors
    double* x, *y;
    spmv_fp64_alloc_pinned((void**)&x, numCols * sizeof(double));
    spmv_fp64_alloc_pinned((void**)&y, numRows * sizeof(double));
    
    // Initialize input vector
    for (int i = 0; i < numCols; i++) {
        x[i] = 1.0;
    }
    
    // Execute SpMV
    spmv_fp64_stats_t stats;
    status = spmv_fp64_execute(matrix, x, y, &opts, &stats);
    
    // Print results
    printf("Kernel time: %.3f ms\n", stats.kernel_time_ms);
    printf("Bandwidth: %.1f GB/s\n", stats.bandwidth_gbps);
    printf("Utilization: %.1f%%\n", stats.utilization_pct);
    
    // Cleanup
    spmv_fp64_free_pinned(x);
    spmv_fp64_free_pinned(y);
    spmv_fp64_destroy_matrix(matrix);
    
    return 0;
}
```