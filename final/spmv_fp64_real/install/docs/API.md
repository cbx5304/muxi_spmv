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

### `spmv_fp64_create_matrix()` (Host CSR Mode)

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

Create CSR matrix handle from host data. The library allocates device memory for CSR data and internal vectors.

**Parameters:**
- `handle`: Output matrix handle
- `numRows`: Number of rows
- `numCols`: Number of columns
- `nnz`: Number of non-zero elements
- `rowPtr`: Host row pointer array (size: numRows + 1)
- `colIdx`: Host column index array (size: nnz)
- `values`: Host value array (size: nnz)
- `opts`: Execution options (NULL for defaults)

**Returns:** Status code

**Memory Ownership:** Library owns all host and device memory. Automatically freed by `spmv_fp64_destroy_matrix()`.

---

### `spmv_fp64_create_matrix_device()` (Zero-copy Mode) ⭐ NEW

```c
spmv_fp64_status_t spmv_fp64_create_matrix_device(
    spmv_fp64_matrix_handle_t* handle,
    int numRows,
    int numCols,
    int nnz,
    const int* d_rowPtr,
    const int* d_colIdx,
    const double* d_values,
    const spmv_fp64_opts_t* opts);
```

Create CSR matrix handle from **device pointers**. Zero-copy mode - the library does NOT allocate or copy CSR data.

**Parameters:**
- `handle`: Output matrix handle
- `numRows`: Number of rows
- `numCols`: Number of columns
- `nnz`: Number of non-zero elements
- `d_rowPtr`: **Device** row pointer array (size: numRows + 1)
- `d_colIdx`: **Device** column index array (size: nnz)
- `d_values`: **Device** value array (size: nnz)
- `opts`: Execution options (NULL for defaults)

**Returns:** Status code

**Memory Ownership:**
- **User owns device CSR data** (d_rowPtr, d_colIdx, d_values)
- Library does NOT allocate internal vectors (use `spmv_fp64_execute_device()` instead)
- `spmv_fp64_destroy_matrix()` only frees the handle, NOT your CSR data

**Use Case:** Best for users who already have CSR data on device and want to avoid redundant copies. Ideal for iterative algorithms where CSR data persists on device.

---

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

### Host-pointer Mode (Library manages H2D/D2H copies)

### `spmv_fp64_execute()`

```c
spmv_fp64_status_t spmv_fp64_execute(
    spmv_fp64_matrix_handle_t handle,
    const double* x,
    double* y,
    const spmv_fp64_opts_t* opts,
    spmv_fp64_stats_t* stats);
```

Execute SpMV: y = A * x (Host pointers for x and y)

**Note:** Only works for matrices created with `spmv_fp64_create_matrix()` or `spmv_fp64_create_matrix_from_file()`. Returns `SPMV_FP64_ERROR_NOT_SUPPORTED` for device-pointer mode matrices.

---

### Device-pointer Mode (No H2D/D2H copies) ⭐ NEW

### `spmv_fp64_execute_device()`

```c
spmv_fp64_status_t spmv_fp64_execute_device(
    spmv_fp64_matrix_handle_t handle,
    const double* d_x,
    double* d_y,
    const spmv_fp64_opts_t* opts,
    spmv_fp64_stats_t* stats);
```

Execute SpMV: y = A * x with **device pointers** for x and y.

**Parameters:**
- `handle`: Matrix handle (created by `spmv_fp64_create_matrix()` OR `spmv_fp64_create_matrix_device()`)
- `d_x`: **Device** input vector (numCols elements)
- `d_y`: **Device** output vector (numRows elements)
- `opts`: Execution options
- `stats`: Performance statistics (optional)

**Returns:** Status code

**Key Benefits:**
- No H2D/D2H copies - maximum performance for iterative algorithms
- Works with both host CSR and device CSR matrix handles
- User manages x/y device memory lifecycle

---

### `spmv_fp64_execute_device_general()` ⭐ NEW

```c
spmv_fp64_status_t spmv_fp64_execute_device_general(
    spmv_fp64_matrix_handle_t handle,
    double alpha,
    double beta,
    const double* d_x,
    double* d_y,
    const spmv_fp64_opts_t* opts,
    spmv_fp64_stats_t* stats);
```

Execute general SpMV: y = alpha * A * x + beta * y with **device pointers**.

---

### One-shot Execution (No handle needed) ⭐ Recommended for single execution

### `spmv_fp64_execute_direct()`

```c
spmv_fp64_status_t spmv_fp64_execute_direct(
    int numRows,
    int nnz,
    const int* d_rowPtr,
    const int* d_colIdx,
    const double* d_values,
    const double* d_x,
    double* d_y,
    cudaStream_t stream);
```

Execute SpMV directly with device CSR data and device vectors - **no matrix handle needed**.

**Parameters:**
- `numRows`: Number of rows
- `nnz`: Number of non-zero elements
- `d_rowPtr`: Device row pointer array
- `d_colIdx`: Device column index array
- `d_values`: Device value array
- `d_x`: Device input vector
- `d_y`: Device output vector
- `stream`: CUDA stream (0 for default)

**Returns:** Status code

**Use Case:** Single execution without handle creation overhead. Best for one-off SpMV operations.

---

### `spmv_fp64_execute_direct_scaled()` ⭐ NEW

```c
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
```

Execute scaled SpMV: **y = alpha * A * x**, no matrix handle needed.

**Parameters:**
- `alpha`: Scaling factor for matrix product
- Other parameters same as `spmv_fp64_execute_direct()`

**Returns:** Status code

---

### `spmv_fp64_execute_direct_general()` ⭐ NEW

```c
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
```

Execute general SpMV: **y = alpha * A * x + beta * y**, no matrix handle needed.

**Parameters:**
- `alpha`: Scaling factor for matrix product
- `beta`: Scaling factor for existing y vector
- Other parameters same as `spmv_fp64_execute_direct()`

**Returns:** Status code

**Note:** When `beta != 0`, `d_y` must contain valid input data (as y_old).

---

### Direct Execution Mode Use Cases

| API | Operation | Use Case |
|-----|-----------|----------|
| `execute_direct` | y = A * x | Basic SpMV, single execution |
| `execute_direct_scaled` | y = α * A * x | Need result scaling |
| `execute_direct_general` | y = α*A*x + β*y | Axpy operations, iterative solvers |

**Memory Management:** User manages all device memory (CSR data + vectors). Library allocates no memory.

---

### `spmv_fp64_execute_scaled()` (with handle)

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

### Example 1: Host CSR Mode (Original API)

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

### Example 2: Device CSR Mode (Zero-copy) ⭐ NEW

```c
#include "spmv_fp64.h"
#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    // User-provided CSR data on device
    int numRows = 10000;
    int nnz = 100000;
    
    // Allocate and fill device CSR arrays (user manages)
    int* d_rowPtr, *d_colIdx;
    double* d_values;
    cudaMalloc(&d_rowPtr, (numRows + 1) * sizeof(int));
    cudaMalloc(&d_colIdx, nnz * sizeof(int));
    cudaMalloc(&d_values, nnz * sizeof(double));
    // ... fill with your data ...
    
    // Create matrix handle (zero-copy - library doesn't copy)
    spmv_fp64_matrix_handle_t matrix;
    spmv_fp64_create_matrix_device(&matrix, numRows, numRows, nnz,
                                    d_rowPtr, d_colIdx, d_values, NULL);
    
    // Allocate device vectors (user manages)
    double* d_x, *d_y;
    cudaMalloc(&d_x, numRows * sizeof(double));
    cudaMalloc(&d_y, numRows * sizeof(double));
    // ... initialize d_x ...
    
    // Execute SpMV (no H2D/D2H copies - maximum performance)
    spmv_fp64_opts_t opts = SPMV_FP64_BENCHMARK_OPTS;
    spmv_fp64_stats_t stats;
    spmv_fp64_execute_device(matrix, d_x, d_y, &opts, &stats);
    
    printf("Kernel time: %.3f ms\n", stats.kernel_time_ms);
    printf("Bandwidth: %.1f GB/s\n", stats.bandwidth_gbps);
    
    // Cleanup - library only frees handle, user frees device memory
    spmv_fp64_destroy_matrix(matrix);
    cudaFree(d_rowPtr);
    cudaFree(d_colIdx);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);
    
    return 0;
}
```

### Example 3: One-shot Direct Execution ⭐ NEW

```c
#include "spmv_fp64.h"
#include <cuda_runtime.h>

int main() {
    int numRows = 10000;
    int nnz = 100000;
    
    // All data must be on device
    int* d_rowPtr, *d_colIdx;
    double* d_values, *d_x, *d_y;
    cudaMalloc(&d_rowPtr, (numRows + 1) * sizeof(int));
    cudaMalloc(&d_colIdx, nnz * sizeof(int));
    cudaMalloc(&d_values, nnz * sizeof(double));
    cudaMalloc(&d_x, numRows * sizeof(double));
    cudaMalloc(&d_y, numRows * sizeof(double));
    // ... fill with your data ...
    
    // Execute directly - no handle creation needed
    spmv_fp64_execute_direct(numRows, nnz, d_rowPtr, d_colIdx, 
                             d_values, d_x, d_y, 0);
    cudaDeviceSynchronize();
    
    // Cleanup - user manages all memory
    cudaFree(d_rowPtr);
    cudaFree(d_colIdx);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);
    
    return 0;
}
```
```