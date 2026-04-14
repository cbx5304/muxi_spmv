# SPMV FP64 Library

Optimized FP64 Sparse Matrix-Vector Multiplication library for real-world sparse matrices.

## Features

- **Auto GPU Detection**: Automatically detects warp size (64 for Mars X201, 32 for NVIDIA)
- **Optimal Kernels**: Uses best kernel for each GPU type
  - Mars X201: TPR=8 kernel
  - RTX 4090: __ldg kernel
- **Pinned Memory**: Critical for end-to-end performance (+150% to +186%)
- **Clean API**: Simple C-style interface

## Performance

### Best Case (Matrices with Column Index Locality)

| GPU        | Kernel Time | Bandwidth  | Utilization |
|------------|-------------|------------|-------------|
| Mars X201  | 0.420 ms    | 897 GB/s   | 48.7%       |
| RTX 4090   | 0.402 ms    | 907 GB/s   | 90.2%       |

*Results from matrices with column index locality (better L2 cache hit rate).*

### Typical Case (Random Column Distribution)

| GPU        | Kernel+H2D | Bandwidth  | Utilization |
|------------|------------|------------|-------------|
| Mars X201  | 0.58 ms    | 357 GB/s   | 19.4%       |
| RTX 4090   | 0.86 ms    | 326 GB/s   | 32.4%       |

*Results from matrices with random column index distribution (lower L2 cache hit rate).*

### Key Factors Affecting Performance

| Factor | High BW | Low BW | Impact |
|--------|---------|--------|--------|
| Column locality | Yes | No | L2 cache hit rate |
| avgNnz/row | >10 | <10 | Thread utilization |
| Matrix size | Large | Small | Kernel overhead |

**Note**: Performance depends heavily on matrix structure. Matrices with column index locality (e.g., band matrices, structured grids) achieve much higher bandwidth than random sparse matrices.

## Quick Start

### Prerequisites

- CUDA 11.6+ (NVIDIA) or cu-bridge (Mars X201)
- CMake 3.18+

### Build for NVIDIA RTX 4090

```bash
./scripts/build_rtx.sh
```

### Build for Mars X201

```bash
# Set environment first
export PATH=$PATH:$HOME/cu-bridge/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/cu-bridge/lib:$HOME/cu-bridge/lib64

./scripts/build_mars.sh
```

### Usage Example

```c
#include "spmv_fp64.h"

int main() {
    // Load matrix
    spmv_fp64_matrix_handle_t matrix;
    spmv_fp64_opts_t opts = SPMV_FP64_DEFAULT_OPTS;
    
    spmv_fp64_create_matrix_from_file(&matrix, "matrix.mtx", &opts);
    
    // Allocate pinned memory (CRITICAL!)
    double* x, *y;
    spmv_fp64_alloc_pinned((void**)&x, numCols * sizeof(double));
    spmv_fp64_alloc_pinned((void**)&y, numRows * sizeof(double));
    
    // Execute SpMV
    spmv_fp64_execute(matrix, x, y, &opts, NULL);
    
    // Cleanup
    spmv_fp64_free_pinned(x);
    spmv_fp64_free_pinned(y);
    spmv_fp64_destroy_matrix(matrix);
    
    return 0;
}
```

## API Modes

### 1. Host CSR Mode (Library manages all memory)

```c
// User provides host CSR data, library handles device allocation and copies
spmv_fp64_create_matrix(&handle, numRows, numCols, nnz, 
                         h_rowPtr, h_colIdx, h_values, opts);
spmv_fp64_execute(handle, h_x, h_y, opts, &stats);
```

### 2. Device CSR Mode (Zero-copy) ⭐ NEW

```c
// User provides device CSR pointers, library does NOT allocate or copy CSR data
spmv_fp64_create_matrix_device(&handle, numRows, numCols, nnz,
                                d_rowPtr, d_colIdx, d_values, opts);
spmv_fp64_execute_device(handle, d_x, d_y, opts, &stats);
```

### 3. Direct Execution Mode (No handle) ⭐ Recommended for single execution

```c
// One-shot execution, no handle creation, user manages all device memory
// Basic: y = A * x
spmv_fp64_execute_direct(numRows, nnz, d_rowPtr, d_colIdx, 
                         d_values, d_x, d_y, stream);

// Scaled: y = alpha * A * x
spmv_fp64_execute_direct_scaled(alpha, numRows, nnz, d_rowPtr, 
                                d_colIdx, d_values, d_x, d_y, stream);

// General: y = alpha * A * x + beta * y
spmv_fp64_execute_direct_general(alpha, beta, numRows, nnz, d_rowPtr,
                                 d_colIdx, d_values, d_x, d_y, stream);
```

## Documentation

- [BUILD.md](docs/BUILD.md) - Build instructions
- [API.md](docs/API.md) - API documentation
- [GPU_OPTIMIZATION.md](docs/GPU_OPTIMIZATION.md) - GPU-specific optimization notes

## Key Optimization Techniques

### Mars X201 (Warp=64)

1. **TPR=8**: 8 threads per row, 8 rows per warp
2. **PreferL1 Cache**: Critical for performance
3. **Pinned Memory**: +186% end-to-end improvement

### NVIDIA RTX 4090 (Warp=32)

1. **__ldg Cache Hints**: Uses texture/read-only cache
2. **Large L2 Cache**: 72MB can cache entire x vector
3. **Pinned Memory**: +152% end-to-end improvement

## License

See LICENSE file for details.

## Version History

- **1.0.0** (2026-04-12): Initial release with optimal kernels for both GPU types