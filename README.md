# SPMV FP64 Library

Optimized FP64 Sparse Matrix-Vector Multiplication library for real-world sparse matrices.

## Features

- **Auto GPU Detection**: Automatically detects warp size (64 for Mars X201, 32 for NVIDIA)
- **Optimal Kernels**: Uses best kernel for each GPU type
  - Mars X201: TPR=8 kernel (48.7% bandwidth utilization)
  - RTX 4090: __ldg kernel (88.8% bandwidth utilization)
- **Pinned Memory**: Critical for end-to-end performance (+150% to +186%)
- **Clean API**: Simple C-style interface

## Performance

| GPU        | Kernel Time | Bandwidth  | Utilization |
|------------|-------------|------------|-------------|
| Mars X201  | 0.420 ms    | 897 GB/s   | 48.7%       |
| RTX 4090   | 0.425 ms    | 893 GB/s   | 88.8%       |

Results from 10 real matrices (1.26M rows, avgNnzPerRow ≈ 10.71).

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