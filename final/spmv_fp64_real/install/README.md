# SPMV FP64 Library - Trial Version

## License

**Trial License Expires: 2026-05-07**

This is a trial version of the SPMV FP64 library. The library will stop working after the expiration date. Please contact the vendor for a permanent license.

The license check is performed automatically when:
- Calling `spmv_fp64_check_license()`
- Creating a matrix with `spmv_fp64_create_matrix()` or `spmv_fp64_create_matrix_device()`
- Using direct execution with `spmv_fp64_execute_direct()`

If the license is expired, the library will return `SPMV_FP64_ERROR_LICENSE_EXPIRED`.

## Contents

```
install/
├── include/
│   └── spmv_fp64.h          # Library header file
├── lib/
│   ├── libspmv_fp64.so      # Shared library (Linux)
│   └── cmake/
│       └── spmv_fp64/       # CMake configuration files
├── docs/
│   ├── API.md               # API documentation (English)
│   └── PERFORMANCE_TEST_REPORT.md  # Performance report (English)
├── docs_zh/
│   ├── API.md               # API documentation (Chinese)
│   └── PERFORMANCE_TEST_REPORT.md  # Performance report (Chinese)
└── README.md                # This file
```

## Usage

### Check License

```cpp
#include "spmv_fp64.h"

// Check license before use
spmv_fp64_status_t status = spmv_fp64_check_license();
if (status == SPMV_FP64_ERROR_LICENSE_EXPIRED) {
    printf("License expired on %s!\n", spmv_fp64_get_license_expiry());
    return 1;
}
printf("License valid until %s\n", spmv_fp64_get_license_expiry());
```

### Basic Example

```cpp
#include "spmv_fp64.h"

int main() {
    // License is checked automatically during create_matrix
    
    // Create matrix from host arrays
    spmv_fp64_matrix_handle_t matrix;
    spmv_fp64_status_t status = spmv_fp64_create_matrix(
        &matrix, numRows, numCols, nnz,
        h_rowPtr, h_colIdx, h_values, NULL);
    
    if (status == SPMV_FP64_ERROR_LICENSE_EXPIRED) {
        printf("License expired!\n");
        return 1;
    }
    
    // Execute SpMV
    spmv_fp64_execute(matrix, h_x, h_y, NULL, NULL);
    
    // Cleanup
    spmv_fp64_destroy_matrix(matrix);
    return 0;
}
```

### Compilation

```bash
# Using shared library
nvcc -I./install/include -L./install/lib -lspmv_fp64 your_code.cu -o your_app

# Run with library path
LD_LIBRARY_PATH=./install/lib:$LD_LIBRARY_PATH ./your_app

# Or using CMake
cmake -DSPMV_FP64_ROOT=./install ..
```

## Supported Platforms

| Platform | Warp Size | Optimal Kernel |
|----------|-----------|----------------|
| NVIDIA RTX 4090 | 32 | `ldg_kernel` |
| Mars X201 (国产GPU) | 64 | `tpr_kernel<64,8>` |

## Performance

| GPU | Kernel Time | Bandwidth | Utilization |
|-----|-------------|-----------|-------------|
| RTX 4090 | 0.355 ms | 584.7 GB/s | 58.0% |
| Mars X201 | 0.293 ms | 704.6 GB/s | 38.2% |

Test conditions: 1M rows, avgNnz=10, FP64 precision.

## API Summary

| Function | Description |
|----------|-------------|
| `spmv_fp64_check_license()` | Check license validity |
| `spmv_fp64_get_license_expiry()` | Get license expiration date string |
| `spmv_fp64_create_matrix()` | Create matrix from host arrays |
| `spmv_fp64_create_matrix_device()` | Create matrix from device arrays (zero-copy) |
| `spmv_fp64_execute()` | Execute SpMV: y = A * x |
| `spmv_fp64_execute_direct()` | One-shot SpMV without handle |
| `spmv_fp64_destroy_matrix()` | Destroy matrix handle |

## Contact

For permanent license or support, please contact the vendor.

---
Version: 1.0.0
License Expiry: 2026-05-07