# SPMV FP32 Library for Mars X201

## Overview

Optimized FP32 Sparse Matrix-Vector Multiplication library for **Mars X201** GPU (Warp=64).

## Performance Results

### Real Matrix Performance (avgNnz=62.36, 3M rows)

Test matrix: spMatMultiDnVec_9.bin (3122329 rows, 194704259 nnz)

**Effective Bandwidth Formula**: `bytes = nnz * 12 + rows * 8` (per nnz: values+colIdx+x[colIdx], per row: rowPtr+y)

| Library | Latency | Effective BW | Utilization | Speedup |
|---------|---------|--------------|-------------|---------|
| cuSPARSE CSR_ALG2 | 3.289 ms | 720 GB/s | 39% | 1.0x |
| nctigpu | 3.277 ms | 722 GB/s | 39% | 1.0x |
| **fp32 optimized** | **2.065 ms** | **1138 GB/s** | **62%** | **1.59x** |

**Performance Ceiling**: After exhaustive optimization testing (TPR variants, cache configs, block sizes, ILP, register cache, prefetch strategies), **62% utilization (1138 GB/s) appears to be the practical limit** for this matrix type on Mars X201. The bottleneck is fundamental: random x-vector access through unpredictable colIdx values.

### TPR Optimization Analysis (avgNnz=62.4)

| TPR | Latency | Bandwidth | Utilization |
|-----|---------|-----------|-------------|
| TPR=4 | 5.200 ms | 452 GB/s | 24.5% |
| TPR=8 | 2.733 ms | 859 GB/s | 46.6% |
| **TPR=16** | **2.065 ms** | **1137 GB/s** | **61.7%** ✓ |
| TPR=32 | 2.130 ms | 1103 GB/s | 59.8% |
| TPR=64 | 2.974 ms | 790 GB/s | 42.9% |

### Optimization Techniques Tested (All Showed No Improvement)

| Technique | Result | Reason |
|-----------|--------|--------|
| __ldg cache hints | No improvement | Mars X201 doesn't benefit from __ldg for FP32 |
| Block size 128/512 | Same as 256 | Memory-bound kernel, block size has minimal effect |
| Cache config (L1/Shared) | No difference | Random access pattern bypasses cache benefits |
| ILP dual accumulators | No improvement | Memory latency dominates, compute ILP ineffective |
| Warp-level gather | Same as baseline | Random colIdx prevents coalescing |
| Register cache | -50% performance | Cache search overhead exceeds benefits |
| Dual-row processing | -4% performance | Increased register pressure |

### Why 62% is the Ceiling

1. **Random x-vector access**: colIdx values are unpredictable, causing scattered memory reads
2. **L2 cache limitation**: Mars X201's 8MB L2 cannot cache the 12MB x vector effectively
3. **Memory system design**: Warp=64 architecture has different memory latency characteristics than NVIDIA's Warp=32

### Adaptive TPR Algorithm

| avgNnz Range | Optimal TPR | Rows/Warp | Expected Utilization |
|--------------|-------------|-----------|---------------------|
| < 16 | 4 | 16 | ~25% |
| 16-40 | 8 | 8 | ~42% |
| **40-80** | **16** | **4** | **~62%** ✓ |
| 80-128 | 32 | 2 | ~60% |
| >= 128 | 64 | 1 | ~43% |

## Key Optimization Techniques

1. **Adaptive TPR**: Select optimal threads-per-row based on avgNnz
2. **TPR=16 for avgNnz~62**: 4 rows/warp, 61.7% bandwidth utilization
3. **L1 Cache Preference**: `cudaFuncCachePreferL1` for random access
4. **Pinned Memory**: `cudaMallocHost` for faster PCIe transfer

## Usage

```cpp
#include "spmv_fp32.h"

// Direct execution (one-shot, device pointers)
spmv_fp32_execute_direct(numRows, nnz, d_rowPtr, d_colIdx, d_values, d_x, d_y, 0);

// With handle (for iterative use)
spmv_fp32_matrix_handle_t handle;
spmv_fp32_create_matrix_device(&handle, numRows, numCols, nnz, 
    d_rowPtr, d_colIdx, d_values, NULL);

// Multiple iterations (kernel-only, no H2D/D2H)
for (int i = 0; i < iterations; i++) {
    spmv_fp32_execute_device(handle, d_x, d_y, NULL, NULL);
}

spmv_fp32_destroy_matrix(handle);
```

## Installation

```bash
# Include path
-I/path/to/install_X201/include

# Library path
-L/path/to/install_X201/lib -lspmv_fp32

# Runtime
export LD_LIBRARY_PATH=/path/to/install_X201/lib:$LD_LIBRARY_PATH
```

## Compilation

```bash
# Using cu-bridge (Mars X201)
export PATH=$HOME/cu-bridge/bin:$HOME/cu-bridge/compiler/bin:$PATH
export LD_LIBRARY_PATH=$HOME/cu-bridge/lib:$HOME/cu-bridge/lib64:$LD_LIBRARY_PATH

pre_make nvcc -O3 -I./include your_program.cu ./lib/libspmv_fp32.so -o your_program
```

## Test Matrix Characteristics

| Matrix | Rows | NNZ | avgNnz | Format |
|--------|------|-----|--------|--------|
| spMatMultiDnVec_* | 3122329 | 194704259 | 62.36 | FP32 real |

## Files

| File | Description |
|------|-------------|
| lib/libspmv_fp32.so | Shared library |
| include/spmv_fp32.h | Header file |
| include/spmv_fp32_impl.cuh | Implementation details |

## License

Trial version valid until: 2026-05-07

---
Version: 1.0.1
Platform: Mars X201 (Warp=64)
Date: 2026-04-14