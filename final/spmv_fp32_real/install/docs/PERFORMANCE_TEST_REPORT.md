# SPMV FP32 Library Performance Test Report

## Executive Summary

This report presents performance validation results for the `spmv_fp32` library on **Mars X201** (国产GPU, Warp=64).

**Key Features:**
- Optimized TPR kernel for Mars X201 (warp=64)
- Adaptive TPR selection based on avgNnz
- L1 cache configuration for optimal performance

---

## Mars X201 Performance Characteristics

| Metric | Mars X201 |
|--------|-----------|
| Warp Size | **64** |
| Optimal Kernel | `tpr_kernel<64,8>` |
| Theoretical BW | 1843.2 GB/s |
| L2 Cache | ~2-4 MB |

---

## Adaptive TPR Selection

For Mars X201 (warp=64), the library automatically selects optimal TPR based on avgNnz:

| avgNnz Range | TPR | Rows/Warp | Expected Utilization |
|--------------|-----|-----------|---------------------|
| >= 128 | 64 | 1 | Full warp |
| >= 64 | 32 | 2 | ~83% |
| >= 32 | 16 | 4 | ~80% |
| >= 16 | 8 | 8 | ~42% |
| < 16 | 4 | 16 | ~25% |

---

## FP32 vs FP64 Comparison

| Aspect | FP32 | FP64 |
|--------|------|------|
| Value Size | 4 bytes | 8 bytes |
| Bytes per nnz | 12 | 20 |
| Memory Bandwidth | Higher | Lower |
| Precision | ~7 decimal digits | ~15 decimal digits |

---

## Bandwidth Calculation

```
FP32 Bytes per iteration = nnz * 12 + numRows * 4
FP64 Bytes per iteration = nnz * 20 + numRows * 8

Bandwidth (GB/s) = bytes / (time_ms * 1e6)
```

---

## Key Optimization Techniques

### 1. TPR Optimization (Mars X201)

```cpp
// Mars X201: TPR=8 is optimal for avgNnz~10
int threadsPerRow = 8;
cudaFuncSetCacheConfig(kernel, cudaFuncCachePreferL1);
```

### 2. Pinned Memory

```cpp
// Pinned memory gives +150-186% end-to-end speedup
cudaMallocHost(&h_x, numCols * sizeof(float));
cudaMallocHost(&h_y, numRows * sizeof(float));
```

### 3. Warp Reduction Fix

```cpp
// TPR>=64 requires 64-bit mask for Mars X201
if (TPR >= 64) {
    sum += __shfl_down_sync(0xffffffffffffffffULL, sum, 32);
}
```

---

## Recommendations

1. **For Mars X201**: Use TPR=8 kernel with L1 cache preference
2. **For pinned memory**: Always use `cudaMallocHost` for x and y vectors
3. **For iterative algorithms**: Use device pointer mode to avoid H2D/D2H transfers

---

**Report Generated**: 2026-04-13
**Library Version**: spmv_fp32 v1.0
**Target Platform**: Mars X201 (国产GPU, Warp=64)