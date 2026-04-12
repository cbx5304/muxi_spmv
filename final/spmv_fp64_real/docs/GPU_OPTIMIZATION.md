# GPU Optimization Notes

## Overview

This document explains the different optimization strategies used for Mars X201 (warp=64) and NVIDIA RTX 4090 (warp=32).

## Hardware Differences

| Parameter | Mars X201 | RTX 4090 | Impact |
|-----------|-----------|----------|--------|
| **Warp Size** | **64** | 32 | Most critical difference |
| **L2 Cache** | **~2-4MB** | 72MB | Determines performance limit |
| Theoretical BW | 1843 GB/s | 1008 GB/s | Mars higher but harder to achieve |
| SM Count | 104 | 128 | Less impact |

## Why Different Strategies?

### Warp Size Impact

The warp size fundamentally changes the optimal parallelization strategy:

```
NVIDIA (Warp=32):
- 1 warp per row is sufficient
- With avgNnz=10, each thread processes 10/32 ≈ 0.3 elements
- But warp-level reduction is efficient

Mars X201 (Warp=64):
- 1 warp per row gives poor utilization
- With avgNnz=10, utilization = 10/64 = 15.6%
- Need TPR=8 (8 threads per row) for better parallelism
- 64/8 = 8 rows per warp → much better utilization
```

### L2 Cache Impact

The L2 cache size determines how well the x vector can be cached:

```
x vector size = numCols × 8 bytes = 1.26M × 8 = 10.8 MB

Mars X201 (L2 ~2-4MB):
- Cannot cache entire x vector
- Each x[col] access often goes to DRAM
- Random access pattern → poor locality
- This limits utilization to ~48.7%

RTX 4090 (L2 72MB):
- Can cache entire x vector
- High cache hit rate for x[col] accesses
- __ldg adds texture cache benefits
- Achieves ~88.8% utilization
```

## Mars X201 Optimization (TPR=8)

### Kernel Architecture

```cpp
template<int WarpSize, int TPR>
__global__ void tpr_kernel(...) {
    // Warp processes WarpSize/TPR rows
    int rowsPerWarp = WarpSize / TPR;  // 64/8 = 8
    
    // Determine row assignment
    int row = warpId * rowsPerWarp + lane / TPR;
    int threadInRow = lane % TPR;
    
    // Each thread processes elements at stride TPR
    for (int i = rowStart + threadInRow; i < rowEnd; i += TPR) {
        sum += values[i] * x[colIdx[i]];
    }
    
    // Reduce within TPR group (not full warp)
    for (int offset = TPR/2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
}
```

### Key Configuration

```cpp
// Optimal settings for Mars X201
const int WarpSize = 64;
const int TPR = 8;
const int blockSize = 256;

// CRITICAL: PreferL1 cache configuration
cudaFuncSetCacheConfig(tpr_kernel<64, 8>, cudaFuncCachePreferL1);
```

### Why TPR=8?

Testing on 10 real matrices:

| TPR | Bandwidth | Utilization | Rows/Warp |
|-----|-----------|-------------|-----------|
| 1   | 296 GB/s  | 16.1%       | 1         |
| 2   | 596 GB/s  | 32.4%       | 2         |
| 4   | 890 GB/s  | 48.3%       | 4         |
| **8** | **897 GB/s** | **48.7%** | **8** |
| 16  | 769 GB/s  | 41.8%       | 1         |

TPR=8 is optimal because:
1. 8 rows per warp gives good parallelism
2. 8 threads per row is enough for sparse rows
3. Larger TPR reduces rows/warp, hurting utilization

## NVIDIA RTX 4090 Optimization (__ldg)

### Kernel Architecture

```cpp
template<int WarpSize>
__global__ void ldg_kernel(...) {
    // 1 warp per row (standard vector kernel)
    int row = warpId;
    int lane = threadIdx.x % WarpSize;
    
    // Use __ldg for read-only cache hints
    for (int i = rowStart + lane; i < rowEnd; i += WarpSize) {
        int col = __ldg(&colIdx[i]);
        sum += __ldg(&values[i]) * __ldg(&x[col]);
    }
    
    // Standard warp reduction
    sum = warpReduceSum<WarpSize>(sum);
}
```

### Why __ldg?

The `__ldg()` function directs data through the texture cache (read-only cache):

1. **Separate cache path**: Uses texture cache alongside L2
2. **Better for repeated reads**: x[col] may be read multiple times
3. **Complements large L2**: L2 caches x, texture cache adds more

Performance improvement with __ldg:

| Kernel | Bandwidth | Improvement |
|--------|-----------|-------------|
| Baseline | 848 GB/s | Base |
| __ldg | 893 GB/s | +5% |

## Warp Reduction Difference

### 32-Thread Warp (NVIDIA)

```cpp
// 32-bit mask
sum += __shfl_down_sync(0xffffffff, sum, 16);
sum += __shfl_down_sync(0xffffffff, sum, 8);
sum += __shfl_down_sync(0xffffffff, sum, 4);
sum += __shfl_down_sync(0xffffffff, sum, 2);
sum += __shfl_down_sync(0xffffffff, sum, 1);
```

### 64-Thread Warp (Mars X201)

```cpp
// First step uses 64-bit mask!
sum += __shfl_down_sync(0xffffffffffffffffULL, sum, 32);

// Remaining steps use 32-bit mask
sum += __shfl_down_sync(0xffffffff, sum, 16);
// ... rest is same
```

The 64-bit mask (`0xffffffffffffffffULL`) is CRITICAL for Mars X201. Without it, only 32 lanes participate in the first reduction step.

## Pinned Memory Importance

Pinned (page-locked) memory is CRITICAL for end-to-end performance:

| Platform | Improvement |
|----------|-------------|
| Mars X201 | +186% |
| RTX 4090 | +152% |

### Why?

1. **Direct DMA transfer**: GPU can DMA directly to pinned memory
2. **No staging buffer**: Avoids copying to intermediate buffer
3. **Higher PCIe utilization**: Full PCIe bandwidth available

### Usage

```cpp
// Allocate pinned memory
cudaMallocHost(&x, numCols * sizeof(double));

// ... use x ...

// Free pinned memory
cudaFreeHost(x);
```

## Performance Analysis

### Bandwidth Calculation

```cpp
// FP64 SpMV bytes transferred:
// - Read: values (8B × nnz), colIdx (4B × nnz), x (8B × nnz random)
// - Write: y (8B × numRows)

double bytes = nnz * (8 + 4 + 8) + nnz * 8 + numRows * 8;
// Simplified: nnz * 20 + numRows * 8

double bandwidth = bytes / (time_ms * 1e6);  // GB/s
```

### Utilization Calculation

```cpp
double utilization = bandwidth / theoretical_bandwidth * 100;

// Mars X201: 897 / 1843 = 48.7%
// RTX 4090: 893 / 1008 = 88.8%
```

## Common Pitfalls

### 1. Wrong Warp Reduction Mask

```cpp
// WRONG for Mars X201 (warp=64):
sum += __shfl_down_sync(0xffffffff, sum, 32);  // Only 32 lanes!

// CORRECT for Mars X201:
sum += __shfl_down_sync(0xffffffffffffffffULL, sum, 32);  // All 64 lanes
```

### 2. Not Setting Cache Config

```cpp
// WRONG for Mars X201:
// No cache config set → uses default (may be PreferShared)

// CORRECT for Mars X201:
cudaFuncSetCacheConfig(kernel, cudaFuncCachePreferL1);
```

### 3. Not Using Pinned Memory

```cpp
// WRONG:
double* x = (double*)malloc(numCols * sizeof(double));

// CORRECT:
cudaMallocHost(&x, numCols * sizeof(double));
```

### 4. Using ILP on NVIDIA

ILP optimization is HARMFUL on NVIDIA RTX 4090:

| ILP Level | Mars Effect | RTX Effect |
|-----------|-------------|------------|
| ILP2 (2 accum) | +0.5% | -14% |
| ILP4 (4 accum) | +0.5% | -28% |

Reason: SpMV is memory-bound, not compute-bound. ILP adds register pressure without helping memory throughput.

## Summary

| Platform | Key Optimization | Reason |
|----------|------------------|--------|
| Mars X201 | TPR=8 | Better parallelism for warp=64 |
| Mars X201 | PreferL1 | More cache for rowPtr access |
| Mars X201 | 64-bit mask | Full warp participation |
| RTX 4090 | __ldg | Texture cache + large L2 |
| Both | Pinned memory | Critical for PCIe transfers |
| Both | No ILP | Memory-bound, not compute-bound |