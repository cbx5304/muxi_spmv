# Advanced Optimization Techniques Analysis Report

## Document Information

- **Date**: 2026-04-10
- **Test Platform**: Mars X201 (Domestic GPU) vs RTX 4090 (NVIDIA)
- **Test Matrix**: Real matrix (1.26M rows, 13.5M NNZ, avgNnz=10.71)

---

## Executive Summary

**Conclusion**: Advanced optimization techniques (multi-stream, batched processing, grid scaling) provide **no benefit** for SpMV workloads on either platform.

The optimal configuration remains:
- **Mars X201**: 8t/row, blockSize=128, 1 stream, 1x grid
- **RTX 4090**: 4t/row, blockSize=256, 1 stream, 1x grid

---

## Test Results

### Test 1: Multi-Stream Parallelization

#### Mars X201

| Streams | Time (μs) | Bandwidth (GB/s) | Utilization |
|---------|-----------|------------------|-------------|
| **1** | **323.02** | **833.7** | **45.2%** |
| 2 | 320.93 | 839.2 | 45.5% |
| 4 | 323.97 | 831.3 | 45.1% |
| 8 | 341.20 | 789.3 | 42.8% |

**Result**: No statistically significant benefit from multi-stream.

#### RTX 4090

| Streams | Time (μs) | Bandwidth (GB/s) | Utilization |
|---------|-----------|------------------|-------------|
| **1** | **204.78** | **1315.2** | **130.5%** |
| 2 | 201.40 | 1337.2 | 132.7% |
| 4 | 201.49 | 1336.7 | 132.6% |
| 8 | 201.98 | 1333.4 | 132.3% |

**Result**: Marginal improvement (~1.5%), not worth the complexity.

---

### Test 2: Batched Processing (L2 Cache Optimization)

#### Mars X201

| Batch Size | Batches | Time (μs) | Bandwidth (GB/s) | Utilization |
|------------|---------|-----------|------------------|-------------|
| **Full (1256K)** | **1** | **318.60** | **845.3** | **45.9%** |
| 500K | 3 | 332.83 | 809.2 | 43.9% |
| 300K | 5 | 346.47 | 777.3 | 42.2% |
| 200K | 7 | 361.94 | 744.1 | 40.4% |
| 100K | 13 | 404.02 | 666.6 | 36.2% |

**Result**: Batching **hurts** performance. Single batch is optimal.

#### RTX 4090

| Batch Size | Batches | Time (μs) | Bandwidth (GB/s) | Utilization |
|------------|---------|-----------|------------------|-------------|
| **Full (1256K)** | **1** | **204.65** | **1316.0** | **130.6%** |
| 500K | 3 | 211.05 | 1276.1 | 126.6% |
| 300K | 5 | 218.03 | 1235.3 | 122.5% |
| 200K | 7 | 224.93 | 1197.3 | 118.8% |
| 100K | 13 | 246.15 | 1094.1 | 108.5% |

**Result**: Batching **hurts** performance. Single batch is optimal.

---

### Test 3: Grid Configuration

#### Mars X201

| Grid Multiplier | Blocks | Time (μs) | Bandwidth (GB/s) | Utilization |
|-----------------|--------|-----------|------------------|-------------|
| **1x** | **78558** | **318.58** | **845.4** | **45.9%** |
| 2x | 157116 | 481.42 | 559.4 | 30.4% |
| 4x | 314232 | 807.68 | 333.4 | 18.1% |
| 8x | 628464 | 1488.88 | 180.9 | 9.8% |
| 16x | 1256928 | 2813.24 | 95.7 | 5.2% |

**Result**: 1x grid is strongly optimal. Larger grids severely degrade performance.

#### RTX 4090

| Grid Multiplier | Blocks | Time (μs) | Bandwidth (GB/s) | Utilization |
|-----------------|--------|-----------|------------------|-------------|
| **1x** | **19640** | **204.64** | **1316.1** | **130.6%** |
| 2x | 39280 | 211.21 | 1275.1 | 126.5% |
| 4x | 78560 | 230.93 | 1166.2 | 115.7% |
| 8x | 157120 | 270.33 | 996.2 | 98.8% |
| 16x | 314240 | 349.08 | 771.5 | 76.5% |

**Result**: 1x grid is optimal. Larger grids degrade performance (less severe than Mars).

---

## Root Cause Analysis

### Why Multi-Stream Doesn't Help

```
SpMV is memory-bound, not compute-bound
- Each row requires random memory access to x[colIdx[j]]
- Memory bandwidth is the bottleneck
- Multiple streams cannot increase memory bandwidth
- Streams help when there are independent compute tasks
```

### Why Batched Processing Hurts

```
1. Extra kernel launch overhead
   - Each batch requires a separate kernel launch
   - Launch overhead: ~5-10 μs per launch
   - More batches = more overhead

2. No L2 cache benefit
   - Mars X201 L2: ~2-4 MB (too small for x vector)
   - RTX 4090 L2: 72 MB (can cache entire x vector)
   - Batching doesn't improve cache locality for SpMV

3. Memory access pattern unchanged
   - Random access to x[colIdx[j]] remains random
   - Batching doesn't change access pattern
```

### Why Grid Scaling Hurts

```
1. Thread divergence increases
   - Larger grids = more warps with uneven workloads
   - Warps with less work wait for others
   - Increased synchronization overhead

2. Cache thrashing
   - More warps competing for limited L1/L2 cache
   - Cache lines evicted before reuse

3. Memory bandwidth saturation
   - Optimal grid already saturates memory bandwidth
   - More threads don't increase available bandwidth
```

---

## Final Recommendations

### Mars X201 Configuration

```cpp
// Optimal configuration for Mars X201
int threadsPerRow = 8;   // 8t/row
int blockSize = 128;
int gridSize = (numRows * threadsPerRow + blockSize - 1) / blockSize;

cudaFuncSetCacheConfig(vector_kernel<8>, cudaFuncCachePreferL1);
vector_kernel<8><<<gridSize, blockSize>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
```

### RTX 4090 Configuration

```cpp
// Optimal configuration for RTX 4090
int threadsPerRow = 4;   // 4t/row
int blockSize = 256;
int gridSize = (numRows * threadsPerRow + blockSize - 1) / blockSize;

cudaFuncSetCacheConfig(vector_kernel<4>, cudaFuncCachePreferL1);
vector_kernel<4><<<gridSize, blockSize>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
```

### What NOT to Use

| Technique | Recommendation | Reason |
|-----------|----------------|--------|
| Multi-stream | ❌ Don't use | No benefit for memory-bound SpMV |
| Batched processing | ❌ Don't use | Hurts performance due to kernel launch overhead |
| Grid scaling > 1x | ❌ Don't use | Severely degrades performance |
| Shared memory caching | ❌ Don't use | Random access overhead too high |

---

## Performance Comparison

| Metric | Mars X201 | RTX 4090 | Ratio |
|--------|-----------|----------|-------|
| Optimal TPR | 8 | 4 | - |
| Optimal BS | 128 | 256 | - |
| Best time | 318.58 μs | 204.64 μs | RTX 1.56x faster |
| Best bandwidth | 845.4 GB/s | 1316.1 GB/s | RTX 1.56x higher |
| Utilization | 45.9% | 130.6% | RTX 2.84x higher |

The RTX 4090's advantage comes from its 72MB L2 cache, which allows x-vector caching and >100% effective bandwidth utilization.

---

## Test Files

- `tests/benchmark/test_advanced_techniques.cu` - Test source code
- `doc/analysis/fp64_final_optimization_report_2026_04_10.md` - Main optimization report
- `doc/gpu_optimization_differences/mars_x201_vs_rtx4090_guide.md` - GPU development guide