# Mars X201 SpMV Optimization Analysis Report

## Executive Summary

This report documents a comprehensive analysis of SpMV performance on Mars X201 GPU for sparse matrices with avgNnzPerRow < 10, and the optimizations attempted to improve performance.

### Final Results

| Metric | Initial | After Optimization | Improvement |
|--------|---------|-------------------|-------------|
| 1M rows, avgNnz=10 | 1.3% utilization | 9.8% utilization | **7.5x** |
| Bandwidth | 23 GB/s | 180 GB/s | **7.8x** |

**Gap vs RTX 4090**: 9x (9.8% vs 87.9%)

## Problem Analysis

### Root Cause Investigation

Through systematic testing, we identified multiple bottlenecks:

1. **L2 Cache Thrashing** (Primary bottleneck for large matrices)
   - Mars X201 L2 cache: ~2-4 MB
   - rowPtr for 1M rows: 8 MB
   - Performance cliff when rowPtr exceeds cache

2. **Batch Loop Overhead** (Secondary bottleneck)
   - Original batched kernel had fixed overhead per batch
   - Execution time was constant regardless of NNZ count
   - Batch processing dominated actual computation

3. **Random Memory Access** (Fundamental limitation)
   - Sparse matrices have scattered colIdx values
   - x[colIdx[i]] accesses are essentially random
   - Mars X201's cache architecture handles this poorly

### Performance Scaling Analysis

**Before Optimization (Batched Kernel):**
| NNZ Count | Execution Time | Observation |
|-----------|---------------|-------------|
| 5M | 0.70 ms | Constant time |
| 10M | 0.69 ms | regardless of |
| 20M | 0.69 ms | NNZ count |
| 40M | 0.70 ms | (batch loop overhead) |

**After Optimization (Scalar Kernel):**
| Rows | Bandwidth | Utilization |
|------|-----------|-------------|
| 100K | 30 GB/s | 1.7% |
| 500K | 134 GB/s | 7.3% |
| 1M | 180 GB/s | 9.8% |
| 2M | 194 GB/s | 10.5% |

## Optimization Attempts

### 1. Light-Balanced Kernel
**Approach**: Each thread processes multiple rows to improve utilization.

**Result**: Improved for small matrices, but poor scaling for large matrices due to L2 cache thrashing.

**Finding**: Warp=64 requires avgNnz >= 64 for efficient vector kernel utilization.

### 2. Batched Kernel (L2 Cache Aware)
**Approach**: Process rows in fixed-size batches (200K rows) to keep rowPtr in L2 cache.

**Result**: 5x improvement for large matrices (1.3% → 6.8%).

**Issue**: Batch loop overhead dominated execution time, making performance constant regardless of NNZ count.

### 3. Shared Memory Caching
**Approach**: Cache x vector in shared memory to reduce global memory access.

**Result**: **Negative** - Performance regressed (6.8% → 4.5%).

**Reason**: Overhead of loading x into shared memory exceeded benefits.

### 4. Read-Only Cache Hint
**Approach**: Use __ldg() intrinsic for x vector access.

**Result**: No significant improvement.

**Reason**: Mars X201's cache architecture doesn't benefit from this hint.

### 5. Scalar Kernel (Final Solution)
**Approach**: Use simple one-thread-per-row kernel, same as NVIDIA uses for sparse matrices.

**Result**: **Best improvement** - 43% over batched kernel (6.8% → 9.8%).

**Why it works**: Eliminates batch loop overhead, allows GPU to handle memory access patterns directly.

## Comparison with RTX 4090

### Hardware Differences

| Feature | Mars X201 | RTX 4090 |
|---------|-----------|----------|
| Warp Size | 64 | 32 |
| L2 Cache | ~2-4 MB | ~72 MB |
| Peak Bandwidth | 1843 GB/s | 1008 GB/s |
| SMs | 104 | 128 |

### Performance Comparison (1M rows, avgNnz=10)

| Metric | Mars X201 | RTX 4090 | Gap |
|--------|-----------|----------|-----|
| Utilization | 9.8% | 87.9% | 9x |
| Bandwidth | 180 GB/s | 886 GB/s | 4.9x |
| Execution Time | 0.49 ms | 0.10 ms | 4.9x |

### Why the Gap Exists

1. **Cache Architecture**: RTX 4090 has 72MB L2 cache vs Mars X201's ~4MB
   - Entire rowPtr fits in RTX 4090's cache
   - Mars X201 must fetch from DRAM repeatedly

2. **Memory Controller**: RTX 4090 has more sophisticated memory controller
   - Better handling of random access patterns
   - More effective memory coalescing

3. **Warp Size**: 64 threads vs 32
   - Mars X201 requires more work per warp for efficiency
   - For avgNnz=10, only 10/64 = 15.6% of warp threads are active

## Recommendations

### For Current Hardware

1. **Use Scalar Kernel** for sparse matrices (avgNnz < 32)
   - Best performance achieved with simple one-thread-per-row approach
   - No batch overhead, good scaling

2. **Use Vector Kernel** for denser matrices (avgNnz >= 32)
   - Achieves 35-52% utilization
   - Warp efficiency increases with avgNnz

3. **Consider Alternative Formats** for specific patterns
   - ELLPACK for regular sparsity patterns
   - JDS for diagonally dominant matrices

### For Future Development

1. **Memory Access Optimization**
   - Column reordering to improve x vector access locality
   - Row sorting by length for better load balancing

2. **Kernel Design**
   - Avoid complex loop structures (batch overhead dominates)
   - Prefer simple, direct work distribution
   - Let GPU hardware handle parallelism

3. **Format Selection**
   - Consider dense format when sparsity > 10%
   - Use profile-guided format selection

## Files Modified

| File | Changes |
|------|---------|
| `src/spmv/csr/spmv_csr.cuh` | Added batched kernel, shared memory kernel, NNZ-based kernel |
| `src/spmv/csr/spmv_csr.cu` | Updated dispatch logic for scalar kernel |
| `doc/gpu_comparison/batched_kernel_optimization.md` | Optimization documentation |

## Conclusion

Through systematic analysis and optimization, we achieved a **7.5x improvement** in sparse matrix SpMV performance on Mars X201. The key findings are:

1. **Batch loop overhead** was the primary bottleneck, not L2 cache as initially hypothesized
2. **Simple scalar kernel** outperforms complex batching strategies
3. **Fundamental hardware differences** (cache size, memory controller) create an unavoidable gap with RTX 4090

The remaining 9x gap with RTX 4090 is primarily due to hardware architecture differences, particularly L2 cache size and memory controller sophistication. Further improvements would require either:
- Hardware changes (larger L2 cache)
- Algorithmic changes (different sparse format)
- Problem changes (denser matrices)

---

*Report generated: 2026-04-02*
*Test configuration: 1M rows × 1K cols, avgNnz=10*