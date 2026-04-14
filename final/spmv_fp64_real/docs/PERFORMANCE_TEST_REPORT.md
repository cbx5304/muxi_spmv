# SPMV FP64 Library Performance & Accuracy Test Report

## Executive Summary

This report presents comprehensive performance and accuracy validation results for the `spmv_fp64_real` library on two GPU platforms:
- **NVIDIA RTX 4090** (Warp=32)
- **Mars X201** (国产GPU, Warp=64)

**Key Findings**:
- ✅ All correctness tests PASSED on both platforms
- ✅ Mars X201 achieves higher kernel bandwidth (704.6 GB/s vs 584.7 GB/s)
- ✅ RTX 4090 achieves higher bandwidth utilization (58% vs 38%)
- ✅ Profiler tools (hcTracer/nsys) confirm CUDA Events measurements are accurate

---

## Test Configuration

### Matrix Parameters

| Parameter | Value |
|-----------|-------|
| Rows | 1,000,000 |
| Columns | 1,000,000 |
| avgNnzPerRow | 10 |
| Total NNZ | 10,000,000 |
| Pattern | Band + Random (column locality) |
| Data Type | FP64 (double) |

### Test Setup

| Platform | GPU | Warp Size | Theoretical BW | Profiler Tool |
|----------|-----|-----------|----------------|---------------|
| RTX 4090 | NVIDIA GeForce RTX 4090 | 32 | 1008 GB/s | nsys |
| Mars X201 | Mars 01 | 64 | 1843.2 GB/s | hcTracer |

### Bandwidth Calculation

```
Bytes per iteration = nnz * 20 + numRows * 8
                    = 10,000,000 * 20 + 1,000,000 * 8
                    = 208,000,000 bytes (208 MB)

Bandwidth (GB/s) = bytes / (time_ms * 1e6)
Utilization (%)  = bandwidth / theoreticalBW * 100
```

---

## Performance Results

### Summary Table

| Metric | RTX 4090 | Mars X201 |
|--------|----------|-----------|
| Kernel Type | `ldg_kernel<32>` | `tpr_kernel<64,8>` |
| Avg Kernel Time | **0.355 ms** | **0.293 ms** |
| Bandwidth | **584.7 GB/s** | **704.6 GB/s** |
| Utilization | **58.0%** | **38.2%** |
| Correctness | ✅ PASSED | ✅ PASSED |

### Detailed Profiler Data

#### RTX 4090 (nsys)

```
Kernel: void spmv_fp64_impl::ldg_kernel<(int)32>(...)
Instances: 111 (warmup + benchmark + verify)
Avg Time: 355,027 ns (0.355 ms)
Min: 354,849 ns
Max: 356,225 ns
StdDev: 167 ns (very stable)
```

**Memory Transfer Summary**:
- Host-to-Device: 140 MB total (5 transfers)
- Device-to-Host: 8 MB (1 transfer)
- Total GPU memory operations: 17.5 ms

#### Mars X201 (hcTracer)

```
Kernel: void spmv_fp64_impl::tpr_kernel<64, 8>(...)
Instances: 111 (warmup + benchmark + verify)
Avg Time: 292,719 us (0.293 ms)
Min: 288,000 us
Max: 301,568 us
StdDev: ~3.6 us
```

---

## Correctness Verification

### Verification Method

1. GPU computes SpMV on test matrix
2. CPU computes reference SpMV (first 100 rows)
3. Compare GPU vs CPU results

### Results

| Platform | Errors (out of 100 samples) | Status |
|----------|------------------------------|--------|
| RTX 4090 | 0 | ✅ PASSED |
| Mars X201 | 0 | ✅ PASSED |

**Relative Error Threshold**: 1e-10

---

## Kernel Analysis

### RTX 4090: ldg_kernel<32>

The `__ldg` kernel uses:
- Warp size: 32
- 1 warp per row (standard CSR-Vector approach)
- `__ldg()` for explicit read-only cache loading
- L1 cache preference for CSR data

**Why 58% utilization?**
- Random column indices prevent perfect memory coalescing
- L2 cache (72MB) helps cache x vector partially
- Memory-bound operation limits achievable bandwidth

### Mars X201: tpr_kernel<64, 8>

The TPR=8 kernel uses:
- Warp size: 64
- 8 threads per row (TPR optimization)
- L1 cache preference (critical for Mars X201!)
- Inter-warp coordination for row aggregation

**Why 38% utilization?**
- L2 cache (~2-4MB) smaller than RTX 4090
- Cannot cache entire x vector (8MB)
- Warp size 64 requires more threads to hide latency
- Random access pattern limits memory efficiency

---

## Platform Comparison

### Mars X201 vs RTX 4090

| Aspect | Mars X201 | RTX 4090 | Winner |
|--------|-----------|----------|--------|
| Kernel Time | 0.293 ms | 0.355 ms | Mars (+18%) |
| Bandwidth | 704.6 GB/s | 584.7 GB/s | Mars (+21%) |
| Utilization | 38.2% | 58.0% | RTX (+52%) |
| L2 Cache | ~2-4 MB | 72 MB | RTX (+18x) |

**Key Insight**: Mars X201 achieves **higher raw bandwidth** despite **lower utilization** due to:
1. Higher theoretical bandwidth (1843 vs 1008 GB/s)
2. TPR=8 optimization specifically tuned for warp=64

---

## Test Files

### profiler_test.cu

The profiler test uses CUDA Events for accurate timing:

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start, 0);
for (int i = 0; i < 100; i++) {
    spmv_fp64_execute_direct(numRows, nnz, d_rowPtr, d_colIdx, d_values, d_x, d_y, 0);
}
cudaEventRecord(stop, 0);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&total_time, start, stop);
```

### Other Tests

- `simple_test.cu` - Basic validation (1000 rows)
- `benchmark_test.cu` - Multi-size benchmark
- `comprehensive_test.cu` - Full API coverage

---

## Conclusions

### Performance Validation

✅ **Both platforms meet performance expectations**:
- RTX 4090: 58% utilization matches typical CSR-Vector performance
- Mars X201: 38% utilization is expected for small-L2-cache architecture

### Accuracy Validation

✅ **All tests pass correctness checks**:
- Zero errors in 100-sample verification
- FP64 precision maintained on both platforms

### Profiler Validation

✅ **Profiler tools confirm measurements**:
- hcTracer (Mars): 0.293 ms avg, matches CUDA Events 0.295 ms
- nsys (RTX): 0.355 ms avg, matches CUDA Events 0.356 ms

### Recommendations

1. **For RTX 4090**: Use `__ldg` kernel for best performance
2. **For Mars X201**: Use TPR=8 kernel with L1 cache preference
3. **For both**: Use pinned memory (`cudaMallocHost`) for best end-to-end performance
4. **For iterative algorithms**: Consider device pointer mode to avoid H2D/D2H transfers

---

## Appendix: Raw Test Output

### RTX 4090 Output

```
========================================
  SPMV FP64 Profiler Test (CUDA Events)
========================================

GPU: NVIDIA GeForce RTX 4090
Warp Size: 32
Memory: 25.25 GB
Theoretical BW: 1008.0 GB/s

Matrix: 1000000 rows, avgNnz=10, total nnz=10000000

Warmup runs (10 iterations)...
Benchmark runs (100 iterations)...

=== Performance Results ===
Kernel time (avg): 0.356 ms
Bandwidth: 584.7 GB/s
Utilization: 58.0%
Kernel type: __ldg

=== Correctness Check ===
Correctness: PASSED (0 errors in 100 sample rows)
```

### Mars X201 Output

```
========================================
  SPMV FP64 Profiler Test (CUDA Events)
========================================

GPU: Mars 01
Warp Size: 64
Memory: 68.28 GB
Theoretical BW: 1843.2 GB/s

Matrix: 1000000 rows, avgNnz=10, total nnz=10000000

Warmup runs (10 iterations)...
Benchmark runs (100 iterations)...

=== Performance Results ===
Kernel time (avg): 0.295 ms
Bandwidth: 704.6 GB/s
Utilization: 38.2%
Kernel type: TPR=8

=== Correctness Check ===
Correctness: PASSED (0 errors in 100 sample rows)
```

---

**Report Generated**: 2026-04-13
**Library Version**: spmv_fp64 v1.0
**Test Configuration**: 1M rows, avgNnz=10, CUDA Events benchmark