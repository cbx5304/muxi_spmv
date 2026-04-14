# SPMV FP32 Performance Test Report for Mars X201

## Executive Summary

Performance validation for `spmv_fp32` library on **Mars X201** GPU (Warp=64, Peak BW=1843.2 GB/s).

**Key Results:**
- Kernel bandwidth: **1111 GB/s** (61.1% utilization)
- Improvement over nctigpu: **+32.8%**
- Optimal TPR: **32** for avgNnz=85 matrices

---

## Test Configuration

| Parameter | Value |
|-----------|-------|
| GPU | Mars X201 (国产) |
| Warp Size | 64 |
| Peak Bandwidth | 1843.2 GB/s |
| Test Cases | pressure_0, pressure_10, pressure_50 |
| Precision | FP32 |

---

## Matrix Characteristics

| Matrix | Rows | NNZ | avgNnz | Sparsity |
|--------|------|-----|--------|----------|
| pressure_0 | 288769 | 24700225 | 85.54 | 0.03% |
| pressure_10 | 288769 | 24846809 | 86.04 | 0.03% |
| pressure_50 | 288769 | 24684849 | 85.48 | 0.03% |

**Matrix Format**: MatrixMarket coordinate real general (FP32)

---

## TPR Optimization Results

### TPR Sweep Test (pressure_0)

| TPR | BlockSize | Time(ms) | BW(GB/s) | Util(%) | Rows/Warp |
|-----|-----------|----------|----------|---------|-----------|
| 4 | 256 | 0.995 | 299 | 16.2% | 16 |
| 8 | 256 | 0.372 | 799 | 43.3% | 8 |
| 16 | 256 | 0.278 | 1072 | 58.2% | 4 |
| **32** | **256** | **0.264** | **1127** | **61.2%** | **2** ⭐ |
| 64 | 256 | 0.331 | 899 | 48.8% | 1 |

**Best Configuration**: TPR=32, blockSize=256

### TPR Selection Logic

```cpp
// Adaptive TPR based on avgNnz
if (avgNnz >= 128) TPR = 64;    // 1 row/warp
else if (avgNnz >= 64) TPR = 32; // 2 rows/warp ⭐ optimal for avgNnz=85
else if (avgNnz >= 32) TPR = 16; // 4 rows/warp
else if (avgNnz >= 16) TPR = 8;  // 8 rows/warp
else TPR = 4;                   // 16 rows/warp
```

**Validation**: avgNnz=85 → TPR=32 → Correct!

---

## Performance Comparison

### Kernel Performance (hcTracer verified)

| Metric | nctigpu | optimized | Improvement |
|--------|---------|-----------|-------------|
| Kernel Time (avg) | 0.3565 ms | 0.2684 ms | **+32.8%** |
| Kernel BW (avg) | 836 GB/s | 1111 GB/s | **+32.8%** |

### Detailed Results

| Case | ncti Kern(ms) | opt Kern(ms) | ncti BW | opt BW | Speedup |
|------|---------------|--------------|---------|--------|---------|
| pressure_0 | 0.355 | 0.268 | 838 | 1107 | 1.32x |
| pressure_10 | 0.354 | 0.268 | 845 | 1111 | 1.32x |
| pressure_50 | 0.353 | 0.268 | 842 | 1108 | 1.32x |

---

## End-to-End Performance

### Full Transfer Mode (Matrix + Vector)

| Mode | E2E Time | E2E BW | Notes |
|------|----------|--------|-------|
| Pageable | 23 ms | 13 GB/s | Baseline |
| Pinned | 15.7 ms | +24% faster | cudaMallocHost |

### Iterative Mode (Matrix Pre-loaded)

| Metric | Value | Improvement |
|--------|-------|-------------|
| Iteration Time | 0.35 ms | **67x faster** vs full transfer |
| Only x/y transfer + kernel | - | Skip matrix transfer |

---

## Bandwidth Utilization Analysis

| Library | Kernel BW | Peak BW | Utilization |
|---------|-----------|---------|-------------|
| nctigpu | 836 GB/s | 1843 GB/s | 45.8% |
| **optimized** | **1111 GB/s** | 1843 GB/s | **61.1%** ⭐ |

**Physical Limit**: Cannot exceed peak bandwidth (1843.2 GB/s)

---

## Accuracy Verification

| Case | Max Difference | Status |
|------|----------------|--------|
| pressure_0 | 9.5e-06 | ✅ PASS |
| pressure_10 | 1.2e-04 | ✅ PASS |
| pressure_50 | 1.2e-04 | ✅ PASS |

---

## Optimization Techniques Summary

| Technique | Effect | Implementation |
|-----------|--------|----------------|
| **TPR=32** | +32% kernel | Adaptive selection based on avgNnz |
| **L1 Cache** | Required | cudaFuncCachePreferL1 |
| **Pinned Memory** | +24% E2E | cudaMallocHost for x/y vectors |
| **Matrix Pre-load** | +6700% iterative | Keep matrix on GPU for iterations |

---

## Recommendations

### Single Execution
- Use pinned memory for x/y vectors
- E2E time ~15-23ms depending on mode

### Iterative Algorithms
- **Pre-load matrix on GPU** (one-time preprocessing)
- Only transfer x/y each iteration
- Achieve ~0.35ms per iteration

### Best Practice
```cpp
// Allocate pinned memory
cudaMallocHost(&h_x, numCols * sizeof(float));
cudaMallocHost(&h_y, numRows * sizeof(float));

// Create matrix handle (matrix stays on GPU)
spmv_fp32_matrix_handle_t handle;
spmv_fp32_create_matrix_device(&handle, ...);

// Iterative loop
for (int iter = 0; iter < maxIter; iter++) {
    cudaMemcpy(d_x, h_x, ...);
    spmv_fp32_execute_device(handle, d_x, d_y, ...);
    cudaMemcpy(h_y, d_y, ...);
}
```

---

## Files Generated

| File | Location |
|------|----------|
| Library | install_X201/lib/libspmv_fp32.so |
| Header | install_X201/include/spmv_fp32.h |
| API Docs | install_X201/docs/API.md |
| hcTracer Profile | /home/chenbinxiangc/spmv_comp/kernel_profile/ |

---

**Report Generated**: 2026-04-14
**Library Version**: spmv_fp32 v1.0.0
**Platform**: Mars X201 (国产GPU, Warp=64)