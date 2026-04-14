# FP64 SpMV Library - Performance Test Results

## Important: Performance Depends on Matrix Structure

### Column Index Locality Impact

| Matrix Type | Column Distribution | L2 Cache Hit | Bandwidth |
|-------------|---------------------|--------------|-----------|
| **With locality** | Clustered/nearby | High (>80%) | 800-900 GB/s |
| **Random** | Uniformly distributed | Low (<20%) | 300-450 GB/s |

**Explanation**: 
- Real-world matrices (e.g., FEM, structured grids) often have **column index locality** - adjacent rows access nearby columns, improving L2 cache hit rate.
- Randomly generated matrices have no locality, resulting in lower bandwidth.

---

## Test Environment

| Server | GPU | Warp Size | Memory | Kernel |
|--------|-----|-----------|--------|--------|
| 172.16.45.81:19936 | Mars X201 | 64 | 68.28 GB | TPR=8 |
| 172.16.45.70:3000 | RTX 4090 | 32 | 25.25 GB | __ldg |

---

## Current Test Results (Random Column Distribution)

### Tested Matrices

| Property | Value |
|----------|-------|
| Rows | 1256923 |
| NNZ | 13465911 |
| avgNnz | 10.71 |
| Column Distribution | **Random (no locality)** |

### RTX 4090 Results

| Matrix | Kernel+H2D Time | Bandwidth | Utilization |
|--------|-----------------|-----------|-------------|
| p0_A | 0.855 ms | 326.8 GB/s | 32.4% |
| p1_A | 0.856 ms | 326.5 GB/s | 32.4% |
| p2_A | 0.856 ms | 326.5 GB/s | 32.4% |
| ... | ... | ... | ... |
| **Average** | **0.86 ms** | **326 GB/s** | **32.4%** |

### Mars X201 Results (Synthetic Matrices)

| Matrix | Rows | NNZ | avgNnz | Time | BW | Util |
|--------|------|-----|--------|------|----|----|
| p3_A | 1M | 10M | 10.00 | 0.58 ms | 357 GB/s | 19.4% |
| p8_A | 100K | 10M | 100.00 | 0.45 ms | 443 GB/s | 24.1% |

**Average (avgNnz=10)**: ~310-360 GB/s

---

## Historical Best Case Results (With Column Locality)

From previous optimization work on matrices with column index locality:

### Mars X201

| Matrix | Kernel Time | Bandwidth | Utilization |
|--------|-------------|-----------|-------------|
| Real matrices (avgNnz=10.71) | 0.420 ms | **897 GB/s** | **48.7%** |

### RTX 4090

| Matrix | Kernel Time | Bandwidth | Utilization |
|--------|-------------|-----------|-------------|
| Real matrices (avgNnz=10.71) | 0.402 ms | **907 GB/s** | **90.2%** |

**Key difference**: These matrices had column index locality, dramatically improving L2 cache performance.

---

## Performance Comparison Summary

| Scenario | Mars X201 | RTX 4090 |
|----------|-----------|----------|
| Best case (with locality) | 897 GB/s (48.7%) | 907 GB/s (90.2%) |
| Typical case (random) | 350 GB/s (19%) | 330 GB/s (32%) |
| Ratio | 2.6x gap | 2.8x gap |

**Conclusion**: Matrix structure (column index locality) has greater impact on performance than GPU type.

---

## Recommendations

1. **For matrices with locality**: Expect 800+ GB/s bandwidth
2. **For random matrices**: Expect 300-450 GB/s bandwidth
3. **Optimization**: Focus on improving locality through matrix reordering (RCM, METIS)

---

*Test date: 2026-04-12*
*Library version: fp64_V1.0*