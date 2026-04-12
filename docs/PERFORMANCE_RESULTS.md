# FP64 SpMV Library - Performance Test Results

## Test Environment

| Server | GPU | Warp Size | Memory | Kernel |
|--------|-----|-----------|--------|--------|
| 172.16.45.81:19936 | Mars X201 | 64 | 68.28 GB | TPR=8 |
| 172.16.45.70:3000 | RTX 4090 | 32 | 25.25 GB | __ldg |

## Synthetic Matrix Test Results (avgNnz=10)

### Mars X201

| Matrix | Rows | NNZ | avgNnz | Time(ms) | BW(GB/s) | Util(%) |
|--------|------|-----|--------|----------|----------|---------|
| p0_A | 100000 | 1000000 | 10.00 | 0.097 | 215.5 | 11.7 |
| p1_A | 200000 | 2000000 | 10.00 | 0.130 | 320.5 | 17.4 |
| p2_A | 500000 | 5000000 | 10.00 | 0.297 | 350.4 | 19.0 |
| p3_A | 1000000 | 10000000 | 10.00 | 0.581 | 357.9 | 19.4 |
| p4_A | 100000 | 500000 | 5.00 | 0.057 | 188.5 | 10.2 |
| p5_A | 100000 | 2000000 | 20.00 | 0.119 | 341.7 | 18.5 |
| p6_A | 100000 | 3000000 | 30.00 | 0.163 | 373.8 | 20.3 |
| p7_A | 100000 | 5000000 | 50.00 | 0.297 | 339.3 | 18.4 |
| p8_A | 100000 | 10000000 | 100.00 | 0.453 | 443.5 | 24.1 |
| p9_A | 50000 | 400000 | 8.00 | 0.048 | 175.1 | 9.5 |

**Summary**: Average bandwidth = **310.6 GB/s** (16.8% utilization)

### RTX 4090

| Matrix | Rows | NNZ | avgNnz | Time(ms) | BW(GB/s) | Util(%) |
|--------|------|-----|--------|----------|----------|---------|
| p0_A | 100000 | 1000000 | 10.00 | 0.080 | 261.4 | 25.9 |
| p1_A | 200000 | 2000000 | 10.00 | 0.148 | 280.9 | 27.9 |
| p2_A | 500000 | 5000000 | 10.00 | 0.348 | 298.5 | 29.6 |
| p3_A | 1000000 | 10000000 | 10.00 | 0.686 | 303.0 | 30.1 |
| p4_A | 100000 | 500000 | 5.00 | 0.080 | 135.8 | 13.5 |
| p5_A | 100000 | 2000000 | 20.00 | 0.080 | 512.1 | 50.8 |
| p6_A | 100000 | 3000000 | 30.00 | 0.089 | 686.1 | 68.1 |
| p7_A | 100000 | 5000000 | 50.00 | 0.121 | 835.5 | 82.9 |
| p8_A | 100000 | 10000000 | 100.00 | 0.205 | 977.2 | 96.9 |
| p9_A | 50000 | 400000 | 8.00 | 0.045 | 187.7 | 18.6 |

**Summary**: Average bandwidth = **447.8 GB/s** (44.4% utilization)

## Real Matrix Test Results (RTX 4090)

Matrix specs: 1256923 rows, 13465911 NNZ, avgNnz = 10.71

| Matrix | Kernel Time (ms) | Bandwidth (GB/s) | Utilization (%) |
|--------|------------------|------------------|-----------------|
| p0_A | 0.855 | 326.8 | 32.4 |
| p1_A | 0.856 | 326.5 | 32.4 |
| p2_A | 0.856 | 326.5 | 32.4 |
| p3_A | 0.856 | 326.4 | 32.4 |
| p4_A | 0.857 | 325.9 | 32.3 |
| p5_A | 0.856 | 326.5 | 32.4 |
| p6_A | 0.858 | 325.6 | 32.3 |
| p7_A | 0.857 | 325.9 | 32.3 |
| p8_A | 0.856 | 326.5 | 32.4 |
| p9_A | 0.856 | 326.5 | 32.4 |

**Summary**: Average bandwidth = **326.4 GB/s** (32.4% utilization)

## Performance Analysis

### avgNnz Impact on Performance

| avgNnz | Mars X201 BW | RTX 4090 BW | RTX Advantage |
|--------|--------------|-------------|---------------|
| 5 | 188.5 GB/s | 135.8 GB/s | Mars wins |
| 8 | 175.1 GB/s | 187.7 GB/s | RTX wins |
| 10 | 357.9 GB/s | 303.0 GB/s | Mars wins (large matrix) |
| 20 | 341.7 GB/s | 512.1 GB/s | RTX wins |
| 30 | 373.8 GB/s | 686.1 GB/s | RTX wins |
| 50 | 339.3 GB/s | 835.5 GB/s | RTX wins |
| 100 | 443.5 GB/s | 977.2 GB/s | RTX wins |

### Key Observations

1. **Small avgNnz (≤10)**: Mars X201 competitive or better with large matrices
2. **Large avgNnz (>10)**: RTX 4090 significantly better
3. **Synthetic vs Real**: Real matrices show lower bandwidth due to random access patterns
4. **L2 Cache Effect**: RTX's 72MB L2 cache benefits larger matrices

## Library Usage Recommendations

1. **Mars X201**: Optimal for avgNnz ≤ 10 with TPR=8 kernel
2. **RTX 4090**: Optimal for avgNnz > 10 with __ldg kernel
3. **Pinned Memory**: Always use for best end-to-end performance
4. **Benchmark Mode**: Use `SPMV_FP64_BENCHMARK_OPTS` for accurate timing

---

*Test date: 2026-04-12*
*Library version: fp64_V1.0*