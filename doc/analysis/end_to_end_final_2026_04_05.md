# 端到端性能分析 - 最终结果 (2026-04-05)

## 测试矩阵
- 文件: p0_A
- 维度: 1,256,923 × 1,256,923
- NNZ: 13,465,911
- avgNnz: 10.7

## 最终性能结果

### Kernel性能对比

| 平台 | 最优配置 | 利用率 | 耗时(ms) | 带宽(GB/s) |
|------|----------|--------|----------|------------|
| **Mars X201** | 8t/row, BS=512 | **24.5%** | 0.365 | 451 |
| **RTX 4090** | 4t/row+ILP, BS=512 | **234%** | 0.070 | 2362 |

### 端到端性能对比

| 平台 | 端到端耗时 | 端到端利用率 | Kernel占比 | 数据传输占比 |
|------|-----------|-------------|-----------|-------------|
| **Mars X201** | 11.6ms | 0.54% | 3.2% | **96.8%** |
| **RTX 4090** | 11.3ms | 1.0% | 0.6% | **99.4%** |

### 数据传输量

| 传输类型 | 大小 |
|----------|------|
| H2D (rowPtr+colIdx+values+x) | 112.33 MB |
| D2H (y) | 4.79 MB |
| **总计** | 117.12 MB |

## 关键发现

### 1. 数据传输主导端到端时间
- Mars X201: 96.8%时间在数据传输
- RTX 4090: 99.4%时间在数据传输
- **结论**: Kernel优化对端到端性能影响小于5%

### 2. 平台配置差异 (关键!)

| 参数 | Mars X201 | RTX 4090 | 原因 |
|------|-----------|----------|------|
| Warp Size | 64 | 32 | 硬件差异 |
| 最优线程/行 | **8** | **4** | warp大小影响 |
| Block Size | 512 | 512 | 相同 |
| L2 Cache | ~4MB | 72MB | **关键差异** |

### 3. 性能差距分析

| 指标 | 差距 | 原因 |
|------|------|------|
| Kernel利用率 | 234%/24.5% = **9.5x** | L2 Cache大小差异 |
| 端到端耗时 | 11.6ms/11.3ms = **1.03x** | 数据传输主导 |
| Kernel耗时 | 0.365ms/0.070ms = **5.2x** | 架构差异 |

## 最优Kernel代码

### Mars X201 (8 threads/row)
```cpp
template<int BLOCK_SIZE, int SMEM_INTS>
__global__ void spmv_mars_optimal(int numRows, const int* __restrict__ rowPtr,
                                  const int* __restrict__ colIdx,
                                  const float* __restrict__ values,
                                  const float* __restrict__ x, float* __restrict__ y) {
    __shared__ int sharedRowPtr[SMEM_INTS];
    int warpId = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int globalWarpId = blockIdx.x * (BLOCK_SIZE / WARP_SIZE) + warpId;
    int baseRow = globalWarpId * 8;  // 8 rows per warp
    int warpOffset = warpId * 9;     // 8+1 entries
    // ... (详见tests/test_end_to_end.cu)
}
```

### RTX 4090 (4 threads/row with ILP)
```cpp
template<int BLOCK_SIZE, int SMEM_INTS>
__global__ void spmv_rtx_optimal(int numRows, const int* __restrict__ rowPtr,
                                 const int* __restrict__ colIdx,
                                 const float* __restrict__ values,
                                 const float* __restrict__ x, float* __restrict__ y) {
    // 4 threads per row with dual accumulators for ILP
    float sum0 = 0, sum1 = 0;
    for (; idx + 4 < rowEnd; idx += 8) {
        sum0 += values[idx] * __ldg(&x[colIdx[idx]]);
        sum1 += values[idx + 4] * __ldg(&x[colIdx[idx + 4]]);
    }
    // ...
}
```

## 优化建议

### 场景1: 数据驻留在GPU
- 使用最优kernel配置
- Mars X201可达24.5%利用率
- RTX 4090可达234%利用率

### 场景2: 需要频繁数据传输
- 数据传输占96%+时间
- kernel优化收益小于5%
- 考虑使用pinned memory、异步传输等优化数据传输

### 场景3: 多次迭代应用
- 预处理开销可摊销
- 考虑CSR5等格式提升负载均衡
- 注意原子操作开销

## 结论

1. **Mars X201 kernel优化已达硬件极限**: 24.5%利用率
2. **RTX 4090利用L2缓存可达234%**: 大缓存优势明显
3. **端到端性能差距很小**: 11.6ms vs 11.3ms (仅1.03x)
4. **数据传输是主要瓶颈**: 占96%+时间
5. **不同平台需要不同配置**: Mars需8t/row，RTX需4t/row

---

*测试日期: 2026-04-05*
*测试文件: tests/test_end_to_end.cu*