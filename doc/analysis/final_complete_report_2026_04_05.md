# Mars X201 SpMV 穷尽性优化最终报告 (完整版)

## 测试日期: 2026-04-05

---

## 一、重大发现汇总

### 1.1 最优线程配置 (重要更新!)

经过全矩阵验证，发现最优线程配置与之前理解不同：

| 平台 | 最优配置 | 利用率 | 之前认为最优 |
|------|----------|--------|-------------|
| **Mars X201** | **4t/row** | **26.70%** | 8t/row (错误) |
| **RTX 4090** | **2t/row** | **118.98%** | 4t/row (错误) |

**原因分析：**
- Mars X201 (WARP=64): 4t/row → 16行/warp → 更好的行并行度
- RTX 4090 (WARP=32): 2t/row → 16行/warp → 最优平衡

### 1.2 L1缓存配置 (新发现!)

| 配置 | Mars X201 | RTX 4090 |
|------|-----------|----------|
| Default (None) | **22.98%** | 119.41% |
| PreferL1 | 26.65% | 119.43% |
| PreferShared | **26.66%** | 118.21% |
| PreferEqual | 26.64% | 118.61% |

**关键发现**：Mars X201必须显式设置缓存配置，否则性能下降14%！

```cpp
// 必须添加这行！
cudaFuncSetCacheConfig(kernel, cudaFuncCachePreferL1);
```

---

## 二、完整测试矩阵

### 2.1 Kernel优化

| 技术 | Mars效果 | RTX效果 | 结论 |
|------|---------|---------|------|
| `__ldg` 缓存 | +11% | 基准 | ⭐⭐⭐ 必须 |
| Dual Accum (ILP) | +10% | +5% | ⭐⭐⭐ 推荐 |
| 4t/row (Mars) / 2t/row (RTX) | 最优 | 最优 | ⭐⭐⭐ 关键 |

### 2.2 内存优化

| 技术 | Mars效果 | RTX效果 | 端到端影响 | 结论 |
|------|---------|---------|-----------|------|
| **Pinned Memory** | +33% | +20% | **+140%** | ⭐⭐⭐ 关键 |
| Multi-stream (2流) | +8% | +8% | +6% | ⭐⭐ 推荐 |
| L1缓存配置 | **+16%** | 0% | +10% | ⭐⭐⭐ Mars必须 |

### 2.3 数据布局优化

| 技术 | Mars效果 | RTX效果 | 结论 |
|------|---------|---------|------|
| RCM列重排序 | +1.8% | **+11.4%** | 仅RTX有效 |

### 2.4 无效优化

| 技术 | 结果 | 原因 |
|------|------|------|
| Shared Memory Cache | -4% | 矩阵太大无局部性 |
| CSR5格式 | -30% | 原子操作开销 |
| Block Size调整 | 0% | L2缓存是瓶颈 |

---

## 三、最优配置代码

### 3.1 Mars X201

```cpp
// === 关键配置 ===

// 1. Pinned Memory (必须!)
float* h_x;
cudaMallocHost(&h_x, numCols * sizeof(float));

// 2. 线程配置
const int THREADS_PER_ROW = 4;    // 4t/row最优!
const int BLOCK_SIZE = 512;
const int NUM_STREAMS = 2;

// 3. L1缓存配置 (必须!)
cudaFuncSetCacheConfig(spmv_kernel, cudaFuncCachePreferL1);

// 4. Kernel模板
template<int BLOCK_SIZE, int TPR>
__global__ void spmv_kernel(int numRows, const int* __restrict__ rowPtr,
                             const int* __restrict__ colIdx,
                             const float* __restrict__ values,
                             const float* __restrict__ x, float* __restrict__ y) {
    // ... (TPR=4 for Mars X201)
    float sum0 = 0, sum1 = 0;
    for (; idx + TPR < rowEnd; idx += TPR * 2) {
        sum0 += values[idx] * __ldg(&x[colIdx[idx]]);
        sum1 += values[idx + TPR] * __ldg(&x[colIdx[idx + TPR]]);
    }
    // ...
}
```

### 3.2 RTX 4090

```cpp
// === 关键配置 ===

// 1. Pinned Memory
cudaMallocHost(&h_x, numCols * sizeof(float));

// 2. 线程配置
const int THREADS_PER_ROW = 2;    // 2t/row最优!
const int BLOCK_SIZE = 512;

// 3. RCM列重排序 (RTX有效)
applyRCMColumnReordering();

// 4. Kernel (TPR=2 for RTX 4090)
```

---

## 四、性能对比

### 4.1 真实矩阵测试 (p0_A ~ p9_A)

| 指标 | Mars X201 | RTX 4090 | 说明 |
|------|-----------|----------|------|
| Kernel利用率 | 26.70% | 118.98% | RTX 4.5x高 |
| Kernel耗时 | 0.335ms | 0.074ms | RTX 4.5x快 |
| 端到端耗时 | 0.84ms | 1.87ms | **Mars 2.2x快** |

### 4.2 关键发现

1. **Mars端到端更快**: 数据传输效率更高
2. **RTX kernel更快**: L2缓存优势(72MB vs 4MB)
3. **L2 Cache是根本瓶颈**: 无法通过软件解决

---

## 五、优化总结

### 5.1 Mars X201关键优化

| 优化 | 提升 | 必要性 |
|------|------|--------|
| Pinned Memory | +140%端到端 | 必须 |
| **L1缓存配置** | **+16%** | **必须!** |
| 4t/row配置 | 最优 | 必须 |
| DualAccum | +10% | 推荐 |

### 5.2 RTX 4090关键优化

| 优化 | 提升 | 必要性 |
|------|------|--------|
| Pinned Memory | +140%端到端 | 必须 |
| **2t/row配置** | 最优 | 必须 |
| **RCM重排序** | **+11.4%** | **推荐** |

---

## 六、测试文件索引

| 文件 | 用途 | 关键结果 |
|------|------|----------|
| test_optimal_thread_config.cu | 线程配置测试 | 4t/2t最优 |
| test_l1_cache_config.cu | L1缓存测试 | Mars必须配置 |
| test_low_density_optimized.cu | 稀疏矩阵测试 | 线程数影响 |
| test_real_matrices_benchmark.cu | 全矩阵基准 | 端到端数据 |

---

## 七、结论

1. **Mars X201可达26.70%利用率**: 通过正确配置(4t/row + L1缓存)
2. **不同平台需要不同配置**: 4t/row (Mars) vs 2t/row (RTX)
3. **L1缓存配置是关键发现**: Mars必须显式配置
4. **端到端Mars更快**: 数据传输效率优势

---

*报告生成: 2026-04-05*
*关键发现: L1缓存配置对Mars X201至关重要*