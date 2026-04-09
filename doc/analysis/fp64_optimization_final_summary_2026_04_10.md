# FP64 SpMV 完整优化报告 - Mars X201 vs RTX 4090

## 文档信息

- **创建日期**: 2026-04-10
- **测试平台**: Mars X201 (国产GPU) vs RTX 4090 (NVIDIA)
- **测试矩阵**: 真实矩阵 (1.26M行, 13.5M NNZ, avgNnz=10.71)
- **任务状态**: ✅ 完成

---

## 🎯 核心结论

### 性能对比总结

| 指标 | Mars X201 | RTX 4090 | 对比 |
|------|-----------|----------|------|
| **最优内核** | Vector 8t/row | Vector 4t/row | 差异化配置 |
| **内核时间** | 318 μs | 204 μs | RTX快1.56x |
| **有效带宽** | 847 GB/s | 1321 GB/s | RTX高1.56x |
| **带宽利用率** | 45.9% | 131% | RTX高2.85x |
| **端到端时间** | 1.96 ms | 1.27 ms | RTX快1.54x |

### 关键发现

1. **Mars X201需要8t/row** (不是4t/row!)，比4t/row快8.4%
2. **RTX 4090需要4t/row**，与NVIDIA标准配置一致
3. **高级优化技术均无效**: 多流、分批、网格缩放均无收益
4. **根本瓶颈**: L2 Cache大小差异 (2-4MB vs 72MB)

---

## 最优配置代码

### Mars X201 (Warp=64)

```cpp
// 1. Pinned Memory (必须!)
double* h_x;
cudaMallocHost(&h_x, numCols * sizeof(double));

// 2. 线程配置 - Vector 8t/row最优!
int threadsPerRow = 8;   // 8t/row最优 (不是4t/row!)
int blockSize = 128;
int gridSize = (numRows * threadsPerRow + blockSize - 1) / blockSize;

// 3. Cache配置 - PreferL1
cudaFuncSetCacheConfig(vector_kernel<8>, cudaFuncCachePreferL1);

// 4. Vector 8t/row Kernel
template<int TPR = 8>
__global__ void vector_kernel(
    int numRows, const int* __restrict__ rowPtr, const int* __restrict__ colIdx,
    const double* __restrict__ values, const double* __restrict__ x, double* __restrict__ y)
{
    const int WARP_SIZE = 64;  // Mars X201 warp size
    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int laneId = threadIdx.x % WARP_SIZE;
    int row = warpId * (WARP_SIZE / TPR) + laneId / TPR;
    
    if (row >= numRows) return;
    
    int rowStart = rowPtr[row], rowEnd = rowPtr[row + 1];
    double sum = 0.0;
    
    for (int i = rowStart + (laneId % TPR); i < rowEnd; i += TPR) {
        sum += values[i] * x[colIdx[i]];
    }
    
    for (int offset = TPR / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    if (laneId % TPR == 0) {
        y[row] = sum;
    }
}
```

### RTX 4090 (Warp=32)

```cpp
// 配置差异
int threadsPerRow = 4;   // 4t/row最优
int blockSize = 256;
cudaFuncSetCacheConfig(vector_kernel<4>, cudaFuncCachePreferL1);
```

---

## 穷尽性测试结果

### 1. 线程/行 (TPR) 测试

#### Mars X201

| TPR | 时间(μs) | 带宽(GB/s) | 利用率 |
|-----|----------|------------|--------|
| 2 | 620 | 434 | 23.6% |
| 4 | 345 | 780 | 42.3% |
| **8** | **318** | **847** | **45.9%** ⭐ |
| 16 | 362 | 743 | 40.3% |
| 32 | 538 | 501 | 27.2% |

#### RTX 4090

| TPR | 时间(μs) | 带宽(GB/s) | 利用率 |
|-----|----------|------------|--------|
| 2 | 213 | 1266 | 125.6% |
| **4** | **204** | **1321** | **131.0%** ⭐ |
| 8 | 204 | 1319 | 130.9% |
| 16 | 216 | 1245 | 123.5% |
| 32 | 404 | 666 | 66.1% |

### 2. Block Size 测试

| 平台 | 最优BS | 说明 |
|------|--------|------|
| Mars X201 | 128 | 64-512差异不大 |
| RTX 4090 | 256 | 64-512差异不大 |

### 3. 高级优化技术测试

| 技术 | Mars X201 | RTX 4090 | 结论 |
|------|-----------|----------|------|
| 多流并行 | 无收益 | ~1%收益 | ❌ 不推荐 |
| 分批处理 | 性能下降 | 性能下降 | ❌ 不推荐 |
| 网格缩放 | 性能严重下降 | 性能下降 | ❌ 不推荐 |

---

## 根本原因分析

### 为什么Mars X201利用率只有45.9%？

```
根本原因: L2 Cache太小 + 随机内存访问

1. x向量大小: 10.8 MB (1.26M × 8B)
2. Mars X201 L2: ~2-4 MB ❌ 无法缓存
3. RTX 4090 L2: 72 MB ✅ 完全缓存

结果:
- Mars: 每次访问都走DRAM → 45.9%利用率
- RTX: 后续访问命中L2 → 131%利用率 (>100%表示数据重用)
```

### 为什么Mars X201需要8t/row？

```
Warp Size = 64 (Mars) vs 32 (RTX)

假设: threadsPerRow = 4
- Mars: 每warp处理 64/4 = 16行
- RTX: 每warp处理 32/4 = 8行

问题: Mars每个线程处理更少元素
- avgNnz=10.71, 4线程各处理~2.7元素
- 线程很快完成计算，等待内存访问

解决方案: 增加到8t/row
- 每线程处理~1.35元素
- 更多线程并发隐藏内存延迟
- 利用率从42.3%提升到45.9%
```

### 为什么高级优化技术无效？

```
SpMV是内存受限，非计算受限

1. 多流并行:
   - 无法增加内存带宽
   - 适用于计算受限任务

2. 分批处理:
   - 额外kernel启动开销
   - 不改变随机访问模式

3. 网格缩放:
   - 已饱和内存带宽
   - 更多线程导致cache thrashing
```

---

## 测试文件索引

### 核心测试代码

| 文件 | 用途 |
|------|------|
| `tests/benchmark/test_comprehensive_optimization.cu` | 穷尽性参数搜索 |
| `tests/benchmark/test_advanced_techniques.cu` | 高级优化技术测试 |
| `tests/benchmark/test_vector_vs_scalar_timed.cu` | Vector vs Scalar对比 |

### 分析报告

| 文件 | 内容 |
|------|------|
| `doc/analysis/fp64_final_optimization_report_2026_04_10.md` | 最终优化报告 |
| `doc/analysis/advanced_techniques_analysis_2026_04_10.md` | 高级技术分析 |
| `doc/gpu_optimization_differences/mars_x201_vs_rtx4090_guide.md` | GPU开发差异指南 |

---

## 经验总结

### 开发注意事项

1. **必须使用Pinned Memory** - 端到端性能提升140%
2. **必须设置PreferL1** - Mars X201必须
3. **不要使用printf调试** - 使用日志库
4. **不要指定sm_xx架构** - Mars X201不支持
5. **使用pre_make编译** - Mars X201必须

### 平台差异速查

| 特性 | Mars X201 | RTX 4090 |
|------|-----------|----------|
| Warp Size | 64 | 32 |
| 最优TPR | 8 | 4 |
| 最优BS | 128 | 256 |
| L2 Cache | ~2-4 MB | 72 MB |
| 编译命令 | pre_make nvcc | nvcc |
| GPU监控 | ht-smi | nvidia-smi |

---

## 结论

**FP64 SpMV优化工作完成**:

1. ✅ 穷尽性参数搜索 - 发现8t/row是最优
2. ✅ 两平台验证完成 - 正确性和性能已验证
3. ✅ 根因分析完成 - L2 Cache和Warp Size是关键
4. ✅ 高级技术测试完成 - 均无收益
5. ✅ 文档完整 - 可供后续开发参考

**最终性能**:
- Mars X201: 318 μs, 847 GB/s, 45.9%利用率
- RTX 4090: 204 μs, 1321 GB/s, 131%利用率
- 差距: RTX快1.56x (主要因L2 Cache差异)