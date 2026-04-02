# Mars X201 SpMV 性能分析与优化最终报告

## 执行摘要

本报告记录了针对Mars X201 GPU上稀疏矩阵向量乘法(SpMV)的全面性能分析与优化工作。

### 最终成果 (2026-04-02 更新)

| 指标 | 初始状态 | Scalar优化 | Merge-based优化 | 总提升 |
|------|---------|-----------|-----------------|--------|
| 带宽利用率 (avgNnz=10) | 1.3% | 9.8% | **13.9%** | **10.7x** |
| 带宽利用率 (avgNnz=28) | - | 10.8% | **26.1%** | - |
| 带宽 (avgNnz=10) | 23 GB/s | 180 GB/s | **256 GB/s** | **11.1x** |

**与RTX 4090差距**: 6.3x (13.9% vs 88%) - 从原来的9x缩小

## 关键发现：Merge-based Kernel优化

### 问题背景

对于Mars X201 (warp=64) 处理稀疏矩阵 (avgNnz < 32):
- Vector kernel: 利用率 = avgNnz/64，对于avgNnz=10只有15.6%
- Scalar kernel: 9.8%利用率
- Light-balanced kernel: 同样约10%

### Merge-based Kernel优势

Merge-based kernel使用merge-path算法：
- 将行迭代和NNZ迭代合并为单一的merge path
- 沿path均匀划分工作给各个warp
- 消除线程空闲，实现负载均衡

### 性能对比

| avgNnz | Scalar/Light-balanced | Merge-based | 改进 |
|--------|----------------------|-------------|------|
| 10 | 9.8% | **13.9%** | +42% |
| 16 | 9.6% | **19.8%** | +106% |
| 24 | 10.7% | **21.9%** | +105% |
| 28 | 10.8% | **26.1%** | +142% |
| 32 | 37.0% (vector) | 25.7% | vector更优 |
| 64 | 52.0% (vector) | 44.8% | vector更优 |

### 最优Kernel选择策略

```
if (avgNnzPerRow < 32) {
    // 稀疏矩阵: 使用merge-based kernel
    spmv_merge_based(matrix, x, y, stream);
} else {
    // 较密矩阵: 使用vector kernel
    spmv_csr_vector_kernel<64><<<...>>>(...);
}
```

## 问题分析

### 根本原因

通过系统性测试，我们识别出多个瓶颈：

1. **L2 Cache抖动** (大矩阵主要瓶颈)
   - Mars X201 L2 cache: ~2-4 MB
   - 1M行的rowPtr: 8 MB
   - 当rowPtr超过cache容量时性能急剧下降

2. **Vector Kernel线程空闲** (稀疏矩阵主要瓶颈)
   - Warp=64，每warp处理一行
   - 对于avgNnz=10，只有10/64=15.6%的线程活跃
   - 大量计算资源浪费

3. **随机内存访问** (根本限制)
   - 稀疏矩阵的colIdx值分散
   - x[colIdx[i]]访问本质上是随机的
   - Mars X201的cache架构处理随机访问效率低

### 性能缩放分析

| 矩阵规模 | Scalar Kernel | Merge-based Kernel | 改进 |
|---------|--------------|-------------------|------|
| 1M行, avgNnz=10 | 9.8% | **13.9%** | +42% |
| 1M行, avgNnz=16 | 9.6% | **19.8%** | +106% |
| 1M行, avgNnz=28 | 10.8% | **26.1%** | +142% |
| 1M行, avgNnz=64 | 52.0% (vector) | 44.8% | vector更优 |

## 优化尝试总结

### 成功的优化

| 方案 | 结果 | 发现 |
|------|------|------|
| **Merge-based Kernel** | 13.9-26.1%利用率 | **最佳方案** - 对avgNnz<32最优 |
| Vector Kernel | 52%利用率 | 对avgNnz>=32最优 |
| Scalar Kernel | 9.8%利用率 | 基准方案 |

### 失败的优化尝试

| 方案 | 结果 | 原因 |
|------|------|------|
| 共享内存缓存 | 4.5% (回退33%) | 加载开销超过收益 |
| __ldg只读缓存提示 | 无改进 | Mars X201不支持此优化 |
| 不同grid size | 无改进 | 瓶颈是内存访问，不是并行度 |
| ELLPACK格式 | 1.06x, 180%内存开销 | 行长度变化大时不适用 |
| CSR5格式 | 8.7%利用率 | 原子操作开销过大 |

## 与RTX 4090对比

### 硬件差异

| 特性 | Mars X201 | RTX 4090 |
|------|-----------|----------|
| Warp大小 | 64 | 32 |
| L2 Cache | ~2-4 MB | ~72 MB |
| 峰值带宽 | 1843 GB/s | 1008 GB/s |
| SM数量 | 104 | 128 |

### 性能对比 (1M行, avgNnz=10)

| 指标 | Mars X201 | RTX 4090 | 差距 |
|------|-----------|----------|------|
| 利用率 | 13.9% | 88.0% | 6.3x |
| 带宽 | 256 GB/s | 887 GB/s | 3.5x |
| 执行时间 | 0.32 ms | 0.10 ms | 3.2x |

### 差距原因

1. **Cache架构**: RTX 4090有72MB L2 cache vs Mars X201的~4MB
   - 整个rowPtr可以放入RTX 4090的cache
   - Mars X201需要反复从DRAM获取

2. **内存控制器**: RTX 4090有更复杂的内存控制器
   - 更好地处理随机访问模式
   - 更有效的内存合并

3. **Warp大小**: 64线程 vs 32
   - Mars X201需要更多工作才能让warp高效
   - Merge-based kernel部分解决了这个问题

## 推荐方案

### Kernel选择策略

| avgNnzPerRow | 推荐Kernel | 预期利用率 |
|--------------|-----------|-----------|
| < 32 | Merge-based | 14-26% |
| 32-64 | Vector | 35-52% |
| > 64 | Vector | 50%+ |

### 代码修改

关键修改在 `src/spmv/csr/spmv_csr.cu`:
```cpp
// Mars X201 optimization: Use merge-based for sparse matrices
if (avgNnzPerRow < 32) {
    spmv_merge_based<float>(matrix, x, y, stream);
} else {
    spmv_csr_vector_kernel<float, 64, false><<<gridSize, blockSize>>>(...);
}
```

## 文件变更

| 文件 | 变更 |
|------|------|
| `src/spmv/csr/spmv_csr.cu` | 更新kernel调度逻辑，使用merge-based |
| `src/spmv/csr5/spmv_csr5.cu` | Merge-based kernel实现 |
| `doc/gpu_comparison/*.md` | 优化分析报告 |

## 结论

通过系统性分析和优化，我们在Mars X201上实现了**10.7x性能提升**（从1.3%到13.9%利用率）。关键发现：

1. **Merge-based kernel**是处理稀疏矩阵 (avgNnz < 32) 的最优方案
2. **Vector kernel**对于较密矩阵 (avgNnz >= 32) 仍然最优
3. **硬件架构差异**造成了与RTX 4090之间不可避免的6.3x差距

与RTX 4090的差距从9x缩小到6.3x，主要得益于merge-based kernel更好地利用了warp=64架构。

---

*报告更新时间: 2026-04-02*
*测试配置: 1M行 × 1K列, avgNnz=10*