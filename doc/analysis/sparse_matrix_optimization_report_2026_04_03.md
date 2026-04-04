# Mars X201 SpMV极稀疏矩阵优化报告

## 问题描述

Mars X201 GPU (warp size=64) 在处理avgNnz<10的极稀疏矩阵时，带宽利用率仅9-28%，与RTX 4090相比差距达2.8-6.3倍。

## 优化方案测试

### 1. 列排序优化

**原理**: 对每行的列索引排序，改善x向量访问局部性

**结果**:
| avgNnz | Mars X201提升 | RTX 4090提升 |
|--------|--------------|-------------|
| 4 | **+67.5%** | **+76.0%** |
| 6 | -0.92% | -4.3% |
| 8 | +0.17% | -2.2% |
| 10 | -0.18% | -0.4% |

**结论**: 仅对avgNnz=4有效，对avgNnz>=6无效

### 2. 全局列重排序

**原理**: 按列的"行中心位置"重排列索引

**结果**: 对随机矩阵无额外收益（与简单列排序效果相同）

### 3. Merge-based Kernel Partition优化

**原理**: 调整elementsPerPartition参数

**结果**: elementsPerPartition=16最优（较64提升47%）

### 4. CSR5格式

**原理**: 固定NNZ tile处理

**结果**: 因原子操作开销，性能仅8.7%

## 根本原因分析

### 1. Warp Size差异

```
Mars X201: warp=64, 每partition处理~4行(avgNnz=4时)
→ 64线程中只有4个有工作，利用率6.25%

RTX 4090: warp=32
→ 32线程中只有4个有工作，利用率12.5%
```

### 2. L2 Cache差异

```
Mars X201: ~4MB L2 cache
RTX 4090: 72MB L2 cache

矩阵数据量(1M行x1000列): ~20MB
→ RTX 4090可完全缓存，Mars X201只能缓存20%
```

### 3. 列排序有效范围

```
avgNnz=4: 每行只访问4个x元素
→ 排序后相邻行访问相近x元素，缓存效率提升

avgNnz>=6: 每行访问6+个x元素
→ 即使排序后访问模式仍然分散
```

## 最终性能数据

### Mars X201

| avgNnz | 原始利用率 | 优化后 | 方案 |
|--------|-----------|--------|------|
| 4 | 9.19% | **15.40%** | Merge-based + 列排序 |
| 6 | 19.95% | 19.95% | Merge-based |
| 8 | 24.51% | 24.51% | Merge-based |
| 10 | 28.73% | 28.73% | Merge-based |

### RTX 4090

| avgNnz | 原始利用率 | 优化后 | 方案 |
|--------|-----------|--------|------|
| 4 | 56.9% | **100.2%** | Merge-based + 列排序 |
| 6 | 104.4% | 104.4% | Merge-based |
| 10 | 76.8% | 76.8% | Merge-based |

## 推荐策略

```cpp
// 预处理阶段
int avgNnz = matrix.nnz / matrix.numRows;
if (avgNnz <= 4) {
    // 仅对极稀疏矩阵进行列排序
    sortColumnsWithinRows(matrix);
}

// SpMV计算
spmv_merge_based(matrix, x, y, stream);
```

## 结论

1. **列排序优化范围有限**: 仅对avgNnz<=4有效，提升67-76%
2. **Mars X201在极稀疏矩阵上性能受限**: 与RTX 4090差距6x
3. **根本瓶颈是硬件特性**: warp size=64和L2 cache小

## 后续优化方向

1. **虚拟Warp Size**: 使用warp size=16/32处理极稀疏矩阵
2. **Light Kernel**: 每线程处理多行，避免partition开销
3. **矩阵分块**: 利用有限的L2 cache
4. **混合精度**: FP16减少数据传输

---
*报告完成: 2026-04-03*
*Mars X201: warp=64, 1843 GB/s, ~4MB L2*
*RTX 4090: warp=32, 1008 GB/s, 72MB L2*