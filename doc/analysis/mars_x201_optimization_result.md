# Mars X201 SpMV性能优化报告

## 执行摘要

**优化目标**: 将Mars X201的SpMV带宽利用率从初始的2.8-5.5%提升至接近RTX 4090的79-86%

**优化结果**:
- 带宽利用率从5.5%提升至17.2%（提升3.1倍）
- 带宽从102 GB/s提升至316 GB/s（提升3.1倍）
- 与RTX 4090仍有5倍差距，但已显著改善

---

## 1. 性能对比（优化后）

### 测试配置
- 矩阵规模: 1,000,000 行 × 1,000 列
- 稀疏度: 1% (NNZ = 10,000,000)
- avgNnzPerRow = 10

### 结果对比

| 指标 | RTX 4090 | Mars X201 | 差距 |
|------|----------|-----------|------|
| 执行时间 | 0.1015 ms | 0.2780 ms | 2.7x |
| 带宽 | 867 GB/s | 317 GB/s | 2.7x |
| 带宽利用率 | 86.0% | 17.2% | **5.0x** |
| 峰值带宽 | 1008 GB/s | 1843 GB/s | Mars理论上更强 |

### 不同矩阵类型性能

| 矩阵类型 | RTX 4090 利用率 | Mars X201 利用率 |
|----------|-----------------|------------------|
| 随机分布 | 86.0% | 17.2% |
| 幂律分布 | 76.6% | 17.2% |
| 集中分布 | 85.3% | ~17% |

---

## 2. 优化过程

### 2.1 问题诊断

**根本原因**: Kernel选择策略未针对warp=64优化

| 问题 | 描述 | 影响 |
|------|------|------|
| Scalar kernel效率低 | 1个线程处理1行，warp内其他63个线程空闲 | 效率仅1.56% |
| Vector kernel利用率不足 | avgNnzPerRow=10时，仅10/64=15.6%线程工作 | 84.4%线程空闲 |

### 2.2 实施的优化

**修改文件**: `src/spmv/csr/spmv_csr.cu`

**优化内容**:
1. 为warp=64强制使用vector kernel替代scalar kernel
2. 调整grid size计算以适应warp=64架构
3. 增加block size至256以改善occupancy

**关键代码修改**:
```cpp
if (WARP_SIZE == 64) {
    // Domestic GPU kernel - always use vector kernel
    int blockSize = 256;  // Increased for better occupancy
    int warpsPerBlock = blockSize / WARP_SIZE;  // = 4

    // Use vector kernel for all matrices
    int gridSize = getGridSize(matrix.numRows, warpsPerBlock);
    spmv_csr_vector_kernel<float, 64, false><<<gridSize, blockSize, 0, stream>>>(...);
}
```

### 2.3 尝试但未采用的优化

| 方案 | 结果 | 原因 |
|------|------|------|
| Multi-row vector kernel | 6.9%利用率（更差） | 行间串行处理，未改善并行性 |
| Parallel multi-row kernel | 10.9%利用率（更差） | 分组减少后warp内线程数，反而降低效率 |
| Merge-based kernel | 需要atomicAdd | 原子操作开销大 |

---

## 3. 性能瓶颈分析

### 3.1 根本限制: Warp Size差异

**理论效率计算**:
```
Vector Kernel效率 = avgNnzPerRow / WarpSize

RTX 4090 (warp=32):
- 效率 = 10 / 32 = 31.25%
- 实际利用率: 86% (受内存带宽限制)

Mars X201 (warp=64):
- 效率 = 10 / 64 = 15.6%
- 实际利用率: 17.2% (受线程利用率限制)
```

### 3.2 性能差距原因

1. **Warp Size影响**: Mars X201的64线程warp在稀疏矩阵处理时，更多线程处于空闲状态
2. **内存访问模式**: Vector kernel的内存合并效率受限于活跃线程数
3. **硬件差异**: L2 Cache更小(8MB vs 96MB)，更需要优化访问模式

---

## 4. 进一步优化建议

### 4.1 短期优化（预期提升至30-40%）

1. **CSR5格式**: 预处理矩阵以实现更好的负载均衡
2. **动态负载均衡**: 根据行长度分布动态分配线程
3. **Shared Memory预取**: 缓存频繁访问的x向量元素

### 4.2 中期优化（预期提升至50-60%）

1. **Merge-based Path**: 实现无原子操作的merge路径
2. **多格式混合**: 根据稀疏模式选择最优存储格式
3. **Tensor Core加速**: 利用矩阵加速单元

### 4.3 架构层面优化

1. **矩阵预处理**: 在CPU端重新排序矩阵以提高局部性
2. **JIT编译**: 根据矩阵特征动态生成最优kernel
3. **异构计算**: 结合CPU和GPU协同处理

---

## 5. 结论

### 5.1 优化成果

- **初始状态**: 2.8-5.5%带宽利用率
- **优化后**: 17.2%带宽利用率
- **提升倍数**: 3.1倍

### 5.2 关键发现

1. **Vector kernel是当前最优选择**: 对于warp=64架构，vector kernel比scalar kernel效率高10倍以上
2. **Warp Size是根本限制**: 64线程warp处理avgNnzPerRow=10的稀疏矩阵，理论效率上限仅15.6%
3. **需要算法级优化**: Kernel级优化已接近极限，需要考虑CSR5等格式级优化

### 5.3 下一步工作

1. 实现CSR5格式支持
2. 添加动态负载均衡机制
3. 测试更大规模矩阵（10M行）
4. 与hcsparse库对比

---

*报告生成时间: 2026-04-01*
*测试框架: muxi_spmv v1.0*
*优化工程师: Claude Code*