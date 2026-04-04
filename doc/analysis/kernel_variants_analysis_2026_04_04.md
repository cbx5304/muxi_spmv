# SpMV Kernel Variants Performance Analysis

## 测试环境
- 矩阵: 1,256,923 × 1,256,923, nnz=13,465,911, avgNnz=10.7
- 测试次数: 5 iterations

## Mars X201 (WARP=64) 结果

| Kernel | 带宽利用率 | 耗时 | 相对性能 |
|--------|-----------|------|---------|
| **Adaptive(B512)** | **26.6%** | **0.335ms** | **1.00x (最优)** |
| VirtualWarp8 | 26.7% | 0.335ms | 1.00x |
| VirtualWarp16 | 23.0% | 0.387ms | 0.86x |
| VirtualWarp32 | 16.5% | 0.542ms | 0.62x |
| Vector | 9.3% | 0.957ms | 0.35x |
| Scalar | 8-20% | 0.44-1.1ms | 0.30-0.75x |

## RTX 4090 (WARP=32) 结果

| Kernel | 带宽利用率 | 耗时 | 相对性能 |
|--------|-----------|------|---------|
| **Adaptive(B512)** | **221%** | **0.074ms** | **1.00x (最优)** |
| VirtualWarp8 | 119% | 0.136ms | 0.54x |
| VirtualWarp16 | 106% | 0.154ms | 0.48x |
| VirtualWarp32 | 67% | 0.241ms | 0.31x |
| Vector | 67% | 0.242ms | 0.31x |
| Scalar | 1.7-113% | 0.14-9.3ms | 0.01-0.64x |

## 关键发现

### 1. Adaptive Kernel是跨平台最优选择
- Mars X201: 26.6% 利用率
- RTX 4090: 221% 利用率 (L2缓存效应)
- 两平台均表现最佳

### 2. VirtualWarp8 与 Adaptive 的差异
**Mars X201**:
- VirtualWarp8: 26.7% ≈ Adaptive: 26.6%
- 差异可忽略

**RTX 4090**:
- VirtualWarp8: 119% vs Adaptive: 221%
- Adaptive快1.86x

**原因分析**:
- Adaptive使用prefetch循环模式
- RTX 4090的大L2缓存使prefetch更有效
- Mars X201的小L2缓存限制了prefetch收益

### 3. Warp Size影响

| 配置 | Mars X201 | RTX 4090 | 差距 |
|------|-----------|----------|------|
| Adaptive | 26.6% | 221% | 8.3x |
| VirtualWarp8 | 26.7% | 119% | 4.5x |
| Vector | 9.3% | 67% | 7.2x |

### 4. 线程分配策略分析

对于avgNnz=10.7:
- **VirtualWarp8**: 每行8线程，每线程处理~1.3元素
- **Adaptive**: 每行4线程，每线程处理~2.7元素
- **Vector**: 每行64/32线程，每线程处理~0.17/0.33元素

**最优策略**: 每线程处理1-3个元素时效率最高

## 优化建议

### Mars X201
1. 使用Adaptive或VirtualWarp8 kernel
2. Block Size: 512
3. 共享内存: 512 ints
4. Prefetch: 可选（收益有限）

### RTX 4090
1. 使用Adaptive kernel
2. Block Size: 512
3. 共享内存: 512 ints
4. Prefetch: 推荐

## 性能差距根本原因

### 硬件因素
1. **L2 Cache**: 4MB vs 72MB (18x差距)
2. **Warp Size**: 64 vs 32 (2x理论利用率差距)
3. **内存控制器**: 效率差异

### 软件因素
1. **cu-bridge开销**: 可能引入额外延迟
2. **编译器优化**: 国产GPU编译器优化能力有限
3. **驱动效率**: 驱动成熟度差异

---

*报告生成: 2026-04-04*
*测试矩阵: p0_A, p1_A, p2_A*