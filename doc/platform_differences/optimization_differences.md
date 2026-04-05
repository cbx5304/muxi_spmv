# Mars X201 vs RTX 4090 SpMV优化差异总结

## 硬件对比

| 特性 | Mars X201 | RTX 4090 |
|------|-----------|----------|
| Warp Size | 64 | 32 |
| 峰值带宽 | 1843 GB/s | 1008 GB/s |
| L2 Cache | ~4MB | 72MB |
| SM数量 | 104 | 128 |

## 🔥 关键发现 (2026-04-05 全矩阵验证)

### 1. 最优线程配置

| 平台 | 最优配置 | 利用率 | 原因 |
|------|----------|--------|------|
| **Mars X201** | **4t/row** | **26.41%** | warp=64, 16行/warp最优 |
| **RTX 4090** | **2t/row** | **119.09%** | warp=32, 16行/warp最优 |

### 2. L1缓存配置 (关键!)

**Mars X201必须显式设置L1缓存配置！**

```cpp
// Mars X201关键优化 (+8%提升)
cudaFuncSetCacheConfig(kernel, cudaFuncCachePreferL1);
```

| 配置 | Mars X201 | RTX 4090 |
|------|-----------|----------|
| Default | 24.47% | 118.27% |
| PreferL1 | **26.41%** | 118.42% |

RTX 4090无需设置，72MB L2缓存足够。

## 优化策略差异

### Mars X201特定优化

```cpp
// 1. L1缓存配置 (关键!)
cudaFuncSetCacheConfig(kernel, cudaFuncCachePreferL1);

// 2. 线程配置: 4t/row
const int THREADS_PER_ROW = 4;  // NOT 8!

// 3. Pinned Memory (端到端+140%)
cudaMallocHost(&h_x, numCols * sizeof(float));

// 4. Block Size: 512
#define BLOCK_SIZE 512

// 5. RCM重排序: 不启用 (+1.8%太小)

// 6. Kernel核心 (4t/row + DualAccum)
int rowsPerWarp = WARP_SIZE / 4;  // 64/4 = 16 rows
int baseRow = globalWarpId * rowsPerWarp;

// Dual accumulator for ILP
float sum0 = 0, sum1 = 0;
for (; idx + 4 < rowEnd; idx += 8) {
    sum0 += values[idx] * __ldg(&x[colIdx[idx]]);
    sum1 += values[idx + 4] * __ldg(&x[colIdx[idx + 4]]);
}
```

### RTX 4090优化

```cpp
// 1. 线程配置: 2t/row
const int THREADS_PER_ROW = 2;

// 2. Pinned Memory
cudaMallocHost(&h_x, numCols * sizeof(float));

// 3. RCM列重排序 (+11.4%)
applyRCMColumnReordering();

// 4. L1缓存配置: 无需 (72MB L2足够)
```

## 性能差距原因分析

### 1. Warp Size差异 (64 vs 32)

**问题**: Mars X201的warp=64导致极稀疏矩阵线程利用率低
```
avgNnz=4, warp=64: 理论利用率 = 4/64 = 6.25%
avgNnz=4, warp=32: 理论利用率 = 4/32 = 12.5%
```

**解决**: Adaptive Warp策略，每warp处理16行，每行4线程

### 2. L2 Cache差异 (~4MB vs 72MB)

**问题**: Mars X201无法缓存大矩阵的x向量
```
1M行 x 1000列矩阵: ~16MB数据
Mars X201 L2: 只能缓存25%
RTX 4090 L2: 可以完全缓存
```

**解决**: 使用__ldg指令通过只读缓存访问x向量

### 3. 共享内存架构差异

**发现**: Mars X201的共享内存性能与大块分配正相关
```
SMEM=68 ints: 3.7%
SMEM=512 ints: 24.7%
```

**建议**: 始终使用较大的共享内存分配

## 编译差异

### Mars X201

```bash
# 使用cu-bridge
export PATH=$HOME/cu-bridge/CUDA_DIR/bin:$PATH
export LIBRARY_PATH=$HOME/cu-bridge/CUDA_DIR/lib64:$LIBRARY_PATH

# 编译命令
nvcc -DWARP_SIZE=64 ...

# 注意事项
# - 不支持printf调试
# - 不支持sm_xx架构号
# - cmake命令前加pre_make
```

### RTX 4090

```bash
# 标准CUDA
export PATH=/usr/local/cuda/bin:$PATH
nvcc -DWARP_SIZE=32 ...
```

## 最终性能对比 (10个真实矩阵 p0_A~p9_A)

### Kernel性能

| 平台 | 配置 | 平均利用率 | 平均耗时 |
|------|------|-----------|----------|
| Mars X201 | 4t/row + L1 cache | **26.41%** | 0.337ms |
| RTX 4090 | 2t/row | **119.09%** | 0.074ms |

差距: **4.5x** (kernel层面)

### 端到端性能

| 指标 | Mars X201 | RTX 4090 |
|------|-----------|----------|
| H2D传输 | 0.138ms | 0.211ms |
| Kernel | 0.336ms | 0.056ms |
| D2H传输 | 0.369ms | 1.607ms |
| **总计** | **0.848ms** | **1.874ms** |

**Mars X201端到端更快: 2.2x** (传输效率优势)

## 经验教训

1. **L1缓存配置关键**: Mars X201必须显式设置cudaFuncCachePreferL1 (+8%)
2. **线程配置不同**: Mars X201用4t/row，RTX 4090用2t/row
3. **Warp Size影响最优配置**: 两平台最优都是16行/warp
4. **L2 Cache是瓶颈**: Mars X201的4MB L2无法缓存5MB x向量
5. **CSR5无效**: 原子操作开销太大，性能下降70%
6. **Pinned Memory关键**: 端到端+140%提升
7. **RCM仅RTX有效**: Mars X201无提升，RTX +11.4%

## 无效优化

| 技术 | 结果 | 原因 |
|------|------|------|
| CSR5 | 8.7% (-70%) | 原子操作开销 |
| Merge-based | 14.4% | 原子操作开销 |
| Shared Memory | -4% | 矩阵太大无局部性 |
| 行重排序 | -5% E2E | 额外恢复开销 |

---
*文档更新: 2026-04-05*
*验证矩阵: 10个真实矩阵 (p0_A ~ p9_A)*
*项目: muxi_spmv*