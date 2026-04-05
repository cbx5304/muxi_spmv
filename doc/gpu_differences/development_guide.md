# Mars X201 vs RTX 4090 开发差异指南

本文档记录在Mars X201（国产GPU）和RTX 4090上开发SpMV时的关键差异和注意点。

---

## 0. 核心结论（2026-04-05 最终验证）

### ⚠️ 不同平台需要不同配置！

| 配置项 | Mars X201 (warp=64) | RTX 4090 (warp=32) |
|--------|---------------------|---------------------|
| **最优线程/行** | **4t/row** | **4t/row 或 2t/row** |
| **最优Block Size** | 512 | 256 或 512 |
| **L1缓存配置** | 必须设置 (+8%) | 可选 |
| **__ldg缓存提示** | 有效 (+11%) | 无效果（自动优化） |
| **Pinned Memory** | 关键 (+140% E2E) | 有效 (+20% E2E) |

### 格式选择结论

| 格式 | Mars X201 | RTX 4090 | 结论 |
|------|-----------|----------|------|
| **CSR Vector (4t/row)** | **26.63%** ✅ | **118.46%** ✅ | 最优方案 |
| CSR5-style | 14.97% ❌ | 86.83% ❌ | 原子操作开销 |
| Merge-based | 2.43% ❌ | 20.55% ❌ | 分区开销巨大 |

**结论**: CSR Vector (4t/row) 是两个平台的最优方案，不要使用CSR5或Merge-based处理avgNnz<64的稀疏矩阵。

---

## 1. 硬件差异

| 参数 | Mars X201 | RTX 4090 | 影响 |
|------|-----------|----------|------|
| Warp Size | 64 | 32 | 决定kernel设计 |
| L2 Cache | ~4 MB | 72 MB | 影响缓存策略 |
| 峰值带宽 | 1843 GB/s | 1008 GB/s | 带宽上限 |
| SM数量 | 104 | 128 | 并行度 |
| 内存架构 | HBM2e | GDDR6X | 访问特性 |

---

## 2. 编译差异

### 2.1 Mars X201编译方法

```bash
# 使用cu-bridge编译
export PATH=/home/chenbinxiangc/cu-bridge/bin:$PATH
export LD_LIBRARY_PATH=/home/chenbinxiangc/cu-bridge/CUDA_DIR/lib64:$LD_LIBRARY_PATH

# 使用pre_make包装器
pre_make cmake .. -DWARP_SIZE_64=ON
pre_make make -j8
```

### 2.2 RTX 4090编译方法

```bash
# 标准CUDA编译
cmake .. -DCUDA_ARCH=89
make -j8
```

### 2.3 注意事项

1. **不要指定sm_xx**：Mars X201不支持
2. **不要指定CUDA版本**：会自动替换
3. **使用find_package让CMake自动查找**：不要硬编码路径

---

## 3. Kernel设计差异

### 3.1 Warp Size影响

**Mars X201 (Warp=64)**:
```cpp
// 需要更多工作才能有效利用
constexpr int WARP_SIZE = 64;

// Warp-level reduction
value += __shfl_down_sync(0xffffffffffffffff, value, 32);
value += __shfl_down_sync(0xffffffffffffffff, value, 16);
value += __shfl_down_sync(0xffffffffffffffff, value, 8);
value += __shfl_down_sync(0xffffffffffffffff, value, 4);
value += __shfl_down_sync(0xffffffffffffffff, value, 2);
value += __shfl_down_sync(0xffffffffffffffff, value, 1);
```

**RTX 4090 (Warp=32)**:
```cpp
constexpr int WARP_SIZE = 32;

// Warp-level reduction
value += __shfl_down_sync(0xffffffff, value, 16);
value += __shfl_down_sync(0xffffffff, value, 8);
value += __shfl_down_sync(0xffffffff, value, 4);
value += __shfl_down_sync(0xffffffff, value, 2);
value += __shfl_down_sync(0xffffffff, value, 1);
```

### 3.2 寄存器使用

- Mars X201: 每SM寄存器数量不同，warp=64需要更多寄存器
- RTX 4090: 标准CUDA寄存器分配

### 3.3 共享内存

- Mars X201: 不同架构，共享内存使用效率不同
- 建议: 测试验证共享内存优化是否有效（在Mars X201上可能反而降低性能）

---

## 4. 性能特征差异

### 4.0 最终测试结果 (真实矩阵, 2026-04-05)

**测试矩阵**: 1.26M行, avgNnz=10.7, X向量=4.79MB

| 平台 | 最优Kernel | 利用率 | E2E时间 | 分析 |
|------|-----------|--------|---------|------|
| **Mars X201** | CSR 4t/row | **26.63%** | 0.85ms | L2缓存限制 |
| **RTX 4090** | CSR 4t/row | **118.46%** | 1.87ms | L2缓存效应 |

**关键发现**:
- Mars X201已达硬件极限，无法通过软件优化突破
- RTX 4090超带宽利用率是因为72MB L2缓存x向量
- Mars X201端到端更快（传输效率优势）

### 4.1 矩阵类型性能对比

| 矩阵类型 | Mars X201 | RTX 4090 | 说明 |
|----------|-----------|----------|------|
| Random (avgNnz=10) | 21% | 88% | RTX优势明显 |
| Banded (bw=30) | 95% | ~90% | Mars X201反而更好 |
| 稠密矩阵 | 40%+ | 85%+ | RTX优势 |

### 4.2 列数对性能的影响

**Mars X201**: 列数越多，性能越好（在一定程度上）
- cols=500: 9-11%
- cols=1000: 21%
- cols=2000: 33%

**RTX 4090**: 列数越少，性能越好
- cols=500: 166-221%
- cols=1000: 88%
- cols=2000: 67%

**原因**: 
- RTX 4090的72MB L2可以缓存小x向量
- Mars X201的4MB L2无法缓存，依赖访问模式

### 4.3 随机访问惩罚

- Mars X201: ~4-5x惩罚
- RTX 4090: ~2-3x惩罚（且有L2缓存缓解）

---

## 5. 不支持的特性

### 5.1 Mars X201不支持的功能

| 功能 | 状态 | 替代方案 |
|------|------|----------|
| `__ldg` 只读缓存 | 不支持 | 直接访问 |
| `prefetch` 指令 | 不支持 | 不使用 |
| `sm_xx` 算力指定 | 不支持 | 不指定 |
| printf调试 | 有问题 | 使用日志库 |
| 动态并行 | 不支持 | 单层kernel |

### 5.2 调试方法

Mars X201调试:
```cpp
// 使用日志库
#include "/c/Users/Lenovo/cbx/muxi_print_bug/log.h"
LOG_INFO("value = %f", value);
```

RTX 4090调试:
```cpp
// 可以使用printf
printf("value = %f\n", value);
```

---

## 6. 优化策略差异

### 6.0 关键优化效果（最终验证）

| 优化技术 | Mars X201 | RTX 4090 | 端到端影响 | 推荐度 |
|----------|-----------|----------|-----------|--------|
| **Pinned Memory** | +33% | +20% | **+140%** | ⭐⭐⭐ 关键 |
| **L1缓存配置** | **+8%** | 0% | +8% | ⭐⭐⭐ Mars必须 |
| **__ldg缓存提示** | +11% | 0% | +10% | ⭐⭐ Mars启用 |
| 多流(2流) | +8% | +8% | +8% | ⭐⭐ |
| CSR5格式 | **-44%** | -27% | 负面 | ❌ 不推荐 |
| Merge-based | **-91%** | -83% | 负面 | ❌ 不推荐 |

### 6.1 有效优化（两平台）

| 优化 | Mars X201 | RTX 4090 |
|------|-----------|----------|
| Merge-based kernel | ✅ 有效 | ✅ 有效 |
| Partition优化 | ✅ ePP=16最优 | ✅ ePP=32最优 |
| 向量化访问 | ✅ 有效 | ✅ 有效 |

### 6.2 无效优化（Mars X201）

| 优化 | 结果 | 原因 |
|------|------|------|
| 共享内存缓存 | -33% | 加载开销>收益 |
| __ldg只读缓存 | 0% | 硬件不支持 |
| prefetch | 0% | 硬件不支持 |

### 6.3 需要不同策略的优化

| 优化 | Mars X201 | RTX 4090 |
|------|-----------|----------|
| Block size | 256 (4 warps) | 256 (8 warps) |
| 分区策略 | ePP=16 | ePP=32 |
| 预处理 | 列重排序重要 | 可选 |

---

## 7. 性能测试差异

### 7.1 测试命令

Mars X201:
```bash
CUDA_VISIBLE_DEVICES=7 ./test_runner --rows 1000000 --cols 1000 --sparsity 0.01 --type random --merge --measure 50
```

RTX 4090:
```bash
./test_runner --rows 1000000 --cols 1000 --sparsity 0.01 --type random --merge --measure 50
```

### 7.2 性能监控

Mars X201:
```bash
ht-smi --show-usage --show-memory
ht-smi --show-hbm-bandwidth
```

RTX 4090:
```bash
nvidia-smi
nvidia-smi dmon
```

---

## 8. 最佳实践

### 8.1 跨平台代码

```cpp
// 使用编译时常量区分
#ifdef WARP_SIZE
    constexpr int WARP_SIZE_LOCAL = WARP_SIZE;
#else
    constexpr int WARP_SIZE_LOCAL = 32;
#endif

// 或运行时检测
int warpSize;
cudaDeviceGetAttribute(&warpSize, cudaDevAttrWarpSize, 0);
```

### 8.2 性能调优流程

1. **先在RTX 4090上验证正确性**
2. **移植到Mars X201**
3. **调整warp相关参数**
4. **测试不同配置**
5. **针对Mars X201特性优化**

### 8.3 性能差距预期

| 场景 | 预期差距 |
|------|----------|
| 随机稀疏矩阵 | 3-5x |
| 有局部性的矩阵 | 1-2x |
| 稠密矩阵 | 1.5-2x |

---

## 9. 常见问题

### Q1: 为什么Mars X201上共享内存优化反而降低性能？

A: Mars X201的共享内存架构与NVIDIA不同，加载开销可能超过缓存收益。需要实际测试验证。

### Q2: 为什么Mars X201上列数越多性能越好？

A: Mars X201的L2 cache小，无法有效缓存x向量。更大的列数意味着每行处理更多数据，摊薄了随机访问的开销。

### Q3: 如何判断是否需要预处理？

A: 如果应用场景是多次迭代（如迭代求解器），预处理开销可被摊销，建议使用列重排序。如果是单次计算，可能不值得。

---

*文档更新: 2026-04-05*
*适用于: Mars X201 (国产GPU) vs RTX 4090 (NVIDIA)*
*状态: **穷尽性优化完成 - 硬件极限确认***