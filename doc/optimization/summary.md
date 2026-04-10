# SpMV优化手段汇总

## 文档信息
- **创建日期**: 2026-04-10
- **测试平台**: Mars X201 (国产GPU) vs RTX 4090 (NVIDIA)
- **测试矩阵**: 真实矩阵 (1.26M行, 13.5M NNZ, avgNnz=10.71)

---

## 优化效果总览

| 优化手段 | Mars X201 | RTX 4090 | 推荐使用 |
|----------|-----------|----------|----------|
| **Pinned Memory** | +140% | +120% | ⭐⭐⭐ 必须 |
| **Vector 8t/row (Mars)** | +8.4% | - | ⭐⭐⭐ Mars必须 |
| **Vector 4t/row (RTX)** | - | 最优 | ⭐⭐⭐ RTX最优 |
| **L1 Cache配置** | +8% | 微效 | ⭐⭐ 推荐使用 |
| 多流并行 | 无效 | ~1% | ❌ 不推荐 |
| 分批处理 | 负优化 | 负优化 | ❌ 不推荐 |
| 网格缩放 | 严重负优化 | 负优化 | ❌ 不推荐 |
| CSR5格式 | -44% | - | ❌ 不推荐 |
| RCM重排序 | 1-15% | 3% | ⭐ 效果有限 |

---

## 详细测试数据

### 测试矩阵特征

| 参数 | 值 |
|------|-----|
| 行数 | 1,256,923 |
| 列数 | ~1,300,000 |
| NNZ | 13,465,911 |
| avgNnz | 10.71 |
| x向量大小 | 10.8 MB |

### GPU硬件对比

| 特性 | Mars X201 | RTX 4090 |
|------|-----------|----------|
| Warp Size | 64 | 32 |
| 理论带宽 | 1843 GB/s | 1008 GB/s |
| L2 Cache | ~2-4 MB | 72 MB |
| SM数量 | 104 | 128 |

---

## 有效优化手段

### 1. Pinned Memory (端到端优化)

**原理**: 使用页锁定内存，避免CPU-GPU传输时的额外拷贝。

**效果**:

| 平台 | 端到端时间(优化前) | 端到端时间(优化后) | 提升 |
|------|-------------------|-------------------|------|
| Mars X201 | 5.6 ms | 1.96 ms | **+186%** |
| RTX 4090 | 3.2 ms | 1.27 ms | **+152%** |

**代码修改**:
```cpp
// 原来: 普通内存
double* h_x = (double*)malloc(numCols * sizeof(double));

// 优化后: Pinned Memory
double* h_x;
cudaMallocHost(&h_x, numCols * sizeof(double));
```

**适用场景**: 所有端到端SpMV计算

---

### 2. 线程/行配置优化

**原理**: 根据Warp Size选择最优线程分配策略。

**Mars X201 (Warp=64)**:

| TPR | 时间(μs) | 带宽(GB/s) | 利用率 |
|-----|----------|------------|--------|
| 2 | 620 | 434 | 23.6% |
| 4 | 345 | 780 | 42.3% |
| **8** | **318** | **847** | **45.9%** |
| 16 | 362 | 743 | 40.3% |

**结论**: Mars X201最优为**8t/row**

**RTX 4090 (Warp=32)**:

| TPR | 时间(μs) | 带宽(GB/s) | 利用率 |
|-----|----------|------------|--------|
| 2 | 213 | 1266 | 125.6% |
| **4** | **204** | **1321** | **131.0%** |
| 8 | 204 | 1319 | 130.9% |

**结论**: RTX 4090最优为**4t/row**

**代码修改**:
```cpp
// Mars X201
int threadsPerRow = 8;
int blockSize = 128;
int gridSize = (numRows * threadsPerRow + blockSize - 1) / blockSize;
vector_kernel<8><<<gridSize, blockSize>>>(...);

// RTX 4090
int threadsPerRow = 4;
int blockSize = 256;
int gridSize = (numRows * threadsPerRow + blockSize - 1) / blockSize;
vector_kernel<4><<<gridSize, blockSize>>>(...);
```

---

### 3. L1 Cache配置

**原理**: 优先使用L1 Cache而非共享内存。

**效果**:
- Mars X201: +8%
- RTX 4090: 效果不明显但无害

**代码修改**:
```cpp
cudaFuncSetCacheConfig(vector_kernel, cudaFuncCachePreferL1);
```

---

## 无效优化手段

### 1. 多流并行

**测试结果**:

| Streams | Mars X201 | RTX 4090 |
|---------|-----------|----------|
| 1 | 323 μs | 205 μs |
| 2 | 321 μs | 201 μs |
| 4 | 324 μs | 201 μs |
| 8 | 341 μs | 202 μs |

**原因**: SpMV是内存受限任务，多流无法增加内存带宽。

**结论**: 不推荐使用

---

### 2. 分批处理

**测试结果** (Mars X201):

| 批大小 | 时间(μs) | 相对性能 |
|--------|----------|----------|
| 全量(1256K) | 319 | 100% |
| 500K | 333 | -4% |
| 300K | 346 | -8% |
| 100K | 404 | -21% |

**原因**: 增加kernel启动开销，无L2 cache收益。

**结论**: 不推荐使用

---

### 3. 网格缩放

**测试结果** (Mars X201):

| 网格倍数 | 时间(μs) | 相对性能 |
|----------|----------|----------|
| 1x | 319 | 100% |
| 2x | 481 | -34% |
| 4x | 808 | -60% |
| 8x | 1489 | -79% |

**原因**: 线程发散增加，cache thrashing。

**结论**: 严禁使用

---

### 4. CSR5格式

**测试结果**:
- Mars X201: 性能下降44%
- 预处理开销大，原子操作代价高

**原因**: avgNnz=10.71时，原子操作开销超过负载均衡收益。

**结论**: 不推荐用于低密度矩阵

---

### 5. RCM重排序

**测试结果**:

| 矩阵 | Mars X201改进 |
|------|--------------|
| p0_A | +6% |
| p1_A | +15% |
| p2_A | +2% |
| p3_A | **-14%** |
| p4_A | +3% |

**原因**: 真实矩阵缺乏结构特性，重排序效果有限且可能负优化。

**结论**: 效果有限，谨慎使用

---

## 最终最优配置

### Mars X201

```cpp
// 线程配置
int threadsPerRow = 8;   // 8t/row最优
int blockSize = 128;
int gridSize = (numRows * 8 + 127) / 128;

// Cache配置
cudaFuncSetCacheConfig(vector_kernel<8>, cudaFuncCachePreferL1);

// 内存分配
double* h_x;
cudaMallocHost(&h_x, numCols * sizeof(double));  // Pinned Memory

// 内核调用
vector_kernel<8><<<gridSize, blockSize>>>(
    numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y
);
```

### RTX 4090

```cpp
// 线程配置
int threadsPerRow = 4;   // 4t/row最优
int blockSize = 256;
int gridSize = (numRows * 4 + 255) / 256;

// Cache配置
cudaFuncSetCacheConfig(vector_kernel<4>, cudaFuncCachePreferL1);

// 内存分配
double* h_x;
cudaMallocHost(&h_x, numCols * sizeof(double));

// 内核调用
vector_kernel<4><<<gridSize, blockSize>>>(
    numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y
);
```

---

## 详细文档索引

| 文档 | 内容 |
|------|------|
| [Pinned Memory优化](../techniques/pinned_memory.md) | 原理、实现、性能数据 |
| [线程配置优化](../techniques/threads_per_row.md) | TPR选择原理、不同GPU差异 |
| [L1 Cache配置](../techniques/l1_cache_config.md) | Cache配置原理、代码实现 |
| [无效技术分析](../techniques/ineffective_techniques.md) | 多流、分批、网格、CSR5分析 |
| [GPU开发差异](../gpu_differences.md) | Mars X201 vs RTX 4090开发指南 |