# FP64 SpMV优化对比报告 - 最终版

## 文档信息

- **创建日期**: 2026-04-09
- **更新日期**: 2026-04-09 (两平台验证完成!)
- **测试平台**: Mars X201 (国产GPU) vs RTX 4090 (NVIDIA)
- **测试矩阵**: 真实矩阵 (1.26M行, 13.5M NNZ, avgNnz=10.71)

---

## 🔥🔥🔥 重大发现 (2026-04-09 两平台验证完成!)

### Vector 4t/row Kernel 在两平台都是最优选择！

#### Mars X201 结果

| 内核 | 时间(μs) | 带宽(GB/s) | 利用率 | 加速比 |
|------|----------|------------|--------|--------|
| **Vector 4t/row** | **345** | **779** | **42.3%** | **1.68x** ⭐⭐⭐ |
| Simple Scalar | 580 | 464 | 25.2% | 1.00x |

#### RTX 4090 结果

| 内核 | 时间(μs) | 带宽(GB/s) | 利用率 | 加速比 |
|------|----------|------------|--------|--------|
| **Vector 4t/row** | **211** | **1278** | **126.8%** | **1.42x** ⭐⭐⭐ |
| Simple Scalar | 299 | 902 | 89.4% | 1.00x |

#### 两平台对比总结

| 指标 | Mars X201 | RTX 4090 | 对比 |
|------|-----------|----------|------|
| Vector Kernel时间 | 345 μs | 211 μs | RTX快1.63x |
| Vector Kernel带宽 | 779 GB/s | 1278 GB/s | RTX高1.64x |
| Vector Kernel利用率 | 42.3% | 126.8% | RTX高3x |
| Vector加速比 | 1.68x | 1.42x | **Mars提升更大** |

**关键发现**:
1. **Vector kernel在两平台都更优** - 分别加速1.68x和1.42x
2. **正确性验证通过** - 两平台最大误差 < 1e-19
3. **Mars X201提升更显著** - 从25%到42% (+17%)
4. **RTX 4090利用率超100%** - 说明使用了数据重用

---

## 执行摘要

经过穷尽性优化测试，确定了Mars X201 GPU上FP64 SpMV性能瓶颈的**根本原因**：

> **L2 Cache大小差异 (2-4MB vs 72MB) 导致随机内存访问性能差距**

**核心结论**：
1. **Vector 4t/row是最优内核** (利用率42.3% vs 25.2%)
2. 软件优化可以显著提升性能 (通过增加线程隐藏延迟)
3. Mars X201端到端性能优于RTX 4090 (Pinned Memory优化)
4. blockSize=128 + PreferShared 是最优配置

---

## 🔥 穷尽性优化测试结果 (2026-04-09)

### 测试1: Block Size对比 (Scalar + PreferEqual)

| Block Size | 时间(μs) | 带宽(GB/s) | 利用率 |
|------------|----------|------------|--------|
| **128** | **2808** | **99.5** | **5.4%** ⭐ |
| 256 | 2945 | 94.9 | 5.1% |
| 64 | 3195 | 87.4 | 4.7% |
| 1024 | 4375 | 63.9 | 3.5% |
| 512 | 6584626 | 0.0 | 0.0% ❌ |

**关键发现**: blockSize=128最优，blockSize=512导致严重性能下降

### 测试2: Threads Per Row对比 (Vector Kernel + PreferEqual)

| Threads/Row | 时间(μs) | 带宽(GB/s) | 利用率 |
|-------------|----------|------------|--------|
| **32** | **2887** | **96.8** | **5.3%** ⭐ |
| 8 | 3002 | 93.1 | 5.0% |
| 16 | 3047 | 91.7 | 5.0% |
| 4 | 3014 | 92.7 | 5.0% |
| 2 | 3096 | 90.2 | 4.9% |

**关键发现**: 32t/row最优，但与简单Scalar相比差异不大

### 测试3: 最优配置组合对比

| 配置 | 时间(μs) | 带宽(GB/s) | 利用率 | 加速比 |
|------|----------|------------|--------|--------|
| **Scalar+PreferShared (bs=128)** | **2849** | **98.1** | **5.3%** | **1.107x** ⭐ |
| Scalar+PreferEqual (bs=256) | 2968 | 94.1 | 5.1% | 1.063x |
| Scalar+PreferEqual (bs=128) | 2987 | 93.5 | 5.1% | 1.056x |
| Aligned+PreferEqual (bs=128) | 2994 | 93.3 | 5.1% | 1.053x |

**关键发现**: blockSize=128 + PreferShared 是最优组合

### 测试4: 多流并行

| Stream数量 | 时间(μs) | 带宽(GB/s) | 加速比 |
|------------|----------|------------|--------|
| **2 Streams** | **2804** | **99.6** | **1.058x** ⭐ |
| Single Stream | 2967 | 94.2 | 1.000x |
| 4 Streams | 3213 | 86.9 | 0.923x |
| 8 Streams | 3389 | 82.4 | 0.875x |

**关键发现**: 2流最优，4+流反而降低性能

---

## 最终最优配置

### Mars X201最优配置 (FP64) - 更新版

```cpp
// 1. Pinned Memory (必须!)
double* h_x;
cudaMallocHost(&h_x, numCols * sizeof(double));

// 2. 线程配置 - Vector Kernel最优!
int threadsPerRow = 4;   // 4t/row最优
int blockSize = 128;
int gridSize = (numRows * threadsPerRow + blockSize - 1) / blockSize;

// 3. Cache配置 (关键优化!)
cudaFuncSetCacheConfig(vector_kernel<4>, cudaFuncCachePreferShared);

// 4. 最优内核 (Vector 4t/row) - 1.68x加速!
template<int THREADS_PER_ROW>
__global__ void vector_kernel(
    int numRows, const int* __restrict__ rowPtr, const int* __restrict__ colIdx,
    const double* __restrict__ values, const double* __restrict__ x, double* __restrict__ y)
{
    const int WARP_SIZE = 64;  // Mars X201 warp size
    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int laneId = threadIdx.x % WARP_SIZE;
    int row = warpId * (WARP_SIZE / THREADS_PER_ROW) + laneId / THREADS_PER_ROW;
    
    if (row >= numRows) return;
    
    int rowStart = rowPtr[row], rowEnd = rowPtr[row + 1];
    double sum = 0.0;
    
    // Each thread processes stride elements
    for (int i = rowStart + (laneId % THREADS_PER_ROW); i < rowEnd; i += THREADS_PER_ROW) {
        sum += values[i] * x[colIdx[i]];
    }
    
    // Warp reduction
    for (int offset = THREADS_PER_ROW / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    if (laneId % THREADS_PER_ROW == 0) {
        y[row] = sum;
    }
}

// 5. 备选: 简单Scalar (如果Vector kernel不可用)
// 时间约580us，利用率25%
```
        sum += values[i] * x[colIdx[i]];
    }
    y[tid] = sum;
}

// 5. 可选: 2流并行 (+5.8%)
cudaStream_t stream0, stream1;
cudaStreamCreate(&stream0);
cudaStreamCreate(&stream1);
int halfRows = numRows / 2;
int gs0 = (halfRows + 127) / 128;
int gs1 = (numRows - halfRows + 127) / 128;
scalar_spmv_kernel<<<gs0, 128, 0, stream0>>>(halfRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
scalar_spmv_kernel<<<gs1, 128, 0, stream1>>>(numRows - halfRows, d_rowPtr + halfRows, d_colIdx, d_values, d_x, d_y + halfRows);
```

### RTX 4090最优配置 (FP64)

```cpp
// 1. Pinned Memory
double* h_x;
cudaMallocHost(&h_x, numCols * sizeof(double));

// 2. 线程配置
int blockSize = 256;
int threadsPerRow = 2;

// 3. Cache配置 (或不设置)
cudaFuncSetCacheConfig(kernel, cudaFuncCachePreferL1);

// 4. ILP优化 (可选)
__global__ void scalar_ilp2_kernel(...)
{
    double sum0 = 0.0, sum1 = 0.0;
    // ILP2优化
}
```

---

## 优化技术有效性汇总表

| 优化技术 | Mars X201 | RTX 4090 | 说明 |
|----------|-----------|----------|------|
| **Pinned Memory** | **+140%** ⭐⭐⭐ | +20% | **最关键优化** |
| **PreferShared Cache** | **+10.7%** ⭐⭐ | -17% ❌ | Mars关键优化 |
| **blockSize=128** | **+5.8%** ⭐ | 0% | Mars优化 |
| **2流并行** | **+5.8%** ⭐ | +2% | Mars优化 |
| ILP双累加 | +9.3% ⭐ | +11.8% ⭐ | 两平台都有效 |
| Vectorized Load | +8.0% | +11.8% | 两平台都有效 |
| `__ldg`预取 | +1.8% | +11.8% | RTX更有效 |
| 共享内存缓存 | **-1000x** ❌ | -10% | 随机访问开销 |
| CSR5格式 | **-44%** ❌ | -20% | 原子操作 |
| Merge-based | **-91%** ❌ | -30% | 不适合avgNnz<10 |
| RCM重排序 | +1-15% | +3% | 效果有限 |

---

## 测试平台对比

| 参数 | Mars X201 | RTX 4090 | 差异影响 |
|------|-----------|----------|----------|
| Warp Size | **64** | 32 | 线程配置差异 |
| L2 Cache | **~2-4 MB** | **72 MB** | **关键瓶颈** |
| 理论带宽 | 1843 GB/s | 1008 GB/s | Mars更高 |
| SM数量 | 104 | 128 | 相近 |

---

## 性能对比结果

### 1. 内核执行性能 (FP64, avgNnz=10.71)

| 指标 | Mars X201 | RTX 4090 | 差距 |
|------|-----------|----------|------|
| 内核时间 | 3254 μs | 258 μs | RTX快12.6x |
| 有效带宽 | 86 GB/s | 1082 GB/s | RTX高12.6x |
| 带宽利用率 | **4.7%** | **107%** | RTX高22.8x |

### 2. 端到端性能 (含Pinned Memory优化)

| 精度 | Mars X201 | RTX 4090 | Mars加速比 |
|------|-----------|----------|------------|
| FP32 | **3.33 ms** | 5.03 ms | **1.51x** ✅ |
| FP64 | **4.90 ms** | 7.70 ms | **1.57x** ✅ |

### 3. 不同avgNnz下的性能

| avgNnz | Mars X201利用率 | RTX 4090利用率 | 差距 |
|--------|-----------------|----------------|------|
| 4 | 6.1% | 58.0% | 9.5x |
| 8 | 8.9% | 78.0% | 8.8x |
| 10 | 27.0% | 107.0% | 4.0x |
| 16 | 10.7% | 98.0% | 9.2x |
| 32 | 13.2% | 117.0% | 8.9x |
| 64 | 13.7% | 127.0% | 9.3x |

---

## 根因分析

### 为什么RTX 4090内核快12.6倍？

```
SpMV x向量大小: 1.26M × 8B = 10.1 MB

RTX 4090:
  L2 Cache: 72 MB > 10.1 MB ✓
  → x向量可完全缓存
  → 后续访问命中L2
  → 107%利用率

Mars X201:
  L2 Cache: ~2-4 MB < 10.1 MB ✗
  → x向量无法缓存
  → 每次访问都需要DRAM
  → 27%利用率
```

### L2缓存影响验证

| 访问模式 | 矩阵大小 | Mars X201利用率 | RTX 4090利用率 |
|----------|----------|-----------------|----------------|
| 顺序访问 | 500K行 | **63.8%** | 120%+ |
| 顺序访问 | 1.26M行 | **54.3%** | 120%+ |
| 随机访问 | 500K行 | 47.6% | 100%+ |
| **随机访问** | **1.26M行** | **27.9%** | **107%** |
| 真实矩阵 | 1.26M行 | **27.9%** | **107%** |

**关键证据**: 顺序访问可达54-64%，随机访问只有27%，与真实矩阵完全一致！

---

## 穷尽优化测试结果

### 优化技术有效性汇总

| 优化技术 | Mars X201 | RTX 4090 | 说明 |
|----------|-----------|----------|------|
| **Pinned Memory** | **+140%** ⭐ | +20% | **唯一重大突破** |
| L1缓存配置 | +8% ⭐ | 0% | Mars必须设置 |
| 4t/row vs 8t/row | +2% | - | 线程配置 |
| `__ldg`预取 | 0% | 0% | L2太小无效 |
| ILP双累加 | **-36%** ❌ | +3-31% | Mars有害 |
| 共享内存缓存 | **-1000x** ❌ | -10% | 随机访问开销 |
| CSR5格式 | **-44%** ❌ | -20% | 原子操作 |
| Merge-based | **-91%** ❌ | -30% | 不适合avgNnz<10 |
| RCM重排序 | +1-15% | +3% | 效果有限 |
| 多流(2流) | +8% | +2% | 有帮助 |

### 图例

- ⭐ 推荐使用
- ❌ 禁止使用 (有害)
- 0% 无效果

---

## 最终优化配置

### Mars X201配置

```cpp
// 1. Pinned Memory (必须!)
double* h_x;
cudaMallocHost(&h_x, numCols * sizeof(double));

// 2. 线程配置
int threadsPerRow = 4;   // 4t/row最优
int blockSize = 256;

// 3. L1缓存配置 (必须!)
cudaFuncSetCacheConfig(kernel, cudaFuncCachePreferL1);

// 4. 最优内核 (简单SCALAR)
__global__ void scalar_spmv_kernel(
    int numRows, const int* rowPtr, const int* colIdx,
    const double* values, const double* x, double* y)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numRows) return;
    
    double sum = 0.0;
    for (int i = rowPtr[tid]; i < rowPtr[tid + 1]; i++) {
        sum += values[i] * x[colIdx[i]];
    }
    y[tid] = sum;
}
```

### RTX 4090配置

```cpp
// 1. Pinned Memory
double* h_x;
cudaMallocHost(&h_x, numCols * sizeof(double));

// 2. 线程配置
int threadsPerRow = 2;   // 2t/row最优
int blockSize = 256;

// 3. ILP优化 (可选)
__global__ void scalar_ilp2_kernel(...)
{
    // 使用双累加器
    double sum0 = 0.0, sum1 = 0.0;
    // ...
}
```

---

## 关键经验教训

### 1. 硬件架构决定优化方向

> L2 Cache大小差异是根本原因，软件优化无法弥补硬件差距。

### 2. 同一优化技术在两平台效果相反

> ILP优化: Mars X201 -36%, RTX 4090 +31%

### 3. 简单往往是最优

> Mars X201编译器已高度优化，手动优化反而有害。

### 4. 端到端优化比内核优化更重要

> Pinned Memory带来的+140%提升远超任何内核优化。

### 5. 接受硬件限制

> 27%带宽利用率是Mars X201的真实矩阵正常性能水平。

---

## 后续建议

### 短期优化

1. **启用Pinned Memory** - 已实现，+140%端到端提升
2. **设置L1缓存配置** - 已实现，+8%内核提升
3. **使用4t/row配置** - 已实现，+2%提升

### 中长期优化

1. **矩阵分块处理** - 对大规模矩阵分块，利用有限的L2缓存
2. **混合精度计算** - 对精度要求不高的部分使用FP32
3. **迭代求解器优化** - 使用缓存友好的预处理技术
4. **多流并行** - 2流提供+8%提升

### 不推荐的优化

1. ❌ ILP优化 - 对Mars X201有害
2. ❌ 共享内存缓存 - 随机访问开销太大
3. ❌ CSR5/Merge-based格式 - 原子操作开销大于收益

---

## 文档索引

### 分析报告

- `doc/analysis/fp64_final_optimization_report_2026_04_09.md` - 最终优化报告
- `doc/analysis/fp64_root_cause_analysis_2026_04_08.md` - 根因分析
- `doc/analysis/fp64_exhaustive_optimization_2026_04_08.md` - 穷尽优化测试

### GPU对比

- `doc/gpu_comparison/mars_x201_vs_rtx4090.md` - 硬件对比与优化指南
- `doc/gpu_comparison/test_files_index.md` - 测试文件索引

### 测试代码

- `tests/benchmark/test_l2_cache_effect.cu` - L2缓存影响测试
- `tests/benchmark/test_rcm_reordering.cu` - RCM重排序测试
- `tests/benchmark/test_rtx4090_baseline.cu` - RTX 4090基线测试
- `tests/benchmark/test_hctracer_comprehensive.cu` - hcTracer全面测试

---

## 结论

**Mars X201 FP64 SpMV优化工作已完成**：

1. ✅ 穷尽性分析完成 - 确定L2 Cache是根本瓶颈
2. ✅ 所有可行优化已测试 - 软件优化无法突破硬件限制
3. ✅ 最优配置已确定 - Pinned Memory + 4t/row + L1配置
4. ✅ 端到端性能优于RTX 4090 - 1.57x加速
5. ✅ 文档已完善 - 便于后续项目接手

**最终性能指标**：

| 指标 | 值 |
|------|-----|
| 内核带宽利用率 | 27% (硬件限制) |
| 端到端时间 | 4.90 ms (FP64) |
| 相对RTX 4090 | 1.57x更快 |

**接受27%的带宽利用率，这是Mars X201在真实稀疏矩阵上的正常性能水平。**