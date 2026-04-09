# Mars X201 vs RTX 4090 GPU性能对比与优化指南

## 文档信息

- 创建日期: 2026-04-09
- 目的: 记录国产GPU Mars X201与NVIDIA RTX 4090的性能差异及优化策略差异

---

## 硬件规格对比

| 参数 | Mars X201 (国产) | RTX 4090 (NVIDIA) | 差异影响 |
|------|------------------|-------------------|----------|
| **Warp Size** | **64** | **32** | 关键差异，影响线程配置 |
| L2 Cache | ~2-4 MB | **72 MB** | **最关键差异** |
| 理论带宽 | 1843 GB/s | 1008 GB/s | Mars带宽更高但难利用 |
| SM数量 | 104 | 128 | 相近 |
| FP64性能 | 强 | 相对弱 | Mars适合科学计算 |
| 架构 | 类CUDA | Ada Lovelace | 编程模型兼容 |

---

## 核心性能差异根因

### L2 Cache大小是决定性因素

```
SpMV问题规模:
  x向量大小: 1.26M × 8B = 10.1 MB (FP64)

RTX 4090:
  L2 Cache: 72 MB > 10.1 MB ✓
  → x向量可完全缓存于L2
  → 后续访问命中L2
  → 带宽利用率: 107%

Mars X201:
  L2 Cache: ~2-4 MB < 10.1 MB ✗
  → x向量无法缓存
  → 每次访问都需DRAM
  → 带宽利用率: 27%
```

### 随机访问模式的影响

| 访问模式 | Mars X201利用率 | RTX 4090利用率 |
|----------|-----------------|----------------|
| 顺序访问 | 54-71% | 120%+ |
| 随机访问 | 27-48% | 100%+ |
| 真实矩阵 | 27% | 107% |

---

## SpMV优化策略对比

### 最优线程配置

| 平台 | 最优配置 | 利用率 | 原因 |
|------|----------|--------|------|
| **Mars X201** | **4t/row** | 27% | warp=64，4t/row=16行/warp |
| **RTX 4090** | **2t/row** | 107% | warp=32，2t/row=16行/warp |

**代码示例:**
```cpp
#if WARP_SIZE == 64
    const int THREADS_PER_ROW = 4;   // Mars X201
#else
    const int THREADS_PER_ROW = 2;   // RTX 4090
#endif
```

### L1缓存配置

| 平台 | 是否必须 | 最优配置 | 效果 |
|------|----------|----------|------|
| **Mars X201** | **必须** | **PreferEqual/PreferShared** | **+22%提升** ⭐ |
| RTX 4090 | 可选 | PreferL1 | 无明显影响 |

```cpp
// Mars X201必须设置 (新发现!)
cudaFuncSetCacheConfig(kernel, cudaFuncCachePreferEqual);  // 或 PreferShared

// 注意: PreferL1反而最慢！
```

### ILP优化效果

| 平台 | ILP2效果 | ILP4效果 | 建议 |
|------|----------|----------|------|
| Mars X201 | 0.64-0.85x (有害!) | 0.80-0.93x | **禁用** |
| RTX 4090 | 1.03-1.31x | 1.00-1.10x | 启用ILP2 |

**关键洞察**: Mars X201编译器已高度优化，手动ILP反而干扰。

### RCM矩阵重排序

| 平台 | 效果 | 建议 |
|------|------|------|
| Mars X201 | +1-15% (可能负优化) | 不推荐 |
| RTX 4090 | +3% | 可选 |

### Pinned Memory

| 平台 | 端到端提升 | 推荐度 |
|------|-----------|--------|
| Mars X201 | **+140%** | ⭐⭐⭐ 必须 |
| RTX 4090 | +20% | ⭐⭐ 推荐 |

```cpp
// 两平台都推荐
double* h_x;
cudaMallocHost(&h_x, numCols * sizeof(double));  // 关键!
```

---

## 优化技术有效性对比表

| 优化技术 | Mars X201 | RTX 4090 | 说明 |
|----------|-----------|----------|------|
| L1缓存配置 | ✅ 必须设置 | ⚪ 无影响 | Mars瓶颈在L2 |
| `__ldg`预取 | ⚪ 无效 | ⚪ 无效 | L2太小无法缓存 |
| ILP双累加 | ❌ 有害 | ✅ 有效 | 编译器已优化 |
| 共享内存 | ❌ -1000x | ❌ -10% | 随机访问开销大 |
| CSR5格式 | ❌ -44% | ❌ -20% | 原子操作开销 |
| Merge-based | ❌ -91% | ❌ -30% | 不适合avgNnz<10 |
| RCM重排序 | ⚪ +1-15% | ⚪ +3% | 矩阵无结构特性 |
| **Pinned Memory** | ✅ **+140%** | ✅ +20% | **最有效优化** |
| 多流并行 | ✅ +8% | ⚪ +2% | 2流最优 |

图例: ✅推荐 ⚪可选 ❌不推荐

---

## 编译与开发差异

### Mars X201特殊要求

```bash
# 1. 使用pre_make包装器
pre_make cmake ..
pre_make make

# 2. 不要指定sm_xx和CUDA_VERSION
# 错误: set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -sm_70")
# 正确: 让cu-bridge自动处理

# 3. 环境变量
export PATH=~/cu-bridge/bin:$PATH
export LD_LIBRARY_PATH=~/cu-bridge/lib:$LD_LIBRARY_PATH

# 4. 调试: 不用printf，使用日志库
# /c/Users/Lenovo/cbx/muxi_print_bug
```

### RTX 4090标准配置

```bash
# 标准CUDA编译
cmake -DCMAKE_CUDA_ARCHITECTURES=89 ..
make
```

---

## 性能分析工具

| 工具 | Mars X201 | RTX 4090 |
|------|-----------|----------|
| 性能分析 | hcTracer | nsys/ncu |
| GPU监控 | ht-smi | nvidia-smi |
| 事件计时 | cudaEvent返回0 | cudaEvent正常 |

### hcTracer用法

```bash
# Mars X201
CUDA_VISIBLE_DEVICES=7 hcTracer --hctx --odname results ./program

# 输出: tracer_out_*.json
# 解析: python3 parse_tracer.py
```

---

## 关键代码模板

### Mars X201最优SpMV内核

```cpp
__global__ void scalar_spmv_kernel(
    int numRows,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const double* __restrict__ values,
    const double* __restrict__ x,
    double* __restrict__ y)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numRows) return;

    int rowStart = rowPtr[tid];
    int rowEnd = rowPtr[tid + 1];
    double sum = 0.0;

    // 简单循环，不使用ILP
    for (int i = rowStart; i < rowEnd; i++) {
        sum += values[i] * x[colIdx[i]];
    }
    y[tid] = sum;
}

// 配置
int threadsPerRow = 4;
int blockSize = 256;
cudaFuncSetCacheConfig(scalar_spmv_kernel, cudaFuncCachePreferL1);
```

### RTX 4090最优SpMV内核

```cpp
__global__ void scalar_ilp2_kernel(
    int numRows,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const double* __restrict__ values,
    const double* __restrict__ x,
    double* __restrict__ y)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numRows) return;

    int rowStart = rowPtr[tid];
    int rowEnd = rowPtr[tid + 1];
    double sum0 = 0.0, sum1 = 0.0;  // ILP2

    int i = rowStart;
    for (; i + 1 < rowEnd; i += 2) {
        sum0 += values[i] * x[colIdx[i]];
        sum1 += values[i+1] * x[colIdx[i+1]];
    }
    
    double sum = sum0 + sum1;
    for (; i < rowEnd; i++) {
        sum += values[i] * x[colIdx[i]];
    }
    y[tid] = sum;
}

// 配置
int threadsPerRow = 2;
int blockSize = 256;
```

---

## 最终性能总结

### 端到端性能 (10矩阵平均, Pinned Memory)

| 精度 | Mars X201 | RTX 4090 | Mars加速比 |
|------|-----------|----------|------------|
| FP32 | 3.33 ms | 5.03 ms | 1.51x ✅ |
| FP64 | 4.90 ms | 7.70 ms | 1.57x ✅ |

**关键**: Mars X201端到端更快是因为Pinned Memory优化了数据传输。

### 内核性能

| 指标 | Mars X201 | RTX 4090 | 差距 |
|------|-----------|----------|------|
| 内核时间 | 3254 μs | 258 μs | RTX快12.6x |
| 带宽利用率 | 4.7% | 107% | RTX高22.8x |

---

## 经验教训

### 1. 不要盲目复制优化策略

> 同一优化技术（如ILP）在两平台效果完全相反！

### 2. 硬件架构决定优化方向

> L2 Cache大小差异决定了性能瓶颈的本质不同。

### 3. 简单往往是最优

> 对Mars X201，简单SCALAR内核是最佳选择，复杂优化反而有害。

### 4. 端到端优化比内核优化更重要

> Pinned Memory带来的+140%提升远超任何内核优化。

---

## 文档索引

- `doc/analysis/fp64_final_optimization_report_2026_04_09.md` - 最终优化报告
- `doc/analysis/fp64_root_cause_analysis_2026_04_08.md` - 根因分析
- `doc/analysis/fp64_exhaustive_optimization_2026_04_08.md` - 穷尽优化测试
- `doc/lessons/domestic_gpu_experience.md` - 国产GPU开发经验