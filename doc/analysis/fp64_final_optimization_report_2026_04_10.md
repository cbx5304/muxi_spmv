# FP64 SpMV最终优化报告 - 两平台验证版

## 文档信息

- **创建日期**: 2026-04-10
- **测试平台**: Mars X201 (国产GPU) vs RTX 4090 (NVIDIA)
- **测试矩阵**: 真实矩阵 (1.26M行, 13.5M NNZ, avgNnz=10.71)

---

## 🔥🔥🔥 重大发现 (2026-04-10 穷尽性测试)

### 穷尽性测试揭示新最优配置！

#### Mars X201 最优配置 (更新)

| 内核 | 时间(μs) | 带宽(GB/s) | 利用率 | 说明 |
|------|----------|------------|--------|------|
| **Vector 8t/row** | **318** | **847** | **45.9%** | ⭐⭐⭐ 最优! |
| Vector 4t/row | 345 | 780 | 42.3% | 之前最优 |
| Vector 2t/row | 620 | 434 | 23.6% | 太少线程 |

**关键发现**: 8t/row比4t/row快**8.4%**！Warp=64需要更多线程隐藏延迟。

#### RTX 4090 最优配置

| 内核 | 时间(μs) | 带宽(GB/s) | 利用率 | 说明 |
|------|----------|------------|--------|------|
| **Vector 4t/row** | **204** | **1321** | **131.0%** | ⭐⭐⭐ 最优! |
| Vector 2t/row | 213 | 1266 | 125.6% | 接近最优 |
| Vector 8t/row | 204 | 1319 | 130.9% | 同样优秀 |

---

## 两平台最优配置对比

| 参数 | Mars X201 | RTX 4090 | 差异原因 |
|------|-----------|----------|----------|
| **最优线程/行** | **8t/row** | **4t/row** | Warp=64需要更多线程隐藏延迟 |
| 最优Block Size | 128 | 256 | 架构差异 |
| 最优Cache配置 | PreferL1 | PreferL1 | 两平台一致 |
| 内核时间 | 318 μs | 204 μs | RTX快1.56x |
| 有效带宽 | 847 GB/s | 1321 GB/s | RTX高1.56x |
| 带宽利用率 | **45.9%** | **131%** | L2 Cache差异 |

---

## 最终最优配置代码

### Mars X201 (国产GPU, Warp=64)

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

// 4. Vector 8t/row Kernel (最优)
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
    
    // Each thread processes stride elements
    for (int i = rowStart + (laneId % TPR); i < rowEnd; i += TPR) {
        sum += values[i] * x[colIdx[i]];
    }
    
    // Warp reduction
    for (int offset = TPR / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    if (laneId % TPR == 0) {
        y[row] = sum;
    }
}
```

### RTX 4090 (NVIDIA, Warp=32)

```cpp
// 1. Pinned Memory
double* h_x;
cudaMallocHost(&h_x, numCols * sizeof(double));

// 2. 线程配置 - Vector 4t/row最优
int threadsPerRow = 4;
int blockSize = 256;
int gridSize = (numRows * threadsPerRow + blockSize - 1) / blockSize;

// 3. Cache配置
cudaFuncSetCacheConfig(vector_kernel<4>, cudaFuncCachePreferL1);

// 4. Vector 4t/row Kernel (同上，TPR=4)
```

---

## 端到端性能 (Pinned Memory)

| 平台 | 端到端时间 | 端到端带宽 |
|------|------------|------------|
| **Mars X201** | **1.96 ms** | 137 GB/s |
| **RTX 4090** | **1.27 ms** | 213 GB/s |

**注意**: Mars X201端到端性能优于之前测试，可能与内存传输优化有关。

---

## 穷尽性测试结果详情

### Vector Kernel 线程/行对比

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

### Block Size 影响分析

#### Mars X201

| BS | 8t/row时间 | 利用率 | 说明 |
|----|------------|--------|------|
| 64 | 319 us | 45.9% | 可接受 |
| **128** | **318 us** | **45.9%** | ⭐ 最优 |
| 256 | 320 us | 45.9% | 略慢 |
| 512 | 324 us | 44.8% | 更慢 |

#### RTX 4090

| BS | 4t/row时间 | 利用率 | 说明 |
|----|------------|--------|------|
| 64 | 204 us | 131.8% | 可接受 |
| 128 | 204 us | 131.9% | 可接受 |
| **256** | **204 us** | **132.1%** | ⭐ 最优 |
| 512 | 204 us | 131.9% | 可接受 |

### Cache配置影响

| Cache配置 | Mars X201 | RTX 4090 |
|-----------|-----------|----------|
| PreferShared | 318 us | 204 us |
| **PreferL1** | **318 us** | **204 us** |
| PreferEqual | 318 us | 204 us |

**结论**: Cache配置对FP64影响不大，两平台可统一使用PreferL1。

---

## 根本原因分析

### 为什么Mars X201需要8t/row而RTX 4090只需要4t/row？

```
关键因素: Warp Size差异

Mars X201: Warp=64
- 每warp需要处理 64/TPR 行
- TPR=8时: 每8个线程处理1行，共8行/warp
- 更好隐藏内存延迟

RTX 4090: Warp=32  
- 每warp需要处理 32/TPR 行
- TPR=4时: 每4个线程处理1行，共8行/warp
- 等效的并行度
```

### 为什么RTX 4090利用率超过100%？

```
L2 Cache优势:
- RTX 4090: 72MB L2 Cache > 10.8MB x向量
- x向量可完全缓存，后续访问命中L2
- 等效于数据重用，带宽超过理论值

Mars X201: ~2-4MB L2 Cache < 10.8MB x向量
- 无法缓存x向量，每次访问都需要DRAM
- 带宽受限于真实DRAM带宽
```

---

## 测试文件索引

### 核心测试文件

- `tests/benchmark/test_comprehensive_optimization.cu` - 穷尽性参数搜索测试
- `tests/benchmark/test_vector_vs_scalar_timed.cu` - Vector vs Scalar对比测试
- `tests/benchmark/test_vector_correctness.cu` - 正确性验证测试

### 分析报告

- `doc/analysis/fp64_optimization_comparison_report_2026_04_09.md` - 两平台对比报告
- `doc/analysis/fp64_root_cause_analysis_2026_04_08.md` - 根因分析
- `doc/gpu_comparison/mars_x201_vs_rtx4090.md` - GPU硬件对比

---

## 结论

**Mars X201 FP64 SpMV优化工作最终完成**：

1. ✅ 穷尽性参数搜索完成 - 发现8t/row是最优配置
2. ✅ 两平台验证完成 - 正确性和性能都已验证
3. ✅ 根因分析完成 - L2 Cache和Warp Size是关键
4. ✅ 最优代码已确定 - 可直接使用

**最终性能指标**：

| 指标 | Mars X201 | RTX 4090 |
|------|-----------|----------|
| 最优内核 | Vector 8t/row | Vector 4t/row |
| 内核时间 | 318 μs | 204 μs |
| 有效带宽 | 847 GB/s | 1321 GB/s |
| 带宽利用率 | **45.9%** | **131%** |
| 端到端时间 | 1.96 ms | 1.27 ms |

**关键经验**: 对于Warp=64的GPU，使用8t/row而非4t/row！