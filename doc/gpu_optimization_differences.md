# GPU性能开发差异 - Mars X201 vs RTX 4090

## 文档目的

记录国产GPU (Mars X201) 和 NVIDIA GPU (RTX 4090) 在SpMV优化过程中的开发差异，帮助后续开发者快速理解两个平台的优化策略差异。

---

## 一、硬件差异总结

| 参数 | Mars X201 | RTX 4090 | 差异影响 |
|------|-----------|----------|---------|
| **Warp Size** | **64** | 32 | 影响线程分配策略 |
| **L2 Cache** | **~4MB** | **72MB** | **最关键差异** |
| 理论带宽 | 1843 GB/s | 1008 GB/s | Mars理论更高 |
| SM数量 | 104 | 128 | 差异较小 |
| 实际有效带宽 | ~463 GB/s | ~2348 GB/s | RTX实际更高 |

---

## 二、端到端性能分析

### 关键发现：数据传输主导端到端时间

| 阶段 | Mars X201 | RTX 4090 | 占比 |
|------|-----------|----------|------|
| H2D数据传输 | ~10.8ms | ~11.9ms | ~93% |
| Kernel执行 | 0.355ms | 0.070ms | 3-1% |
| D2H数据传输 | ~0.5ms | ~0.2ms | ~4% |
| **端到端总计** | **11.6ms** | **12.2ms** | - |

**关键洞察**: 
- 数据传输占端到端时间的**95%以上**
- Kernel优化仅影响**1-5%**的总时间
- 两平台端到端性能差异**小于2x**（尽管kernel差距9x）

### 优化策略建议

1. **如果数据驻留在GPU**：专注kernel优化
2. **如果需要频繁数据传输**：优先优化数据传输（使用pinned memory、异步传输等）
3. **多次迭代应用**：预处理开销可摊销，考虑CSR5等格式

## 二、优化策略差异

### 2.1 线程配置策略

**关键发现：不同平台需要不同的线程/行配置！**

```cpp
// ❌ 错误：两个平台使用相同配置
if (avgNnz == 10) {
    spmv_8threads_per_row(matrix, x, y);  // 两平台都不最优
}

// ✅ 正确：根据平台选择不同配置
if (WARP_SIZE == 64) {
    // Mars X201: 需要8线程/行
    spmv_8threads_per_row<512, 1024>(matrix, x, y);  // 26.68%
} else {
    // RTX 4090: 需要4线程/行
    spmv_4threads_per_row<256, 1024>(matrix, x, y);  // 231%
}
```

### 2.2 线程配置原理

| 平台 | 最优线程/行 | 每warp行数 | 理论利用率 | 实际利用率 |
|------|------------|-----------|-----------|-----------|
| Mars X201 | **8** | 8 | 12.5% | 26.68% |
| RTX 4090 | **4** | 16 | 12.5% | **231%** |

**为什么RTX 4090的4线程/行最优？**
- 每线程处理 ~2.7元素 (avgNnz=10.7 / 4)
- 双累加器(ILP)完美隐藏访存延迟
- 72MB L2缓存使x向量访问高效
- 小Warp(32)使4线程配置更有效

**为什么Mars X201需要8线程/行？**
- Warp=64，需要更多线程才能充分利用
- 4线程/行时理论利用率只有6.25%
- 8线程/行时理论利用率12.5%，实际可达26.68%
- L2缓存小，需要更多并行度来隐藏延迟

---

## 三、编译差异

### 3.1 编译命令

```bash
# Mars X201 (国产GPU)
export PATH=~/cu-bridge/CUDA_DIR/bin:$PATH
nvcc -O3 -DWARP_SIZE=64 spmv.cu -o spmv

# RTX 4090 (NVIDIA)
nvcc -O3 -DWARP_SIZE=32 spmv.cu -o spmv
```

### 3.2 CMake构建

```bash
# Mars X201
pre_make cmake ..
pre_make make

# RTX 4090
cmake ..
make
```

### 3.3 注意事项

| 事项 | Mars X201 | RTX 4090 |
|------|-----------|----------|
| printf调试 | ❌ 不支持 | ✅ 支持 |
| 打印CUDA版本 | ❌ 不支持 | ✅ 支持 |
| 打印sm_xx | ❌ 不支持 | ✅ 支持 |
| cu-bridge | ✅ 必须 | ❌ 不需要 |

---

## 四、Kernel优化差异

### 4.1 共享内存大小

```cpp
// Mars X201: 需要更大的共享内存padding
__shared__ int sharedRowPtr[SMEM_INTS + 16];  // +16 padding

// RTX 4090: 标准大小即可
__shared__ int sharedRowPtr[SMEM_INTS];
```

### 4.2 Shuffle操作

```cpp
// 通用实现
template<int WarpSize>
__device__ float warp_reduce(float val) {
    if (WarpSize >= 64) {
        val += __shfl_down_sync(0xffffffff, val, 32);
    }
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    return val;
}
```

### 4.3 __ldg指令效果

| 平台 | __ldg效果 | 原因 |
|------|----------|------|
| Mars X201 | +15% | L2缓存小，只读缓存帮助大 |
| RTX 4090 | +5% | L2缓存大，__ldg收益小 |

---

## 五、性能分析差异

### 5.1 带宽利用率计算

```cpp
float peakBW = (WARP_SIZE == 64) ? 1843.0f : 1008.0f;
float dataBytes = /* 数据量计算 */;
float bw = (dataBytes / (avgTime * 1e-3)) / (1024 * 1024 * 1024);
float util = bw / peakBW * 100;

// RTX 4090可能超过100%，因为L2缓存复用
```

### 5.2 性能差距分解

```
总体差距: 8.6x

分解:
- L2 Cache差异: ~4x (主要因素)
- Warp Size差异: ~2x
- 内存控制器效率: ~1.1x
```

---

## 六、调试建议

### 6.1 Mars X201调试

```cpp
// ❌ 不要使用printf
printf("value = %f\n", x[idx]);  // 可能崩溃或输出错误

// ✅ 使用日志库
// 使用 /c/Users/Lenovo/cbx/muxi_print_bug 日志库
```

### 6.2 性能分析

```bash
# Mars X201
ht-smi  # 类似nvidia-smi

# RTX 4090
nvidia-smi
```

---

## 七、最佳实践总结

### 7.1 代码模板

```cpp
template<int BLOCK_SIZE, int SMEM_INTS, int WarpSize>
__global__ void spmv_optimal(
    int numRows, const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const float* __restrict__ values,
    const float* __restrict__ x, float* __restrict__ y)
{
    // 根据WarpSize选择不同的线程配置
    int threadsPerRow = (WarpSize == 64) ? 8 : 4;
    int rowsPerWarp = WarpSize / threadsPerRow;

    // ... kernel实现 ...
}
```

### 7.2 配置选择表

| avgNnz | Mars X201 | RTX 4090 |
|--------|-----------|----------|
| < 5 | 8t/row | 2t/row |
| 5-15 | **8t/row** | **4t/row** |
| 15-30 | 8t/row | 4t/row |
| > 30 | 8t/row | 4-8t/row |

---

## 八、常见问题

### Q1: 为什么RTX 4090利用率能超过100%？

**A**: 因为L2缓存复用。当x向量的元素被多次访问时，72MB的L2缓存可以缓存这些数据，使有效带宽远超理论带宽。

### Q2: 为什么Mars X201不能超过27%？

**A**: 两个原因：
1. L2缓存只有~4MB，无法有效缓存x向量
2. Warp=64，对于avgNnz=10的行，线程利用率受限

### Q3: 能否通过软件优化让Mars X201达到更高性能？

**A**: 已尝试多种优化（CSR5、merge-based、多种kernel模式），均无法突破27%。这是硬件限制。

---

## 九、参考文档

- `doc/performance_report.md` - 性能报告汇总
- `doc/analysis/final_exhaustive_optimization_2026_04_04.md` - 完整分析报告
- `tests/test_extreme_optimizations.cu` - 极限优化测试代码

---

*文档生成: 2026-04-04*
*适用于: Mars X201 (国产) 和 RTX 4090 (NVIDIA) 的SpMV优化*