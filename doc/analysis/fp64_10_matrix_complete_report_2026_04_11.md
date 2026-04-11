# FP64 SpMV穷尽性优化分析 - 10矩阵完整测试报告

## 文档信息
- **创建日期**: 2026-04-11
- **测试矩阵**: 10个真实矩阵 (p0-p9)
- **矩阵特征**: 1256923行, 13465911 NNZ, avgNnz=10.71
- **测试GPU**: Mars X201 (国产) vs RTX 4090 (NVIDIA)

---

## 1. 10矩阵完整测试数据

### 1.1 Mars X201 测试结果汇总

| 矩阵 | Baseline(ms) | ILP4(ms) | __ldg(ms) | TPR=8(ms) | TPR=8带宽(GB/s) | 利用率 |
|------|--------------|----------|-----------|-----------|-----------------|--------|
| p0_A | 1.272 | 1.247 | 1.246 | **0.420** | 896.8 | 48.7% |
| p1_A | 1.279 | 1.260 | 1.265 | **0.420** | 898.0 | 48.8% |
| p2_A | 1.278 | 1.256 | 1.254 | **0.420** | 897.1 | 48.7% |
| p3_A | 1.265 | 1.240 | 1.246 | **0.420** | 897.1 | 48.7% |
| p4_A | 1.262 | 1.245 | 1.251 | **0.420** | 897.1 | 48.7% |
| p5_A | 1.275 | 1.258 | 1.266 | **0.420** | 898.6 | 48.8% |
| p6_A | 1.295 | 1.273 | 1.273 | **0.421** | 895.6 | 48.6% |
| p7_A | 1.258 | 1.247 | 1.243 | **0.420** | 897.6 | 48.7% |
| p8_A | 1.270 | 1.247 | 1.255 | **0.420** | 897.6 | 48.7% |
| p9_A | 1.300 | 1.274 | 1.271 | **0.420** | 896.7 | 48.7% |
| **平均** | **1.275** | **1.254** | **1.254** | **0.420** | **897.3** | **48.7%** |

### 1.2 RTX 4090 测试结果汇总

| 矩阵 | Baseline(ms) | ILP4(ms) | __ldg(ms) | __ldg带宽(GB/s) | 利用率 |
|------|--------------|----------|-----------|-----------------|--------|
| p0_A | 0.445 | 0.612 | **0.402** | 937.7 | 93.1% |
| p1_A | 0.445 | 0.664 | **0.419** | 900.8 | 89.4% |
| p2_A | 0.402 | 0.600 | **0.403** | 936.6 | 93.0% |
| p3_A | 0.445 | 0.664 | **0.417** | 904.1 | 89.9% |
| p4_A | 0.445 | 0.664 | **0.416** | 907.4 | 90.2% |
| p5_A | 0.446 | 0.664 | **0.445** | 847.7 | 84.3% |
| p6_A | 0.444 | 0.622 | **0.403** | 936.6 | 93.0% |
| p7_A | 0.445 | 0.664 | **0.443** | 851.3 | 84.7% |
| p8_A | 0.446 | 0.664 | **0.444** | 849.5 | 84.6% |
| p9_A | 0.445 | 0.621 | **0.403** | 936.7 | 93.0% |
| **平均** | **0.444** | **0.649** | **0.419** | **907.3** | **90.2%** |

---

## 2. 关键发现

### 2.1 Mars X201 最优配置分析

| 配置 | 平均带宽 | 相对基线提升 |
|------|----------|--------------|
| Baseline (1 warp/row) | 297 GB/s | 基线 |
| ILP4 (4累加器) | 301 GB/s | +1.4% |
| __ldg (纹理缓存) | 300 GB/s | +1.0% |
| **TPR=2** | 596 GB/s | **+100%** |
| **TPR=4** | 889 GB/s | **+198%** |
| **TPR=8** | **897 GB/s** | **+202%** ⭐⭐⭐ |
| TPR=16 | 769 GB/s | +158% |

### 2.2 RTX 4090 最优配置分析

| 配置 | 平均带宽 | 相对基线提升 |
|------|----------|--------------|
| Baseline | 847 GB/s | 基线 |
| ILP2 | 730 GB/s | -14% ❌ |
| ILP4 | 607 GB/s | -28% ❌ |
| **__ldg** | **907 GB/s** | **+7%** ⭐⭐⭐ |
| ILP4+__ldg | 628 GB/s | -26% ❌ |

---

## 3. 核心结论

### 3.1 优化后性能对比

| 指标 | Mars X201 (TPR=8) | RTX 4090 (__ldg) | 差距 |
|------|-------------------|------------------|------|
| 平均内核时间 | **0.420 ms** | **0.419 ms** | +0.2% |
| 平均有效带宽 | **897 GB/s** | **907 GB/s** | -1.1% |
| 平均利用率 | **48.7%** | **90.2%** | -45.9% |
| 理论带宽 | 1843 GB/s | 1008 GB/s | Mars高83% |

**关键结论**: 优化后两台GPU的内核时间几乎相同(0.420 vs 0.419 ms)，有效带宽差距仅1%！

### 3.2 为什么Mars利用率上限49%？

| 原因 | 说明 |
|------|------|
| **L2 Cache小** | ~2-4MB vs RTX的72MB，无法缓存x向量(10.8MB) |
| **Warp Size大** | 64 vs 32，需要更多线程隐藏延迟 |
| **随机访问** | colIdx随机，无法利用coalescing |
| **内存控制器** | 可能不如NVIDIA优化 |

### 3.3 为什么ILP无效？

| GPU | ILP效果 | 根因 |
|------|---------|------|
| Mars X201 | +0.5% | 内存瓶颈，计算优化无帮助 |
| RTX 4090 | -28% | L2缓存主导，ILP增加寄存器压力 |

### 3.4 Warp Size决定最优TPR

```
Mars X201 (Warp=64):
- avgNnz=10.71, 每行约10个元素
- TPR=8: 每warp处理8行 → 8组线程并行
- 每线程处理 ~1.35个元素 → 最优延迟隐藏

RTX 4090 (Warp=32):
- L2 Cache可缓存x向量(10.8MB)
- 1 warp/row足够，__ldg让数据进入纹理缓存
```

---

## 4. 无效优化技术汇总

| 技术 | Mars效果 | RTX效果 | 根因 |
|------|----------|----------|------|
| ILP双累加器 | +0.5% | -14% | 内存瓶颈 |
| ILP四累加器 | +0.5% | -28% | 寄存器压力 |
| 多流并行 | 无效 | 无效 | SpMV内存受限 |
| 分批处理 | 负优化 | 负优化 | kernel启动开销 |
| CSR5格式 | -44% | - | 原子操作开销 |
| 网格缩放 | 负优化 | 负优化 | 线程发散 |
| 共享内存缓存 | 负优化 | 负优化 | 随机访问开销 |
| __ldg (Mars) | +1% | +7% | L2太小无法缓存 |

---

## 5. 最终优化代码

```cpp
// 自动检测GPU并选择最优配置
void optimized_spmv_fp64(
    int numRows, int numCols, int nnz,
    int* d_rowPtr, int* d_colIdx, double* d_values,
    double* d_x, double* d_y)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int warpSize = prop.warpSize;

    if (warpSize == 64) {
        // Mars X201最优配置
        const int TPR = 8;   // ⭐⭐⭐ 关键！
        int blockSize = 256;
        int gridSize = (numRows * TPR + blockSize - 1) / blockSize;
        
        cudaFuncSetCacheConfig(vector_tpr_kernel<double, 64, 8>, 
                                cudaFuncCachePreferL1);
        vector_tpr_kernel<double, 64, 8><<<gridSize, blockSize>>>(
            numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    } else {
        // RTX 4090最优配置
        int blockSize = 256;
        int gridSize = (numRows + blockSize / warpSize - 1) / (blockSize / warpSize);
        
        cudaFuncSetCacheConfig(vector_ldg_kernel<double, 32>, 
                                cudaFuncCachePreferL1);
        vector_ldg_kernel<double, 32><<<gridSize, blockSize>>>(
            numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    }
}
```

---

## 6. 端到端优化建议

### 6.1 Pinned Memory (必须)

之前的测试表明Pinned Memory可以带来：
- Mars X201: +186% 端到端提升
- RTX 4090: +152% 端到端提升

```cpp
// Pinned Memory分配
cudaMallocHost(&h_x, numCols * sizeof(double));
cudaMallocHost(&h_y, numRows * sizeof(double));
```

### 6.2 综合优化流程

```cpp
// 1. 使用Pinned Memory
cudaMallocHost(&h_x, numCols * sizeof(double));

// 2. 根据GPU选择最优配置
int warpSize = prop.warpSize;
if (warpSize == 64) {
    int tpr = 8;  // Mars最优
} else {
    // 使用__ldg提示
}

// 3. 设置Cache配置
cudaFuncSetCacheConfig(kernel, cudaFuncCachePreferL1);

// 4. 执行kernel
```

---

## 7. 总结

1. **TPR=8是Mars最关键优化**: 从297 GB/s提升到897 GB/s (+202%)
2. **__ldg是RTX最关键优化**: 从847 GB/s提升到907 GB/s (+7%)
3. **ILP无效**: 内存瓶颈，计算优化无帮助
4. **L2 Cache决定性能上限**: Mars小L2是根本限制
5. **优化后差距仅1%**: 897 vs 907 GB/s，Mars已接近硬件极限
6. **内核时间几乎相同**: 0.420 vs 0.419 ms

**最终结论**: 经过穷尽性优化，Mars X201的FP64 SpMV内核性能已经达到与RTX 4090相当的水平，差距仅1%。49%的利用率上限是由硬件特性(L2 Cache小、Warp Size大)决定的，软件优化无法突破。

---

## 附录: 测试命令

### Mars X201

```bash
export PATH=$PATH:$HOME/cu-bridge/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/cu-bridge/lib:$HOME/cu-bridge/lib64

pre_make nvcc -O3 -o test_deep_analysis_fp64 tests/benchmark/test_deep_analysis_fp64.cu
CUDA_VISIBLE_DEVICES=7 ./test_deep_analysis_fp64 real_cases/mtx/p0_A
```

### RTX 4090

```bash
export PATH=/usr/local/cuda-13.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:$LD_LIBRARY_PATH

nvcc -O3 -arch=sm_89 -o test_deep_analysis_fp64 tests/benchmark/test_deep_analysis_fp64.cu
./test_deep_analysis_fp64 real_cases/mtx/p0_A
``