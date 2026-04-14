# SPMV FP64 库性能与精度测试报告

## 执行摘要

本报告展示 `spmv_fp64_real` 库在两个 GPU 平台上的完整性能和精度验证结果：
- **NVIDIA RTX 4090** (Warp=32)
- **Mars X201** (国产GPU, Warp=64)

**关键发现**:
- ✅ 两平台所有正确性测试均通过
- ✅ Mars X201 实现更高的内核带宽 (704.6 GB/s vs 584.7 GB/s)
- ✅ RTX 4090 实现更高的带宽利用率 (58% vs 38%)
- ✅ Profiler 工具 (hcTracer/nsys) 确认 CUDA Events 测量准确

---

## 测试配置

### 矩阵参数

| 参数 | 值 |
|-----------|-------|
| 行数 | 1,000,000 |
| 列数 | 1,000,000 |
| 每行平均非零元 | 10 |
| 总非零元数 | 10,000,000 |
| 矩阵模式 | Band + Random (列局部性) |
| 数据类型 | FP64 (double) |

### 测试环境

| 平台 | GPU | Warp Size | 理论带宽 | Profiler工具 |
|----------|-----|-----------|----------------|---------------|
| RTX 4090 | NVIDIA GeForce RTX 4090 | 32 | 1008 GB/s | nsys |
| Mars X201 | Mars 01 | 64 | 1843.2 GB/s | hcTracer |

### 带宽计算

```
每次迭代传输字节 = nnz * 20 + numRows * 8
                 = 10,000,000 * 20 + 1,000,000 * 8
                 = 208,000,000 bytes (208 MB)

带宽 (GB/s) = bytes / (time_ms * 1e6)
利用率 (%)  = bandwidth / theoreticalBW * 100
```

---

## 性能结果

### 结果汇总表

| 指标 | RTX 4090 | Mars X201 |
|--------|----------|-----------|
| 内核类型 | `ldg_kernel<32>` | `tpr_kernel<64,8>` |
| 平均内核时间 | **0.355 ms** | **0.293 ms** |
| 带宽 | **584.7 GB/s** | **704.6 GB/s** |
| 利用率 | **58.0%** | **38.2%** |
| 正确性 | ✅ 通过 | ✅ 通过 |

### 详细Profiler数据

#### RTX 4090 (nsys)

```
内核: void spmv_fp64_impl::ldg_kernel<(int)32>(...)
调用次数: 111 (warmup + benchmark + verify)
平均时间: 355,027 ns (0.355 ms)
最小: 354,849 ns
最大: 356,225 ns
标准差: 167 ns (非常稳定)
```

**内存传输汇总**:
- Host-to-Device: 140 MB (5次传输)
- Device-to-Host: 8 MB (1次传输)
- GPU内存操作总耗时: 17.5 ms

#### Mars X201 (hcTracer)

```
内核: void spmv_fp64_impl::tpr_kernel<64, 8>(...)
调用次数: 111 (warmup + benchmark + verify)
平均时间: 292,719 us (0.293 ms)
最小: 288,000 us
最大: 301,568 us
标准差: ~3.6 us
```

---

## 正确性验证

### 验证方法

1. GPU在测试矩阵上执行SpMV
2. CPU计算参考SpMV (前100行)
3. 比较GPU与CPU结果

### 结果

| 平台 | 错误数 (100个样本) | 状态 |
|----------|------------------------------|--------|
| RTX 4090 | 0 | ✅ 通过 |
| Mars X201 | 0 | ✅ 通过 |

**相对误差阈值**: 1e-10

---

## 内核分析

### RTX 4090: ldg_kernel<32>

`__ldg` 内核使用:
- Warp size: 32
- 每行1个warp (标准CSR-Vector方案)
- `__ldg()` 显式只读缓存加载
- L1缓存偏好设置用于CSR数据

**为什么58%利用率?**
- 随机列索引阻止完美内存合并
- L2缓存(72MB)有助于部分缓存x向量
- 内存受限操作限制可实现的带宽

### Mars X201: tpr_kernel<64, 8>

TPR=8 内核使用:
- Warp size: 64
- 每行8个线程 (TPR优化)
- L1缓存偏好设置 (对Mars X201至关重要!)
- 跨warp协调进行行聚合

**为什么38%利用率?**
- L2缓存(~2-4MB)比RTX 4090小
- 无法缓存整个x向量(8MB)
- Warp size 64需要更多线程来隐藏延迟
- 随机访问模式限制内存效率

---

## 平台对比

### Mars X201 vs RTX 4090

| 方面 | Mars X201 | RTX 4090 | 优胜者 |
|--------|-----------|----------|--------|
| 内核时间 | 0.293 ms | 0.355 ms | Mars (+18%) |
| 带宽 | 704.6 GB/s | 584.7 GB/s | Mars (+21%) |
| 利用率 | 38.2% | 58.0% | RTX (+52%) |
| L2缓存 | ~2-4 MB | 72 MB | RTX (+18x) |

**关键洞察**: Mars X201 实现**更高原始带宽**尽管**更低利用率**，原因:
1. 更高的理论带宽 (1843 vs 1008 GB/s)
2. TPR=8 优化专门针对 warp=64 调优

---

## 测试文件

### profiler_test.cu

Profiler测试使用CUDA Events进行精确计时:

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start, 0);
for (int i = 0; i < 100; i++) {
    spmv_fp64_execute_direct(numRows, nnz, d_rowPtr, d_colIdx, d_values, d_x, d_y, 0);
}
cudaEventRecord(stop, 0);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&total_time, start, stop);
```

### 其他测试

- `simple_test.cu` - 基础验证 (1000行)
- `benchmark_test.cu` - 多规模基准测试
- `comprehensive_test.cu` - 完整API覆盖

---

## 结论

### 性能验证

✅ **两平台均满足性能预期**:
- RTX 4090: 58%利用率匹配典型CSR-Vector性能
- Mars X201: 38%利用率符合小L2缓存架构预期

### 精度验证

✅ **所有测试通过正确性检查**:
- 100样本验证中零错误
- 两平台均保持FP64精度

### Profiler验证

✅ **Profiler工具确认测量结果**:
- hcTracer (Mars): 0.293 ms平均，匹配CUDA Events 0.295 ms
- nsys (RTX): 0.355 ms平均，匹配CUDA Events 0.356 ms

### 建议

1. **RTX 4090**: 使用 `__ldg` 内核获得最佳性能
2. **Mars X201**: 使用 TPR=8 内核配合L1缓存偏好设置
3. **两平台**: 使用 pinned memory (`cudaMallocHost`) 获得最佳端到端性能
4. **迭代算法**: 考虑使用设备指针模式避免H2D/D2H传输

---

## 附录: 原始测试输出

### RTX 4090 输出

```
========================================
  SPMV FP64 Profiler Test (CUDA Events)
========================================

GPU: NVIDIA GeForce RTX 4090
Warp Size: 32
Memory: 25.25 GB
Theoretical BW: 1008.0 GB/s

Matrix: 1000000 rows, avgNnz=10, total nnz=10000000

Warmup runs (10 iterations)...
Benchmark runs (100 iterations)...

=== Performance Results ===
Kernel time (avg): 0.356 ms
Bandwidth: 584.7 GB/s
Utilization: 58.0%
Kernel type: __ldg

=== Correctness Check ===
Correctness: PASSED (0 errors in 100 sample rows)
```

### Mars X201 输出

```
========================================
  SPMV FP64 Profiler Test (CUDA Events)
========================================

GPU: Mars 01
Warp Size: 64
Memory: 68.28 GB
Theoretical BW: 1843.2 GB/s

Matrix: 1000000 rows, avgNnz=10, total nnz=10000000

Warmup runs (10 iterations)...
Benchmark runs (100 iterations)...

=== Performance Results ===
Kernel time (avg): 0.295 ms
Bandwidth: 704.6 GB/s
Utilization: 38.2%
Kernel type: TPR=8

=== Correctness Check ===
Correctness: PASSED (0 errors in 100 sample rows)
```

---

**报告生成日期**: 2026-04-13
**库版本**: spmv_fp64 v1.0
**测试配置**: 1M行, avgNnz=10, CUDA Events基准测试