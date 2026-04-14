# SPMV FP32 库性能测试报告

## 概要

本报告展示 `spmv_fp32` 库在 **Mars X201** (国产GPU, Warp=64) 上的性能验证结果。

**关键特性:**
- 为 Mars X201 (warp=64) 优化的TPR内核
- 基于 avgNnz 的自适应TPR选择
- L1缓存配置以获得最优性能

---

## Mars X201 性能特性

| 指标 | Mars X201 |
|------|-----------|
| Warp大小 | **64** |
| 最优内核 | `tpr_kernel<64,8>` |
| 理论带宽 | 1843.2 GB/s |
| L2缓存 | ~2-4 MB |

---

## 自适应TPR选择

对于 Mars X201 (warp=64)，库根据 avgNnz 自动选择最优TPR：

| avgNnz范围 | TPR | 每warp行数 | 预期利用率 |
|--------------|-----|-----------|---------------------|
| >= 128 | 64 | 1 | 全warp |
| >= 64 | 32 | 2 | ~83% |
| >= 32 | 16 | 4 | ~80% |
| >= 16 | 8 | 8 | ~42% |
| < 16 | 4 | 16 | ~25% |

---

## FP32 vs FP64 对比

| 方面 | FP32 | FP64 |
|--------|------|------|
| 值大小 | 4字节 | 8字节 |
| 每nnz字节 | 12 | 20 |
| 内存带宽 | 更高 | 更低 |
| 精度 | ~7位小数 | ~15位小数 |

---

## 带宽计算

```
FP32 每迭代字节 = nnz * 12 + numRows * 4
FP64 每迭代字节 = nnz * 20 + numRows * 8

带宽 (GB/s) = 字节 / (时间_ms * 1e6)
```

---

## 关键优化技术

### 1. TPR优化 (Mars X201)

```cpp
// Mars X201: TPR=8 对 avgNnz~10 最优
int threadsPerRow = 8;
cudaFuncSetCacheConfig(kernel, cudaFuncCachePreferL1);
```

### 2. 固定内存

```cpp
// 固定内存带来 +150-186% 端到端加速
cudaMallocHost(&h_x, numCols * sizeof(float));
cudaMallocHost(&h_y, numRows * sizeof(float));
```

### 3. Warp归约修复

```cpp
// TPR>=64 对 Mars X201 需要64位掩码
if (TPR >= 64) {
    sum += __shfl_down_sync(0xffffffffffffffffULL, sum, 32);
}
```

---

## 建议

1. **对于 Mars X201**: 使用TPR=8内核并配置L1缓存偏好
2. **对于固定内存**: 始终使用 `cudaMallocHost` 为x和y向量分配内存
3. **对于迭代算法**: 使用设备指针模式避免H2D/D2H传输

---

**报告生成时间**: 2026-04-13
**库版本**: spmv_fp32 v1.0
**目标平台**: Mars X201 (国产GPU, Warp=64)