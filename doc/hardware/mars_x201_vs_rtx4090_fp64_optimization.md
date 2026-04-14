# Mars X201 vs RTX 4090 FP64 SpMV优化开发差异指南

## 文档信息
- **创建日期**: 2026-04-12
- **目的**: 记录两种GPU达到相同性能的开发差异和注意事项
- **目标读者**: 新接手项目的开发人员

---

## 1. 核心硬件差异

### 1.1 关键参数对比

| 参数 | Mars X201 | RTX 4090 | 开发影响 |
|------|-----------|----------|----------|
| **Warp Size** | **64** | 32 | ⭐⭐⭐ **最关键差异** |
| **L2 Cache** | **~2-4MB** | 72MB | ⭐⭐⭐ **决定性能上限** |
| 理论带宽 | 1843 GB/s | 1008 GB/s | Mars更高 |
| SM数量 | 104 | 128 | - |
| 寄存器/SM | 131072 | 65536 | Mars更多 |

### 1.2 Warp Size差异的影响

```
NVIDIA RTX 4090 (Warp=32):
- 1个warp处理1行足够
- 32线程归约
- TPR优化效果有限

Mars X201 (Warp=64):
- 需要TPR=8才能达到最优
- 64线程需要更多并行隐藏延迟
- 代码模板必须适配64线程
```

---

## 2. 优化策略差异

### 2.1 Mars X201最优策略

```cpp
// ⭐⭐⭐ 最关键配置
const int TPR = 8;  // threadsPerRow = 8

// Warp内线程分配
int rowsPerWarp = WarpSize / TPR;  // 64/8 = 8行/warp
int threadInRow = lane % TPR;       // 0-7

// 必须设置Cache配置
cudaFuncSetCacheConfig(kernel, cudaFuncCachePreferL1);
```

### 2.2 RTX 4090最优策略

```cpp
// ⭐⭐⭐ 最关键配置
// 使用__ldg让数据进入纹理缓存

for (int i = rowStart + lane; i < rowEnd; i += warpSize) {
    int col = __ldg(&colIdx[i]);
    sum += __ldg(&values[i]) * __ldg(&x[col]);
}

// Warp归约
for (int offset = warpSize/2; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
}
```

---

## 3. 编译环境差异

### 3.1 Mars X201编译要求

```bash
# ⚠️ 必须使用pre_make nvcc
export PATH=$PATH:$HOME/cu-bridge/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/cu-bridge/lib:$HOME/cu-bridge/lib64

pre_make nvcc -O3 -o test_xxx tests/benchmark/test_xxx.cu

# ⚠️ 禁止使用以下选项:
# -arch=sm_xx        (不支持)
# -DCUDA_VERSION     (不支持)
```

### 3.2 RTX 4090编译要求

```bash
# 标准NVIDIA编译
nvcc -O3 -arch=sm_89 -o test_xxx tests/benchmark/test_xxx.cu
```

---

## 4. 无效优化技术对比

| 技术 | Mars效果 | RTX效果 | 为什么无效？ |
|------|----------|----------|--------------|
| **ILP双累加器** | +0.5% | **-14%** | 内存瓶颈 |
| **ILP四累加器** | +0.5% | **-28%** | 寄存器压力 |
| CSR5格式 | -44% | - | 原子操作开销 |
| 多流并行 | 0% | 1% | SpMV内存受限 |
| 共享内存缓存 | 负优化 | 负优化 | 随机访问开销 |

**重要**: ILP优化在RTX上**有害**！不要使用！

---

## 5. 性能上限差异分析

### 5.1 为什么Mars上限48.7%？

| 原因 | 说明 |
|------|------|
| **L2 Cache太小** | ~2-4MB无法缓存x向量(10.8MB) |
| **随机访问** | colIdx随机，无法利用coalescing |
| **Warp=64** | 需要更多线程隐藏延迟 |

### 5.2 为什么RTX上限88.8%？

| 原因 | 说明 |
|------|------|
| **L2 Cache大** | 72MB可完全缓存x向量 |
| **__ldg有效** | 利用纹理缓存进一步加速 |
| **Warp=32** | 延迟隐藏需求较低 |

---

## 6. 端到端优化差异

### 6.1 Pinned Memory

```cpp
// ⭐⭐⭐ 两平台都必须！
cudaMallocHost(&h_x, numCols * sizeof(double));
cudaMallocHost(&h_y, numRows * sizeof(double));
```

| 平台 | 端到端提升 |
|------|------------|
| Mars X201 | +186% |
| RTX 4090 | +152% |

### 6.2 端到端最终性能

| 精度 | Mars X201 | RTX 4090 | Mars更快 |
|------|-----------|----------|----------|
| FP32 | 3.33 ms | 5.03 ms | **1.51x** ✅ |
| FP64 | 4.93 ms | 7.57 ms | **1.54x** ✅ |

---

## 7. 代码适配模板

### 7.1 自动检测GPU类型

```cpp
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
int warpSize = prop.warpSize;

if (warpSize == 64) {
    // Mars X201配置
    const int TPR = 8;
    cudaFuncSetCacheConfig(kernel, cudaFuncCachePreferL1);
    vector_tpr_kernel<double, 64, 8><<<gridSize, blockSize>>>(...);
} else {
    // RTX 4090配置
    vector_ldg_kernel<double, 32><<<gridSize, blockSize>>>(...);
}
```

### 7.2 Warp归约适配

```cpp
// Mars X201 (Warp=64)
#pragma unroll
for (int offset = 32; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
}

// RTX 4090 (Warp=32)
#pragma unroll
for (int offset = 16; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
}
```

---

## 8. 测试和验证差异

### 8.1 Mars X201特殊注意事项

1. **不能使用printf调试**: 国产GPUprintf有问题
2. **使用GPU7**: `CUDA_VISIBLE_DEVICES=7`
3. **工作目录**: `/home/chenbinxiangc/spmv_muxi`
4. **性能分析**: 使用`/opt/hpcc/bin/hcTracer`

### 8.2 RTX 4090注意事项

1. **正常printf可用**: 标准CUDA行为
2. **工作目录**: `/home/test/cbx/spmv_muxi`
3. **性能分析**: 使用`nvprof`或Nsight

---

## 9. 反汇编工具差异

### 9.1 Mars X201工具

```bash
# 反汇编.so文件
htobjdump --print-code --source ****/aabbb.so > 1.txt

# 反汇编机器码
echo "0x10,0x00,... | /opt/x201-3.2.0.0/restricted/htgpu_llvm/bin/llvm-mc --arch=htc --disassemble

# 生成汇编文件
pre_make nvcc ./test.cpp -S -aop -lineinfo
```

### 9.2 RTX 4090工具

```bash
# 反汇编
cuobjdump -sass program.exe

# 生成汇编
nvcc -S -o program.sass program.cu
```

---

## 10. 常见问题和解决方案

### 10.1 Mars X201编译错误

| 问题 | 解决方案 |
|------|----------|
| `-arch=sm_xx`报错 | 删除该选项，国产GPU不支持 |
| CUDA_VERSION未定义 | 不使用该宏，使用运行时检测 |
| printf不输出 | 使用日志库或文件输出 |

### 10.2 性能调优差异

| 场景 | Mars策略 | RTX策略 |
|------|----------|----------|
| 低利用率 | 增加TPR | 检查__ldg使用 |
| 寄存器溢出 | 减少TPR | 减少ILP |
| 共享内存不足 | 减少使用 | 可使用更多 |

---

## 11. 开发流程建议

### 11.1 新项目开发流程

1. **先在RTX开发**: NVIDIA工具链更成熟
2. **适配Mars**: 修改warp size相关代码
3. **验证正确性**: 两平台都验证误差
4. **性能测试**: 使用真实矩阵测试

### 11.2 性能对比标准

| 指标 | Mars目标 | RTX目标 |
|------|----------|----------|
| FP64利用率 | 48.7% | 88.8% |
| 内核时间 | ~0.42 ms | ~0.42 ms |
| 端到端时间 | < 5 ms | < 8 ms |

---

## 附录: 服务器信息

| 服务器 | 地址 | SSH端口 | GPU | 工作目录 |
|--------|------|---------|-----|----------|
| Mars X201 | chenbinxiangc@172.16.45.81 | 19936 | 国产GPU | `/home/chenbinxiangc/spmv_muxi` |
| RTX 4090 | test@172.16.45.70 | 3000 | NVIDIA | `/home/test/cbx/spmv_muxi` |