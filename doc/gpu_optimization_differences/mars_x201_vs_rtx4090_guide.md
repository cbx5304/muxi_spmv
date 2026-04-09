# Mars X201 vs RTX 4090 GPU开发差异指南

## 文档信息

- **创建日期**: 2026-04-10
- **目标**: 记录国产GPU (Mars X201) 与 NVIDIA RTX 4090 的开发差异，帮助开发者快速适配两平台

---

## 🔥 核心差异速查表

| 特性 | Mars X201 (国产) | RTX 4090 (NVIDIA) | 影响 |
|------|------------------|-------------------|------|
| **Warp Size** | **64** | 32 | ⭐⭐⭐ 关键差异 |
| **L2 Cache** | **~2-4 MB** | 72 MB | ⭐⭐⭐ 性能瓶颈 |
| **理论带宽** | 1843 GB/s | 1008 GB/s | Mars更高 |
| **SM数量** | 104 | 128 | 相近 |
| **编译器** | hc (通过cu-bridge) | nvcc | 编译命令不同 |
| **FP64最优TPR** | **8t/row** | 4t/row | ⭐⭐ 配置差异 |
| **最优BlockSize** | 128 | 256 | 配置差异 |

---

## 1. 编译环境差异

### 1.1 编译命令

**Mars X201 (国产GPU):**
```bash
# 必须使用pre_make包装
export PATH=$PATH:~/cu-bridge/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/cu-bridge/lib:~/cu-bridge/lib64

# 编译命令
pre_make nvcc -O3 -o output source.cu

# CMake编译
pre_make cmake ..
pre_make make
```

**RTX 4090 (NVIDIA):**
```bash
# 直接使用nvcc
nvcc -O3 -o output source.cu

# CMake编译
cmake ..
make
```

### 1.2 CMakeLists.txt 注意事项

```cmake
# 不要硬编码CUDA路径
# 错误示例:
# set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")

# 正确做法: 让find_package自动查找
find_package(CUDA REQUIRED)

# 不要指定CUDA架构 (国产GPU不支持)
# 错误示例:
# set(CMAKE_CUDA_ARCHITECTURES "89")

# 正确做法: 留空或条件判断
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    # 国产GPU不设置架构
else()
    # NVIDIA GPU可以设置
endif()
```

### 1.3 不支持的CUDA特性

| 特性 | Mars X201 | RTX 4090 | 替代方案 |
|------|-----------|----------|----------|
| `sm_xx` 架构指定 | ❌ 不支持 | ✅ 支持 | 不指定，让编译器自动处理 |
| `CUDA_VERSION` 宏 | ❌ 不支持 | ✅ 支持 | 使用条件编译或运行时检测 |
| `printf` in kernel | ⚠️ 有问题 | ✅ 支持 | 使用自定义日志库 |
| `__CUDA_ARCH__` | ⚠️ 部分支持 | ✅ 支持 | 谨慎使用 |

---

## 2. Warp Size差异 (关键!)

### 2.1 影响分析

```
Mars X201: Warp = 64 threads
RTX 4090:  Warp = 32 threads

影响:
1. Warp级归约需要不同的偏移量
2. 每 warp 处理的行数不同
3. 线程利用率和延迟隐藏能力不同
```

### 2.2 自适应代码模板

```cpp
// 方法1: 编译时检测 (推荐)
#ifdef __CUDA_ARCH__
    #if __CUDA_ARCH__ >= 700  // NVIDIA Volta+
        #define WARP_SIZE 32
    #else
        #define WARP_SIZE 64  // Mars X201
    #endif
#else
    #define WARP_SIZE 32  // 默认NVIDIA
#endif

// 方法2: 运行时检测 (更可靠)
__device__ int getWarpSize() {
    // Mars X201 没有可靠的__CUDA_ARCH__检测
    // 建议通过环境变量或编译选项传递
    #ifdef MARS_X201
        return 64;
    #else
        return 32;
    #endif
}

// 方法3: 模板参数 (最灵活)
template<int WARP_SIZE>
__global__ void spmv_kernel(...) {
    // 编译时确定
}
```

### 2.3 Warp Shuffle注意事项

```cpp
// Warp Shuffle在Mars X201上的注意事项
template<int TPR>
__device__ double warpReduce(double val) {
    // TPR = Threads Per Row
    // Mars X201: WARP_SIZE=64
    // RTX 4090: WARP_SIZE=32
    
    // 关键: 偏移量必须与TPR匹配
    for (int offset = TPR / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}
```

---

## 3. L2 Cache差异 (性能瓶颈!)

### 3.1 问题分析

```
场景: SpMV x向量访问
- x向量大小: 10.8 MB (1.26M × 8B)
- Mars X201 L2: ~2-4 MB ❌ 无法缓存
- RTX 4090 L2: 72 MB ✅ 完全缓存

结果:
- Mars X201: 每次访问都走DRAM → 45.9%利用率
- RTX 4090: 后续访问命中L2 → 131%利用率
```

### 3.2 优化策略差异

| 策略 | Mars X201 | RTX 4090 | 说明 |
|------|-----------|----------|------|
| **矩阵分块** | ⭐⭐⭐ 推荐 | 不需要 | 减少工作集大小 |
| **RCM重排序** | ⭐ 效果有限 | ⭐⭐⭐ 有效 | 提升缓存局部性 |
| **数据预取** | ⚠️ 效果有限 | ⭐⭐ 有效 | L2太小预取无效 |
| **共享内存缓存** | ❌ 不推荐 | ⭐ 可考虑 | 随机访问开销大 |
| **Pinned Memory** | ⭐⭐⭐ 必须 | ⭐⭐ 推荐 | 端到端优化 |

### 3.3 Mars X201特有优化

```cpp
// 对于大矩阵，考虑分块处理
// 每块控制在 L2 Cache可缓存范围内
int blockSize = 300000;  // ~300K行
for (int start = 0; start < numRows; start += blockSize) {
    int end = min(start + blockSize, numRows);
    // 处理子块
    spmv_kernel<<<...>>>(end - start, ...);
}
```

---

## 4. 最优配置差异

### 4.1 FP64 SpMV最优配置

**Mars X201:**
```cpp
int threadsPerRow = 8;   // 8t/row最优!
int blockSize = 128;
cudaFuncSetCacheConfig(kernel, cudaFuncCachePreferL1);
cudaMallocHost(&h_x, numCols * sizeof(double));  // Pinned Memory必须
```

**RTX 4090:**
```cpp
int threadsPerRow = 4;   // 4t/row最优
int blockSize = 256;
cudaFuncSetCacheConfig(kernel, cudaFuncCachePreferL1);
cudaMallocHost(&h_x, numCols * sizeof(double));
```

### 4.2 为什么Mars需要更多线程/行?

```
Warp Size = 64 (Mars) vs 32 (RTX)

假设: threadsPerRow = 4
- Mars: 每 warp 处理 64/4 = 16 行
- RTX:  每 warp 处理 32/4 = 8 行

问题: Mars每个线程处理更少元素 (avgNnz=10.71时)
- Mars: 每行10.71元素，4线程各处理~2.7元素
- 线程很快完成计算，等待内存访问

解决方案: 增加到8t/row
- Mars: 每线程处理~1.35元素
- 更多线程并发隐藏内存延迟
- 利用率从42.3%提升到45.9%
```

---

## 5. 性能分析工具差异

### 5.1 性能分析命令

| 功能 | Mars X201 | RTX 4090 |
|------|-----------|----------|
| GPU状态 | `ht-smi` | `nvidia-smi` |
| 性能分析 | `hcTracer --hctx ./prog` | `nsys profile ./prog` |
| 时间测量 | cudaEvent (可能返回0) | cudaEvent |

### 5.2 hcTracer使用 (Mars X201)

```bash
# 基本用法
CUDA_VISIBLE_DEVICES=7 hcTracer --hctx --odname results ./program

# 输出目录
# 结果保存在 tracer_out_YYYYMMDDHHMMSS/ 目录

# 限制: 
# - 不能使用printf (会崩溃)
# - 需要root权限安装
```

### 5.3 cudaEvent问题 (Mars X201)

```cpp
// Mars X201上cudaEventElapsedTime可能返回0
// 解决方案: 使用hcTracer或多次循环计时

// 方法1: 使用hcTracer (推荐)
// 方法2: 增加迭代次数
int iterations = 100;  // 足够多的迭代
auto start = std::chrono::high_resolution_clock::now();
for (int i = 0; i < iterations; i++) {
    kernel<<<...>>>();
}
cudaDeviceSynchronize();
auto end = std::chrono::high_resolution_clock::now();
float ms = std::chrono::duration<float>(end - start).count() * 1000 / iterations;
```

---

## 6. 调试注意事项

### 6.1 Mars X201调试

```cpp
// 不要使用printf!
// 错误示例:
// printf("value = %f\n", val);

// 正确做法: 使用日志库
// 路径: /c/Users/Lenovo/cbx/muxi_print_bug
#include "muxi_print.h"

// 使用方法
MUXI_PRINT("value = %f\n", val);
```

### 6.2 错误检查

```cpp
// 两平台通用的错误检查
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// 注意: Mars X201可能不返回所有错误信息
// 建议添加cudaDeviceSynchronize()后检查
kernel<<<...>>>();
CHECK_CUDA(cudaDeviceSynchronize());  // 确保kernel执行完毕
```

---

## 7. 常见问题与解决方案

### 7.1 编译问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `cuda_runtime.h` not found | 未设置cu-bridge环境 | `source ~/.bashrc` 或手动设置PATH |
| `sm_xx` not supported | Mars不支持架构指定 | 移除 `-gencode arch=compute_xx` |
| 链接错误 | 库路径问题 | 添加cu-bridge的lib到LD_LIBRARY_PATH |

### 7.2 运行时问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| Kernel不执行 | GPU选择错误 | `export CUDA_VISIBLE_DEVICES=7` |
| 结果错误 | Warp size不匹配 | 检查WARP_SIZE宏定义 |
| 性能低下 | 配置未优化 | 使用8t/row + PreferL1 |

### 7.3 性能问题

| 现象 | 可能原因 | 诊断方法 |
|------|----------|----------|
| 利用率<10% | 随机访问 | 分析列索引分布 |
| 利用率>100% | L2缓存效应 | 正常，RTX 4090特有 |
| 端到端慢 | 未使用Pinned Memory | 使用cudaMallocHost |

---

## 8. 最佳实践总结

### 8.1 Mars X201开发清单

- [ ] 使用 `pre_make` 编译命令
- [ ] 设置 `CUDA_VISIBLE_DEVICES=7`
- [ ] 不指定 `sm_xx` 架构
- [ ] 不使用 `printf`，使用日志库
- [ ] 使用 `WARP_SIZE=64`
- [ ] 使用 `8t/row` 线程配置
- [ ] 使用 `blockSize=128`
- [ ] 设置 `cudaFuncCachePreferL1`
- [ ] 使用 `cudaMallocHost` (Pinned Memory)
- [ ] 大矩阵考虑分块处理

### 8.2 RTX 4090开发清单

- [ ] 使用标准 `nvcc` 编译
- [ ] 可指定 `sm_89` 架构
- [ ] 使用 `WARP_SIZE=32`
- [ ] 使用 `4t/row` 线程配置
- [ ] 使用 `blockSize=256`
- [ ] 使用 `cudaMallocHost` (Pinned Memory)
- [ ] 可使用 `printf` 调试
- [ ] 可使用 `nvidia-smi` 和 `nsys`

---

## 9. 性能对比参考

### 9.1 FP64 SpMV (avgNnz=10.71)

| 指标 | Mars X201 | RTX 4090 | 差距 |
|------|-----------|----------|------|
| 最优内核 | Vector 8t/row | Vector 4t/row | - |
| 内核时间 | 318 μs | 204 μs | RTX快1.56x |
| 有效带宽 | 847 GB/s | 1321 GB/s | RTX高1.56x |
| 带宽利用率 | 45.9% | 131% | RTX高2.85x |
| 端到端时间 | 1.96 ms | 1.27 ms | RTX快1.54x |

### 9.2 关键发现

1. **内核层面RTX更快**: L2 Cache是决定因素
2. **Mars带宽利用率可接受**: 45.9%是合理的
3. **配置必须差异化**: 8t/row vs 4t/row
4. **端到端差距小于内核差距**: Pinned Memory帮助

---

## 附录A: 环境变量配置

```bash
# Mars X201 ~/.bashrc 添加:
export PATH=$PATH:~/cu-bridge/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/cu-bridge/lib:~/cu-bridge/lib64
export CUDA_VISIBLE_DEVICES=7

# 验证配置
which nvcc  # 应该指向cu-bridge/bin/nvcc
ht-smi      # 应该显示GPU信息
```

## 附录B: 测试命令

```bash
# Mars X201
CUDA_VISIBLE_DEVICES=7 ./test_comprehensive real_cases/mtx/p0_A

# RTX 4090
./test_comprehensive real_cases/mtx/p0_A
```

## 附录C: 相关文档

- `doc/analysis/fp64_final_optimization_report_2026_04_10.md` - 最新优化报告
- `doc/gpu_comparison/mars_x201_vs_rtx4090.md` - GPU硬件对比
- `tests/benchmark/test_comprehensive_optimization.cu` - 穷尽性测试代码