# SPMV FP64 库

面向真实稀疏矩阵的优化FP64稀疏矩阵-向量乘法库。

## 特性

- **自动GPU检测**：自动检测warp大小（Mars X201为64，NVIDIA为32）
- **最优内核**：针对每种GPU类型使用最佳内核
  - Mars X201：TPR=8内核
  - RTX 4090：__ldg内核
- **Pinned内存**：端到端性能的关键（+150%至+186%）
- **简洁API**：简单的C风格接口
- **多种API模式**：支持主机指针模式、设备指针模式（零拷贝）和直接执行模式

## 性能表现

### 最佳情况（具有列索引局部性的矩阵）

| GPU        | 内核时间 | 带宽     | 利用率 |
|------------|----------|----------|--------|
| Mars X201  | 0.420 ms | 897 GB/s | 48.7%  |
| RTX 4090   | 0.402 ms | 907 GB/s | 90.2%  |

*数据来源于具有列索引局部性的矩阵（L2缓存命中率较高）。*

### 典型情况（随机列分布）

| GPU        | 内核+拷贝 | 带宽     | 利用率 |
|------------|-----------|----------|--------|
| Mars X201  | 0.58 ms   | 357 GB/s | 19.4%  |
| RTX 4090   | 0.86 ms   | 326 GB/s | 32.4%  |

*数据来源于随机列索引分布的矩阵（L2缓存命中率较低）。*

### 影响性能的关键因素

| 因素 | 高带宽 | 低带宽 | 影响 |
|------|--------|--------|------|
| 列索引局部性 | 有 | 无 | L2缓存命中率 |
| 平均每行NNZ | >10 | <10 | 线程利用率 |
| 矩阵规模 | 大 | 小 | 内核启动开销 |

**注意**：性能很大程度上取决于矩阵结构。具有列索引局部性的矩阵（如带状矩阵、结构化网格）比随机稀疏矩阵能获得更高的带宽。

## API模式

### 1. 主机CSR模式（库管理所有内存）

```c
// 用户提供主机CSR数据，库负责设备内存分配和拷贝
spmv_fp64_create_matrix(&handle, numRows, numCols, nnz, 
                         h_rowPtr, h_colIdx, h_values, opts);
spmv_fp64_execute(handle, h_x, h_y, opts, &stats);
```

### 2. 设备CSR模式（零拷贝）⭐ 新增

```c
// 用户提供设备CSR指针，库不分配或拷贝CSR数据
spmv_fp64_create_matrix_device(&handle, numRows, numCols, nnz,
                                d_rowPtr, d_colIdx, d_values, opts);
spmv_fp64_execute_device(handle, d_x, d_y, opts, &stats);
```

### 3. 直接执行模式（无需句柄）⭐ 推荐

```c
// 一次性执行，无需创建矩阵句柄，用户管理所有设备内存
// 基础: y = A * x
spmv_fp64_execute_direct(numRows, nnz, d_rowPtr, d_colIdx, 
                         d_values, d_x, d_y, stream);

// 缩放: y = alpha * A * x
spmv_fp64_execute_direct_scaled(alpha, numRows, nnz, d_rowPtr, 
                                d_colIdx, d_values, d_x, d_y, stream);

// 通用: y = alpha * A * x + beta * y
spmv_fp64_execute_direct_general(alpha, beta, numRows, nnz, d_rowPtr,
                                 d_colIdx, d_values, d_x, d_y, stream);
```

## 快速开始

### 前置条件

- CUDA 11.6+（NVIDIA）或 cu-bridge（Mars X201）
- C++编译器支持C++11

### 在NVIDIA RTX 4090上构建

```bash
./scripts/build_rtx.sh
```

### 在Mars X201上构建

```bash
# 首先设置环境
export PATH=$PATH:$HOME/cu-bridge/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/cu-bridge/lib:$HOME/cu-bridge/lib64

./scripts/build_mars.sh
```

### 使用示例

```c
#include "spmv_fp64.h"

int main() {
    // 加载矩阵
    spmv_fp64_matrix_handle_t matrix;
    spmv_fp64_opts_t opts = SPMV_FP64_DEFAULT_OPTS;
    
    spmv_fp64_create_matrix_from_file(&matrix, "matrix.mtx", &opts);
    
    // 分配pinned内存（关键！）
    double* x, *y;
    spmv_fp64_alloc_pinned((void**)&x, numCols * sizeof(double));
    spmv_fp64_alloc_pinned((void**)&y, numRows * sizeof(double));
    
    // 执行SpMV
    spmv_fp64_execute(matrix, x, y, &opts, NULL);
    
    // 清理
    spmv_fp64_free_pinned(x);
    spmv_fp64_free_pinned(y);
    spmv_fp64_destroy_matrix(matrix);
    
    return 0;
}
```

## 文档

### 中文文档
- [构建指南](BUILD.md) - 构建说明
- [API文档](API.md) - API详细说明
- [性能结果](PERFORMANCE_RESULTS.md) - 性能测试结果
- [理论分析](THEORY_ANALYSIS.md) - GPU优化理论分析

### 英文文档
- [BUILD.md](../docs/BUILD.md) - Build instructions
- [API.md](../docs/API.md) - API documentation
- [GPU_OPTIMIZATION.md](../docs/GPU_OPTIMIZATION.md) - GPU-specific notes

## 关键优化技术

### Mars X201（Warp=64）

1. **TPR=8**：每行8线程，每warp处理8行
2. **PreferL1缓存**：对性能至关重要
3. **Pinned内存**：+186%端到端提升

### NVIDIA RTX 4090（Warp=32）

1. **__ldg缓存提示**：使用纹理/只读缓存
2. **大L2缓存**：72MB可缓存整个x向量
3. **Pinned内存**：+152%端到端提升

## 内存所有权模型

| 模式 | CSR内存 | x/y向量 | 创建方式 |
|------|---------|---------|----------|
| 主机CSR模式 | 库拥有 | 库管理拷贝 | `create_matrix` |
| 设备CSR模式 | 用户拥有 | 用户提供设备指针 | `create_matrix_device` |
| 直接执行 | 用户拥有全部 | 用户提供全部 | `execute_direct` |

## 许可证

详见LICENSE文件。

## 版本历史

- **1.0.0**（2026-04-12）：初始发布，支持两种GPU类型的最优内核
- **新增API**：设备CSR模式、直接执行模式