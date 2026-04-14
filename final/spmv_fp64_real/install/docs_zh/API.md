# API文档

## 概述

SPMV FP64库提供简洁的C风格API，用于优化的FP64稀疏矩阵-向量乘法。

## 类型定义

### `spmv_fp64_matrix_handle_t`

CSR矩阵的不透明句柄。通过`spmv_fp64_create_matrix()`创建，通过`spmv_fp64_destroy_matrix()`销毁。

### `spmv_fp64_status_t`

错误码：

| 值 | 描述 |
|-----|------|
| `SPMV_FP64_SUCCESS` | 操作成功 |
| `SPMV_FP64_ERROR_INVALID_INPUT` | 输入参数无效 |
| `SPMV_FP64_ERROR_MEMORY` | 内存分配错误 |
| `SPMV_FP64_ERROR_CUDA` | CUDA运行时错误 |
| `SPMV_FP64_ERROR_NOT_SUPPORTED` | 功能不支持 |
| `SPMV_FP64_ERROR_INTERNAL` | 内部库错误 |

### `spmv_fp64_opts_t`

执行选项：

| 字段 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `stream` | `cudaStream_t` | 0 | CUDA流用于异步执行 |
| `sync_after_exec` | `int` | 1 | 执行后同步 |
| `benchmark_mode` | `int` | 0 | 启用性能基准测试 |

### `spmv_fp64_stats_t`

性能统计（仅在`benchmark_mode = 1`时填充）：

| 字段 | 类型 | 描述 |
|------|------|------|
| `kernel_time_ms` | `double` | 内核执行时间（毫秒） |
| `bandwidth_gbps` | `double` | 有效带宽（GB/s） |
| `utilization_pct` | `double` | 带宽利用率（%） |
| `theoretical_bw` | `double` | 理论峰值带宽（GB/s） |
| `warp_size` | `int` | GPU warp大小（32或64） |
| `optimal_tpr` | `int` | 最优每行线程数 |
| `gpu_name` | `const char*` | GPU设备名称 |

## 矩阵管理

### `spmv_fp64_create_matrix()`（主机CSR模式）

```c
spmv_fp64_status_t spmv_fp64_create_matrix(
    spmv_fp64_matrix_handle_t* handle,
    int numRows,
    int numCols,
    int nnz,
    const int* rowPtr,
    const int* colIdx,
    const double* values,
    const spmv_fp64_opts_t* opts);
```

从主机数据创建CSR矩阵句柄。库为CSR数据和内部向量分配设备内存。

**参数：**
- `handle`：输出矩阵句柄
- `numRows`：行数
- `numCols`：列数
- `nnz`：非零元素数量
- `rowPtr`：主机行指针数组（大小：numRows + 1）
- `colIdx`：主机列索引数组（大小：nnz）
- `values`：主机值数组（大小：nnz）
- `opts`：执行选项（NULL使用默认值）

**返回：** 状态码

**内存所有权：** 库拥有所有主机和设备内存。由`spmv_fp64_destroy_matrix()`自动释放。

---

### `spmv_fp64_create_matrix_device()`（零拷贝模式）⭐ 新增

```c
spmv_fp64_status_t spmv_fp64_create_matrix_device(
    spmv_fp64_matrix_handle_t* handle,
    int numRows,
    int numCols,
    int nnz,
    const int* d_rowPtr,
    const int* d_colIdx,
    const double* d_values,
    const spmv_fp64_opts_t* opts);
```

从**设备指针**创建CSR矩阵句柄。零拷贝模式 - 库不分配或拷贝CSR数据。

**参数：**
- `handle`：输出矩阵句柄
- `numRows`：行数
- `numCols`：列数
- `nnz`：非零元素数量
- `d_rowPtr`：**设备**行指针数组（大小：numRows + 1）
- `d_colIdx`：**设备**列索引数组（大小：nnz）
- `d_values`：**设备**值数组（大小：nnz）
- `opts`：执行选项（NULL使用默认值）

**返回：** 状态码

**内存所有权：**
- **用户拥有设备CSR数据**（d_rowPtr、d_colIdx、d_values）
- 库不分配内部向量（使用`spmv_fp64_execute_device()`代替）
- `spmv_fp64_destroy_matrix()`只释放句柄，不释放您的CSR数据

**使用场景：** 适合已在设备上有CSR数据的用户，可避免冗余拷贝。适合迭代算法中CSR数据在设备上持久化的场景。

---

### `spmv_fp64_create_matrix_from_file()`

```c
spmv_fp64_status_t spmv_fp64_create_matrix_from_file(
    spmv_fp64_matrix_handle_t* handle,
    const char* filename,
    const spmv_fp64_opts_t* opts);
```

从Matrix Market（.mtx）文件加载CSR矩阵。

### `spmv_fp64_destroy_matrix()`

```c
spmv_fp64_status_t spmv_fp64_destroy_matrix(
    spmv_fp64_matrix_handle_t handle);
```

销毁矩阵句柄并释放所有内存。

**注意：** 在零拷贝模式下，只释放句柄本身，不释放用户提供的CSR数据。

### `spmv_fp64_get_matrix_info()`

```c
spmv_fp64_status_t spmv_fp64_get_matrix_info(
    spmv_fp64_matrix_handle_t handle,
    int* numRows,
    int* numCols,
    int* nnz);
```

获取矩阵维度信息。

## SpMV执行

### 主机指针模式（库管理H2D/D2H拷贝）

### `spmv_fp64_execute()`

```c
spmv_fp64_status_t spmv_fp64_execute(
    spmv_fp64_matrix_handle_t handle,
    const double* x,
    double* y,
    const spmv_fp64_opts_t* opts,
    spmv_fp64_stats_t* stats);
```

执行SpMV：y = A * x（x和y为主机指针）

**注意：** 仅适用于通过`spmv_fp64_create_matrix()`或`spmv_fp64_create_matrix_from_file()`创建的矩阵。对设备指针模式矩阵返回`SPMV_FP64_ERROR_NOT_SUPPORTED`。

---

### 设备指针模式（无H2D/D2H拷贝）⭐ 新增

### `spmv_fp64_execute_device()`

```c
spmv_fp64_status_t spmv_fp64_execute_device(
    spmv_fp64_matrix_handle_t handle,
    const double* d_x,
    double* d_y,
    const spmv_fp64_opts_t* opts,
    spmv_fp64_stats_t* stats);
```

执行SpMV：y = A * x，使用**设备指针**。

**参数：**
- `handle`：矩阵句柄（由`spmv_fp64_create_matrix()`或`spmv_fp64_create_matrix_device()`创建）
- `d_x`：**设备**输入向量（numCols个元素）
- `d_y`：**设备**输出向量（numRows个元素）
- `opts`：执行选项
- `stats`：性能统计（可选）

**返回：** 状态码

**关键优势：**
- 无H2D/D2H拷贝 - 迭代算法的最佳性能
- 同时支持主机CSR和设备CSR矩阵句柄
- 用户管理x/y设备内存生命周期

---

### `spmv_fp64_execute_device_general()` ⭐ 新增

```c
spmv_fp64_status_t spmv_fp64_execute_device_general(
    spmv_fp64_matrix_handle_t handle,
    double alpha,
    double beta,
    const double* d_x,
    double* d_y,
    const spmv_fp64_opts_t* opts,
    spmv_fp64_stats_t* stats);
```

执行通用SpMV：y = alpha * A * x + beta * y，使用**设备指针**。

---

### 直接执行模式（无需句柄）⭐ 推荐单次执行

### `spmv_fp64_execute_direct()`

```c
spmv_fp64_status_t spmv_fp64_execute_direct(
    int numRows,
    int nnz,
    const int* d_rowPtr,
    const int* d_colIdx,
    const double* d_values,
    const double* d_x,
    double* d_y,
    cudaStream_t stream);
```

直接使用设备CSR数据和设备向量执行SpMV - **无需矩阵句柄**。

**参数：**
- `numRows`：行数
- `nnz`：非零元素数量
- `d_rowPtr`：设备行指针数组
- `d_colIdx`：设备列索引数组
- `d_values`：设备值数组
- `d_x`：设备输入向量
- `d_y`：设备输出向量
- `stream`：CUDA流（0为默认）

**返回：** 状态码

**使用场景：** 单次执行无需句柄创建开销。适合一次性的SpMV操作。

---

### `spmv_fp64_execute_direct_scaled()` ⭐ 新增

```c
spmv_fp64_status_t spmv_fp64_execute_direct_scaled(
    double alpha,
    int numRows,
    int nnz,
    const int* d_rowPtr,
    const int* d_colIdx,
    const double* d_values,
    const double* d_x,
    double* d_y,
    cudaStream_t stream);
```

执行缩放SpMV：**y = alpha * A * x**，无需矩阵句柄。

**参数：**
- `alpha`：矩阵乘积的缩放因子
- 其他参数同`spmv_fp64_execute_direct()`

**返回：** 状态码

---

### `spmv_fp64_execute_direct_general()` ⭐ 新增

```c
spmv_fp64_status_t spmv_fp64_execute_direct_general(
    double alpha,
    double beta,
    int numRows,
    int nnz,
    const int* d_rowPtr,
    const int* d_colIdx,
    const double* d_values,
    const double* d_x,
    double* d_y,
    cudaStream_t stream);
```

执行通用SpMV：**y = alpha * A * x + beta * y**，无需矩阵句柄。

**参数：**
- `alpha`：矩阵乘积的缩放因子
- `beta`：现有y向量的缩放因子
- 其他参数同`spmv_fp64_execute_direct()`

**返回：** 状态码

**注意：** 当`beta != 0`时，`d_y`必须包含有效的输入数据（作为y_old）。

---

### 直接执行模式使用场景

| API | 操作 | 适用场景 |
|-----|------|---------|
| `execute_direct` | y = A * x | 基础SpMV，单次执行 |
| `execute_direct_scaled` | y = α * A * x | 需要结果缩放 |
| `execute_direct_general` | y = α*A*x + β*y | Axpy操作，迭代求解器 |

**内存管理：** 用户管理所有设备内存（CSR数据 + 向量）。库不分配任何内存。

### `spmv_fp64_execute_scaled()`

```c
spmv_fp64_status_t spmv_fp64_execute_scaled(
    spmv_fp64_matrix_handle_t handle,
    double alpha,
    const double* x,
    double* y,
    const spmv_fp64_opts_t* opts,
    spmv_fp64_stats_t* stats);
```

执行缩放SpMV：y = alpha * A * x

### `spmv_fp64_execute_general()`

```c
spmv_fp64_status_t spmv_fp64_execute_general(
    spmv_fp64_matrix_handle_t handle,
    double alpha,
    double beta,
    const double* x,
    double* y,
    const spmv_fp64_opts_t* opts,
    spmv_fp64_stats_t* stats);
```

执行通用SpMV：y = alpha * A * x + beta * y

## 工具函数

## 工具函数

### `spmv_fp64_alloc_pinned()`

```c
spmv_fp64_status_t spmv_fp64_alloc_pinned(void** ptr, size_t size);
```

分配pinned（页锁定）主机内存。

**对性能至关重要！** Pinned内存提供：
- Mars X201：+186%端到端提升
- RTX 4090：+152%端到端提升

使用`cudaFreeHost()`或`spmv_fp64_free_pinned()`释放。

### `spmv_fp64_free_pinned()`

```c
spmv_fp64_status_t spmv_fp64_free_pinned(void* ptr);
```

释放pinned内存。

### `spmv_fp64_get_device_info()`

```c
spmv_fp64_status_t spmv_fp64_get_device_info(
    int* warpSize,
    const char** name,
    size_t* memory);
```

获取GPU设备信息。

### `spmv_fp64_get_theoretical_bandwidth()`

```c
spmv_fp64_status_t spmv_fp64_get_theoretical_bandwidth(double* bandwidth);
```

获取理论峰值内存带宽。

### `spmv_fp64_get_error_string()`

```c
const char* spmv_fp64_get_error_string(spmv_fp64_status_t status);
```

获取可读的错误描述。

### `spmv_fp64_get_version()`

```c
const char* spmv_fp64_get_version(void);
```

获取库版本字符串。

## 默认选项

```c
static const spmv_fp64_opts_t SPMV_FP64_DEFAULT_OPTS = {
    .stream = 0,
    .sync_after_exec = 1,
    .benchmark_mode = 0
};

static const spmv_fp64_opts_t SPMV_FP64_BENCHMARK_OPTS = {
    .stream = 0,
    .sync_after_exec = 1,
    .benchmark_mode = 1       // 启用基准测试
};
```

## 完整示例

### 示例1：主机CSR模式（原始API）

```c
#include "spmv_fp64.h"
#include <stdio.h>

int main(int argc, char** argv) {
    // 初始化选项
    spmv_fp64_opts_t opts = SPMV_FP64_BENCHMARK_OPTS;
    
    // 加载矩阵
    spmv_fp64_matrix_handle_t matrix;
    spmv_fp64_status_t status = spmv_fp64_create_matrix_from_file(
        &matrix, argv[1], &opts);
    if (status != SPMV_FP64_SUCCESS) {
        fprintf(stderr, "错误: %s\n", spmv_fp64_get_error_string(status));
        return 1;
    }
    
    // 获取维度
    int numRows, numCols, nnz;
    spmv_fp64_get_matrix_info(matrix, &numRows, &numCols, &nnz);
    
    // 为向量分配pinned内存
    double* x, *y;
    spmv_fp64_alloc_pinned((void**)&x, numCols * sizeof(double));
    spmv_fp64_alloc_pinned((void**)&y, numRows * sizeof(double));
    
    // 初始化输入向量
    for (int i = 0; i < numCols; i++) {
        x[i] = 1.0;
    }
    
    // 执行SpMV
    spmv_fp64_stats_t stats;
    status = spmv_fp64_execute(matrix, x, y, &opts, &stats);
    
    // 打印结果
    printf("内核时间: %.3f ms\n", stats.kernel_time_ms);
    printf("带宽: %.1f GB/s\n", stats.bandwidth_gbps);
    printf("利用率: %.1f%%\n", stats.utilization_pct);
    
    // 清理
    spmv_fp64_free_pinned(x);
    spmv_fp64_free_pinned(y);
    spmv_fp64_destroy_matrix(matrix);
    
    return 0;
}
```

### 示例2：设备CSR模式（零拷贝）⭐ 新增

```c
#include "spmv_fp64.h"
#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    // 用户提供的设备CSR数据
    int numRows = 10000;
    int nnz = 100000;
    
    // 分配并填充设备CSR数组（用户管理）
    int* d_rowPtr, *d_colIdx;
    double* d_values;
    cudaMalloc(&d_rowPtr, (numRows + 1) * sizeof(int));
    cudaMalloc(&d_colIdx, nnz * sizeof(int));
    cudaMalloc(&d_values, nnz * sizeof(double));
    // ... 用您的数据填充 ...
    
    // 创建矩阵句柄（零拷贝 - 库不拷贝）
    spmv_fp64_matrix_handle_t matrix;
    spmv_fp64_create_matrix_device(&matrix, numRows, numRows, nnz,
                                    d_rowPtr, d_colIdx, d_values, NULL);
    
    // 分配设备向量（用户管理）
    double* d_x, *d_y;
    cudaMalloc(&d_x, numRows * sizeof(double));
    cudaMalloc(&d_y, numRows * sizeof(double));
    // ... 初始化 d_x ...
    
    // 执行SpMV（无H2D/D2H拷贝 - 最大性能）
    spmv_fp64_opts_t opts = SPMV_FP64_BENCHMARK_OPTS;
    spmv_fp64_stats_t stats;
    spmv_fp64_execute_device(matrix, d_x, d_y, &opts, &stats);
    
    printf("内核时间: %.3f ms\n", stats.kernel_time_ms);
    printf("带宽: %.1f GB/s\n", stats.bandwidth_gbps);
    
    // 清理 - 库只释放句柄，用户释放设备内存
    spmv_fp64_destroy_matrix(matrix);
    cudaFree(d_rowPtr);
    cudaFree(d_colIdx);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);
    
    return 0;
}
```

### 示例3：一次性直接执行 ⭐ 新增

```c
#include "spmv_fp64.h"
#include <cuda_runtime.h>

int main() {
    int numRows = 10000;
    int nnz = 100000;
    
    // 所有数据必须在设备上
    int* d_rowPtr, *d_colIdx;
    double* d_values, *d_x, *d_y;
    cudaMalloc(&d_rowPtr, (numRows + 1) * sizeof(int));
    cudaMalloc(&d_colIdx, nnz * sizeof(int));
    cudaMalloc(&d_values, nnz * sizeof(double));
    cudaMalloc(&d_x, numRows * sizeof(double));
    cudaMalloc(&d_y, numRows * sizeof(double));
    // ... 用您的数据填充 ...
    
    // 直接执行 - 无需创建句柄
    spmv_fp64_execute_direct(numRows, nnz, d_rowPtr, d_colIdx, 
                             d_values, d_x, d_y, 0);
    cudaDeviceSynchronize();
    
    // 清理 - 用户管理所有内存
    cudaFree(d_rowPtr);
    cudaFree(d_colIdx);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);
    
    return 0;
}
```