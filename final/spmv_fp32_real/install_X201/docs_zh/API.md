# SPMV FP32 库 API 参考文档

## 概述

SPMV FP32 库为FP32精度稀疏矩阵提供优化的稀疏矩阵-向量乘法(SpMV)。

## 错误码

```c
typedef enum {
    SPMV_FP32_SUCCESS = 0,              // 成功
    SPMV_FP32_ERROR_INVALID_INPUT,      // 输入参数无效
    SPMV_FP32_ERROR_MEMORY,             // 内存分配错误
    SPMV_FP32_ERROR_CUDA,               // CUDA运行时错误
    SPMV_FP32_ERROR_NOT_SUPPORTED,      // 功能不支持
    SPMV_FP32_ERROR_INTERNAL,           // 内部库错误
    SPMV_FP32_ERROR_LICENSE_EXPIRED     // 许可证过期
} spmv_fp32_status_t;
```

## 矩阵管理

### spmv_fp32_create_matrix

从主机数组创建CSR矩阵。

```c
spmv_fp32_status_t spmv_fp32_create_matrix(
    spmv_fp32_matrix_handle_t* handle,
    int numRows,
    int numCols,
    int nnz,
    const int* rowPtr,
    const int* colIdx,
    const float* values,
    const spmv_fp32_opts_t* opts);
```

**参数:**
- `handle`: 输出矩阵句柄
- `numRows`: 行数
- `numCols`: 列数
- `nnz`: 非零元素数
- `rowPtr`: 行指针数组 (numRows+1个元素)
- `colIdx`: 列索引数组 (nnz个元素)
- `values`: 值数组 (nnz个元素)
- `opts`: 执行选项 (NULL使用默认值)

### spmv_fp32_create_matrix_device

从设备数组创建CSR矩阵(零拷贝模式)。

```c
spmv_fp32_status_t spmv_fp32_create_matrix_device(
    spmv_fp32_matrix_handle_t* handle,
    int numRows,
    int numCols,
    int nnz,
    const int* d_rowPtr,
    const int* d_colIdx,
    const float* d_values,
    const spmv_fp32_opts_t* opts);
```

### spmv_fp32_destroy_matrix

销毁矩阵句柄并释放资源。

```c
spmv_fp32_status_t spmv_fp32_destroy_matrix(spmv_fp32_matrix_handle_t handle);
```

## SpMV执行

### spmv_fp32_execute

执行SpMV: y = A * x

```c
spmv_fp32_status_t spmv_fp32_execute(
    spmv_fp32_matrix_handle_t handle,
    const float* x,
    float* y,
    const spmv_fp32_opts_t* opts,
    spmv_fp32_stats_t* stats);
```

### spmv_fp32_execute_scaled

执行带缩放的SpMV: y = alpha * A * x

```c
spmv_fp32_status_t spmv_fp32_execute_scaled(
    spmv_fp32_matrix_handle_t handle,
    float alpha,
    const float* x,
    float* y,
    const spmv_fp32_opts_t* opts,
    spmv_fp32_stats_t* stats);
```

### spmv_fp32_execute_general

执行通用SpMV: y = alpha * A * x + beta * y

```c
spmv_fp32_status_t spmv_fp32_execute_general(
    spmv_fp32_matrix_handle_t handle,
    float alpha,
    float beta,
    const float* x,
    float* y,
    const spmv_fp32_opts_t* opts,
    spmv_fp32_stats_t* stats);
```

### spmv_fp32_execute_device

使用设备向量执行SpMV: d_y = A * d_x

```c
spmv_fp32_status_t spmv_fp32_execute_device(
    spmv_fp32_matrix_handle_t handle,
    const float* d_x,
    float* d_y,
    const spmv_fp32_opts_t* opts,
    spmv_fp32_stats_t* stats);
```

### spmv_fp32_execute_direct

一次性SpMV执行(无需句柄)。

```c
spmv_fp32_status_t spmv_fp32_execute_direct(
    int numRows,
    int nnz,
    const int* d_rowPtr,
    const int* d_colIdx,
    const float* d_values,
    const float* d_x,
    float* d_y,
    cudaStream_t stream);
```

**注意:** 所有指针必须是设备指针。用户管理所有内存。

## 工具函数

### spmv_fp32_alloc_pinned

分配固定内存用于向量(推荐用于最佳性能)。

```c
spmv_fp32_status_t spmv_fp32_alloc_pinned(void** ptr, size_t size);
```

### spmv_fp32_free_pinned

释放固定内存。

```c
spmv_fp32_status_t spmv_fp32_free_pinned(void* ptr);
```

### 许可证函数

```c
spmv_fp32_status_t spmv_fp32_check_license(void);
const char* spmv_fp32_get_license_expiry(void);
```

## 执行选项

```c
typedef struct {
    cudaStream_t stream;      // CUDA流 (0 = 默认)
    int sync_after_exec;      // 执行后同步 (默认: 1)
    int benchmark_mode;       // 启用基准测试 (默认: 0)
} spmv_fp32_opts_t;
```

## 性能统计

```c
typedef struct {
    double kernel_time_ms;    // 内核+传输时间 (ms)
    double bandwidth_gbps;    // 有效带宽 (GB/s)
    double theoretical_bw;    // 峰值带宽 (GB/s)
    double utilization_pct;   // 带宽利用率 (%)
    int warp_size;            // GPU warp大小
    int optimal_tpr;          // 最优线程/行
    const char* gpu_name;     // GPU名称
} spmv_fp32_stats_t;
```

## 带宽计算

对于FP32 SpMV:
- 每nnz字节 = 4 (值) + 4 (列索引) + 4 (x[col]) = 12字节
- y输出字节 = numRows * 4字节

```
带宽 (GB/s) = (nnz * 12 + numRows * 4) / (time_ms * 1e6)
```

---
版本: 1.0.0
最后更新: 2026-04-13