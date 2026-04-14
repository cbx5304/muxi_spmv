# SPMV FP64 Library - Trial Version (Mars X201)

## 许可证

**试用许可证过期日期: 2026-05-07**

这是 SPMV FP64 库的试用版本。许可证过期后库将停止工作。请联系供应商获取永久许可证。

许可证检查会在以下时机自动执行：
- 调用 `spmv_fp64_check_license()`
- 使用 `spmv_fp64_create_matrix()` 或 `spmv_fp64_create_matrix_device()` 创建矩阵
- 使用 `spmv_fp64_execute_direct()` 直接执行

如果许可证过期，库将返回 `SPMV_FP64_ERROR_LICENSE_EXPIRED`。

## 目录结构

```
install_x201/
├── include/
│   └── spmv_fp64.h          # 库头文件
├── lib/
│   └── libspmv_fp64.so      # 共享库 (Mars X201专用)
├── docs/
│   ├── API.md               # API文档 (英文)
│   └── PERFORMANCE_TEST_REPORT.md  # 性能报告
├── docs_zh/
│   ├── API.md               # API文档 (中文)
│   └── PERFORMANCE_TEST_REPORT.md  # 性能报告
└── README.md                # 本文件
```

## 编译说明 (Mars X201)

**重要**: 此版本专为 Mars X201 (国产GPU, Warp=64) 编译，使用 `pre_make nvcc` 命令。

```bash
# 编译用户程序 (必须使用 pre_make nvcc)
export PATH=$HOME/cu-bridge/bin:$PATH
export LD_LIBRARY_PATH=$HOME/cu-bridge/CUDA_DIR/lib64:$LD_LIBRARY_PATH
CUDA_VISIBLE_DEVICES=7 pre_make nvcc -I./install_x201/include -L./install_x201/lib -lspmv_fp64 your_code.cu -o your_app

# 运行
CUDA_VISIBLE_DEVICES=7 LD_LIBRARY_PATH=./install_x201/lib:$LD_LIBRARY_PATH ./your_app
```

## Mars X201 性能特性

| 特性 | Mars X201 |
|------|-----------|
| Warp Size | **64** (NVIDIA为32) |
| 最优Kernel | `tpr_kernel<64,8>` |
| TPR优化 | **每行8线程** (关键优化!) |
| L1缓存配置 | `cudaFuncCachePreferL1` (必须设置) |

### 性能数据 (真实算例测试)

#### 与nctigpu (spmv_1)对比

| Metric | nctigpu (spmv_1) | Optimized spmv_fp64 | 提升 |
|--------|------------------|---------------------|------|
| Kernel Time | 1.15 ms | **0.36 ms** | **3.2x faster** |
| Bandwidth | 242 GB/s | **776 GB/s** | **+320%** |
| Utilization | 13% | **42%** | **+229%** |

测试条件: 1.26M行, avgNnz=10.7, FP64精度, 7组真实算例平均

#### 自适应TPR优化

库自动根据avgNnz选择最优TPR：
- avgNnz >= 128: TPR=64 (1 row/warp, full warp)
- avgNnz >= 64: TPR=32 (2 rows/warp)
- avgNnz >= 32: TPR=16 (4 rows/warp)
- avgNnz >= 16: TPR=8 (8 rows/warp)
- avgNnz < 16: TPR=4 (16 rows/warp)

**重要修复**: TPR=64已修复warp reduction bug（使用64位mask）。

### 关键优化代码

```cpp
// Mars X201 必须使用以下配置
if (warpSize == 64) {
    int threadsPerRow = 8;   // ⭐ 关键优化: 8线程/行
    cudaFuncSetCacheConfig(kernel, cudaFuncCachePreferL1);  // 必须!
}

// Pinned Memory (必须使用)
cudaMallocHost(&h_x, numCols * sizeof(double));
```

## 使用示例

### 许可证检查

```cpp
#include "spmv_fp64.h"

// 检查许可证
spmv_fp64_status_t status = spmv_fp64_check_license();
if (status == SPMV_FP64_ERROR_LICENSE_EXPIRED) {
    printf("许可证已过期! 过期日期: %s\n", spmv_fp64_get_license_expiry());
    return 1;
}
printf("许可证有效期至: %s\n", spmv_fp64_get_license_expiry());
```

### 基本使用

```cpp
#include "spmv_fp64.h"

int main() {
    // 许可证会在创建矩阵时自动检查
    
    spmv_fp64_matrix_handle_t matrix;
    spmv_fp64_status_t status = spmv_fp64_create_matrix(
        &matrix, numRows, numCols, nnz,
        h_rowPtr, h_colIdx, h_values, NULL);
    
    if (status == SPMV_FP64_ERROR_LICENSE_EXPIRED) {
        printf("许可证过期!\n");
        return 1;
    }
    
    // 执行 SpMV
    spmv_fp64_execute(matrix, h_x, h_y, NULL, NULL);
    
    // 清理
    spmv_fp64_destroy_matrix(matrix);
    return 0;
}
```

## 注意事项

### Mars X201 特殊要求

1. **编译命令**: 必须使用 `pre_make nvcc`，不能用普通 `nvcc`
2. **CUDA版本**: 仅支持 CUDA 11.6 及以下语法
3. **调试**: 不要使用 printf（国产GPU printf 有问题），使用日志库
4. **GPU选择**: 必须使用 `CUDA_VISIBLE_DEVICES=7`
5. **Warp Size**: 64 vs NVIDIA的32，影响寄存器/共享内存使用
6. **架构**: 不支持 sm_xx 架构指定，留空即可

### 不要做的事

❌ 不要使用 `nvcc` 直接编译（必须用 `pre_make nvcc`）
❌ 不要在代码中使用国产GPU特有接口（保持CUDA兼容）
❌ 不要强制指定 CUDA_ARCH 或 CUDA库路径
❌ 不要使用 printf 调试GPU代码

## 支持的GPU

| GPU | Warp Size | 最优Kernel |
|------|-----------|------------|
| **Mars X201** | 64 | `tpr_kernel<64,8>` |
| NVIDIA RTX系列 | 32 | `ldg_kernel` |

**注意**: 此安装包 (`install_x201`) 专为 Mars X201 编译。如需 NVIDIA GPU 版本，请使用 `install` 目录。

## 联系方式

永久许可证或技术支持请联系供应商。

---
版本: 1.0.0
许可证过期: 2026-05-07
目标平台: Mars X201 (国产GPU, Warp=64)
