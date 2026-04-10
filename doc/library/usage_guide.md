# Muxi SpMV 库使用指南

## 概述

Muxi SpMV 是一个高性能稀疏矩阵-向量乘法(SpMV)库，支持国产GPU(Mars X201)和NVIDIA GPU。

### 特性

- 支持多种稀疏矩阵格式：CSR, COO, CSR5, BSR
- 支持FP32和FP64精度
- 自动适配不同GPU架构(NVIDIA warp=32, Mars X201 warp=64)
- 最优内核自动选择
- Pinned Memory端到端优化

---

## 编译

### 环境要求

| 组件 | 版本要求 |
|------|----------|
| CUDA | 11.0+ (NVIDIA) / cu-bridge (Mars X201) |
| CMake | 3.18+ |
| C++编译器 | C++14支持 |

### NVIDIA GPU编译

```bash
# 克隆仓库
git clone <repo_url> muxi_spmv
cd muxi_spmv

# 创建构建目录
mkdir build && cd build

# 配置和编译
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)

# 安装 (可选)
make install
```

### 国产GPU (Mars X201) 编译

```bash
# 设置环境变量
export PATH=$PATH:~/cu-bridge/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/cu-bridge/lib:~/cu-bridge/lib64

# 使用pre_make编译
mkdir build && cd build
pre_make cmake -DCMAKE_BUILD_TYPE=Release ..
pre_make make -j$(nproc)
```

**注意**:
- 不要指定CUDA架构 (`sm_xx`)
- 所有cmake/make命令前加`pre_make`
- 设置`CUDA_VISIBLE_DEVICES=7`指定GPU

---

## API接口

### 1. 端到端接口 (CPU指针 → GPU计算 → CPU结果)

适用于数据在CPU内存，需要自动处理数据传输的场景。

```cpp
#include <muxi_spmv.h>

// FP64示例
int numRows = 1000000, numCols = 1000000, nnz = 10000000;
int* h_rowPtr = ...;  // CPU行指针数组
int* h_colIdx = ...;  // CPU列索引数组
double* h_values = ...;  // CPU值数组
double* h_x = ...;  // CPU输入向量
double* h_y = ...;  // CPU输出向量

// 调用端到端接口 (自动使用Pinned Memory优化)
spmv_status_t status = muxi_spmv::spmv_csr_raw_host<double>(
    numRows, numCols, nnz,
    h_rowPtr, h_colIdx, h_values,
    h_x, h_y
);
```

### 2. GPU内核接口 (GPU指针 → GPU计算)

适用于数据已在GPU内存，只需执行计算的场景。

```cpp
#include <muxi_spmv.h>

// GPU内存指针
int* d_rowPtr;   // GPU行指针
int* d_colIdx;   // GPU列索引
double* d_values;  // GPU值
double* d_x;     // GPU输入向量
double* d_y;     // GPU输出向量

// 分配GPU内存
cudaMalloc(&d_rowPtr, (numRows+1) * sizeof(int));
cudaMalloc(&d_colIdx, nnz * sizeof(int));
cudaMalloc(&d_values, nnz * sizeof(double));
cudaMalloc(&d_x, numCols * sizeof(double));
cudaMalloc(&d_y, numRows * sizeof(double));

// 拷贝数据到GPU
cudaMemcpy(d_rowPtr, h_rowPtr, ...);
// ... 其他拷贝

// 调用GPU计算接口
spmv_status_t status = muxi_spmv::spmv_csr_raw_device<double>(
    numRows, numCols, nnz,
    d_rowPtr, d_colIdx, d_values,
    d_x, d_y
);
```

### 3. 高级接口 (使用CSRMatrix结构)

```cpp
#include <muxi_spmv.h>

// 创建CSR矩阵结构
muxi_spmv::CSRMatrix<double> A;
A.numRows = numRows;
A.numCols = numCols;
A.nnz = nnz;
A.allocateHost(numRows, numCols, nnz);
A.allocateDevice();

// 填充数据...
A.copyToDevice();

// 分配向量
double *d_x, *d_y;
cudaMalloc(&d_x, A.numCols * sizeof(double));
cudaMalloc(&d_y, A.numRows * sizeof(double));

// 执行SpMV
muxi_spmv::spmv_opts_t opts = muxi_spmv::spmv_default_opts();
spmv_status_t status = muxi_spmv::spmv_csr<double>(A, d_x, d_y, 1.0, 0.0, opts);
```

---

## 完整示例

### 端到端示例 (推荐)

```cpp
#include <muxi_spmv.h>
#include <cstdio>

int main() {
    // 矩阵参数
    int numRows = 1000000;
    int numCols = 1000000;
    int nnz = 10000000;

    // 分配CPU内存
    int* h_rowPtr = (int*)malloc((numRows+1) * sizeof(int));
    int* h_colIdx = (int*)malloc(nnz * sizeof(int));
    double* h_values = (double*)malloc(nnz * sizeof(double));
    double* h_x = (double*)malloc(numCols * sizeof(double));
    double* h_y = (double*)malloc(numRows * sizeof(double));

    // 初始化数据...
    // h_rowPtr, h_colIdx, h_values, h_x

    // 调用端到端SpMV (自动处理GPU传输)
    spmv_status_t status = muxi_spmv::spmv_csr_raw_host<double>(
        numRows, numCols, nnz,
        h_rowPtr, h_colIdx, h_values,
        h_x, h_y
    );

    if (status != SPMV_SUCCESS) {
        printf("SpMV failed with error: %d\n", status);
        return 1;
    }

    // h_y 现在包含结果

    // 清理
    free(h_rowPtr);
    free(h_colIdx);
    free(h_values);
    free(h_x);
    free(h_y);

    return 0;
}
```

---

## 数据类型

### 状态码

```cpp
typedef enum {
    SPMV_SUCCESS = 0,               // 成功
    SPMV_ERROR_INVALID_HANDLE = -1, // 无效句柄
    SPMV_ERROR_INVALID_MATRIX = -2, // 无效矩阵
    SPMV_ERROR_INVALID_VECTOR = -3, // 无效向量
    SPMV_ERROR_MEMORY_ALLOC = -4,   // 内存分配失败
    SPMV_ERROR_MEMORY_COPY = -5,    // 内存拷贝失败
    SPMV_ERROR_DEVICE = -6,         // 设备错误
    SPMV_ERROR_UNSUPPORTED_FORMAT = -7, // 不支持的格式
} spmv_status_t;
```

### 执行选项

```cpp
typedef struct {
    spmv_operation_t operation;  // 操作类型
    int sync;                    // 0=异步, 1=同步
    int use_tensor_core;         // 使用Tensor Core
} spmv_opts_t;

// 获取默认选项
spmv_opts_t opts = spmv_default_opts();
```

---

## 性能建议

### 最优配置

| 平台 | 线程/行 | Block Size | Cache配置 |
|------|---------|------------|-----------|
| Mars X201 | 8 | 128 | PreferL1 |
| NVIDIA RTX 4090 | 4 | 256 | PreferL1 |

### 端到端优化

1. **使用Pinned Memory**: 端到端接口自动使用，提升2.9x性能
2. **避免频繁内存分配**: 复用GPU内存缓冲区
3. **批量处理**: 多个向量一起计算

---

## 编译为独立库

### 构建静态库

```bash
mkdir build && cd build
cmake -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release ..
make
```

### 构建动态库

```bash
cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release ..
make
```

### 安装

```bash
make install
# 头文件: /usr/local/include/muxi_spmv/
# 库文件: /usr/local/lib/libmuxi_spmv.a
```

### 使用已安装库

```cmake
# CMakeLists.txt
find_package(muxi_spmv REQUIRED)
target_link_libraries(your_target muxi_spmv)
```

---

## 常见问题

### Q: 编译时找不到cuda_runtime.h
A: 设置CUDA环境变量:
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Q: Mars X201运行时找不到GPU
A: 设置GPU设备:
```bash
export CUDA_VISIBLE_DEVICES=7
```

### Q: Mars X201编译报错 "sm_xx not supported"
A: 移除架构指定，让编译器自动处理:
```cmake
# 不要设置:
# set(CMAKE_CUDA_ARCHITECTURES "80")
```

---

## 文件结构

```
muxi_spmv/
├── include/
│   └── muxi_spmv.h          # 公共API头文件
├── src/
│   ├── api/
│   │   └── spmv_api.cpp     # API实现
│   ├── spmv/
│   │   └── csr/
│   │       └── spmv_csr.cu  # CSR内核
│   └── utils/
│       └── common.h         # 通用定义
├── lib/
│   └── libmuxi_spmv.a       # 静态库
└── doc/
    └── library/
        └── usage_guide.md   # 本文档
```