# 构建指南

## 前置条件

### NVIDIA RTX 4090

- CUDA Toolkit 11.6+ 或 12.x/13.x
- nvcc编译器
- 支持C++11的C++编译器

### Mars X201（国产GPU）

- cu-bridge层已安装
- GPU7访问权限（`CUDA_VISIBLE_DEVICES=7`）
- `pre_make`包装器用于nvcc编译

## 构建方法

### 方法1：直接使用nvcc（推荐）

#### NVIDIA RTX 4090

```bash
cd final/spmv_fp64_real
./scripts/build_direct_rtx.sh
```

或手动编译：

```bash
# 如果需要，设置CUDA路径
export PATH=/usr/local/cuda-13.1/bin:$PATH

mkdir -p build_direct
nvcc -O3 -arch=sm_89 -Xcompiler -fPIC -I./include -I./src -c src/spmv_fp64.cu -o build_direct/spmv_fp64.o
nvcc -shared -arch=sm_89 -Xcompiler -fPIC -o build_direct/libspmv_fp64.so build_direct/spmv_fp64.o
nvcc -O3 -arch=sm_89 -I./include -I./src examples/simple_example.cu -Lbuild_direct -lspmv_fp64 -o build_direct/spmv_example -Xlinker -rpath,build_direct
```

输出文件：
- `build_direct/libspmv_fp64.so`
- `build_direct/spmv_example`

#### Mars X201

```bash
# SSH连接到Mars服务器
ssh -p 19936 chenbinxiangc@172.16.45.81

# 进入项目目录
cd ~/spmv_muxi/final/spmv_fp64_real

# 使用pre_make包装器构建
mkdir -p build_direct
pre_make nvcc -O3 -Xcompiler -fPIC -I./include -I./src -c src/spmv_fp64.cu -o build_direct/spmv_fp64.o
pre_make nvcc -shared -Xcompiler -fPIC -o build_direct/libspmv_fp64.so build_direct/spmv_fp64.o
pre_make nvcc -O3 -I./include -I./src examples/simple_example.cu -Lbuild_direct -lspmv_fp64 -o build_direct/spmv_example

# 使用GPU7运行，并设置库路径
CUDA_VISIBLE_DEVICES=7 LD_LIBRARY_PATH=build_direct:$LD_LIBRARY_PATH ./build_direct/spmv_example matrix.mtx
```

## 已验证平台

| 平台 | Warp大小 | 内核 | 状态 |
|------|----------|------|------|
| Mars X201 | 64 | TPR=8 | ✅ 已验证 |
| RTX 4090 | 32 | __ldg | ✅ 已验证 |

## 重要注意事项

### Mars X201注意事项

1. **不要设置CUDA架构**：国产GPU不支持`sm_xx`标志
2. **使用pre_make**：所有nvcc命令必须使用`pre_make`包装器
3. **GPU选择**：始终使用GPU7（`CUDA_VISIBLE_DEVICES=7`）
4. **工作目录**：必须在`$HOME/cbx/spmv_muxi/`或`$HOME/spmv_muxi/`下工作
5. **不要使用printf**：使用`$HOME/muxi_print_bug`的日志库进行调试

### RTX 4090注意事项

1. **设置CUDA路径**：如果nvcc不在PATH中，设置`CUDA_PATH=/usr/local/cuda-13.1`
2. **架构**：RTX 4090使用`-arch=sm_89`

## 测试新增API接口 ⭐ 新增

库提供三种API模式：

1. **主机CSR模式** - 库管理所有内存（原始API）
2. **设备CSR模式** - 用户提供设备CSR指针（零拷贝）
3. **直接执行模式** - 无需句柄的一次性执行

### 测试所有API

```bash
# 构建测试程序
nvcc -O3 -arch=sm_89 -I./include -I./src -o build/test_new_api examples/test_new_api.cu build/libspmv_fp64.so

# 运行测试
LD_LIBRARY_PATH=build:$LD_LIBRARY_PATH ./build/test_new_api
```

预期输出：
```
=== Test 1: Host CSR API === PASSED
=== Test 2: Device CSR API === PASSED
=== Test 3: Direct Execution API === PASSED
=== Test 4: Error Handling === PASSED
=== All Tests Complete ===
```

### Mars X201测试

```bash
# SSH连接到Mars服务器
ssh -p 19936 chenbinxiangc@172.16.45.81
cd ~/spmv_muxi/final/spmv_fp64_real

# 使用pre_make构建
export PATH=/opt/hpcc/tools/cu-bridge/tools:$PATH
pre_make nvcc -O3 -I./include -I./src -o build/test_new_api examples/test_new_api.cu build/libspmv_fp64.so

# 使用GPU7运行
CUDA_VISIBLE_DEVICES=7 LD_LIBRARY_PATH=build:$LD_LIBRARY_PATH ./build/test_new_api
```

## 问题排查

### CUDA头文件未找到

如果看到：
```
fatal error: cuda_runtime.h: No such file or directory
```

解决方案：添加CUDA包含路径：
```bash
export PATH=/usr/local/cuda/bin:$PATH
```

### cu-bridge未找到

如果看到：
```
pre_make: command not found
```

解决方案：检查cu-bridge安装或使用直接nvcc路径。

### GPU权限错误

如果看到：
```
CUDA_ERROR_NO_DEVICE
```

解决方案：指定正确的GPU：
```bash
CUDA_VISIBLE_DEVICES=7 ./build_direct/spmv_example matrix.mtx
```

### CUDA 13+ API变更

如果看到：
```
class "cudaDeviceProp" has no member "memoryClockRate"
```

这是预期的 - CUDA 13+移除了已弃用的时钟字段。库使用GPU名称检测代替。

## 安装

系统级安装：

```bash
sudo cp build_direct/libspmv_fp64.so /usr/local/lib/
sudo cp include/spmv_fp64.h /usr/local/include/
sudo ldconfig
```

## 在您的项目中使用

安装后：

```cmake
include_directories(/usr/local/include)
link_directories(/usr/local/lib)
target_link_libraries(your_target PRIVATE spmv_fp64)
```

或直接在代码中使用：

```cpp
#include "spmv_fp64.h"

// 链接：-L/usr/local/lib -lspmv_fp64
```