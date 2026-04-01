# 国产GPU服务器编译运行经验

## 服务器信息（已更新2026-04-01）

- **地址**: chenbinxiangc@172.16.45.81
- **SSH端口**: 19936
- **操作系统**: Kylin (Arm64)
- **GPU**: Mars X201 (8卡)
- **工作目录**: /home/chenbinxiangc/spmv_muxi
- **指定GPU**: GPU7 (使用CUDA_VISIBLE_DEVICES=7)
- **原服务器(172.16.45.80)已永久停用**

## SSH连接命令

```bash
ssh -p 19936 chenbinxiangc@172.16.45.81
```

## 编译环境

### 编译器位置
- **国产CUDA编译器**: `~/cu-bridge/CUDA_DIR/bin/nvcc` (实际是htcc的软链接)
- **htcc编译器**: `/opt/hpcc/htgpu_llvm/bin/htcc`

### 推荐的编译方式（已验证可行）

由于cmake无法识别cu-bridge的nvcc作为CUDA编译器，推荐直接使用nvcc编译：

```bash
# 设置环境变量
export PATH=~/cu-bridge/CUDA_DIR/bin:$PATH
export LIBRARY_PATH=~/cu-bridge/CUDA_DIR/lib64:/opt/hpcc/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=~/cu-bridge/CUDA_DIR/lib64:/opt/hpcc/lib:$LD_LIBRARY_PATH

# 直接编译（推荐）
~/cu-bridge/CUDA_DIR/bin/nvcc -O3 -DWARP_SIZE=64 -Isrc \
    src/spmv/csr/spmv_csr.cu \
    src/utils/device_info.cu \
    src/api/spmv_api.cu \
    tests/correctness/test_spmv.cu \
    -o test_spmv

# 或使用编译脚本
./scripts/build_domestic.sh all
```

### CMake编译（存在问题）

cmake目前无法自动识别cu-bridge的nvcc。pre_make cmake仍有问题：
```
CMake Error: Couldn't find CUDA library root.
```

解决方案待研究：可能需要显式指定CMAKE_CUDA_COMPILER或修改cmake检测逻辑。

## 关键注意事项

### 1. Warp Size = 64

**这是最重要的差异！**

国产GPU的warp size是64，而不是NVIDIA的32。这意味着：
- 每个warp有64个线程
- 寄存器压力翻倍
- 共享内存使用量翻倍
- 需要重新设计kernel以适应64线程的warp

### 2. CUDA版本兼容性

国产GPU支持CUDA 11.6及以下的语法。代码中应避免：
- 使用CUDA 12+新增的API
- 使用CUDA_VERSION宏（国产GPU可能不支持）
- 使用printf进行GPU调试（国产GPU的printf有问题）
- 指定sm_xx算力版本号（国产GPU不支持）

### 3. GPU调试

**不要使用printf**进行GPU调试。使用：
```bash
# 使用指定的日志库
$HOME/cbx/muxi_print_bug/
```

### 4. GPU监控

```bash
# 使用ht-smi代替nvidia-smi
ht-smi
```

### 5. 指定GPU

```bash
# 只使用GPU7
export CUDA_VISIBLE_DEVICES=7
```

## 常见问题及解决方案

### 问题1: nvcc找不到
**解决方案**: 使用`~/cu-bridge/CUDA_DIR/bin/nvcc`

### 问题2: 链接错误找不到cudart
**解决方案**: 国产GPU不使用libcudart，使用cu-bridge的nvcc会自动处理

### 问题3: CUDA_VERSION未定义
**解决方案**: 不要使用CUDA_VERSION宏，国产GPU不支持

### 问题4: cudaDeviceProp字段不存在
**解决方案**: 使用cudaDeviceGetAttribute获取属性，而不是直接访问prop字段

### 问题5: sm_xx不支持
**解决方案**: 不要在编译时指定-arch=sm_xx，国产GPU不支持

## 性能优化建议

### Mars X201 GPU关键参数（实测）
- Warp Size: **64**（重要！NVIDIA为32）
- SM Count: 104
- Global Memory: 68.28 GB
- Peak Bandwidth: 1843.20 GB/s
- Registers/SM: 131072
- Shared Mem/SM: 64 KB
- Max Threads/SM: 2048
- Max Blocks/SM: 16

### 优化建议
1. **寄存器使用**: 每个线程最多使用64个寄存器（由于Warp Size翻倍，需谨慎）
2. **共享内存**: 每个SM只有64KB，需要谨慎使用
3. **Block大小**: 推荐128-256线程（2-4个warp）
4. **带宽优化**: 目标是达到峰值带宽的80%以上

## 文件路径约定

- 工作目录: `/home/chenbinxiangc/spmv_muxi/`
- cu-bridge: `~/cu-bridge/`
- 日志库: `$HOME/cbx/muxi_print_bug/`
- **严禁在其他目录写入或修改文件**