# 国产GPU服务器编译运行经验

## 服务器信息

- **地址**: chenbinxiangc@172.16.45.80
- **操作系统**: Kylin (Arm64)
- **GPU**: Mars X201 (8卡)
- **工作目录**: ~/cbx/spmv_muxi/
- **指定GPU**: GPU7 (使用CUDA_VISIBLE_DEVICES=7)

## 编译环境

### 编译器位置
- **国产CUDA编译器**: `~/cu-bridge/CUDA_DIR/bin/nvcc` (实际是cucc的软链接)
- **htcc编译器**: `/opt/hpcc/htgpu_llvm/bin/htcc`

### 推荐的编译方式

```bash
# 使用cu-bridge的nvcc（推荐，兼容CUDA语法）
~/cu-bridge/CUDA_DIR/bin/nvcc -O3 device_info.cu -o device_info

# 或者使用环境变量设置PATH
export PATH=~/cu-bridge/CUDA_DIR/bin:$PATH
nvcc -O3 device_info.cu -o device_info
```

### CMake项目编译

```bash
# 创建build目录
mkdir -p build && cd build

# 设置环境变量
export PATH=~/cu-bridge/CUDA_DIR/bin:$PATH
export LD_LIBRARY_PATH=/opt/hpcc/lib:$LD_LIBRARY_PATH

# 使用pre_make cmake（如果系统支持）
pre_make cmake ..
pre_make make
```

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

### 3. GPU调试

**不要使用printf**进行GPU调试。使用：
```bash
# 使用指定的日志库
~/cbx/muxi_print_bug/
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

## 性能优化建议

1. **寄存器使用**: 每个线程最多使用64个寄存器
2. **共享内存**: 每个SM只有64KB，需要谨慎使用
3. **Block大小**: 推荐128-256线程（2-4个warp）
4. **带宽优化**: 目标是达到峰值带宽的80%以上

## 文件路径约定

- 工作目录: `~/cbx/spmv_muxi/`
- cu-bridge: `~/cu-bridge/`
- 日志库: `~/cbx/muxi_print_bug/`
- **严禁在其他目录写入或修改文件**