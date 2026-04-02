# GPU性能开发差异指南

本文档记录了Mars X201 (国产GPU, warp=64) 和 RTX 4090 (NVIDIA, warp=32) 之间的开发差异和注意事项。

## 硬件规格对比

| 参数 | RTX 4090 | Mars X201 |
|------|----------|-----------|
| 架构 | NVIDIA Ada Lovelace | 国产自研 |
| Warp Size | 32 | **64** |
| SM数量 | 128 | 104 |
| 峰值带宽 | 1008 GB/s | 1843 GB/s |
| 显存 | 24 GB | 68 GB |

## 关键开发差异

### 1. Warp Size影响

**最重要差异**：Mars X201使用64线程的warp，而NVIDIA使用32线程。

**影响范围**：
- Kernel设计：Vector kernel每warp处理一行时，warp=64需要更多元素才能充分利用
- 理论利用率：`max_utilization = min(avgNnzPerRow / warpSize, 100%)`
- 寄存器使用：warp=64时每个SM的寄存器分配不同
- 共享内存：warp级操作需要适配

**实际影响**：
| avgNnzPerRow | RTX 4090 (warp=32) | Mars X201 (warp=64) |
|--------------|-------------------|---------------------|
| 10 | 理论31.3% | 理论15.6% |
| 32 | 理论100% | 理论50% |
| 64 | 理论100% | 理论100% |

### 2. 编译环境差异

**RTX 4090**：
```bash
# 标准CUDA编译
nvcc -O3 -DWARP_SIZE=32 ...
```

**Mars X201**：
```bash
# 使用cu-bridge编译
export PATH=$HOME/cu-bridge/CUDA_DIR/bin:$PATH
export LD_LIBRARY_PATH=$HOME/cu-bridge/CUDA_DIR/lib64:$LD_LIBRARY_PATH
nvcc -O3 -DWARP_SIZE=64 ...
```

**注意事项**：
1. 使用`pre_make cmake`和`pre_make make`替代标准cmake/make
2. 不要在CMakeLists.txt中强制指定CUDA库路径
3. 使用`find_package`自动查找库
4. 不支持打印sm_xx架构版本号
5. 仅支持CUDA 11.6及以下语法

### 3. 调试差异

**Mars X201限制**：
- `printf`在kernel中不可靠，避免使用
- 可使用`$HOME/cbx/muxi_print_bug`日志库
- 使用`ht-smi`替代`nvidia-smi`

### 4. 性能调优策略

**Mars X201特定优化**：
1. **Block Size**：推荐256 (4 warps per block)
2. **Tile Size**：对于CSR5，使用256 (64 * 4)
3. **Shared Memory**：注意warp=64时共享内存分配差异
4. **原子操作**：尽量减少，性能影响更大

## SpMV Kernel选择策略

### Mars X201 (warp=64)

```cpp
if (avgNnzPerRow < 8) {
    // Light balanced kernel - 多行/线程
    // 适用于极稀疏矩阵
} else {
    // Vector kernel - 1 warp/row
    // 适用于avgNnzPerRow >= 8
}
```

### RTX 4090 (warp=32)

```cpp
if (avgNnzPerRow < 32) {
    // Scalar kernel - 1 thread/row
} else {
    // Vector kernel - 1 warp/row
}
```

## 性能测试结果

### 最终性能（2026-04-02）

| GPU | avgNnz=10 | avgNnz=64 | 说明 |
|-----|-----------|-----------|------|
| RTX 4090 | 86.1% | 88.9% | NVIDIA GPU |
| Mars X201 | 17.1% | **77.8%** | 国产GPU |

### 关键发现

1. **密集矩阵性能达标**：Mars X201对于avgNnzPerRow >= 64的矩阵达到77.8%带宽利用率

2. **稀疏矩阵受限**：低利用率是warp=64架构的物理限制，无法通过kernel优化完全解决

3. **CSR5挑战**：当前CSR5实现因原子操作开销，性能不如预期，需要进一步优化

## 开发最佳实践

### 1. 跨平台兼容

```cpp
// 使用编译时宏区分
#if WARP_SIZE == 64
    // Mars X201特定代码
#else
    // NVIDIA特定代码
#endif
```

### 2. 避免硬编码

```cpp
// 错误：硬编码warp大小
constexpr int WARP_SIZE = 32;  // 与宏冲突

// 正确：使用宏或不同名称
constexpr int LOCAL_WARP_SIZE = 64;
```

### 3. 测试流程

1. 在本地编写代码
2. 同步到两台服务器
3. 分别编译测试
4. 对比性能结果
5. 更新文档

### 4. 目录规范

- Mars X201: `$HOME/spmv_muxi/`
- RTX 4090: `$HOME/cbx/spmv_muxi/`
- 禁止在其他目录写入文件

## 常见问题

### Q1: 为什么稀疏矩阵性能低？

A: 这是warp=64架构的物理限制。当avgNnzPerRow < warpSize时，每个warp中只有部分线程有工作。

### Q2: 如何提高稀疏矩阵性能？

A: 考虑以下方案：
1. CSR5格式（需要优化实现）
2. 行重排序预处理
3. 使用其他稀疏格式（如ELLPACK）

### Q3: 编译错误"expected unqualified-id"

A: 检查是否定义了与宏同名的变量，如`constexpr int WARP_SIZE`与`#define WARP_SIZE`冲突。

---

*最后更新: 2026-04-02*
*维护者: Claude Code 自动生成*