# 平台开发差异：Mars X201 vs RTX 4090

## 硬件差异

| 特性 | Mars X201 | RTX 4090 |
|------|-----------|----------|
| Warp Size | 64 | 32 |
| L2 Cache | ~4MB | 72MB |
| 峰值带宽 | 1843 GB/s | 1008 GB/s |
| SM数量 | 104 | 128 |
| 编译器 | htcc (cu-bridge) | nvcc |

## 关键开发差异

### 1. Warp Size影响

**Mars X201 (warp=64)**:
```
问题: 极稀疏矩阵(avgNnz=4)线程利用率低
- 64线程中只有4个有工作 → 6.25%利用率
- 需要虚拟warp或列排序优化

解决方案:
1. 虚拟Warp kernel: 使用warp=8/16
2. 列排序: 改善缓存局部性
```

**RTX 4090 (warp=32)**:
```
优势: 同avgNnz=4下利用率是Mars X201的2倍
- 32线程中4个有工作 → 12.5%利用率
- 标准kernel表现良好
```

### 2. L2 Cache影响

**Mars X201 (~4MB L2)**:
```
问题: 无法缓存大矩阵
- 1M行x1000列矩阵: ~16MB数据
- 只能缓存25%

优化策略:
- 矩阵分块
- 列排序改善局部性
- 批处理kernel
```

**RTX 4090 (72MB L2)**:
```
优势: 可以完全缓存中小矩阵
- 16MB数据可完全放入L2
- 缓存命中率高
```

### 3. 编译差异

**Mars X201**:
```bash
# 使用cu-bridge
export PATH=$HOME/cu-bridge/CUDA_DIR/bin:$PATH
export LIBRARY_PATH=$HOME/cu-bridge/CUDA_DIR/lib64:$LIBRARY_PATH

# 编译时指定warp size
nvcc -DWARP_SIZE=64 ...

# 注意事项
- 不支持printf调试
- 不支持sm_xx架构号
- cmake需要pre_make前缀
```

**RTX 4090**:
```bash
# 标准CUDA
nvcc -DWARP_SIZE=32 ...

# 支持所有CUDA功能
```

### 4. 性能差异总结

| 场景 | Mars X201 | RTX 4090 | 差距原因 |
|------|-----------|----------|----------|
| avgNnz=4 | 15.9% | 100%+ | Warp size + L2 |
| avgNnz=10 | 28.2% | 79% | L2 cache |
| avgNnz=64 | 70-85% | 25% | 带宽优势 |
| Banded矩阵 | 95% | - | 规则访问 |

### 5. 优化策略差异

**Mars X201优化重点**:
1. 极稀疏矩阵(avgNnz<=4): 虚拟warp kernel
2. 稀疏矩阵(avgNnz<32): 列排序 + merge-based
3. 密集矩阵(avgNnz>=64): merge-based表现良好

**RTX 4090优化重点**:
1. 所有稀疏度: 标准merge-based即可
2. 利用大L2 cache
3. 关注kernel launch开销

### 6. 代码适配建议

```cpp
// 平台自适应代码
#ifdef WARP_SIZE
    #if WARP_SIZE == 64
        // Mars X201特定优化
        if (avgNnz <= 4) {
            spmv_virtual_warp<8>(matrix, x, y);
        } else {
            spmv_merge_based(matrix, x, y);
        }
    #else
        // RTX 4090
        spmv_merge_based(matrix, x, y);
    #endif
#endif
```

## 经验教训

1. **Warp Size是关键**: Mars X201的warp=64需要特殊处理
2. **L2 Cache决定性能**: 小cache需要更多优化
3. **调试困难**: Mars X201不支持printf，需要日志库
4. **编译环境**: cu-bridge有特殊要求

---
*文档创建: 2026-04-03*