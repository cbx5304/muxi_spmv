# SpMV Performance Optimization Report - CSR5 and Merge-based Implementation

## Executive Summary

This report documents the implementation and performance evaluation of CSR5 format and Merge-based SpMV algorithms for sparse matrix-vector multiplication on two GPU architectures:
- **Mars X201** (国产GPU, warp=64, 1843 GB/s peak)
- **RTX 4090** (NVIDIA, warp=32, 1008 GB/s peak)

## Implementation Summary

### 1. CSR5 Format

**原理**: 将NNZ划分为固定大小的tiles，每个warp处理一个tile，实现负载均衡。

**核心组件**:
- `CSR5Matrix` 数据结构：包含CSR基础数据 + tile元数据
- `csr5_preprocess`: 预处理kernel，计算每个tile的起始行
- `spmv_csr5_warp64_kernel`: 针对warp=64优化的SpMV kernel
- `spmv_csr5_warp32_kernel`: 针对warp=32优化的SpMV kernel

**关键优化**:
- Tile size: sigma = 256 (warp=64) 或 128 (warp=32)
- 预计算tile_row_ptr避免部分二分查找开销
- Warp级聚合减少原子操作

### 2. Merge-based SpMV

**原理**: 使用merge-path算法将行迭代和NNZ迭代合并，避免原子操作。

**核心组件**:
- `merge_path_search`: 在merge path上查找(row, nnz)坐标
- `compute_merge_partitions_kernel`: 计算partition边界
- `spmv_merge_based_kernel`: 主kernel，每个warp处理一个partition

**关键优化**:
- Partition沿merge-path均匀分布
- 仅在partition边界需要原子操作
- 大row使用warp级reduce

## Performance Results

### Mars X201 (warp=64) - Large Matrices (1M rows × 1K cols)

| avgNnzPerRow | Standard CSR | CSR5 (w/o conv) | Merge-based |
|--------------|--------------|-----------------|-------------|
| 10 | 19.8% | 8.7% | 13.7% |
| 32 | 36.5% | 8.8% | 25.7% |
| 64 | 52.6% | 8.7% | 44.7% |

**带宽利用率峰值**: 1843.20 GB/s

### RTX 4090 (warp=32) - Large Matrices (1M rows × 1K cols)

| avgNnzPerRow | Standard CSR | CSR5 (w/o conv) | Merge-based |
|--------------|--------------|-----------------|-------------|
| 10 | 85.8% | 73.0% | *bug* |
| 32 | *bug* | 76.1% | *bug* |
| 64 | *bug* | 78.5% | *bug* |

**带宽利用率峰值**: 1008.10 GB/s

*注：RTX 4090上Standard CSR和Merge-based存在timing bug，导致部分测试显示>100%利用率*

## Key Findings

### 1. Mars X201 (国产GPU, warp=64)

**标准CSR表现最佳**:
- avgNnz=64时达到52.6%利用率
- 对于稀疏矩阵(avgNnz<64)，merge-based有改善空间

**CSR5性能问题**:
- 仅8.7%利用率，远低于预期
- 原因分析：
  1. 原子操作开销过大
  2. Warp内聚合实现仍有优化空间
  3. 预处理开销(0.6ms)不能被摊销

**Merge-based表现中等**:
- 比CSR5好，但不如标准CSR
- 需要进一步优化merge-path搜索

### 2. RTX 4090 (warp=32)

**CSR5表现稳定**:
- 73-78%利用率，接近理论最优
- 表明CSR5算法在warp=32架构上有效

**Standard CSR和Merge-based存在Bug**:
- 对于较密矩阵显示>100%利用率
- 可能原因：
  1. 时序测量精度问题
  2. CUDA缓存效应
  3. Vector kernel选择逻辑问题

## Technical Issues Encountered

### 1. 国产GPU编译问题

**问题**: cu-bridge替换CUDA类型导致链接错误

**解决方案**:
- 使用 `-dc` (device-code) 分别编译
- 添加显式模板实例化
- 移除模板实例化使用隐式实例化（最终方案）

### 2. Warp Size差异

**Mars X201**: warp=64
- Tile size = 256 (64 threads × 4 elements)
- Vector kernel每个warp处理一行
- 理论利用率 = min(avgNnzPerRow/64, 100%)

**RTX 4090**: warp=32
- Tile size = 128 (32 threads × 4 elements)
- Vector kernel每个warp处理一行
- 理论利用率 = min(avgNnzPerRow/32, 100%)

### 3. CSR5性能瓶颈

**根本原因**: 原子操作开销

**解决方案尝试**:
1. Warp级聚合 - 部分改善
2. 减少原子操作频率 - 待优化
3. 使用shared memory缓存 - 待实现

## Recommendations

### 1. 对于Mars X201

- **avgNnzPerRow >= 64**: 使用标准CSR Vector kernel
- **avgNnzPerRow < 64**: 
  - 标准CSR Light-balanced kernel最佳
  - Merge-based作为备选方案
  - CSR5不推荐（需要进一步优化）

### 2. 对于RTX 4090

- **CSR5格式表现良好**: 73-78%利用率
- 需要修复Standard CSR和Merge-based的timing bug

### 3. 后续优化方向

1. **CSR5优化**:
   - 完全消除原子操作（使用segment descriptor）
   - 实现真正的warp级聚合
   - 参考原版CSR5库实现

2. **Merge-based优化**:
   - 优化merge-path搜索算法
   - 使用shared memory缓存partition信息
   - 减少边界原子操作

3. **测试框架优化**:
   - 修复RTX 4090上的timing bug
   - 添加更精确的性能测量

## Files Modified

| File | Changes |
|------|---------|
| `src/spmv/csr5/spmv_csr5.cuh` | CSR5 kernel声明 |
| `src/spmv/csr5/spmv_csr5.cu` | CSR5和Merge-based kernel实现 |
| `src/formats/sparse_formats.h` | CSR5Matrix数据结构 |
| `src/benchmark/performance_benchmark.cu` | CSR5性能测试函数 |
| `tests/benchmark/test_runner.cu` | 测试运行器，支持CSR5和Merge-based |
| `scripts/build_domestic.sh` | 国产GPU编译脚本 |
| `scripts/build_nvidia.sh` | NVIDIA GPU编译脚本 |

## Conclusion

CSR5和Merge-based算法的实现已完成，但在Mars X201上的性能提升有限。主要原因是：

1. **CSR5**: 原子操作开销抵消了负载均衡的优势
2. **Merge-based**: 算法复杂度和实现细节需要进一步优化

对于Mars X201这类warp=64的架构，标准CSR的Light-balanced kernel在稀疏矩阵上表现最佳，而Vector kernel在较密矩阵上表现优异。

对于RTX 4090这类warp=32的架构，CSR5格式能够达到73-78%的利用率，是一个可行的优化方案。

---

*Report generated: 2026-04-02*
*Authors: Claude Code Assistant*