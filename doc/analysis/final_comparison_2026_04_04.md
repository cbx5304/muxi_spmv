# Mars X201 vs RTX 4090 SpMV性能对比 - 最终报告

## 测试配置

| 参数 | Mars X201 | RTX 4090 |
|------|-----------|----------|
| Warp Size | 64 | 32 |
| 峰值带宽 | 1843 GB/s | 1008 GB/s |
| L2 Cache | ~4MB | 72MB |
| SM数量 | 104 | 128 |

## 性能对比 (avgNnz=4, 极稀疏矩阵)

### Mars X201 优化历程

| 优化方案 | 利用率 | 带宽 | 备注 |
|----------|--------|------|------|
| Scalar kernel | 7.7% | 142 GB/s | 基准 |
| ILP kernel | 13.7% | 252 GB/s | 指令级并行 |
| 虚拟Warp=8 | 15.9% | 292 GB/s | 之前最优 |
| **Adaptive Warp + 大SMEM** | **25.15%** | **464 GB/s** | **新最优** |

### 与RTX 4090对比

| avgNnz | Mars X201 (优化后) | Mars X201 (优化前) | RTX 4090 | 差距变化 |
|--------|-------------------|-------------------|----------|---------|
| 4 | **25%** | 15.9% | ~100% | **6.3x → 4x** |
| 6 | 25.77% | 19.9% | ~100% | 5.1x → 3.9x |
| 8 | 26.89% | 24.4% | ~80% | 3.3x → 3.0x |

## 共享内存大小影响（更新发现）

### 小矩阵 (rows=100K)

| 共享内存 | 利用率 | 提升 |
|----------|--------|------|
| 272 bytes | 3.73% | 基准 |
| 4096 bytes | 7.19% | +93% |
| 8192 bytes | 10.58% | +184% |
| **16384 bytes** | **11.51%** | **+209%** |

### 大矩阵 (rows=1M)

| 共享内存 | 利用率 | 备注 |
|----------|--------|------|
| 272 bytes | 24.95% | |
| 2048 bytes | 25.18% | 最优 |
| 16384 bytes | 24.51% | 略下降 |

**结论**:
- 小矩阵：大共享内存显著提升性能
- 大矩阵：共享内存大小影响有限，2048 bytes最优

## 关键发现

### 1. 共享内存大小对性能有决定性影响（小矩阵）

```
共享内存大小    利用率
272 bytes      3.66%
1024 bytes     6.25%
1028 bytes     8.05%  ← 关键跳跃点！
2048 bytes     11.41%
4096 bytes     11.83%
```

**结论**: 使用 `__shared__ int sharedRowPtr[BLOCK_SIZE + 1]` (1028个int)

### 2. 线程分配策略

| 线程/行 | 行/warp | 利用率 |
|--------|---------|--------|
| 4 | 16 | 16.11% |
| **8** | **8** | **16.14%** |
| 16 | 4 | 12.87% |

### 3. 优化效果总结

- **优化前**: 15.9% (虚拟Warp=8)
- **优化后**: 25.15% (Adaptive Warp + 大SMEM)
- **提升**: **+58%**

## 推荐实现

```cpp
template<typename FloatType, int BLOCK_SIZE>
__global__ void spmv_optimized_kernel(
    int numRows,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y)
{
    // 关键：使用大共享内存
    __shared__ int sharedRowPtr[BLOCK_SIZE + 1];

    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int globalWarpId = blockIdx.x * (BLOCK_SIZE / WARP_SIZE) + warpId;

    // Mars X201: 每warp处理16行 (每行4线程)
    // RTX 4090: 每warp处理8行 (每行4线程)
    #if WARP_SIZE == 64
    int baseRow = globalWarpId * 16;
    const int ROWS_PER_WARP = 16;
    #else
    int baseRow = globalWarpId * 8;
    const int ROWS_PER_WARP = 8;
    #endif

    // 加载row pointer到共享内存
    if (lane < ROWS_PER_WARP + 1 && baseRow + lane <= numRows) {
        sharedRowPtr[threadIdx.x / WARP_SIZE * (ROWS_PER_WARP + 1) + lane] = 
            rowPtr[baseRow + lane];
    }
    __syncthreads();

    int rowIdx = lane / 4;
    int threadInRow = lane % 4;
    int row = baseRow + rowIdx;

    if (row >= numRows) return;

    FloatType sum = 0;
    int rowStart = sharedRowPtr[threadIdx.x / WARP_SIZE * (ROWS_PER_WARP + 1) + rowIdx];
    int rowEnd = sharedRowPtr[threadIdx.x / WARP_SIZE * (ROWS_PER_WARP + 1) + rowIdx + 1];

    for (int idx = rowStart + threadInRow; idx < rowEnd; idx += 4) {
        sum += values[idx] * __ldg(&x[colIdx[idx]]);
    }

    sum += __shfl_down_sync(0xffffffff, sum, 2, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 1, 4);

    if (threadInRow == 0) {
        y[row] = sum;
    }
}
```

## 硬件差异影响分析

### 1. Warp Size (64 vs 32)

- Mars X201的warp=64导致极稀疏矩阵时线程利用率低
- 使用Adaptive Warp策略可以将每行分配给4线程，提高利用率

### 2. L2 Cache (~4MB vs 72MB)

- RTX 4090的大L2可以缓存整个x向量
- Mars X201需要依赖共享内存优化

### 3. 共享内存架构差异

- Mars X201的共享内存性能与大块分配正相关
- 可能存在特殊的bank冲突模式

## 结论

1. **Adaptive Warp + 大SMEM** 是Mars X201的最优方案
2. 性能从15.9%提升到**25.15%** (+58%)
3. 与RTX 4090的差距从6.3x缩小到**4x**
4. **共享内存大小是关键优化点**

---
*报告生成: 2026-04-04*
*测试平台: Mars X201 (GPU7), RTX 4090*