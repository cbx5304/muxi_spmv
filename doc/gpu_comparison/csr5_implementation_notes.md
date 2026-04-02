# CSR5格式开发经验

## 背景

CSR5格式是一种负载均衡的稀疏矩阵格式，旨在解决传统CSR格式在GPU上的线程空闲问题。

## CSR5原理

### 核心思想

1. **Tile划分**：将NNZ划分为固定大小的tiles
2. **固定工作量**：每warp处理一个tile，工作量均匀
3. **预处理**：预计算每个tile的起始行信息

### 理论优势

- 打破"1 warp/row"限制
- 对于稀疏矩阵实现负载均衡
- 理论上可以达到更高的利用率

## 实现细节

### 数据结构

```cpp
template<typename FloatType>
struct CSR5Matrix {
    // CSR基础数据（引用）
    int numRows, numCols, nnz;
    int* d_rowPtr, * d_colIdx;
    FloatType* d_values;
    
    // CSR5元数据
    int sigma;             // tile大小
    int numTiles;          // tile数量
    int* d_tile_row_ptr;   // 每tile起始行
    int* d_tile_nnz_offset; // 行内偏移
};
```

### 预处理Kernel

```cpp
// 计算每个tile的起始行
__global__ void csr5_compute_tile_boundaries_kernel(
    const int* rowPtr, int numRows, int nnz, int sigma,
    int* tile_row_ptr, int* tile_nnz_offset, int numTiles)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numTiles) return;
    
    int tileStart = tid * sigma;
    int row = binary_search_row(rowPtr, numRows, tileStart);
    
    tile_row_ptr[tid] = row;
    tile_nnz_offset[tid] = tileStart - rowPtr[row];
}
```

### SpMV Kernel

```cpp
// 每warp处理一个tile
__global__ void spmv_csr5_kernel(...)
{
    int warpId = ...;
    int tileStart = warpId * sigma;
    int tileEnd = min(tileStart + sigma, nnz);
    
    // 每线程处理 sigma/warpSize 个元素
    int myStart = tileStart + lane * elements_per_thread;
    
    // 处理元素，处理行边界
    for (int idx = myStart; idx < myEnd; idx++) {
        // 需要处理跨行边界的情况
        atomicAdd(&y[row], localSum);
    }
}
```

## 性能测试结果

### Mars X201测试（2026-04-02）

| Kernel | avgNnz=10 | avgNnz=64 |
|--------|-----------|-----------|
| Standard CSR (Vector) | **17.1%** | **77.8%** |
| CSR5 (当前实现) | 8.7% | 10.4% |

### 问题分析

当前CSR5实现性能不佳的原因：

1. **原子操作开销**
   - 每次跨行边界都需要atomicAdd
   - 大量原子操作严重影响性能

2. **Binary Search开销**
   - 每个线程需要确定自己处理的元素属于哪一行
   - 虽然有预计算的tile_row_ptr，但每个线程内部仍需搜索

3. **Warp内聚合缺失**
   - 没有实现warp内同行的结果聚合
   - 直接使用原子操作效率低

## 优化方向

### 1. 减少原子操作

**方法**：Warp内聚合

```cpp
// 同一warp内处理同一行的线程先进行warp级reduce
FloatType warpSum = warp_reduce_sum(localSum);
if (lane == 0) {
    atomicAdd(&y[row], warpSum);  // 只有线程0写入
}
```

### 2. 利用Segment Descriptor

参考标准CSR5实现：
- 预计算每个tile内的行边界（segment descriptor）
- 避免运行时binary search

### 3. 两阶段写入

```cpp
// Phase 1: 每个warp写入临时缓冲区
// Phase 2: 聚合跨warp的部分结果
```

## 经验教训

1. **原子操作是性能杀手**：在GPU编程中应尽量避免或减少

2. **预处理很重要**：更多的预处理数据可以减少运行时开销

3. **测试驱动优化**：先实现基础版本，测量瓶颈，再针对性优化

4. **参考成熟实现**：应参考CSR5原论文的高效实现细节

## 后续工作

1. 实现warp内聚合优化
2. 参考标准CSR5库的实现
3. 考虑其他稀疏格式（如Merge-based方法）

---

*文档创建: 2026-04-02*