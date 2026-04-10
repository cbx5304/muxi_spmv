# Pinned Memory 优化

## 原理

### 什么是Pinned Memory?

Pinned Memory (页锁定内存) 是一种特殊的CPU内存分配方式:
- 晱通内存(Pageable Memory): 可被操作系统换出到磁盘
- Pinned Memory: 锁定在物理内存中，不会被换出

### 为什么能加速?

**普通内存传输过程**:
```
CPU内存 → 临时缓冲区(内核空间) → GPU内存
        [操作系统拷贝]        [DMA传输]
```

**Pinned Memory传输过程**:
```
CPU Pinned内存 → GPU内存
               [DMA直接传输]
```

**关键**: 省去一次内存拷贝，DMA可以直接访问Pinned Memory。

---

## 代码实现

### 端到端示例

```cpp
#include <cuda_runtime.h>

// 原始代码 (普通内存)
void spmv_original(int numCols, double* h_x) {
    double* h_x = (double*)malloc(numCols * sizeof(double));
    double* d_x;
    cudaMalloc(&d_x, numCols * sizeof(double));
    
    // 传输慢 - 需要中间缓冲
    cudaMemcpy(d_x, h_x, numCols * sizeof(double), cudaMemcpyHostToDevice);
    
    // ... 计算内核 ...
    
    cudaMemcpy(h_y, d_y, numRows * sizeof(double), cudaMemcpyDeviceToHost);
    
    free(h_x);
    cudaFree(d_x);
}

// 优化代码 (Pinned Memory)
void spmv_optimized(int numCols, double* h_x) {
    double* h_x;
    cudaMallocHost(&h_x, numCols * sizeof(double));  // Pinned分配
    
    double* d_x;
    cudaMalloc(&d_x, numCols * sizeof(double));
    
    // 传输快 - DMA直接传输
    cudaMemcpy(d_x, h_x, numCols * sizeof(double), cudaMemcpyHostToDevice);
    
    // ... 计算内核 ...
    
    cudaMemcpy(h_y, d_y, numRows * sizeof(double), cudaMemcpyDeviceToHost);
    
    cudaFreeHost(h_x);  // Pinned释放
    cudaFree(d_x);
}
```

### 异步传输

```cpp
// 使用Pinned Memory + 异步传输
cudaStream_t stream;
cudaStreamCreate(&stream);

// 异步传输
cudaMemcpyAsync(d_x, h_x_pinned, size, cudaMemcpyHostToDevice, stream);

// 内核在同一个流中执行
kernel<<<grid, block, 0, stream>>>(...);

// 异步传回
cudaMemcpyAsync(h_y_pinned, d_y, size, cudaMemcpyDeviceToHost, stream);

// 等待完成
cudaStreamSynchronize(stream);
```

---

## 性能数据

### 测试配置
- 矩阵: 1.26M行, 13.5M NNZ
- x向量大小: 10.8 MB
- 平台: Mars X201, RTX 4090

### 端到端时间对比

| 平台 | 普通内存 | Pinned Memory | 提升 |
|------|----------|---------------|------|
| **Mars X201** | 5.6 ms | 1.96 ms | **+186%** |
| **RTX 4090** | 3.2 ms | 1.27 ms | **+152%** |

### 传输带宽对比

| 平台 | 普通内存带宽 | Pinned内存带宽 | 提升 |
|------|--------------|----------------|------|
| Mars X201 | 2.8 GB/s | 12.2 GB/s | 4.4x |
| RTX 4090 | 4.1 GB/s | 13.5 GB/s | 3.3x |

---

## 适用场景

### 必须使用的场景

1. **端到端SpMV计算**: 数据在CPU，结果需要在CPU
2. **频繁CPU-GPU传输**: 每次迭代都需要传输数据
3. **实时处理**: 需要最小化延迟

### 可以不使用的场景

1. **纯GPU计算**: 数据始终在GPU
2. **一次性计算**: 传输时间可忽略

---

## 注意事项

### 内存限制

Pinned Memory占用物理内存，不能被换出:
- 分配过多可能导致系统内存不足
- 建议只对频繁传输的数据使用

### 分配开销

Pinned Memory分配比普通malloc慢:
- 适合长期使用，不适合频繁分配释放
- 可以预分配内存池

### 代码示例: 内存池

```cpp
class PinnedMemoryPool {
private:
    std::vector<void*> pool;
    std::vector<size_t> sizes;
    
public:
    void* allocate(size_t size) {
        // 查找可用块
        for (size_t i = 0; i < pool.size(); i++) {
            if (sizes[i] >= size) {
                void* ptr = pool[i];
                pool.erase(pool.begin() + i);
                sizes.erase(sizes.begin() + i);
                return ptr;
            }
        }
        
        // 分配新块
        void* ptr;
        cudaMallocHost(&ptr, size);
        return ptr;
    }
    
    void deallocate(void* ptr, size_t size) {
        pool.push_back(ptr);
        sizes.push_back(size);
    }
};
```

---

## 与其他优化配合

Pinned Memory是端到端优化的基础，应与以下优化配合使用:

1. **内核优化**: Vector 8t/row (Mars) / 4t/row (RTX)
2. **Cache配置**: PreferL1
3. **异步传输**: cudaMemcpyAsync

完整优化代码:

```cpp
// 完整的端到端优化SpMV
void optimized_spmv_end_to_end(
    int numRows, int numCols, int nnz,
    int* h_rowPtr, int* h_colIdx, double* h_values,
    double* h_x, double* h_y)
{
    // 1. Pinned Memory
    double *h_x_pin, *h_y_pin;
    cudaMallocHost(&h_x_pin, numCols * sizeof(double));
    cudaMallocHost(&h_y_pin, numRows * sizeof(double));
    memcpy(h_x_pin, h_x, numCols * sizeof(double));
    
    // 2. GPU内存
    int *d_rowPtr, *d_colIdx;
    double *d_values, *d_x, *d_y;
    cudaMalloc(&d_rowPtr, (numRows+1) * sizeof(int));
    cudaMalloc(&d_colIdx, nnz * sizeof(int));
    cudaMalloc(&d_values, nnz * sizeof(double));
    cudaMalloc(&d_x, numCols * sizeof(double));
    cudaMalloc(&d_y, numRows * sizeof(double));
    
    // 3. 传输数据
    cudaMemcpy(d_rowPtr, h_rowPtr, (numRows+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdx, h_colIdx, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values, nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x_pin, numCols * sizeof(double), cudaMemcpyHostToDevice);
    
    // 4. Cache配置
    cudaFuncSetCacheConfig(vector_kernel<8>, cudaFuncCachePreferL1);
    
    // 5. 执行内核 (Mars X201: 8t/row)
    int blockSize = 128;
    int gridSize = (numRows * 8 + blockSize - 1) / blockSize;
    vector_kernel<8><<<gridSize, blockSize>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
    
    // 6. 传回结果
    cudaMemcpy(h_y_pin, d_y, numRows * sizeof(double), cudaMemcpyDeviceToHost);
    memcpy(h_y, h_y_pin, numRows * sizeof(double));
    
    // 7. 清理
    cudaFreeHost(h_x_pin);
    cudaFreeHost(h_y_pin);
    cudaFree(d_rowPtr); cudaFree(d_colIdx); cudaFree(d_values);
    cudaFree(d_x); cudaFree(d_y);
}
```