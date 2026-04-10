# 线程/行配置优化

## 原理

### 什么是线程/行 (Threads Per Row, TPR)?

SpMV Vector内核中，每行的非零元素由多个线程并行处理:
- TPR = 2: 每行2个线程处理
- TPR = 4: 每行4个线程处理
- TPR = 8: 每行8个线程处理

### 为什么TPR影响性能?

```
行数据分布:
row_i: [val0, val1, val2, val3, val4, val5, val6, val7, val8, val9]  (10个元素)

TPR=2: 线程0处理val0,val2,val4,val6,val8 (5个元素)
       线程1处理val1,val3,val5,val7,val9 (5个元素)

TPR=4: 线程0处理val0,val4,val8 (3个元素)
       线程1处理val1,val5,val9 (3个元素)
       线程2处理val2,val6 (2个元素)
       线程3处理val3,val7 (2个元素)

TPR=8: 线程0-7各处理1-2个元素
```

**关键权衡**:
- TPR越大: 每线程处理元素越少，并行度越高，但归约开销越大
- TPR越小: 每线程处理元素越多，内存延迟隐藏越差

### Warp Size的影响

**关键**: Warp Size决定最优TPR!

```
Mars X201 (Warp=64):
- TPR=4时，每warp处理 64/4 = 16行
- TPR=8时，每warp处理 64/8 = 8行

RTX 4090 (Warp=32):
- TPR=4时，每warp处理 32/4 = 8行
- TPR=8时，每warp处理 32/8 = 4行
```

---

## 代码实现

### Vector内核模板

```cpp
template<int TPR>  // TPR: Threads Per Row
__global__ void vector_kernel(
    int numRows, 
    const int* __restrict__ rowPtr, 
    const int* __restrict__ colIdx,
    const double* __restrict__ values, 
    const double* __restrict__ x, 
    double* __restrict__ y)
{
    // Warp Size检测
    #ifdef __CUDA_ARCH__
        const int WARP_SIZE = (sizeof(int) == 4 && __CUDA_ARCH__ >= 700) ? 32 : 64;
    #else
        const int WARP_SIZE = 32;
    #endif

    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int laneId = threadIdx.x % WARP_SIZE;
    int row = warpId * (WARP_SIZE / TPR) + laneId / TPR;

    if (row >= numRows) return;

    int rowStart = rowPtr[row], rowEnd = rowPtr[row + 1];
    double sum = 0.0;

    // 每线程处理stride元素
    for (int i = rowStart + (laneId % TPR); i < rowEnd; i += TPR) {
        sum += values[i] * x[colIdx[i]];
    }

    // Warp归约
    for (int offset = TPR / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (laneId % TPR == 0) {
        y[row] = sum;
    }
}
```

### 调用示例

```cpp
// Mars X201 (Warp=64): 最优TPR=8
void spmv_mars_x201(/* params */) {
    const int TPR = 8;
    int blockSize = 128;
    int gridSize = (numRows * TPR + blockSize - 1) / blockSize;
    
    cudaFuncSetCacheConfig(vector_kernel<TPR>, cudaFuncCachePreferL1);
    vector_kernel<TPR><<<gridSize, blockSize>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
}

// RTX 4090 (Warp=32): 最优TPR=4
void spmv_rtx4090(/* params */) {
    const int TPR = 4;
    int blockSize = 256;
    int gridSize = (numRows * TPR + blockSize - 1) / blockSize;
    
    cudaFuncSetCacheConfig(vector_kernel<TPR>, cudaFuncCachePreferL1);
    vector_kernel<TPR><<<gridSize, blockSize>>>(numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
}
```

### 自适应选择

```cpp
int getOptimalTPR(int warpSize, int avgNnzPerRow) {
    if (warpSize == 64) {
        // Mars X201: 需要更多线程隐藏延迟
        if (avgNnzPerRow < 16) return 8;
        return 8;  // Mars X201始终使用8t/row
    } else {
        // NVIDIA: 标准配置
        if (avgNnzPerRow < 8) return 2;
        if (avgNnzPerRow < 32) return 4;
        return 8;
    }
}

int getOptimalBlockSize(int warpSize) {
    return (warpSize == 64) ? 128 : 256;
}
```

---

## 性能数据

### Mars X201 (Warp=64, avgNnz=10.71)

| TPR | 时间(μs) | 带宽(GB/s) | 利用率 | 说明 |
|-----|----------|------------|--------|------|
| 2 | 620 | 434 | 23.6% | 线程太少，延迟隐藏差 |
| 4 | 345 | 780 | 42.3% | 之前认为最优 |
| **8** | **318** | **847** | **45.9%** | ⭐ 实际最优 |
| 16 | 362 | 743 | 40.3% | 归约开销增加 |
| 32 | 538 | 501 | 27.2% | 过度并行 |

**Mars X201结论**: TPR=8比TPR=4快**8.4%**

### RTX 4090 (Warp=32, avgNnz=10.71)

| TPR | 时间(μs) | 带宽(GB/s) | 利用率 | 说明 |
|-----|----------|------------|--------|------|
| 2 | 213 | 1266 | 125.6% | 接近最优 |
| **4** | **204** | **1321** | **131.0%** | ⭐ 最优 |
| 8 | 204 | 1319 | 130.9% | 同样优秀 |
| 16 | 216 | 1245 | 123.5% | 略慢 |
| 32 | 404 | 666 | 66.1% | 过度并行 |

**RTX 4090结论**: TPR=4最优，TPR=8效果相近

---

## 不同矩阵密度的影响

### avgNnz对最优TPR的影响

| avgNnz | Mars X201最优TPR | RTX 4090最优TPR |
|--------|-----------------|-----------------|
| < 4 | 8 | 2 |
| 4-16 | 8 | 4 |
| 16-64 | 8 | 4-8 |
| > 64 | 8-16 | 8 |

### 原理分析

```
理论利用率计算:
- Vector内核利用率 ≈ min(avgNnz / WarpSize, 100%)

Mars X201 (Warp=64):
- avgNnz=10: 理论利用率 = 10/64 = 15.6%
- TPR=8提升到45.9% (通过更多并发隐藏延迟)

RTX 4090 (Warp=32):
- avgNnz=10: 理论利用率 = 10/32 = 31.3%
- L2 Cache加成，实际可达131%
```

---

## Block Size的影响

### Mars X201 (TPR=8)

| Block Size | 时间(μs) | 说明 |
|------------|----------|------|
| 64 | 319 | 可接受 |
| **128** | **318** | ⭐ 最优 |
| 256 | 320 | 差异不大 |
| 512 | 324 | 略慢 |

### RTX 4090 (TPR=4)

| Block Size | 时间(μs) | 说明 |
|------------|----------|------|
| 64 | 204 | 可接受 |
| 128 | 204 | 可接受 |
| **256** | **204** | ⭐ 最优 |
| 512 | 204 | 可接受 |

**结论**: Block Size影响不大，但建议:
- Mars X201: 128
- RTX 4090: 256

---

## 最佳实践

### 推荐配置

```cpp
// 自动检测GPU并选择最优配置
void auto_config_spmv(int numRows, int avgNnzPerRow, 
                      int& tpr, int& blockSize) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    int warpSize = prop.warpSize;  // 32 or 64
    
    if (warpSize == 64) {
        // Mars X201
        tpr = 8;
        blockSize = 128;
    } else {
        // NVIDIA
        tpr = 4;
        blockSize = 256;
    }
}
```

### 完整示例

```cpp
template<typename FloatType>
void optimized_spmv(
    int numRows, int numCols, int nnz,
    const int* d_rowPtr, const int* d_colIdx, const FloatType* d_values,
    const FloatType* d_x, FloatType* d_y)
{
    // 获取设备属性
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int warpSize = prop.warpSize;
    
    // 选择最优配置
    int tpr, blockSize;
    if (warpSize == 64) {
        tpr = 8;
        blockSize = 128;
    } else {
        tpr = 4;
        blockSize = 256;
    }
    
    int gridSize = (numRows * tpr + blockSize - 1) / blockSize;
    
    // Cache配置
    cudaFuncSetCacheConfig(vector_kernel<8>, cudaFuncCachePreferL1);
    
    // 根据TPR调用对应内核
    switch(tpr) {
        case 4:
            vector_kernel<4><<<gridSize, blockSize>>>(
                numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
            break;
        case 8:
            vector_kernel<8><<<gridSize, blockSize>>>(
                numRows, d_rowPtr, d_colIdx, d_values, d_x, d_y);
            break;
    }
}
```