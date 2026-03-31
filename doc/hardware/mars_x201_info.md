# GPU Hardware Information

**Device:** Mars 01

**Generated:** Mar 31 2026 16:55:31

## Basic Information

| Property | Value |
|----------|-------|
| Device ID | 0 |
| Device Name | Mars 01 |
| Compute Capability | 8.0 |

## Memory Specifications

| Property | Value |
|----------|-------|
| Global Memory | 68.28 GB |
| Constant Memory | 2097152.00 KB |
| Memory Bus Width | 4096 bits |
| Memory Clock Rate | 1800 MHz |
| Peak Memory Bandwidth | 1843.20 GB/s |

## Execution Resources

| Property | Value |
|----------|-------|
| SM Count | 104 |
| Warp Size | 64 |
| Max Threads/Block | 1024 |
| Max Threads/SM | 2048 |
| Max Blocks/SM | 16 |

## Register & Shared Memory

| Property | Value |
|----------|-------|
| Registers/Block | 131072 |
| Registers/SM | 131072 |
| Shared Memory/Block | 64.00 KB |
| Shared Memory/SM | 64.00 KB |
| Max Shared Memory (Opt-in) | 64.00 KB |

## Performance Metrics

| Property | Value |
|----------|-------|
| GPU Clock Rate | 1600 MHz |
| FP32 Cores/SM | 64 |
| Peak FP32 FLOPS | 21299.20 TFLOPS |
| Tensor Core Support | Yes |

## Cache & Execution Capabilities

| Property | Value |
|----------|-------|
| L2 Cache Size | 8192.00 KB |
| Concurrent Kernels | 1 |
| Async Copy Engines | 2 |
| Managed Memory | Yes |

## Grid & Dimension Limits

| Property | Value |
|----------|-------|
| Max Threads Dimensions | (1024, 1024, 1024) |
| Max Grid Size | (2147483647, 2147483647, 2147483647) |

## SpMV Optimization Implications

### Warp Size Considerations
- **Extended warp size (64 threads) - Domestic GPU**
- Critical: Kernel design must adapt to 64-thread warps
- Register pressure higher per warp
- Shared memory usage per warp doubled
- Recommended: Use larger block sizes to match warp size

### Shared Memory Strategy
- Available shared memory per SM: 64.00 KB
- Max shared memory per block: 64.00 KB
- For CSR SpMV: Consider using shared memory for row pointer caching

### Register Usage
- Registers per SM: 131072
- Registers per block: 131072
- Max threads per SM: 2048
- Recommended register usage per thread: < 64

### Bandwidth Optimization
- Peak bandwidth: 1843.20 GB/s
- SpMV is memory-bound; optimize for bandwidth utilization
- Target: >80% bandwidth utilization for large matrices

### SM Occupancy
- Target occupancy: 50-100% for SpMV kernels
- Block size recommendation: 128 - 256
