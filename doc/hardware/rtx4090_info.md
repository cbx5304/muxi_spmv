# GPU Hardware Information

**Device:** NVIDIA GeForce RTX 4090

**Generated:** Mar 31 2026 08:37:30

## Basic Information

| Property | Value |
|----------|-------|
| Device ID | 0 |
| Device Name | NVIDIA GeForce RTX 4090 |
| Compute Capability | 8.9 |

## Memory Specifications

| Property | Value |
|----------|-------|
| Global Memory | 25.25 GB |
| Constant Memory | 64.00 KB |
| Memory Bus Width | 384 bits |
| Memory Clock Rate | 10501 MHz |
| Peak Memory Bandwidth | 1008.10 GB/s |

## Execution Resources

| Property | Value |
|----------|-------|
| SM Count | 128 |
| Warp Size | 32 |
| Max Threads/Block | 1024 |
| Max Threads/SM | 1536 |
| Max Blocks/SM | 24 |

## Register & Shared Memory

| Property | Value |
|----------|-------|
| Registers/Block | 65536 |
| Registers/SM | 65536 |
| Shared Memory/Block | 48.00 KB |
| Shared Memory/SM | 100.00 KB |
| Max Shared Memory (Opt-in) | 99.00 KB |

## Performance Metrics

| Property | Value |
|----------|-------|
| GPU Clock Rate | 2520 MHz |
| FP32 Cores/SM | 128 |
| Peak FP32 FLOPS | 82575.36 TFLOPS |
| Tensor Core Support | Yes |

## Cache & Execution Capabilities

| Property | Value |
|----------|-------|
| L2 Cache Size | 73728.00 KB |
| Concurrent Kernels | 1 |
| Async Copy Engines | 2 |
| Managed Memory | Yes |

## Grid & Dimension Limits

| Property | Value |
|----------|-------|
| Max Threads Dimensions | (1024, 1024, 64) |
| Max Grid Size | (2147483647, 65535, 65535) |

## SpMV Optimization Implications

### Warp Size Considerations
- Standard NVIDIA warp size (32 threads)
- Use vector-based and merge-based SpMV kernels
- Each warp processes multiple rows or uses merge-based load balancing

### Shared Memory Strategy
- Available shared memory per SM: 100.00 KB
- Max shared memory per block: 99.00 KB
- For CSR SpMV: Consider using shared memory for row pointer caching

### Register Usage
- Registers per SM: 65536
- Registers per block: 65536
- Max threads per SM: 1536
- Recommended register usage per thread: < 42

### Bandwidth Optimization
- Peak bandwidth: 1008.10 GB/s
- SpMV is memory-bound; optimize for bandwidth utilization
- Target: >80% bandwidth utilization for large matrices

### SM Occupancy
- Target occupancy: 50-100% for SpMV kernels
- Block size recommendation: 64 - 128
