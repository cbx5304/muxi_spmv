# GPU Hardware Comparison

## Overview

| Property | RTX 4090 (NVIDIA) | Mars X201 (Domestic) |
|----------|-------------------|----------------------|
| **Warp Size** | **32** | **64** ⚠️ CRITICAL |
| SM Count | 128 | 104 |
| Global Memory | 25.25 GB | 68.28 GB |
| Memory Bus Width | 384 bits | 4096 bits |
| Peak Bandwidth | 1008.10 GB/s | 1843.20 GB/s |
| GPU Clock | 2520 MHz | 1600 MHz |
| Registers/SM | 65536 | 131072 |
| Shared Mem/SM | 100 KB | 64 KB |
| Max Threads/SM | 1536 | 2048 |
| Max Blocks/SM | 24 | 16 |
| L2 Cache | 73728 KB | 8192 KB |
| Compute Capability | 8.9 | 8.0 |
| Tensor Core | Yes | Yes |

## Measured Bandwidth

| Metric | RTX 4090 | Mars X201 |
|--------|----------|-----------|
| Memset BW | 939.58 GB/s (93.2% eff) | 510.75 GB/s (27.7% eff) |
| D2H BW | 2.07 GB/s | 16.57 GB/s |
| H2D BW | 9.71 GB/s | 25.57 GB/s |

## Critical Design Implications

### Warp Size Difference (32 vs 64)

This is the **most critical** difference for SpMV kernel design:

1. **NVIDIA (warp=32)**
   - Each warp has 32 threads
   - Standard CUDA kernel patterns apply
   - Common block sizes: 128, 256 threads

2. **Domestic GPU (warp=64)**
   - Each warp has 64 threads
   - Requires kernel redesign for 64-thread warps
   - **Double the register pressure per warp**
   - **Double the shared memory usage per warp**
   - Recommended block sizes: 128, 256 threads (2-4 warps)

### Register Strategy

| GPU | Registers/SM | Max Threads/SM | Regs/Thread (max) |
|-----|--------------|----------------|-------------------|
| RTX 4090 | 65536 | 1536 | 42 |
| Mars X201 | 131072 | 2048 | 64 |

- Domestic GPU has more registers but also more threads per SM
- SpMV kernels should aim for < 32 regs/thread for good occupancy

### Shared Memory Strategy

| GPU | Shared Mem/SM | Shared Mem/Block (max) |
|-----|---------------|------------------------|
| RTX 4090 | 100 KB | 99 KB |
| Mars X201 | 64 KB | 64 KB |

- Domestic GPU has less shared memory per SM
- Must be more conservative with shared memory usage
- Consider using shared memory only for row pointer caching

### SpMV Kernel Recommendations

1. **CSR Vector-based kernel**
   - NVIDIA: One warp per row, 32-thread reduction
   - Domestic: One warp per row, 64-thread reduction (or use 2 warps)

2. **Merge-based kernel**
   - Works well for both GPUs
   - Requires careful work distribution

3. **Block size selection**
   - NVIDIA: 128 threads (4 warps)
   - Domestic: 128-256 threads (2-4 warps)

## Compilation Notes

### RTX 4090 (NVIDIA)
```bash
nvcc -O3 -arch=sm_89 device_info.cu -o device_info
```

### Mars X201 (Domestic)
```bash
# Use cu-bridge for CUDA compatibility
~/cu-bridge/CUDA_DIR/bin/nvcc -O3 device_info.cu -o device_info
```

**Important**: Domestic GPU uses `cucc` (linked as `nvcc`) from cu-bridge for CUDA compilation.