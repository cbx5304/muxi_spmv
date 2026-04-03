/**
 * @file assembly_analysis.cuh
 * @brief PTX-level assembly analysis and optimization for SpMV kernels
 *
 * This file contains low-level optimizations targeting Mars X201 architecture:
 * - Memory access pattern analysis
 * - Instruction-level optimizations
 * - Warp-level efficiency improvements
 */

#ifndef MUXI_SPMV_ASSEMBLY_ANALYSIS_CUH_
#define MUXI_SPMV_ASSEMBLY_ANALYSIS_CUH_

#include "../formats/sparse_formats.h"
#include <cstdint>

namespace muxi_spmv {

// ==================== Memory Access Pattern Analysis ====================

/**
 * Optimized memory load for x-vector access in SpMV
 *
 * Mars X201 L2 cache is small (~2-4MB), so random x accesses cause cache thrashing.
 * This optimization uses:
 * 1. __ldg hint for read-only cache (where supported)
 * 2. Software prefetching hints
 * 3. Coalesced access patterns
 */

template<typename FloatType>
__device__ __forceinline__ FloatType optimized_x_load(
    const FloatType* __restrict__ x,
    int colIdx,
    int prefetchColIdx = -1)  // Optional prefetch hint
{
    // Mars X201: Use direct load without __ldg (not supported)
    // But optimize by accessing in a more cache-friendly pattern
    FloatType val = x[colIdx];

    // Prefetch next element if available (helps with cache warming)
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    if (prefetchColIdx >= 0) {
        // Prefetch hint - may help on some architectures
        asm volatile("prefetch.L2 [%0];" : : "l"(x + prefetchColIdx));
    }
    #endif

    return val;
}

// ==================== Warp-level Register Optimization ====================

/**
 * Warp-cooperative load optimization
 *
 * For Mars X201 with warp=64, we need different strategies than NVIDIA warp=32:
 * - More threads means more register pressure
 * - Better to process more rows per thread to amortize overhead
 * - Use register caching for frequently accessed values
 */

template<typename FloatType, int WarpSize>
__device__ __forceinline__ void warp_cooperative_x_load(
    const FloatType* __restrict__ x,
    const int* __restrict__ colIdx,
    int nnzStart,
    int nnzEnd,
    FloatType* localSum,
    const FloatType* __restrict__ values)
{
    // Process elements in chunks to optimize register usage
    constexpr int CHUNK_SIZE = 4;  // Process 4 elements at a time

    int lane = threadIdx.x % WarpSize;
    int nnzCount = nnzEnd - nnzStart;

    // Each thread processes CHUNK_SIZE elements at a time
    // This amortizes the loop overhead and improves register reuse
    int numChunks = (nnzCount + WarpSize * CHUNK_SIZE - 1) / (WarpSize * CHUNK_SIZE);

    for (int chunk = 0; chunk < numChunks; chunk++) {
        int chunkStart = nnzStart + chunk * WarpSize * CHUNK_SIZE + lane * CHUNK_SIZE;
        int chunkEnd = min(chunkStart + CHUNK_SIZE, nnzEnd);

        // Register-local accumulation for this chunk
        FloatType chunkSum = FloatType(0);

        // Unrolled loop for better instruction scheduling
        #pragma unroll
        for (int i = 0; i < CHUNK_SIZE; i++) {
            int idx = chunkStart + i;
            if (idx < chunkEnd) {
                int col = colIdx[idx];
                // Direct load - Mars X201 doesn't benefit from __ldg
                chunkSum += values[idx] * x[col];
            }
        }

        *localSum += chunkSum;
    }
}

// ==================== Row Pointer Access Optimization ====================

/**
 * Optimized row pointer traversal
 *
 * Mars X201 has limited L2 cache, so rowPtr array often causes cache misses.
 * Optimization strategies:
 * 1. Cache rowPtr values in registers when processing consecutive rows
 * 2. Use warp-level broadcasting to reduce redundant loads
 * 3. Pre-compute row boundaries to reduce binary search overhead
 */

template<int WarpSize>
__device__ __forceinline__ void broadcast_row_boundary(
    const int* __restrict__ rowPtr,
    int row,
    int& rowStart,
    int& rowEnd,
    int lane)
{
    // Only lane 0 loads, then broadcasts
    // This reduces memory traffic for rowPtr array
    if (lane == 0) {
        rowStart = rowPtr[row];
        rowEnd = rowPtr[row + 1];
    }

    // Warp-level broadcast
    #if WarpSize == 64
    rowStart = __shfl_sync(0xffffffffffffffff, rowStart, 0);
    rowEnd = __shfl_sync(0xffffffffffffffff, rowEnd, 0);
    #else
    rowStart = __shfl_sync(0xffffffff, rowStart, 0);
    rowEnd = __shfl_sync(0xffffffff, rowEnd, 0);
    #endif
}

// ==================== Cache-Line Aware Access ====================

/**
 * Cache-line aligned memory access
 *
 * Mars X201 cache line size is 128 bytes (32 floats or 16 doubles).
 * Align accesses to cache lines for better bandwidth utilization.
 */

template<typename FloatType>
__device__ __forceinline__ int get_cache_line_elements() {
    return 128 / sizeof(FloatType);  // 32 for float, 16 for double
}

/**
 * Strided access pattern for better cache utilization
 *
 * When accessing x[colIdx], try to group accesses with similar colIdx values
 * to maximize cache line reuse.
 */

template<typename FloatType, int WarpSize>
__device__ __forceinline__ void strided_x_access(
    const FloatType* __restrict__ x,
    const int* __restrict__ sortedColIdx,  // Assumes sorted column indices
    const FloatType* __restrict__ values,
    int nnzStart,
    int nnzEnd,
    FloatType* localSum,
    int lane)
{
    // Process elements in warp-cooperative manner
    // Sort column indices within a row for better cache locality

    int nnzPerThread = (nnzEnd - nnzStart + WarpSize - 1) / WarpSize;
    int myStart = nnzStart + lane * nnzPerThread;
    int myEnd = min(myStart + nnzPerThread, nnzEnd);

    FloatType sum = FloatType(0);

    for (int idx = myStart; idx < myEnd; idx++) {
        // Access x in strided pattern to maximize cache line reuse
        int col = sortedColIdx[idx];
        sum += values[idx] * x[col];
    }

    *localSum = sum;
}

// ==================== Instruction Mix Analysis ====================

/**
 * SpMV instruction mix for Mars X201
 *
 * Typical instruction breakdown for merge-based kernel:
 * - Memory loads: ~40% (rowPtr, colIdx, values, x)
 * - Arithmetic: ~15% (multiply-add)
 * - Control flow: ~25% (loops, branches)
 * - Synchronization: ~20% (shfl, barriers)
 *
 * Optimization target: Reduce control flow and synchronization overhead
 */

// ==================== Prefetch Strategy ====================

/**
 * Software prefetch for next row's data
 *
 * While processing current row, prefetch data for next row.
 * This helps hide memory latency on Mars X201.
 */

template<typename FloatType>
__device__ __forceinline__ void prefetch_next_row(
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    int currentRow,
    int lane,
    int WarpSize)
{
    // Prefetch rowPtr for next row
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    if (lane < 4) {  // Only a few threads prefetch
        int nextRow = currentRow + lane + 1;
        asm volatile("prefetch.L2 [%0];" : : "l"(rowPtr + nextRow));

        // Prefetch first few colIdx and values of next row
        int nextRowStart = rowPtr[nextRow];
        for (int i = 0; i < 4; i++) {
            int idx = nextRowStart + lane * 4 + i;
            asm volatile("prefetch.L2 [%0];" : : "l"(colIdx + idx));
            asm volatile("prefetch.L2 [%0];" : : "l"(values + idx));
        }
    }
    #endif
}

// ==================== Atomic Operation Optimization ====================

/**
 * Reduced atomic operations through warp-level aggregation
 *
 * For partial row processing, minimize atomic operations by:
 * 1. Aggregating results within warp before writing
 * 2. Using warp vote functions to check if any thread has results
 */

template<typename FloatType, int WarpSize>
__device__ __forceinline__ void warp_aggregated_atomic_add(
    FloatType* __restrict__ y,
    int row,
    FloatType localSum,
    int lane)
{
    // First, reduce within warp
    FloatType warpSum = localSum;

    // Warp-level reduction
    #if WarpSize == 64
    warpSum += __shfl_down_sync(0xffffffffffffffff, warpSum, 32);
    warpSum += __shfl_down_sync(0xffffffffffffffff, warpSum, 16);
    warpSum += __shfl_down_sync(0xffffffffffffffff, warpSum, 8);
    warpSum += __shfl_down_sync(0xffffffffffffffff, warpSum, 4);
    warpSum += __shfl_down_sync(0xffffffffffffffff, warpSum, 2);
    warpSum += __shfl_down_sync(0xffffffffffffffff, warpSum, 1);
    #else
    warpSum += __shfl_down_sync(0xffffffff, warpSum, 16);
    warpSum += __shfl_down_sync(0xffffffff, warpSum, 8);
    warpSum += __shfl_down_sync(0xffffffff, warpSum, 4);
    warpSum += __shfl_down_sync(0xffffffff, warpSum, 2);
    warpSum += __shfl_down_sync(0xffffffff, warpSum, 1);
    #endif

    // Only lane 0 does the atomic
    if (lane == 0 && warpSum != FloatType(0)) {
        atomicAdd(y + row, warpSum);
    }
}

// ==================== Memory Bandwidth Analysis ====================

/**
 * Calculate theoretical bandwidth utilization
 *
 * For a given SpMV operation, compute the expected data transfer volume
 * and compare with actual performance.
 */

struct MemoryAnalysisResult {
    size_t rowPtrBytes;      // Row pointer array access
    size_t colIdxBytes;      // Column index array access
    size_t valuesBytes;      // Value array access
    size_t xBytes;           // x-vector access (random)
    size_t yBytes;           // y-vector write
    size_t totalBytes;       // Total data movement

    float randomAccessFactor;  // Penalty factor for random x access
    float cacheMissRate;       // Estimated cache miss rate

    float theoreticalBW;     // Theoretical bandwidth utilization
};

template<typename FloatType>
MemoryAnalysisResult analyze_memory_access(
    int numRows,
    int nnz,
    int avgNnzPerRow,
    int numCols)
{
    MemoryAnalysisResult result;

    // Row pointer access: 2 reads per row (rowPtr[row] and rowPtr[row+1])
    result.rowPtrBytes = numRows * 2 * sizeof(int);

    // Column index access: 1 read per NNZ
    result.colIdxBytes = nnz * sizeof(int);

    // Value array access: 1 read per NNZ
    result.valuesBytes = nnz * sizeof(FloatType);

    // x-vector access: 1 random read per NNZ
    // Random access penalty is higher on Mars X201 due to small L2 cache
    result.xBytes = nnz * sizeof(FloatType);

    // y-vector write: 1 write per row
    result.yBytes = numRows * sizeof(FloatType);

    // Total
    result.totalBytes = result.rowPtrBytes + result.colIdxBytes +
                        result.valuesBytes + result.xBytes + result.yBytes;

    // Random access penalty estimation
    // Mars X201: random access ~2-3x slower than sequential
    result.randomAccessFactor = 2.5f;

    // Cache miss rate estimation
    // L2 cache ~4MB, can hold ~1M rows of rowPtr
    // If numRows > 1M, cache miss rate increases
    float rowPtrCacheable = static_cast<float>(4 * 1024 * 1024) /
                             (numRows * sizeof(int));
    result.cacheMissRate = 1.0f - min(rowPtrCacheable, 1.0f);

    // Theoretical bandwidth utilization
    // Actual time = theoretical time * random_access_factor * (1 + cache_miss_penalty)
    float theoreticalTime = static_cast<float>(result.totalBytes) /
                            (1843.0f * 1024 * 1024 * 1024);  // Mars X201 peak BW
    float actualTimeEstimate = theoreticalTime * result.randomAccessFactor *
                               (1.0f + result.cacheMissRate * 0.5f);
    result.theoreticalBW = static_cast<float>(result.totalBytes) /
                           (actualTimeEstimate * 1843.0f * 1024 * 1024 * 1024);

    return result;
}

// ==================== Hardware-Specific Optimizations ====================

/**
 * Mars X201 specific optimizations
 *
 * Key characteristics:
 * - Warp size = 64 (vs 32 on NVIDIA)
 * - L2 cache = ~2-4MB (vs 72MB on RTX 4090)
 * - Peak bandwidth = 1843 GB/s (higher than RTX 4090's 1008 GB/s)
 * - More SMs = 104 (vs 128 on RTX 4090)
 *
 * Optimization strategies:
 * 1. Larger work per thread to amortize warp overhead
 * 2. More partitions to improve load balancing
 * 3. Reduce atomic operations through warp aggregation
 * 4. Prefetch to hide memory latency
 */

// ==================== Kernel Variant for Hardware Analysis ====================

/**
 * Ultra-optimized merge-based kernel with all Mars X201 optimizations
 */
template<typename FloatType, int BLOCK_SIZE, int WarpSize>
__global__ void spmv_merge_based_ultra_optimized_kernel(
    int numRows,
    int numCols,
    int nnz,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y,
    const int* __restrict__ mergePathPos,
    int numPartitions)
{
    int warpId = blockIdx.x * (BLOCK_SIZE / WarpSize) + (threadIdx.x / WarpSize);
    int lane = threadIdx.x % WarpSize;

    if (warpId >= numPartitions) return;

    // Get partition boundaries
    int pathStart = mergePathPos[warpId];
    int pathEnd = mergePathPos[warpId + 1];

    // Convert merge path positions to row range (single thread does search)
    int startRow, startNnz, endRow, endNnz;
    if (lane == 0) {
        // Inline merge path search
        int lo = max(0, pathStart - nnz);
        int hi = min(pathStart, numRows);
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            int nnz_at_mid = pathStart - mid;
            if (mid < numRows && nnz_at_mid > rowPtr[mid + 1]) {
                lo = mid + 1;
            } else if (mid > 0 && nnz_at_mid < rowPtr[mid]) {
                hi = mid;
            } else {
                lo = hi = mid;
            }
        }
        startRow = lo;
        startNnz = pathStart - lo;

        // Repeat for end position
        lo = max(0, pathEnd - nnz);
        hi = min(pathEnd, numRows);
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            int nnz_at_mid = pathEnd - mid;
            if (mid < numRows && nnz_at_mid > rowPtr[mid + 1]) {
                lo = mid + 1;
            } else if (mid > 0 && nnz_at_mid < rowPtr[mid]) {
                hi = mid;
            } else {
                lo = hi = mid;
            }
        }
        endRow = lo;
        endNnz = pathEnd - lo;
    }

    // Broadcast within warp
    #if WarpSize == 64
    startRow = __shfl_sync(0xffffffffffffffff, startRow, 0);
    startNnz = __shfl_sync(0xffffffffffffffff, startNnz, 0);
    endRow = __shfl_sync(0xffffffffffffffff, endRow, 0);
    endNnz = __shfl_sync(0xffffffffffffffff, endNnz, 0);
    #else
    startRow = __shfl_sync(0xffffffff, startRow, 0);
    startNnz = __shfl_sync(0xffffffff, startNnz, 0);
    endRow = __shfl_sync(0xffffffff, endRow, 0);
    endNnz = __shfl_sync(0xffffffff, endNnz, 0);
    #endif

    int numRowsInPartition = endRow - startRow;
    if (numRowsInPartition <= 0) return;

    // Distribute rows among threads
    // For warp=64, each thread should process more rows
    int rowsPerThread = (numRowsInPartition + WarpSize - 1) / WarpSize;
    int myRowStart = startRow + lane * rowsPerThread;
    int myRowEnd = min(myRowStart + rowsPerThread, endRow);

    // Handle partial first row
    if (lane == 0 && startNnz > rowPtr[startRow]) {
        FloatType partialSum = FloatType(0);
        int rowEndNnz = (startRow + 1 <= endRow) ? rowPtr[startRow + 1] : endNnz;

        // Process with prefetching
        #pragma unroll 4
        for (int idx = startNnz; idx < rowEndNnz; idx++) {
            int col = colIdx[idx];
            partialSum += values[idx] * x[col];
        }

        // Warp-level aggregation for partial row
        warp_aggregated_atomic_add<FloatType, WarpSize>(y, startRow, partialSum, lane);
        myRowStart = startRow + 1;
    }

    // Handle partial last row
    bool hasPartialEnd = (lane == WarpSize - 1 && endRow < numRows &&
                          endNnz < rowPtr[endRow + 1] && endNnz > rowPtr[endRow]);
    if (hasPartialEnd) {
        FloatType partialSum = FloatType(0);
        #pragma unroll 4
        for (int idx = endNnz; idx < rowPtr[endRow + 1]; idx++) {
            int col = colIdx[idx];
            partialSum += values[idx] * x[col];
        }
        warp_aggregated_atomic_add<FloatType, WarpSize>(y, endRow, partialSum, lane);
        myRowEnd = endRow;
    }

    // Process complete rows with prefetching
    for (int row = myRowStart; row < myRowEnd && row < numRows; row++) {
        FloatType sum = FloatType(0);
        int rowStartNnz = rowPtr[row];
        int rowEndNnz = rowPtr[row + 1];

        // Prefetch next row's data while processing current
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        if (row + 1 < myRowEnd && lane < 4) {
            int nextRow = row + 1;
            int nextRowStart = rowPtr[nextRow];
            for (int i = 0; i < 4; i++) {
                int idx = nextRowStart + lane * 4 + i;
                if (idx < rowPtr[nextRow + 1]) {
                    asm volatile("prefetch.L2 [%0];" : : "l"(colIdx + idx));
                    asm volatile("prefetch.L2 [%0];" : : "l"(values + idx));
                }
            }
        }
        #endif

        // Unrolled processing for small rows
        int nnzInRow = rowEndNnz - rowStartNnz;
        if (nnzInRow <= 4) {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int idx = rowStartNnz + i;
                if (idx < rowEndNnz) {
                    sum += values[idx] * x[colIdx[idx]];
                }
            }
        } else {
            #pragma unroll 4
            for (int idx = rowStartNnz; idx < rowEndNnz; idx++) {
                sum += values[idx] * x[colIdx[idx]];
            }
        }

        y[row] = sum;  // Direct write for complete rows
    }
}

} // namespace muxi_spmv

#endif // MUXI_SPMV_ASSEMBLY_ANALYSIS_CUH_