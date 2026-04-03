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
#include <algorithm>

namespace muxi_spmv {

// ==================== Memory Access Pattern Analysis ====================

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
    float rowPtrCacheable = static_cast<float>(4 * 1024 * 1024) /
                             (numRows * sizeof(int));
    result.cacheMissRate = 1.0f - std::min(rowPtrCacheable, 1.0f);

    // Theoretical bandwidth utilization
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
    int rowsPerThread = (numRowsInPartition + WarpSize - 1) / WarpSize;
    int myRowStart = startRow + lane * rowsPerThread;
    int myRowEnd = min(myRowStart + rowsPerThread, endRow);

    // Handle partial first row
    if (lane == 0 && startNnz > rowPtr[startRow]) {
        FloatType partialSum = FloatType(0);
        int rowEndNnz = (startRow + 1 <= endRow) ? rowPtr[startRow + 1] : endNnz;

        #pragma unroll 4
        for (int idx = startNnz; idx < rowEndNnz; idx++) {
            int col = colIdx[idx];
            partialSum += values[idx] * x[col];
        }

        // Warp-level aggregation for partial row
        #if WarpSize == 64
        partialSum += __shfl_down_sync(0xffffffffffffffff, partialSum, 32);
        partialSum += __shfl_down_sync(0xffffffffffffffff, partialSum, 16);
        partialSum += __shfl_down_sync(0xffffffffffffffff, partialSum, 8);
        partialSum += __shfl_down_sync(0xffffffffffffffff, partialSum, 4);
        partialSum += __shfl_down_sync(0xffffffffffffffff, partialSum, 2);
        partialSum += __shfl_down_sync(0xffffffffffffffff, partialSum, 1);
        #else
        partialSum += __shfl_down_sync(0xffffffff, partialSum, 16);
        partialSum += __shfl_down_sync(0xffffffff, partialSum, 8);
        partialSum += __shfl_down_sync(0xffffffff, partialSum, 4);
        partialSum += __shfl_down_sync(0xffffffff, partialSum, 2);
        partialSum += __shfl_down_sync(0xffffffff, partialSum, 1);
        #endif

        if (lane == 0 && partialSum != FloatType(0)) {
            atomicAdd(y + startRow, partialSum);
        }
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

        #if WarpSize == 64
        partialSum += __shfl_down_sync(0xffffffffffffffff, partialSum, 32);
        partialSum += __shfl_down_sync(0xffffffffffffffff, partialSum, 16);
        partialSum += __shfl_down_sync(0xffffffffffffffff, partialSum, 8);
        partialSum += __shfl_down_sync(0xffffffffffffffff, partialSum, 4);
        partialSum += __shfl_down_sync(0xffffffffffffffff, partialSum, 2);
        partialSum += __shfl_down_sync(0xffffffffffffffff, partialSum, 1);
        #else
        partialSum += __shfl_down_sync(0xffffffff, partialSum, 16);
        partialSum += __shfl_down_sync(0xffffffff, partialSum, 8);
        partialSum += __shfl_down_sync(0xffffffff, partialSum, 4);
        partialSum += __shfl_down_sync(0xffffffff, partialSum, 2);
        partialSum += __shfl_down_sync(0xffffffff, partialSum, 1);
        #endif

        if (lane == 0 && partialSum != FloatType(0)) {
            atomicAdd(y + endRow, partialSum);
        }
        myRowEnd = endRow;
    }

    // Process complete rows
    for (int row = myRowStart; row < myRowEnd && row < numRows; row++) {
        FloatType sum = FloatType(0);
        int rowStartNnz = rowPtr[row];
        int rowEndNnz = rowPtr[row + 1];

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

        y[row] = sum;
    }
}

} // namespace muxi_spmv

#endif // MUXI_SPMV_ASSEMBLY_ANALYSIS_CUH_