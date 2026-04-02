/**
 * @file spmv_csr.cuh
 * @brief CSR format SpMV kernel declarations
 *
 * Supports multiple floating-point types and warp sizes (32/64)
 */

#ifndef SPMV_CSR_CUH_
#define SPMV_CSR_CUH_

#include "utils/common.h"
#include "formats/sparse_formats.h"

namespace muxi_spmv {

// Forward declarations - device functions must be declared before use
template<typename FloatType, int WarpSize>
__device__ __forceinline__ FloatType warpReduceSum(FloatType val);

__device__ __forceinline__ int binarySearchRow(
    const int* __restrict__ rowPtr, int numRows, int nnzIdx);

/**
 * @brief CSR SpMV kernel configurations
 */
struct CSRKernelConfig {
    int blockSize;
    int rowsPerBlock;
    int itemsPerThread;
    bool useSharedMem;
    int sharedMemSize;
};

CSRKernelConfig getOptimalConfig(int numRows, int nnz, int avgNnzPerRow, int warpSize);

// ==================== Device Functions ====================

template<typename FloatType, int WarpSize>
__device__ __forceinline__ FloatType warpReduceSum(FloatType val) {
    if (WarpSize >= 64) {
        val += __shfl_down_sync(0xffffffff, val, 32);
    }
    if (WarpSize >= 32) {
        val += __shfl_down_sync(0xffffffff, val, 16);
    }
    if (WarpSize >= 16) {
        val += __shfl_down_sync(0xffffffff, val, 8);
    }
    if (WarpSize >= 8) {
        val += __shfl_down_sync(0xffffffff, val, 4);
    }
    if (WarpSize >= 4) {
        val += __shfl_down_sync(0xffffffff, val, 2);
    }
    if (WarpSize >= 2) {
        val += __shfl_down_sync(0xffffffff, val, 1);
    }
    return val;
}

__device__ __forceinline__ int binarySearchRow(
    const int* __restrict__ rowPtr, int numRows, int nnzIdx)
{
    int lo = 0;
    int hi = numRows;

    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (rowPtr[mid] <= nnzIdx) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo - 1;
}

// ==================== Kernel Functions ====================

/**
 * @brief CSR SpMV kernel - Scalar approach (one thread per row)
 */
template<typename FloatType, bool TRANSPOSE = false>
__global__ void spmv_csr_scalar_kernel(
    int numRows,
    int numCols,
    int nnz,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numRows) {
        FloatType sum = static_cast<FloatType>(0);

        // Use __ldg for read-only cache hints on arrays that benefit from caching
        int rowStart = __ldg(&rowPtr[row]);
        int rowEnd = __ldg(&rowPtr[row + 1]);

        for (int i = rowStart; i < rowEnd; i++) {
            int col = __ldg(&colIdx[i]);
            if (TRANSPOSE) {
                atomicAdd(&y[col], __ldg(&values[i]) * __ldg(&x[row]));
            } else {
                sum += __ldg(&values[i]) * __ldg(&x[col]);
            }
        }

        if (!TRANSPOSE) {
            y[row] = sum;
        }
    }
}

/**
 * @brief CSR SpMV kernel - Vector approach (one warp per row)
 */
template<typename FloatType, int WarpSize, bool TRANSPOSE = false>
__global__ void spmv_csr_vector_kernel(
    int numRows,
    int numCols,
    int nnz,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y)
{
    int row = blockIdx.x * (blockDim.x / WarpSize) + (threadIdx.x / WarpSize);
    int lane = threadIdx.x % WarpSize;

    if (row < numRows) {
        int rowStart = rowPtr[row];
        int rowEnd = rowPtr[row + 1];

        FloatType sum = static_cast<FloatType>(0);

        for (int i = rowStart + lane; i < rowEnd; i += WarpSize) {
            int col = colIdx[i];
            sum += values[i] * x[col];
        }

        sum = warpReduceSum<FloatType, WarpSize>(sum);

        if (lane == 0) {
            y[row] = sum;
        }
    }
}

/**
 * @brief CSR SpMV kernel - Parallel multi-row vector approach
 * Divides warp into lane groups, each group processes one row in parallel
 * This provides better parallelism than sequential multi-row processing
 */
template<typename FloatType, int WarpSize, int RowsPerWarp>
__global__ void spmv_csr_parallel_multirow_kernel(
    int numRows,
    int numCols,
    int nnz,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y)
{
    // Each warp processes RowsPerWarp rows in PARALLEL
    // Warp is divided into RowsPerWarp groups of (WarpSize / RowsPerWarp) threads
    // Each group processes one row in parallel using vector approach

    int warpId = blockIdx.x * (blockDim.x / WarpSize) + (threadIdx.x / WarpSize);
    int lane = threadIdx.x % WarpSize;

    // Determine which row within the warp this thread handles
    int groupSize = WarpSize / RowsPerWarp;  // threads per row
    int rowInWarp = lane / groupSize;         // which row (0 to RowsPerWarp-1)
    int laneInGroup = lane % groupSize;       // position within group

    int baseRow = warpId * RowsPerWarp;
    int row = baseRow + rowInWarp;

    if (row < numRows) {
        int rowStart = rowPtr[row];
        int rowEnd = rowPtr[row + 1];

        FloatType sum = static_cast<FloatType>(0);

        // Each thread in the group processes elements at stride = groupSize
        for (int i = rowStart + laneInGroup; i < rowEnd; i += groupSize) {
            int col = colIdx[i];
            sum += values[i] * x[col];
        }

        // Reduce within the group (not full warp)
        // Use shuffle operations with mask for group
        for (int offset = groupSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        // Thread 0 in each group writes the result
        if (laneInGroup == 0) {
            y[row] = sum;
        }
    }
}

/**
 * @brief CSR SpMV kernel - Multi-row vector approach
 * Each warp processes multiple rows for better utilization when avgNnzPerRow is small
 */
template<typename FloatType, int WarpSize, int RowsPerWarp>
__global__ void spmv_csr_multirow_vector_kernel(
    int numRows,
    int numCols,
    int nnz,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y)
{
    // Each warp processes RowsPerWarp rows
    int warpId = blockIdx.x * (blockDim.x / WarpSize) + (threadIdx.x / WarpSize);
    int lane = threadIdx.x % WarpSize;

    int baseRow = warpId * RowsPerWarp;

    // Process RowsPerWarp rows per warp
    for (int r = 0; r < RowsPerWarp; r++) {
        int row = baseRow + r;
        if (row >= numRows) break;

        int rowStart = rowPtr[row];
        int rowEnd = rowPtr[row + 1];

        FloatType sum = static_cast<FloatType>(0);

        // Distribute elements of this row across threads in warp
        // Each thread processes elements at offsets: lane, lane+WarpSize, lane+2*WarpSize...
        for (int i = rowStart + lane; i < rowEnd; i += WarpSize) {
            int col = colIdx[i];
            sum += values[i] * x[col];
        }

        // Use warp reduce to sum contributions
        sum = warpReduceSum<FloatType, WarpSize>(sum);

        // Thread 0 writes the result for this row
        if (lane == 0) {
            y[row] = sum;
        }
    }
}

/**
 * @brief CSR SpMV kernel - Vector approach with row balancing
 * Dynamically assigns rows to warps based on nnz count
 */
template<typename FloatType, int WarpSize>
__global__ void spmv_csr_balanced_vector_kernel(
    int numRows,
    int numCols,
    int nnz,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y)
{
    // Work distribution: process NNZ elements rather than rows
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = tid / WarpSize;
    int lane = tid % WarpSize;

    // Calculate work per warp based on total NNZ
    int totalWarps = gridDim.x * (blockDim.x / WarpSize);
    int nnzPerWarp = (nnz + totalWarps - 1) / totalWarps;

    int startNnz = warpId * nnzPerWarp;
    int endNnz = min(startNnz + nnzPerWarp, nnz);

    FloatType sum = static_cast<FloatType>(0);
    int currentRow = -1;

    // Process elements assigned to this warp
    for (int idx = startNnz + lane; idx < endNnz; idx += WarpSize) {
        // Find which row this element belongs to
        int row = binarySearchRow(rowPtr, numRows, idx);

        if (row != currentRow && lane == 0) {
            // Write previous row's sum if we had one
            if (currentRow >= 0) {
                y[currentRow] = sum;
            }
            currentRow = row;
            sum = static_cast<FloatType>(0);
        }

        // Accumulate contribution
        int col = colIdx[idx];
        sum += values[idx] * x[col];
    }

    // Write final row's sum
    if (lane == 0 && currentRow >= 0) {
        y[currentRow] = sum;
    }
}

/**
 * @brief Merge-based CSR SpMV kernel (original)
 */
template<typename FloatType, int WarpSize>
__global__ void spmv_csr_merge_kernel(
    int numRows,
    int numCols,
    int nnz,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int idx = tid; idx < nnz; idx += stride) {
        int row = binarySearchRow(rowPtr, numRows, idx);
        FloatType val = values[idx];
        int col = colIdx[idx];
        atomicAdd(&y[row], val * x[col]);
    }
}

/**
 * @brief CSR-Tile kernel for warp=64 optimization
 * Divides matrix into tiles of NNZ elements for better load balancing
 * Each thread block processes one tile cooperatively
 */
template<typename FloatType, int BLOCK_SIZE, int TILE_SIZE>
__global__ void spmv_csr_tile_kernel(
    int numRows,
    int numCols,
    int nnz,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y)
{
    // Shared memory for tile processing
    __shared__ FloatType s_partial[BLOCK_SIZE];
    __shared__ int s_rowIndices[BLOCK_SIZE];
    __shared__ int s_rowStarts[BLOCK_SIZE / 64 + 1];  // For warp-level reduction

    int tid = threadIdx.x;
    int blockId = blockIdx.x;
    int tileSize = TILE_SIZE;

    // Calculate this block's tile range
    int tileStart = blockId * tileSize;
    int tileEnd = min(tileStart + tileSize, nnz);

    // Each thread processes elements from the tile
    FloatType mySum = static_cast<FloatType>(0);
    int myRow = -1;

    // Process elements assigned to this thread
    // Use a strided access pattern for coalescing
    for (int idx = tileStart + tid; idx < tileEnd; idx += BLOCK_SIZE) {
        int row = binarySearchRow(rowPtr, numRows, idx);
        int col = colIdx[idx];
        FloatType val = values[idx] * x[col];

        if (myRow == -1) {
            myRow = row;
            mySum = val;
        } else if (myRow == row) {
            // Same row - accumulate
            mySum += val;
        } else {
            // Different row - flush previous result
            atomicAdd(&y[myRow], mySum);
            myRow = row;
            mySum = val;
        }
    }

    // Store thread's partial result
    s_partial[tid] = mySum;
    s_rowIndices[tid] = myRow;

    __syncthreads();

    // Warp-level reduction by row
    // Each warp reduces results for rows that threads in that warp handled
    int warpId = tid / 64;
    int lane = tid % 64;

    // Simple approach: each thread atomics its result
    // This is necessary because different threads may have different rows
    if (myRow >= 0) {
        atomicAdd(&y[myRow], mySum);
    }
}

/**
 * @brief CSR-Stream kernel optimized for sparse matrices on warp=64
 * Distributes NNZ elements evenly across all threads
 * Each thread processes nnz/totalThreads elements
 */
template<typename FloatType, int BLOCK_SIZE>
__global__ void spmv_csr_stream_kernel(
    int numRows,
    int numCols,
    int nnz,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y)
{
    // Each thread processes a contiguous chunk of NNZ elements
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;

    // Calculate chunk size and range for this thread
    int chunkSize = (nnz + totalThreads - 1) / totalThreads;
    int startIdx = min(tid * chunkSize, nnz);
    int endIdx = min(startIdx + chunkSize, nnz);

    // Process elements in this thread's chunk
    FloatType sum = static_cast<FloatType>(0);
    int currentRow = -1;

    for (int idx = startIdx; idx < endIdx; idx++) {
        // Find which row this element belongs to using binary search
        int row = binarySearchRow(rowPtr, numRows, idx);
        int col = colIdx[idx];
        FloatType val = values[idx] * x[col];

        // Check if we're still on the same row
        if (row != currentRow) {
            // Flush previous row's sum
            if (currentRow >= 0) {
                atomicAdd(&y[currentRow], sum);
            }
            currentRow = row;
            sum = val;
        } else {
            sum += val;
        }
    }

    // Flush final row's sum
    if (currentRow >= 0) {
        atomicAdd(&y[currentRow], sum);
    }
}

/**
 * @brief CSR5-style kernel for optimal load balancing on sparse matrices
 * Divides matrix into tiles of fixed NNZ count, each tile processed by one warp
 * Eliminates thread idle time for sparse matrices
 *
 * Key insight: Instead of "1 warp per row", use "1 warp per tile of NNZ"
 */
template<typename FloatType, int BLOCK_SIZE, int TILE_SIZE, int WARP_SZ>
__global__ void spmv_csr5_kernel(
    int numRows,
    int numCols,
    int nnz,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y,
    const int* __restrict__ tileRowPtr,    // Precomputed: which row each tile starts at
    const int* __restrict__ tileNnzOffset) // Precomputed: NNZ offset within start row
{
    int warpId = blockIdx.x * (BLOCK_SIZE / WARP_SZ) + (threadIdx.x / WARP_SZ);
    int lane = threadIdx.x % WARP_SZ;

    // Calculate total number of tiles
    int numTiles = (nnz + TILE_SIZE - 1) / TILE_SIZE;

    if (warpId >= numTiles) return;

    // Get tile boundaries
    int tileStart = warpId * TILE_SIZE;
    int tileEnd = min(tileStart + TILE_SIZE, nnz);
    int tileNnz = tileEnd - tileStart;

    // Get starting row for this tile (from precomputed data)
    int startRow = tileRowPtr[warpId];

    // Each thread processes TILE_SIZE / WARP_SZ elements from this tile
    // Since TILE_SIZE = WARP_SZ * elements_per_thread
    int elementsPerThread = (TILE_SIZE + WARP_SZ - 1) / WARP_SZ;
    int myStart = tileStart + lane * elementsPerThread;
    int myEnd = min(myStart + elementsPerThread, tileEnd);

    // Find which row our first element belongs to
    int currentRow = -1;
    FloatType localSum = static_cast<FloatType>(0);

    // Use binary search to find initial row
    if (myStart < tileEnd) {
        currentRow = binarySearchRow(rowPtr, numRows, myStart);
    }

    // Process elements assigned to this thread
    for (int idx = myStart; idx < myEnd; idx++) {
        int row = binarySearchRow(rowPtr, numRows, idx);
        int col = colIdx[idx];

        if (row != currentRow) {
            // Contribute previous row's sum
            if (currentRow >= 0) {
                // Use warp-level reduction for same-row contributions
                // Then atomic add to global
                atomicAdd(&y[currentRow], localSum);
            }
            currentRow = row;
            localSum = values[idx] * x[col];
        } else {
            localSum += values[idx] * x[col];
        }
    }

    // Contribute final sum
    if (currentRow >= 0) {
        atomicAdd(&y[currentRow], localSum);
    }
}

/**
 * @brief Batched kernel with read-only cache hint for x vector
 * Uses __ldg() to hint the compiler to use read-only cache for x
 * This can improve performance on some architectures
 */
template<typename FloatType, int BLOCK_SIZE, int ROWS_PER_THREAD, int BATCH_SIZE>
__global__ void spmv_csr_batched_rocache_kernel(
    int numRows,
    int numCols,
    int nnz,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;

    // Calculate how many batches we need
    int numBatches = (numRows + BATCH_SIZE - 1) / BATCH_SIZE;

    // Process batches sequentially
    for (int batch = 0; batch < numBatches; batch++) {
        int batchStartRow = batch * BATCH_SIZE;
        int batchEndRow = min(batchStartRow + BATCH_SIZE, numRows);

        // Each thread processes ROWS_PER_THREAD rows within this batch
        int rowsInBatch = batchEndRow - batchStartRow;
        int threadsForBatch = (rowsInBatch + ROWS_PER_THREAD - 1) / ROWS_PER_THREAD;

        // Process rows assigned to this thread in this batch
        int localTid = tid;
        while (localTid < threadsForBatch) {
            int row = batchStartRow + localTid * ROWS_PER_THREAD;

            for (int r = 0; r < ROWS_PER_THREAD; r++) {
                int currentRow = row + r;
                if (currentRow >= batchEndRow) break;

                int rowStart = rowPtr[currentRow];
                int rowEnd = rowPtr[currentRow + 1];

                FloatType sum = static_cast<FloatType>(0);
                for (int i = rowStart; i < rowEnd; i++) {
                    // Use read-only cache hint for x access
                    sum += values[i] * __ldg(&x[colIdx[i]]);
                }
                y[currentRow] = sum;
            }

            localTid += totalThreads;
        }
    }
}

/**
 * @brief Batched kernel with shared memory cache for x vector
 * Optimized for sparse matrices where x fits in shared memory
 * Shared memory size: X_SHARED_SIZE elements (must be <= numCols)
 */
template<typename FloatType, int BLOCK_SIZE, int ROWS_PER_THREAD, int BATCH_SIZE, int X_SHARED_SIZE>
__global__ void spmv_csr_batched_shared_kernel(
    int numRows,
    int numCols,
    int nnz,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y)
{
    // Shared memory for x vector cache
    __shared__ FloatType s_x[X_SHARED_SIZE];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;

    // Cooperatively load x into shared memory
    // Each thread loads multiple elements if needed
    for (int i = threadIdx.x; i < min(numCols, X_SHARED_SIZE); i += blockDim.x) {
        s_x[i] = x[i];
    }
    __syncthreads();

    // If numCols > X_SHARED_SIZE, we need to handle out-of-range accesses
    // For now, assume numCols <= X_SHARED_SIZE

    // Calculate how many batches we need
    int numBatches = (numRows + BATCH_SIZE - 1) / BATCH_SIZE;

    // Process batches sequentially
    for (int batch = 0; batch < numBatches; batch++) {
        int batchStartRow = batch * BATCH_SIZE;
        int batchEndRow = min(batchStartRow + BATCH_SIZE, numRows);

        // Each thread processes ROWS_PER_THREAD rows within this batch
        int rowsInBatch = batchEndRow - batchStartRow;
        int threadsForBatch = (rowsInBatch + ROWS_PER_THREAD - 1) / ROWS_PER_THREAD;

        // Process rows assigned to this thread in this batch
        int localTid = tid;
        while (localTid < threadsForBatch) {
            int row = batchStartRow + localTid * ROWS_PER_THREAD;

            for (int r = 0; r < ROWS_PER_THREAD; r++) {
                int currentRow = row + r;
                if (currentRow >= batchEndRow) break;

                int rowStart = rowPtr[currentRow];
                int rowEnd = rowPtr[currentRow + 1];

                FloatType sum = static_cast<FloatType>(0);
                // Use shared memory for x access
                for (int i = rowStart; i < rowEnd; i++) {
                    int col = colIdx[i];
                    // Access x from shared memory if in range
                    if (col < X_SHARED_SIZE) {
                        sum += values[i] * s_x[col];
                    } else {
                        // Fallback to global memory for out-of-range columns
                        sum += values[i] * x[col];
                    }
                }
                y[currentRow] = sum;
            }

            localTid += totalThreads;
        }
    }
}

/**
 * @brief NNZ-chunked kernel for sparse matrices on warp=64
 * Each thread block processes a fixed chunk of NNZ elements
 * Threads cooperatively find row boundaries using binary search
 * Uses atomics for row sums, optimized for scattered row access
 */
template<typename FloatType, int BLOCK_SIZE>
__global__ void spmv_csr_nnz_chunked_kernel(
    int numRows,
    int numCols,
    int nnz,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;

    // Each thread processes a chunk of NNZ elements
    int chunkSize = (nnz + totalThreads - 1) / totalThreads;
    int startNnz = min(tid * chunkSize, nnz);
    int endNnz = min(startNnz + chunkSize, nnz);

    if (startNnz >= nnz) return;

    // Binary search to find starting row
    int low = 0, high = numRows;
    int currentRow = 0;
    while (low < high) {
        int mid = (low + high) / 2;
        if (rowPtr[mid] <= startNnz) {
            currentRow = mid;
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    int currentRowEnd = rowPtr[currentRow + 1];
    FloatType localSum = static_cast<FloatType>(0);

    // Process NNZ elements
    for (int idx = startNnz; idx < endNnz; idx++) {
        // Check if we need to advance to next row
        while (idx >= currentRowEnd && currentRow < numRows - 1) {
            // Write previous row's sum
            if (localSum != static_cast<FloatType>(0)) {
                atomicAdd(&y[currentRow], localSum);
                localSum = static_cast<FloatType>(0);
            }
            currentRow++;
            currentRowEnd = rowPtr[currentRow + 1];
        }

        // Process this element
        localSum += values[idx] * x[colIdx[idx]];
    }

    // Write final sum
    if (localSum != static_cast<FloatType>(0)) {
        atomicAdd(&y[currentRow], localSum);
    }
}

/**
 * @brief NNZ-based batched kernel for sparse matrices on warp=64
 * Distributes work by NNZ chunks instead of row batches
 * Each thread block processes a fixed chunk of NNZ elements
 * This approach has O(NNZ) complexity, not O(numRows)
 */
template<typename FloatType, int BLOCK_SIZE, int NNZ_PER_BLOCK>
__global__ void spmv_csr_nnz_batched_kernel(
    int numRows,
    int numCols,
    int nnz,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y)
{
    int blockId = blockIdx.x;
    int totalBlocks = gridDim.x;

    // Calculate this block's NNZ range
    int nnzPerBlock = (nnz + totalBlocks - 1) / totalBlocks;
    int startNnz = blockId * nnzPerBlock;
    int endNnz = min(startNnz + nnzPerBlock, nnz);

    if (startNnz >= nnz) return;

    // Find the starting row for this NNZ range using binary search
    int startRow = 0;
    int low = 0, high = numRows;
    while (low < high) {
        int mid = (low + high) / 2;
        if (rowPtr[mid] <= startNnz) {
            startRow = mid;
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    // Process NNZ elements in this block's range
    // Each thread processes a subset of the NNZ range
    int tid = threadIdx.x;
    int currentRow = startRow;
    FloatType localSum = static_cast<FloatType>(0);
    int currentRowStart = rowPtr[currentRow];
    int currentRowEnd = rowPtr[currentRow + 1];

    for (int idx = startNnz + tid; idx < endNnz; idx += BLOCK_SIZE) {
        // Find which row this NNZ belongs to
        while (idx >= currentRowEnd && currentRow < numRows - 1) {
            currentRow++;
            currentRowStart = rowPtr[currentRow];
            currentRowEnd = rowPtr[currentRow + 1];
        }

        // Process this element
        int col = colIdx[idx];
        FloatType val = values[idx];

        // Check if we're starting a new row
        int nextIdx = idx + BLOCK_SIZE;
        if (nextIdx < endNnz && nextIdx >= currentRowEnd) {
            // Write current row's sum and reset
            atomicAdd(&y[currentRow], localSum);
            localSum = static_cast<FloatType>(0);
        }

        localSum += val * x[col];
    }

    // Write final sum
    if (localSum != static_cast<FloatType>(0)) {
        atomicAdd(&y[currentRow], localSum);
    }
}

/**
 * @brief Batched kernel for sparse matrices on warp=64
 * Processes rows in batches to keep rowPtr data in L2 cache
 * Each batch processes BATCH_SIZE rows before moving to next batch
 * This kernel is optimized for matrices where rowPtr exceeds L2 cache
 */
template<typename FloatType, int BLOCK_SIZE, int ROWS_PER_THREAD, int BATCH_SIZE>
__global__ void spmv_csr_batched_kernel(
    int numRows,
    int numCols,
    int nnz,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;

    // Calculate how many batches we need
    int numBatches = (numRows + BATCH_SIZE - 1) / BATCH_SIZE;

    // Process batches sequentially
    for (int batch = 0; batch < numBatches; batch++) {
        int batchStartRow = batch * BATCH_SIZE;
        int batchEndRow = min(batchStartRow + BATCH_SIZE, numRows);

        // Each thread processes ROWS_PER_THREAD rows within this batch
        int rowsInBatch = batchEndRow - batchStartRow;
        int threadsForBatch = (rowsInBatch + ROWS_PER_THREAD - 1) / ROWS_PER_THREAD;

        // Process rows assigned to this thread in this batch
        int localTid = tid;
        while (localTid < threadsForBatch) {
            int row = batchStartRow + localTid * ROWS_PER_THREAD;

            for (int r = 0; r < ROWS_PER_THREAD; r++) {
                int currentRow = row + r;
                if (currentRow >= batchEndRow) break;

                int rowStart = rowPtr[currentRow];
                int rowEnd = rowPtr[currentRow + 1];

                FloatType sum = static_cast<FloatType>(0);
                for (int i = rowStart; i < rowEnd; i++) {
                    sum += values[i] * x[colIdx[i]];
                }
                y[currentRow] = sum;
            }

            localTid += totalThreads;
        }
    }
}

/**
 * @brief Warp-cooperative kernel for sparse matrices on warp=64
 * Each warp processes multiple rows cooperatively to reduce memory overhead
 * Threads in a warp share rowPtr reads to improve cache efficiency
 */
template<typename FloatType, int BLOCK_SIZE, int WARP_SZ, int ROWS_PER_WARP>
__global__ void spmv_csr_warp_cooperative_kernel(
    int numRows,
    int numCols,
    int nnz,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y)
{
    int warpId = blockIdx.x * (BLOCK_SIZE / WARP_SZ) + (threadIdx.x / WARP_SZ);
    int lane = threadIdx.x % WARP_SZ;
    int totalWarps = gridDim.x * (BLOCK_SIZE / WARP_SZ);

    // Each warp processes ROWS_PER_WARP rows
    int startRow = warpId * ROWS_PER_WARP;

    for (int r = 0; r < ROWS_PER_WARP; r++) {
        int row = startRow + r;
        if (row >= numRows) return;

        // Warp leader reads rowPtr and broadcasts to all lanes
        int rowStart, rowEnd, rowNnz;
        if (lane == 0) {
            rowStart = rowPtr[row];
            rowEnd = rowPtr[row + 1];
            rowNnz = rowEnd - rowStart;
        }
        // Broadcast row info to all threads in warp
        rowStart = __shfl_sync(0xffffffff, rowStart, 0);
        rowEnd = __shfl_sync(0xffffffff, rowEnd, 0);
        rowNnz = __shfl_sync(0xffffffff, rowNnz, 0);

        // Distribute NNZ elements across warp threads
        // Each thread processes a chunk of the row's elements
        FloatType sum = static_cast<FloatType>(0);
        for (int i = rowStart + lane; i < rowEnd; i += WARP_SZ) {
            sum += values[i] * x[colIdx[i]];
        }

        // Warp-level reduction to get final sum
        sum = warpReduceSum<FloatType, WARP_SZ>(sum);

        // Lane 0 writes the result
        if (lane == 0) {
            y[row] = sum;
        }
    }
}

/**
 * @brief Light-weight NNZ-balanced kernel for very sparse matrices
 * Each thread processes a fixed number of rows (optimal for avgNnzPerRow < 8)
 * This is the "scalar" approach but with multiple rows per thread
 */
template<typename FloatType, int BLOCK_SIZE, int ROWS_PER_THREAD>
__global__ void spmv_csr_light_balanced_kernel(
    int numRows,
    int numCols,
    int nnz,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;

    // Each thread processes ROWS_PER_THREAD rows
    // Total rows processed = totalThreads * ROWS_PER_THREAD
    int startRow = tid * ROWS_PER_THREAD;

    for (int r = 0; r < ROWS_PER_THREAD; r++) {
        int row = startRow + r;
        if (row >= numRows) return;

        int rowStart = rowPtr[row];
        int rowEnd = rowPtr[row + 1];

        FloatType sum = static_cast<FloatType>(0);
        for (int i = rowStart; i < rowEnd; i++) {
            sum += values[i] * x[colIdx[i]];
        }
        y[row] = sum;
    }
}

/**
 * @brief Adaptive kernel selector for warp=64
 * Chooses optimal strategy based on matrix characteristics
 */
enum class SpMVStrategy {
    VECTOR,           // 1 warp/row - good for avgNnz >= 16
    LIGHT_BALANCED,   // Multiple rows/thread - good for avgNnz < 16
    STREAM,           // NNZ-based distribution - fallback
    CSR5              // Tile-based - requires preprocessing
};

inline SpMVStrategy selectStrategy(int avgNnzPerRow, int numRows, int nnz) {
    // For Mars X201 (warp=64), use light-balanced for avgNnz < 32
    // Reason: Vector kernel utilization = avgNnzPerRow / 64
    // For avgNnz=10: 10/64 = 15.6% max utilization
    // Light-balanced can achieve much better thread utilization

    // For very sparse matrices
    if (avgNnzPerRow < 8) {
        return SpMVStrategy::LIGHT_BALANCED;
    }
    // For moderately sparse (Mars X201 needs light-balanced for < 32)
    if (avgNnzPerRow < 32) {
        return SpMVStrategy::LIGHT_BALANCED;
    }
    // For denser matrices (avgNnz >= 32), vector kernel is optimal
    // Vector kernel utilization now at least 32/64 = 50%
    return SpMVStrategy::VECTOR;
}

/**
 * @brief CSR-Adaptive kernel optimized for warp=64
 * Dynamically selects between scalar and vector processing based on row length
 * Uses work distribution across thread blocks for better load balancing
 */
template<typename FloatType, int BLOCK_SIZE, int WARP_SZ>
__global__ void spmv_csr_adaptive_kernel(
    int numRows,
    int numCols,
    int nnz,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y,
    const int* __restrict__ rowPartitions,  // Partition boundaries
    int numPartitions)
{
    __shared__ FloatType s_results[BLOCK_SIZE];
    __shared__ int s_rows[BLOCK_SIZE];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = threadIdx.x / WARP_SZ;
    int lane = threadIdx.x % WARP_SZ;

    // Each block processes a partition of rows
    int partition = blockIdx.x;
    if (partition >= numPartitions) return;

    int rowStart = rowPartitions[partition];
    int rowEnd = (partition + 1 < numPartitions) ? rowPartitions[partition + 1] : numRows;

    // Process rows in this partition
    for (int row = rowStart + warpId; row < rowEnd; row += (BLOCK_SIZE / WARP_SZ)) {
        int rowPtrStart = rowPtr[row];
        int rowPtrEnd = rowPtr[row + 1];
        int rowNnz = rowPtrEnd - rowPtrStart;

        FloatType sum = static_cast<FloatType>(0);

        if (rowNnz >= WARP_SZ / 2) {
            // Use vector approach: all threads in warp cooperate
            for (int i = rowPtrStart + lane; i < rowPtrEnd; i += WARP_SZ) {
                int col = colIdx[i];
                sum += values[i] * x[col];
            }
            sum = warpReduceSum<FloatType, WARP_SZ>(sum);
            if (lane == 0) {
                y[row] = sum;
            }
        } else {
            // Use scalar approach: lane 0 handles the row
            if (lane == 0) {
                for (int i = rowPtrStart; i < rowPtrEnd; i++) {
                    int col = colIdx[i];
                    sum += values[i] * x[col];
                }
                y[row] = sum;
            }
        }
    }
}

/**
 * @brief CSR-Vector-Like kernel for extremely sparse matrices
 * Each thread processes multiple rows sequentially
 * Optimized for avgNnzPerRow << warpSize
 */
template<typename FloatType, int BLOCK_SIZE, int ROWS_PER_THREAD>
__global__ void spmv_csr_light_kernel(
    int numRows,
    int numCols,
    int nnz,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread processes ROWS_PER_THREAD rows
    for (int r = 0; r < ROWS_PER_THREAD; r++) {
        int row = tid * ROWS_PER_THREAD + r;
        if (row >= numRows) return;

        int rowStart = rowPtr[row];
        int rowEnd = rowPtr[row + 1];

        FloatType sum = static_cast<FloatType>(0);

        // Process all elements of this row
        for (int i = rowStart; i < rowEnd; i++) {
            int col = colIdx[i];
            sum += values[i] * x[col];
        }

        y[row] = sum;
    }
}

// ==================== Host API ====================

template<typename FloatType>
spmv_status_t spmv_csr(
    const CSRMatrix<FloatType>& matrix,
    const FloatType* x,
    FloatType* y,
    FloatType alpha,
    FloatType beta,
    const spmv_opts_t& opts);

template<typename FloatType>
spmv_status_t spmv_csr_transpose(
    const CSRMatrix<FloatType>& matrix,
    const FloatType* x,
    FloatType* y,
    FloatType alpha,
    FloatType beta,
    const spmv_opts_t& opts);

} // namespace muxi_spmv

#endif // SPMV_CSR_CUH_