/**
 * @file spmv_csr5.cu
 * @brief CSR5 format SpMV kernel implementations
 */

#include "spmv_csr5.cuh"
#include <cstdio>

namespace muxi_spmv {

// ==================== Configuration ====================

CSR5KernelConfig getCSR5Config(int warpSize, int nnz, int avgNnzPerRow) {
    CSR5KernelConfig config;

    // Determine optimal tile size (sigma)
    // For warp=64, use larger tiles
    // For warp=32, use smaller tiles
    if (warpSize == 64) {
        config.sigma = 256;        // 64 * 4 elements per thread
        config.blockSize = 256;    // 4 warps per block
    } else {
        config.sigma = 128;        // 32 * 4 elements per thread
        config.blockSize = 256;    // 8 warps per block
    }

    config.warpSize = warpSize;
    config.warpsPerBlock = config.blockSize / warpSize;
    config.sharedMemSize = 0;

    return config;
}

// ==================== Preprocessing Kernel ====================

template<int BLOCK_SIZE>
__global__ void csr5_compute_tile_boundaries_kernel(
    const int* __restrict__ rowPtr,
    int numRows,
    int nnz,
    int sigma,
    int* __restrict__ tile_row_ptr,
    int* __restrict__ tile_nnz_offset,
    int numTiles)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= numTiles) return;

    // Compute tile start position
    int tileStart = tid * sigma;

    // Binary search to find which row contains this NNZ position
    int row = csr5_binary_search_row(rowPtr, numRows, tileStart);

    tile_row_ptr[tid] = row;
    tile_nnz_offset[tid] = tileStart - rowPtr[row];
}

// ==================== SpMV Kernels ====================

/**
 * CSR5 SpMV kernel - Warp=64 version for Mars X201
 *
 * Each warp processes a tile of Sigma NNZ elements.
 * Each thread processes Sigma/64 elements.
 */
template<typename FloatType, int BLOCK_SIZE, int Sigma>
__global__ void spmv_csr5_warp64_kernel(
    int numRows,
    int numCols,
    int nnz,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y,
    const int* __restrict__ tile_row_ptr,
    const int* __restrict__ tile_nnz_offset,
    int sigma)
{
    constexpr int LOCAL_WARP_SIZE = 64;
    int warpId = blockIdx.x * (BLOCK_SIZE / LOCAL_WARP_SIZE) + (threadIdx.x / LOCAL_WARP_SIZE);
    int lane = threadIdx.x % LOCAL_WARP_SIZE;

    int numTiles = (nnz + sigma - 1) / sigma;
    if (warpId >= numTiles) return;

    // Get tile boundaries
    int tileStart = warpId * sigma;
    int tileEnd = min(tileStart + sigma, nnz);

    // Use precomputed tile metadata
    int startRow = tile_row_ptr[warpId];

    // Each thread processes Sigma/WARP_SIZE elements
    constexpr int ELEMENTS_PER_THREAD = Sigma / WARP_SIZE;
    int myStart = tileStart + lane * ELEMENTS_PER_THREAD;
    int myEnd = min(myStart + ELEMENTS_PER_THREAD, tileEnd);

    // Local accumulation
    FloatType localSum = static_cast<FloatType>(0);
    int currentRow = -1;

    // Find initial row for this thread's elements
    if (myStart < tileEnd) {
        if (lane == 0) {
            // First thread uses precomputed row
            currentRow = startRow;
        } else {
            // Other threads need to search
            currentRow = csr5_binary_search_row(rowPtr, numRows, myStart);
        }
    }

    // Process elements
    int rowStart = (currentRow >= 0) ? rowPtr[currentRow] : 0;
    int rowEnd = (currentRow >= 0) ? rowPtr[currentRow + 1] : 0;

    for (int idx = myStart; idx < myEnd; idx++) {
        // Check if we've crossed into next row
        while (idx >= rowEnd && currentRow < numRows - 1) {
            // Write previous row's result
            if (localSum != static_cast<FloatType>(0)) {
                atomicAdd(&y[currentRow], localSum);
                localSum = static_cast<FloatType>(0);
            }
            currentRow++;
            rowStart = rowPtr[currentRow];
            rowEnd = rowPtr[currentRow + 1];
        }

        if (idx < tileEnd) {
            int col = colIdx[idx];
            localSum += values[idx] * x[col];
        }
    }

    // Write final row's result
    if (localSum != static_cast<FloatType>(0) && currentRow >= 0) {
        atomicAdd(&y[currentRow], localSum);
    }
}

/**
 * CSR5 SpMV kernel - Warp=32 version for NVIDIA GPUs
 */
template<typename FloatType, int BLOCK_SIZE, int Sigma>
__global__ void spmv_csr5_warp32_kernel(
    int numRows,
    int numCols,
    int nnz,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y,
    const int* __restrict__ tile_row_ptr,
    const int* __restrict__ tile_nnz_offset,
    int sigma)
{
    constexpr int LOCAL_WARP_SIZE = 32;
    int warpId = blockIdx.x * (BLOCK_SIZE / LOCAL_WARP_SIZE) + (threadIdx.x / LOCAL_WARP_SIZE);
    int lane = threadIdx.x % LOCAL_WARP_SIZE;

    int numTiles = (nnz + sigma - 1) / sigma;
    if (warpId >= numTiles) return;

    // Get tile boundaries
    int tileStart = warpId * sigma;
    int tileEnd = min(tileStart + sigma, nnz);

    // Use precomputed tile metadata
    int startRow = tile_row_ptr[warpId];

    // Each thread processes Sigma/WARP_SIZE elements
    constexpr int ELEMENTS_PER_THREAD = Sigma / WARP_SIZE;
    int myStart = tileStart + lane * ELEMENTS_PER_THREAD;
    int myEnd = min(myStart + ELEMENTS_PER_THREAD, tileEnd);

    // Local accumulation
    FloatType localSum = static_cast<FloatType>(0);
    int currentRow = -1;

    // Find initial row for this thread's elements
    if (myStart < tileEnd) {
        if (lane == 0) {
            currentRow = startRow;
        } else {
            currentRow = csr5_binary_search_row(rowPtr, numRows, myStart);
        }
    }

    // Process elements
    int rowStart = (currentRow >= 0) ? rowPtr[currentRow] : 0;
    int rowEnd = (currentRow >= 0) ? rowPtr[currentRow + 1] : 0;

    for (int idx = myStart; idx < myEnd; idx++) {
        while (idx >= rowEnd && currentRow < numRows - 1) {
            if (localSum != static_cast<FloatType>(0)) {
                atomicAdd(&y[currentRow], localSum);
                localSum = static_cast<FloatType>(0);
            }
            currentRow++;
            rowStart = rowPtr[currentRow];
            rowEnd = rowPtr[currentRow + 1];
        }

        if (idx < tileEnd) {
            int col = colIdx[idx];
            localSum += values[idx] * x[col];
        }
    }

    // Write final row's result
    if (localSum != static_cast<FloatType>(0) && currentRow >= 0) {
        atomicAdd(&y[currentRow], localSum);
    }
}

// ==================== Host API Implementation ====================

template<typename FloatType>
spmv_status_t csr5_preprocess(
    const CSRMatrix<FloatType>& csr,
    CSR5Matrix<FloatType>& csr5,
    int sigma,
    cudaStream_t stream)
{
    if (csr.nnz == 0) {
        return SPMV_SUCCESS;
    }

    // Reference CSR data
    csr5.referenceFromCSR(csr);

    // Auto-select sigma if not specified
    if (sigma <= 0) {
        CSR5KernelConfig config = getCSR5Config(WARP_SIZE, csr.nnz, csr.nnz / csr.numRows);
        sigma = config.sigma;
    }

    csr5.sigma = sigma;
    int numTiles = (csr.nnz + sigma - 1) / sigma;

    // Allocate metadata
    csr5.allocateTileMetadata(numTiles);
    csr5.allocateDeviceTileMetadata();

    // Launch preprocessing kernel
    int blockSize = 256;
    int gridSize = (numTiles + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, stream);

    csr5_compute_tile_boundaries_kernel<256><<<gridSize, blockSize, 0, stream>>>(
        csr.d_rowPtr, csr.numRows, csr.nnz, sigma,
        csr5.d_tile_row_ptr, csr5.d_tile_nnz_offset, numTiles);

    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float timeMs;
    cudaEventElapsedTime(&timeMs, start, stop);
    csr5.conversionTimeMs = timeMs;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return SPMV_SUCCESS;
}

template<typename FloatType>
spmv_status_t spmv_csr5(
    const CSR5Matrix<FloatType>& matrix,
    const FloatType* x,
    FloatType* y,
    FloatType alpha,
    FloatType beta,
    const spmv_opts_t& opts)
{
    if (matrix.nnz == 0) {
        return SPMV_SUCCESS;
    }

#if SPMV_ENABLE_CHECKS
    if (!matrix.d_rowPtr || !matrix.d_colIdx || !matrix.d_values) {
        return SPMV_ERROR_INVALID_MATRIX;
    }
    if (!matrix.d_tile_row_ptr || !matrix.d_tile_nnz_offset) {
        return SPMV_ERROR_INVALID_MATRIX;
    }
    if (!x || !y) {
        return SPMV_ERROR_INVALID_VECTOR;
    }
#endif

    cudaStream_t stream = opts.stream ? (cudaStream_t)opts.stream : 0;

    // Clear output vector
    cudaMemsetAsync(y, 0, matrix.numRows * sizeof(FloatType), stream);

    // Get configuration
    CSR5KernelConfig config = getCSR5Config(WARP_SIZE, matrix.nnz, matrix.nnz / matrix.numRows);

    int blockSize = config.blockSize;
    int warpsPerBlock = config.warpsPerBlock;
    int numTiles = matrix.numTiles;
    int gridSize = (numTiles + warpsPerBlock - 1) / warpsPerBlock;

    int sigma = matrix.sigma;

    // Launch appropriate kernel based on warp size and sigma
    if (WARP_SIZE == 64) {
        // Mars X201 - warp=64
        if (sigma == 256) {
            spmv_csr5_warp64_kernel<FloatType, 256, 256><<<gridSize, blockSize, 0, stream>>>(
                matrix.numRows, matrix.numCols, matrix.nnz,
                matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values,
                x, y, matrix.d_tile_row_ptr, matrix.d_tile_nnz_offset, sigma);
        } else if (sigma == 512) {
            spmv_csr5_warp64_kernel<FloatType, 256, 512><<<gridSize, blockSize, 0, stream>>>(
                matrix.numRows, matrix.numCols, matrix.nnz,
                matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values,
                x, y, matrix.d_tile_row_ptr, matrix.d_tile_nnz_offset, sigma);
        } else {
            // Generic sigma (slower due to non-constexpr)
            spmv_csr5_warp64_kernel<FloatType, 256, 256><<<gridSize, blockSize, 0, stream>>>(
                matrix.numRows, matrix.numCols, matrix.nnz,
                matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values,
                x, y, matrix.d_tile_row_ptr, matrix.d_tile_nnz_offset, sigma);
        }
    } else {
        // NVIDIA - warp=32
        if (sigma == 128) {
            spmv_csr5_warp32_kernel<FloatType, 256, 128><<<gridSize, blockSize, 0, stream>>>(
                matrix.numRows, matrix.numCols, matrix.nnz,
                matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values,
                x, y, matrix.d_tile_row_ptr, matrix.d_tile_nnz_offset, sigma);
        } else if (sigma == 256) {
            spmv_csr5_warp32_kernel<FloatType, 256, 256><<<gridSize, blockSize, 0, stream>>>(
                matrix.numRows, matrix.numCols, matrix.nnz,
                matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values,
                x, y, matrix.d_tile_row_ptr, matrix.d_tile_nnz_offset, sigma);
        } else {
            spmv_csr5_warp32_kernel<FloatType, 256, 128><<<gridSize, blockSize, 0, stream>>>(
                matrix.numRows, matrix.numCols, matrix.nnz,
                matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values,
                x, y, matrix.d_tile_row_ptr, matrix.d_tile_nnz_offset, sigma);
        }
    }

    if (opts.sync) {
        cudaStreamSynchronize(stream);
    }

    return SPMV_SUCCESS;
}

// ==================== Optimized CSR5 Kernel with Warp Aggregation ====================

/**
 * CSR5 SpMV kernel - Optimized with warp-level aggregation
 *
 * Key optimization: Reduce atomic operations by aggregating results
 * at warp level before writing to global memory.
 */
template<typename FloatType, int BLOCK_SIZE, int WarpSize, int Sigma>
__global__ void spmv_csr5_optimized_kernel(
    int numRows,
    int numCols,
    int nnz,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y,
    const int* __restrict__ tile_row_ptr,
    const int* __restrict__ tile_nnz_offset,
    int sigma)
{
    int warpId = blockIdx.x * (BLOCK_SIZE / WarpSize) + (threadIdx.x / WarpSize);
    int lane = threadIdx.x % WarpSize;

    int numTiles = (nnz + sigma - 1) / sigma;
    if (warpId >= numTiles) return;

    // Get tile boundaries
    int tileStart = warpId * sigma;
    int tileEnd = min(tileStart + sigma, nnz);
    int startRow = tile_row_ptr[warpId];

    // Shared memory for warp-level aggregation
    __shared__ FloatType s_partial[BLOCK_SIZE];
    __shared__ int s_rowIdx[BLOCK_SIZE];
    __shared__ int s_valid[BLOCK_SIZE];  // Valid flag for each thread's result

    int tid = threadIdx.x;

    // Each thread processes Sigma/WarpSize elements
    constexpr int ELEMENTS_PER_THREAD = (Sigma + WarpSize - 1) / WarpSize;
    int myStart = tileStart + lane * ELEMENTS_PER_THREAD;
    int myEnd = min(myStart + ELEMENTS_PER_THREAD, tileEnd);

    // Local accumulation
    FloatType localSum = static_cast<FloatType>(0);
    int currentRow = -1;
    int valid = 0;

    // Find initial row
    if (myStart < tileEnd) {
        if (lane == 0) {
            currentRow = startRow;
        } else {
            // Binary search for initial row
            int lo = 0, hi = numRows;
            int target = myStart;
            while (lo < hi) {
                int mid = (lo + hi) / 2;
                if (rowPtr[mid] <= target) lo = mid + 1;
                else hi = mid;
            }
            currentRow = lo - 1;
        }
        valid = 1;
    }

    // Process elements
    for (int idx = myStart; idx < myEnd; idx++) {
        // Check if we need to move to next row
        while (currentRow >= 0 && currentRow < numRows && idx >= rowPtr[currentRow + 1]) {
            // Store partial result for this row
            if (localSum != static_cast<FloatType>(0)) {
                atomicAdd(&y[currentRow], localSum);
            }
            localSum = static_cast<FloatType>(0);
            currentRow++;
        }

        if (idx < tileEnd && currentRow >= 0 && currentRow < numRows) {
            int col = colIdx[idx];
            localSum += values[idx] * x[col];
        }
    }

    // Store final partial result
    s_partial[tid] = localSum;
    s_rowIdx[tid] = currentRow;
    s_valid[tid] = valid && (localSum != static_cast<FloatType>(0));

    __syncthreads();

    // Warp-level aggregation: each warp aggregates results for the same row
    // This reduces the number of atomic operations
    int localWarpId = tid / WarpSize;
    int laneInWarp = tid % WarpSize;

    // Only first thread of each warp does the aggregation
    if (laneInWarp == 0) {
        for (int i = 0; i < WarpSize; i++) {
            int idx = localWarpId * WarpSize + i;
            if (s_valid[idx]) {
                FloatType sum = s_partial[idx];
                int row = s_rowIdx[idx];

                // Check if next thread has same row
                if (i + 1 < WarpSize) {
                    int nextIdx = localWarpId * WarpSize + i + 1;
                    if (s_valid[nextIdx] && s_rowIdx[nextIdx] == row) {
                        // Aggregate same-row results
                        sum += s_partial[nextIdx];
                        s_valid[nextIdx] = 0;  // Mark as consumed
                        i++;  // Skip next
                    }
                }

                if (sum != static_cast<FloatType>(0)) {
                    atomicAdd(&y[row], sum);
                }
            }
        }
    }
}

// ==================== Merge-based SpMV Kernel ====================

/**
 * Device function: Find merge-path position
 *
 * Given a position k on the merge path, returns (rowIdx, nnzIdx).
 * The merge path combines rowPtr (row boundaries) and NNZ indices.
 */
__device__ __forceinline__ void merge_path_search(
    const int* __restrict__ rowPtr,
    int numRows,
    int nnz,
    int k,           // Position on merge path (0 to numRows + nnz)
    int& rowIdx,     // Output: row index
    int& nnzIdx)     // Output: NNZ index within row
{
    // Binary search on merge path diagonal
    // The merge path has: row indices (0..numRows) and NNZ indices (0..nnz)
    // At position k, we need (rowIdx, nnzIdx) where rowIdx + nnzIdx = k
    // and nnzIdx >= rowPtr[rowIdx] (the row boundary constraint)

    int lo = max(0, k - nnz);
    int hi = min(k, numRows);

    while (lo < hi) {
        int mid = (lo + hi) / 2;
        int nnz_at_mid = k - mid;
        if (mid < numRows && nnz_at_mid > rowPtr[mid + 1]) {
            lo = mid + 1;
        } else if (mid > 0 && nnz_at_mid < rowPtr[mid]) {
            hi = mid;
        } else {
            // Found the boundary
            lo = mid;
            hi = mid;
        }
    }

    rowIdx = lo;
    nnzIdx = k - lo;

    // Clamp to valid range
    if (rowIdx > numRows) rowIdx = numRows;
    if (nnzIdx > nnz) nnzIdx = nnz;
}

/**
 * Merge-based SpMV kernel - Optimized version
 *
 * Uses merge-path algorithm to divide work evenly without atomics.
 * Each warp processes a range of complete rows (or partial rows with exclusive ownership).
 *
 * Key optimizations:
 * 1. Merge-path partitioning aligns with row boundaries
 * 2. No atomic operations - each partition owns its rows exclusively
 * 3. Warp-level cooperation for row processing
 */
template<typename FloatType, int BLOCK_SIZE, int WarpSize>
__global__ void spmv_merge_based_kernel(
    int numRows,
    int numCols,
    int nnz,
    const int* __restrict__ rowPtr,
    const int* __restrict__ colIdx,
    const FloatType* __restrict__ values,
    const FloatType* __restrict__ x,
    FloatType* __restrict__ y,
    const int* __restrict__ mergePathPos,  // Merge path positions for partitions
    int numPartitions)
{
    // Each warp processes one partition
    int warpId = blockIdx.x * (BLOCK_SIZE / WarpSize) + (threadIdx.x / WarpSize);
    int lane = threadIdx.x % WarpSize;

    if (warpId >= numPartitions) return;

    // Get partition boundaries from merge path
    int pathStart = mergePathPos[warpId];
    int pathEnd = mergePathPos[warpId + 1];

    // Convert merge path positions to (row, nnz) coordinates
    int startRow, startNnz;
    int endRow, endNnz;

    // Only lane 0 does the search, then broadcasts
    if (lane == 0) {
        merge_path_search(rowPtr, numRows, nnz, pathStart, startRow, startNnz);
        merge_path_search(rowPtr, numRows, nnz, pathEnd, endRow, endNnz);
    }

    // Broadcast results within warp
    startRow = __shfl_sync(0xffffffff, startRow, 0);
    startNnz = __shfl_sync(0xffffffff, startNnz, 0);
    endRow = __shfl_sync(0xffffffff, endRow, 0);
    endNnz = __shfl_sync(0xffffffff, endNnz, 0);

    // Process rows from startRow to endRow
    // Each thread handles a subset of the rows
    int numRowsInPartition = endRow - startRow;
    if (numRowsInPartition <= 0) return;

    // Distribute rows among threads
    int rowsPerThread = (numRowsInPartition + WarpSize - 1) / WarpSize;
    int myRowStart = startRow + lane * rowsPerThread;
    int myRowEnd = min(myRowStart + rowsPerThread, endRow);

    // Handle partial row at start (if partition starts mid-row)
    if (lane == 0 && startNnz > rowPtr[startRow]) {
        // Process partial first row from startNnz to rowPtr[startRow+1]
        FloatType partialSum = static_cast<FloatType>(0);
        int rowEndNnz = (startRow + 1 <= endRow) ? rowPtr[startRow + 1] : endNnz;
        for (int idx = startNnz; idx < rowEndNnz; idx++) {
            int col = colIdx[idx];
            partialSum += values[idx] * x[col];
        }
        // Write partial result - use atomic only for this partial row
        if (partialSum != static_cast<FloatType>(0)) {
            atomicAdd(&y[startRow], partialSum);
        }
        myRowStart = startRow + 1;  // Skip the partial row
    }

    // Handle partial row at end (if partition ends mid-row)
    // This is done by the last thread
    int partialEndRow = -1;
    int partialEndNnzStart = -1;
    if (lane == WarpSize - 1 && endRow < numRows && endNnz < rowPtr[endRow + 1]) {
        // This partition ends mid-row, save for atomic handling
        partialEndRow = endRow;
        partialEndNnzStart = endNnz;
        myRowEnd = endRow;  // Don't process partial row normally
    }

    // Process complete rows (no atomics needed!)
    for (int row = myRowStart; row < myRowEnd && row < numRows; row++) {
        FloatType rowSum = static_cast<FloatType>(0);
        int rowStartNnz = rowPtr[row];
        int rowEndNnz = rowPtr[row + 1];

        // Each thread processes entire row, or partition rows among warp
        if (rowEndNnz - rowStartNnz > WarpSize) {
            // Large row: partition among warp threads
            int nnzPerThread = (rowEndNnz - rowStartNnz + WarpSize - 1) / WarpSize;
            int myNnzStart = rowStartNnz + lane * nnzPerThread;
            int myNnzEnd = min(myNnzStart + nnzPerThread, rowEndNnz);

            for (int idx = myNnzStart; idx < myNnzEnd; idx++) {
                int col = colIdx[idx];
                rowSum += values[idx] * x[col];
            }

            // Warp reduce for large row
            rowSum = csr5_warp_reduce_sum<FloatType, WarpSize>(rowSum);

            if (lane == 0) {
                y[row] = rowSum;  // Direct write, no atomic!
            }
        } else {
            // Small row: one thread processes entire row
            if (lane == 0 || row - myRowStart < WarpSize) {
                int threadRow = myRowStart + lane;
                if (threadRow < myRowEnd && threadRow < numRows) {
                    FloatType sum = static_cast<FloatType>(0);
                    for (int idx = rowPtr[threadRow]; idx < rowPtr[threadRow + 1]; idx++) {
                        int col = colIdx[idx];
                        sum += values[idx] * x[col];
                    }
                    y[threadRow] = sum;  // Direct write
                }
            }
        }
    }

    // Handle partial end row with atomic
    if (lane == WarpSize - 1 && partialEndRow >= 0 && partialEndRow < numRows) {
        FloatType partialSum = static_cast<FloatType>(0);
        int rowEndNnz = rowPtr[partialEndRow + 1];
        for (int idx = partialEndNnzStart; idx < rowEndNnz; idx++) {
            int col = colIdx[idx];
            partialSum += values[idx] * x[col];
        }
        if (partialSum != static_cast<FloatType>(0)) {
            atomicAdd(&y[partialEndRow], partialSum);
        }
    }
}

// ==================== Merge-based Preprocessing ====================

/**
 * Compute merge path partition positions
 *
 * The merge path has length = numRows + nnz (combining row indices and NNZ indices)
 * We divide this path evenly among partitions for load balancing.
 */
__global__ void compute_merge_partitions_kernel(
    int numRows,
    int nnz,
    int mergePathLength,
    int numPartitions,
    int* __restrict__ mergePathPos)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > numPartitions) return;  // Note: includes numPartitions as valid index

    // Evenly divide the merge path
    int pathPerPartition = (mergePathLength + numPartitions - 1) / numPartitions;
    mergePathPos[tid] = min(tid * pathPerPartition, mergePathLength);
}

/**
 * Host function for merge-based SpMV
 */
template<typename FloatType>
spmv_status_t spmv_merge_based(
    const CSRMatrix<FloatType>& matrix,
    const FloatType* x,
    FloatType* y,
    cudaStream_t stream)
{
    if (matrix.nnz == 0) {
        return SPMV_SUCCESS;
    }

    // Calculate number of partitions (one per warp)
    int blockSize = 256;
    int warpsPerBlock = blockSize / WARP_SIZE;

    // The merge path length = numRows + nnz
    int mergePathLength = matrix.numRows + matrix.nnz;

    // Aim for good load balancing: at least 64 elements per partition
    int targetPartitions = mergePathLength / 64;
    int numSMs = (WARP_SIZE == 64) ? 104 : 128;  // Mars X201 or RTX 4090
    int maxPartitions = numSMs * warpsPerBlock * 4;  // 4 blocks per SM for occupancy

    int numPartitions = max(1, min(targetPartitions, maxPartitions));

    // Allocate partition array (numPartitions + 1 entries)
    int* d_mergePathPos;
    cudaMalloc(&d_mergePathPos, (numPartitions + 1) * sizeof(int));

    // Compute merge path positions
    int gridSize = (numPartitions + 2 + 255) / 256;
    compute_merge_partitions_kernel<<<gridSize, 256, 0, stream>>>(
        matrix.numRows, matrix.nnz, mergePathLength, numPartitions, d_mergePathPos);

    // Clear output
    cudaMemsetAsync(y, 0, matrix.numRows * sizeof(FloatType), stream);

    // Launch kernel
    gridSize = (numPartitions + warpsPerBlock - 1) / warpsPerBlock;

    if (WARP_SIZE == 64) {
        spmv_merge_based_kernel<FloatType, 256, 64><<<gridSize, blockSize, 0, stream>>>(
            matrix.numRows, matrix.numCols, matrix.nnz,
            matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values,
            x, y, d_mergePathPos, numPartitions);
    } else {
        spmv_merge_based_kernel<FloatType, 256, 32><<<gridSize, blockSize, 0, stream>>>(
            matrix.numRows, matrix.numCols, matrix.nnz,
            matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values,
            x, y, d_mergePathPos, numPartitions);
    }

    // Cleanup
    cudaFree(d_mergePathPos);

    return SPMV_SUCCESS;
}

// ==================== Explicit Template Instantiation ====================
// These instantiations ensure the template code is generated for the linker.
// cu-bridge will replace cudaStream_t with the appropriate type.

template spmv_status_t csr5_preprocess<float>(const CSRMatrix<float>&, CSR5Matrix<float>&, int, cudaStream_t);
template spmv_status_t csr5_preprocess<double>(const CSRMatrix<double>&, CSR5Matrix<double>&, int, cudaStream_t);
template spmv_status_t spmv_csr5<float>(const CSR5Matrix<float>&, const float*, float*, float, float, const spmv_opts_t&);
template spmv_status_t spmv_csr5<double>(const CSR5Matrix<double>&, const double*, double*, double, double, const spmv_opts_t&);
template spmv_status_t spmv_merge_based<float>(const CSRMatrix<float>&, const float*, float*, cudaStream_t);
template spmv_status_t spmv_merge_based<double>(const CSRMatrix<double>&, const double*, double*, cudaStream_t);

} // namespace muxi_spmv