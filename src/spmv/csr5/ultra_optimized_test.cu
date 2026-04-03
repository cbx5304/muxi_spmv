/**
 * @file ultra_optimized_test.cu
 * @brief Test ultra-optimized kernel variants for Mars X201
 */

#include "spmv_csr5.cuh"
#include "assembly_analysis.cuh"
#include "../formats/sparse_formats.h"
#include <iostream>
#include <vector>
#include <cmath>

namespace muxi_spmv {

// ==================== Host Function for Ultra-Optimized Kernel ====================

template<typename FloatType>
spmv_status_t spmv_merge_based_ultra_optimized(
    const CSRMatrix<FloatType>& matrix,
    const FloatType* x,
    FloatType* y,
    cudaStream_t stream)
{
    if (matrix.nnz == 0) {
        return SPMV_SUCCESS;
    }

    int blockSize = 256;
    int warpsPerBlock = blockSize / WARP_SIZE;
    int mergePathLength = matrix.numRows + matrix.nnz;

    // Ultra-optimized partition strategy
    int elementsPerPartition;
    int maxPartitionMultiplier;

    if (WARP_SIZE == 64) {
        // Mars X201: Ultra-optimized configuration
        elementsPerPartition = 8;  // More partitions
        maxPartitionMultiplier = 16;
    } else {
        elementsPerPartition = 32;
        maxPartitionMultiplier = 8;
    }

    int numSMs = (WARP_SIZE == 64) ? 104 : 128;
    int targetPartitions = mergePathLength / elementsPerPartition;
    int maxPartitions = numSMs * warpsPerBlock * maxPartitionMultiplier;
    int numPartitions = max(1, min(targetPartitions, maxPartitions));

    // Allocate partition array
    int* d_mergePathPos;
    cudaMalloc(&d_mergePathPos, (numPartitions + 1) * sizeof(int));

    // Compute merge path positions
    int gridSize = (numPartitions + 2 + 255) / 256;
    compute_merge_partitions_kernel<<<gridSize, 256, 0, stream>>>(
        matrix.numRows, matrix.nnz, mergePathLength, numPartitions, d_mergePathPos);

    // Clear output
    cudaMemsetAsync(y, 0, matrix.numRows * sizeof(FloatType), stream);

    // Launch ultra-optimized kernel
    gridSize = (numPartitions + warpsPerBlock - 1) / warpsPerBlock;

    if (WARP_SIZE == 64) {
        spmv_merge_based_ultra_optimized_kernel<FloatType, 256, 64>
            <<<gridSize, blockSize, 0, stream>>>(
            matrix.numRows, matrix.numCols, matrix.nnz,
            matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values,
            x, y, d_mergePathPos, numPartitions);
    } else {
        spmv_merge_based_ultra_optimized_kernel<FloatType, 256, 32>
            <<<gridSize, blockSize, 0, stream>>>(
            matrix.numRows, matrix.numCols, matrix.nnz,
            matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values,
            x, y, d_mergePathPos, numPartitions);
    }

    cudaFree(d_mergePathPos);
    return SPMV_SUCCESS;
}

// ==================== Explicit Instantiation ====================

template spmv_status_t spmv_merge_based_ultra_optimized<float>(
    const CSRMatrix<float>&, const float*, float*, cudaStream_t);
template spmv_status_t spmv_merge_based_ultra_optimized<double>(
    const CSRMatrix<double>&, const double*, double*, cudaStream_t);

} // namespace muxi_spmv