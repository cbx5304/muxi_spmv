/**
 * @file ultra_optimized_test.cu
 * @brief Test ultra-optimized kernel variants for Mars X201
 */

#include "spmv_csr5.cuh"
#include "assembly_analysis.cuh"
#include "../formats/sparse_formats.h"
#include "../../utils/timer.h"
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
    // More aggressive partitioning for Mars X201
    int elementsPerPartition;
    int maxPartitionMultiplier;

    if (WARP_SIZE == 64) {
        // Mars X201: Ultra-optimized configuration
        // Experiment with even smaller partitions
        elementsPerPartition = 8;  // More partitions
        maxPartitionMultiplier = 16;  // Allow more partitions per SM
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

// ==================== Performance Testing ====================

template<typename FloatType>
void test_ultra_optimized_kernel(
    const CSRMatrix<FloatType>& matrix,
    const FloatType* d_x,
    FloatType* d_y,
    FloatType* d_reference,
    int numIterations = 100)
{
    std::cout << "\n=== Ultra-Optimized Kernel Test ===\n";
    std::cout << "Matrix: " << matrix.numRows << " rows, "
              << matrix.numCols << " cols, "
              << matrix.nnz << " nnz\n";
    std::cout << "avgNnzPerRow: " << (matrix.nnz / matrix.numRows) << "\n";

    // Create timer
    GpuTimer timer;

    // Test original merge-based kernel
    std::vector<float> originalTimes;
    for (int i = 0; i < numIterations; i++) {
        timer.start();
        spmv_merge_based<FloatType>(matrix, d_x, d_y, 0);
        timer.stop();
        originalTimes.push_back(timer.elapsed_ms());
    }

    float originalAvg = 0;
    for (float t : originalTimes) originalAvg += t;
    originalAvg /= numIterations;

    // Test ultra-optimized kernel
    std::vector<float> ultraTimes;
    for (int i = 0; i < numIterations; i++) {
        timer.start();
        spmv_merge_based_ultra_optimized<FloatType>(matrix, d_x, d_y, 0);
        timer.stop();
        ultraTimes.push_back(timer.elapsed_ms());
    }

    float ultraAvg = 0;
    for (float t : ultraTimes) ultraAvg += t;
    ultraAvg /= numIterations;

    // Verify correctness
    cudaMemset(d_y, 0, matrix.numRows * sizeof(FloatType));
    spmv_merge_based_ultra_optimized<FloatType>(matrix, d_x, d_y, 0);
    cudaDeviceSynchronize();

    // Compare with reference
    bool correct = true;
    FloatType* h_y = new FloatType[matrix.numRows];
    FloatType* h_ref = new FloatType[matrix.numRows];
    cudaMemcpy(h_y, d_y, matrix.numRows * sizeof(FloatType), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ref, d_reference, matrix.numRows * sizeof(FloatType), cudaMemcpyDeviceToHost);

    for (int i = 0; i < matrix.numRows; i++) {
        FloatType diff = std::abs(h_y[i] - h_ref[i]);
        FloatType tol = std::max(std::abs(h_ref[i]) * 1e-4f, FloatType(1e-6));
        if (diff > tol) {
            correct = false;
            if (i < 10) {
                std::cout << "Mismatch at row " << i << ": "
                          << h_y[i] << " vs " << h_ref[i] << "\n";
            }
        }
    }

    // Calculate bandwidth utilization
    size_t dataBytes = matrix.numRows * sizeof(int) * 2 +  // rowPtr
                       matrix.nnz * sizeof(int) +           // colIdx
                       matrix.nnz * sizeof(FloatType) +     // values
                       matrix.nnz * sizeof(FloatType) +     // x (random access)
                       matrix.numRows * sizeof(FloatType);  // y

    float peakBW = (WARP_SIZE == 64) ? 1843.0f : 1008.0f;  // GB/s
    float originalBW = (dataBytes / (originalAvg * 1e-3)) / (1024 * 1024 * 1024);
    float ultraBW = (dataBytes / (ultraAvg * 1e-3)) / (1024 * 1024 * 1024);

    float originalUtil = originalBW / peakBW * 100;
    float ultraUtil = ultraBW / peakBW * 100;

    std::cout << "\n=== Results ===\n";
    std::cout << "Original Merge-based:\n";
    std::cout << "  Time: " << originalAvg << " ms\n";
    std::cout << "  Bandwidth: " << originalBW << " GB/s\n";
    std::cout << "  Utilization: " << originalUtil << " %\n";

    std::cout << "\nUltra-Optimized:\n";
    std::cout << "  Time: " << ultraAvg << " ms\n";
    std::cout << "  Bandwidth: " << ultraBW << " GB/s\n";
    std::cout << "  Utilization: " << ultraUtil << " %\n";

    std::cout << "\nImprovement: " << ((originalAvg - ultraAvg) / originalAvg * 100) << " %\n";
    std::cout << "Correctness: " << (correct ? "PASS" : "FAIL") << "\n";

    // Memory analysis
    auto analysis = analyze_memory_access<FloatType>(
        matrix.numRows, matrix.nnz,
        matrix.nnz / matrix.numRows, matrix.numCols);

    std::cout << "\n=== Memory Analysis ===\n";
    std::cout << "Total data movement: " << analysis.totalBytes / (1024 * 1024) << " MB\n";
    std::cout << "Random access factor: " << analysis.randomAccessFactor << "\n";
    std::cout << "Cache miss rate: " << analysis.cacheMissRate * 100 << " %\n";
    std::cout << "Theoretical BW: " << analysis.theoreticalBW * 100 << " %\n";

    delete[] h_y;
    delete[] h_ref;
}

// ==================== Partition Strategy Analysis ====================

template<typename FloatType>
void analyze_partition_strategies(
    const CSRMatrix<FloatType>& matrix,
    const FloatType* d_x,
    FloatType* d_y,
    int numIterations = 50)
{
    std::cout << "\n=== Partition Strategy Analysis ===\n";

    GpuTimer timer;
    int blockSize = 256;
    int warpsPerBlock = blockSize / WARP_SIZE;
    int mergePathLength = matrix.numRows + matrix.nnz;
    int numSMs = (WARP_SIZE == 64) ? 104 : 128;

    // Test different elementsPerPartition values
    std::vector<int> elementsPerPartitionValues = {4, 8, 12, 16, 32, 64, 128};

    std::cout << "\nElementsPerPartition | Partitions | Time (ms) | BW Util (%)\n";
    std::cout << "-----------------------------------------------------------\n";

    for (int elementsPerPartition : elementsPerPartitionValues) {
        int maxPartitions = numSMs * warpsPerBlock * 8;
        int targetPartitions = mergePathLength / elementsPerPartition;
        int numPartitions = max(1, min(targetPartitions, maxPartitions));

        // Test this configuration
        std::vector<float> times;
        for (int i = 0; i < numIterations; i++) {
            // Allocate partition array
            int* d_mergePathPos;
            cudaMalloc(&d_mergePathPos, (numPartitions + 1) * sizeof(int));

            int gridSize = (numPartitions + 2 + 255) / 256;
            compute_merge_partitions_kernel<<<gridSize, 256, 0, 0>>>(
                matrix.numRows, matrix.nnz, mergePathLength, numPartitions, d_mergePathPos);

            cudaMemsetAsync(d_y, 0, matrix.numRows * sizeof(FloatType), 0);

            gridSize = (numPartitions + warpsPerBlock - 1) / warpsPerBlock;

            timer.start();
            if (WARP_SIZE == 64) {
                spmv_merge_based_kernel<FloatType, 256, 64>
                    <<<gridSize, blockSize, 0, 0>>>(
                    matrix.numRows, matrix.numCols, matrix.nnz,
                    matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values,
                    d_x, d_y, d_mergePathPos, numPartitions);
            } else {
                spmv_merge_based_kernel<FloatType, 256, 32>
                    <<<gridSize, blockSize, 0, 0>>>(
                    matrix.numRows, matrix.numCols, matrix.nnz,
                    matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values,
                    d_x, d_y, d_mergePathPos, numPartitions);
            }
            timer.stop();
            times.push_back(timer.elapsed_ms());

            cudaFree(d_mergePathPos);
        }

        float avgTime = 0;
        for (float t : times) avgTime += t;
        avgTime /= numIterations;

        // Calculate bandwidth utilization
        size_t dataBytes = matrix.numRows * sizeof(int) * 2 +
                           matrix.nnz * sizeof(int) +
                           matrix.nnz * sizeof(FloatType) +
                           matrix.nnz * sizeof(FloatType) +
                           matrix.numRows * sizeof(FloatType);

        float peakBW = (WARP_SIZE == 64) ? 1843.0f : 1008.0f;
        float bw = (dataBytes / (avgTime * 1e-3)) / (1024 * 1024 * 1024);
        float util = bw / peakBW * 100;

        std::cout << elementsPerPartition << " | " << numPartitions
                  << " | " << avgTime << " | " << util << "\n";
    }
}

// ==================== Explicit Instantiation ====================

template spmv_status_t spmv_merge_based_ultra_optimized<float>(
    const CSRMatrix<float>&, const float*, float*, cudaStream_t);
template spmv_status_t spmv_merge_based_ultra_optimized<double>(
    const CSRMatrix<double>&, const double*, double*, cudaStream_t);

template void test_ultra_optimized_kernel<float>(
    const CSRMatrix<float>&, const float*, float*, float*, int);
template void test_ultra_optimized_kernel<double>(
    const CSRMatrix<double>&, const double*, double*, double*, int);

template void analyze_partition_strategies<float>(
    const CSRMatrix<float>&, const float*, float*, int);
template void analyze_partition_strategies<double>(
    const CSRMatrix<double>&, const double*, double*, int);

} // namespace muxi_spmv