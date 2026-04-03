/**
 * @file deep_analysis_runner.cpp
 * @brief Comprehensive performance analysis tool for SpMV optimization
 *
 * Tests multiple optimization strategies and provides detailed analysis:
 * - Partition strategy optimization
 * - Memory access pattern analysis
 * - Comparison with theoretical limits
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cstdlib>

#include "formats/sparse_formats.h"
#include "spmv/csr/spmv_csr.cuh"
#include "spmv/csr5/spmv_csr5.cuh"
#include "spmv/csr5/assembly_analysis.cuh"
#include "spmv/csr5/ultra_optimized_test.cu"
#include "utils/timer.h"
#include "generators/matrix_generator.h"

using namespace muxi_spmv;

// ==================== Test Configuration ====================

struct TestConfig {
    int numRows;
    int numCols;
    int avgNnzPerRow;
    int numIterations;
};

// ==================== Performance Analysis Functions ====================

void printSeparator() {
    std::cout << "========================================\n";
}

template<typename FloatType>
struct PerformanceResult {
    std::string kernelName;
    float avgTimeMs;
    float bandwidthGBs;
    float utilizationPercent;
    bool correct;
};

template<typename FloatType>
PerformanceResult<FloatType> runKernelTest(
    const std::string& kernelName,
    std::function<void(const CSRMatrix<FloatType>&, const FloatType*, FloatType*, cudaStream_t)> kernelFunc,
    const CSRMatrix<FloatType>& matrix,
    const FloatType* d_x,
    FloatType* d_y,
    FloatType* d_reference,
    int numIterations)
{
    PerformanceResult<FloatType> result;
    result.kernelName = kernelName;

    GpuTimer timer;
    std::vector<float> times;

    // Warmup
    cudaMemset(d_y, 0, matrix.numRows * sizeof(FloatType));
    kernelFunc(matrix, d_x, d_y, 0);
    cudaDeviceSynchronize();

    // Run kernel multiple times
    for (int i = 0; i < numIterations; i++) {
        cudaMemset(d_y, 0, matrix.numRows * sizeof(FloatType));

        timer.start();
        kernelFunc(matrix, d_x, d_y, 0);
        timer.stop();

        times.push_back(timer.elapsed_ms());
    }

    // Calculate average time
    float avgTime = 0;
    for (float t : times) avgTime += t;
    avgTime /= numIterations;
    result.avgTimeMs = avgTime;

    // Verify correctness
    cudaMemset(d_y, 0, matrix.numRows * sizeof(FloatType));
    kernelFunc(matrix, d_x, d_y, 0);
    cudaDeviceSynchronize();

    FloatType* h_y = new FloatType[matrix.numRows];
    FloatType* h_ref = new FloatType[matrix.numRows];
    cudaMemcpy(h_y, d_y, matrix.numRows * sizeof(FloatType), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ref, d_reference, matrix.numRows * sizeof(FloatType), cudaMemcpyDeviceToHost);

    result.correct = true;
    for (int i = 0; i < matrix.numRows && result.correct; i++) {
        FloatType diff = std::abs(h_y[i] - h_ref[i]);
        FloatType tol = std::max(std::abs(h_ref[i]) * 1e-4f, FloatType(1e-6));
        if (diff > tol) {
            result.correct = false;
        }
    }

    delete[] h_y;
    delete[] h_ref;

    // Calculate bandwidth metrics
    size_t totalDataBytes = matrix.numRows * sizeof(int) * 2 +
                           matrix.nnz * sizeof(int) +
                           matrix.nnz * sizeof(FloatType) +
                           matrix.nnz * sizeof(FloatType) +
                           matrix.numRows * sizeof(FloatType);

    float peakBW = (WARP_SIZE == 64) ? 1843.0f : 1008.0f;
    result.bandwidthGBs = (totalDataBytes / (avgTime * 1e-3)) / (1024 * 1024 * 1024);
    result.utilizationPercent = result.bandwidthGBs / peakBW * 100;

    return result;
}

template<typename FloatType>
void printResult(const PerformanceResult<FloatType>& result) {
    std::cout << "Kernel: " << result.kernelName << "\n";
    std::cout << "  Time: " << result.avgTimeMs << " ms\n";
    std::cout << "  Bandwidth: " << result.bandwidthGBs << " GB/s\n";
    std::cout << "  Utilization: " << result.utilizationPercent << " %\n";
    std::cout << "  Correctness: " << (result.correct ? "PASS" : "FAIL") << "\n";
}

// ==================== Comprehensive Analysis ====================

template<typename FloatType>
void runComprehensiveAnalysis(const TestConfig& config) {
    printSeparator();
    std::cout << "SpMV Deep Performance Analysis\n";
    std::cout << "Warp Size: " << WARP_SIZE << "\n";
    printSeparator();

    std::cout << "\nMatrix Configuration:\n";
    std::cout << "  Rows: " << config.numRows << "\n";
    std::cout << "  Columns: " << config.numCols << "\n";
    std::cout << "  Avg NNZ/Row: " << config.avgNnzPerRow << "\n";
    std::cout << "  Iterations: " << config.numIterations << "\n";

    // Generate matrix using existing generator
    std::cout << "\nGenerating matrix...\n";
    CSRMatrix<FloatType> matrix;
    int nnz = config.numRows * config.avgNnzPerRow;

    spmv_status_t status = generators::generateRandomMatrix<FloatType>(
        config.numRows, config.numCols, nnz, matrix);

    if (status != SPMV_SUCCESS) {
        std::cerr << "Failed to generate matrix!\n";
        return;
    }

    std::cout << "  Actual NNZ: " << matrix.nnz << "\n";
    std::cout << "  Actual Avg NNZ/Row: " << (matrix.nnz / matrix.numRows) << "\n";

    // Generate input/output vectors
    FloatType* h_x = new FloatType[config.numCols];
    FloatType* h_reference = new FloatType[config.numRows];

    for (int i = 0; i < config.numCols; i++) {
        h_x[i] = static_cast<FloatType>(rand()) / RAND_MAX;
    }

    // Compute reference on CPU
    std::cout << "\nComputing CPU reference...\n";
    int* h_rowPtr = new int[matrix.numRows + 1];
    int* h_colIdx = new int[matrix.nnz];
    FloatType* h_values = new FloatType[matrix.nnz];

    cudaMemcpy(h_rowPtr, matrix.d_rowPtr, (matrix.numRows + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_colIdx, matrix.d_colIdx, matrix.nnz * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_values, matrix.d_values, matrix.nnz * sizeof(FloatType), cudaMemcpyDeviceToHost);

    for (int row = 0; row < matrix.numRows; row++) {
        FloatType sum = FloatType(0);
        for (int idx = h_rowPtr[row]; idx < h_rowPtr[row + 1]; idx++) {
            sum += h_values[idx] * h_x[h_colIdx[idx]];
        }
        h_reference[row] = sum;
    }

    // Allocate device vectors
    FloatType* d_x;
    FloatType* d_y;
    FloatType* d_reference;
    cudaMalloc(&d_x, config.numCols * sizeof(FloatType));
    cudaMalloc(&d_y, config.numRows * sizeof(FloatType));
    cudaMalloc(&d_reference, config.numRows * sizeof(FloatType));

    cudaMemcpy(d_x, h_x, config.numCols * sizeof(FloatType), cudaMemcpyHostToDevice);
    cudaMemcpy(d_reference, h_reference, config.numRows * sizeof(FloatType), cudaMemcpyHostToDevice);

    // ==================== Kernel Tests ====================

    printSeparator();
    std::cout << "Testing Kernel Variants\n";
    printSeparator();

    std::vector<PerformanceResult<FloatType>> results;

    // Test 1: Original Merge-based (elementsPerPartition=16)
    auto originalMerge = [](const CSRMatrix<FloatType>& m, const FloatType* x, FloatType* y, cudaStream_t s) {
        spmv_merge_based<FloatType>(m, x, y, s);
    };
    results.push_back(runKernelTest("Original Merge (ePP=16)",
        originalMerge, matrix, d_x, d_y, d_reference, config.numIterations));

    // Test 2: Ultra-optimized kernel
    auto ultraOptimized = [](const CSRMatrix<FloatType>& m, const FloatType* x, FloatType* y, cudaStream_t s) {
        spmv_merge_based_ultra_optimized<FloatType>(m, x, y, s);
    };
    results.push_back(runKernelTest("Ultra-Optimized (prefetch+warp_agg)",
        ultraOptimized, matrix, d_x, d_y, d_reference, config.numIterations));

    // Test 3: Scalar kernel (baseline)
    auto scalarKernel = [](const CSRMatrix<FloatType>& m, const FloatType* x, FloatType* y, cudaStream_t s) {
        spmv_csr_scalar<FloatType>(m, x, y, 1.0, 0.0, spmv_opts_t());
    };
    results.push_back(runKernelTest("Scalar (1 thread/row)",
        scalarKernel, matrix, d_x, d_y, d_reference, config.numIterations));

    // ==================== Print Results ====================

    printSeparator();
    std::cout << "Performance Results\n";
    printSeparator();

    for (const auto& result : results) {
        printResult(result);
        std::cout << "\n";
    }

    // ==================== Partition Analysis ====================

    printSeparator();
    std::cout << "Partition Strategy Analysis\n";
    printSeparator();

    analyze_partition_strategies<FloatType>(matrix, d_x, d_y, 30);

    // ==================== Memory Analysis ====================

    printSeparator();
    std::cout << "Memory Access Analysis\n";
    printSeparator();

    auto analysis = analyze_memory_access<FloatType>(
        matrix.numRows, matrix.nnz,
        matrix.nnz / matrix.numRows, matrix.numCols);

    std::cout << "Data Movement Breakdown:\n";
    std::cout << "  rowPtr: " << analysis.rowPtrBytes / 1024 << " KB\n";
    std::cout << "  colIdx: " << analysis.colIdxBytes / 1024 << " KB\n";
    std::cout << "  values: " << analysis.valuesBytes / 1024 << " KB\n";
    std::cout << "  x (random): " << analysis.xBytes / 1024 << " KB\n";
    std::cout << "  y (output): " << analysis.yBytes / 1024 << " KB\n";
    std::cout << "  Total: " << analysis.totalBytes / 1024 << " KB\n";
    std::cout << "\n";
    std::cout << "Access Patterns:\n";
    std::cout << "  Random Access Factor: " << analysis.randomAccessFactor << "\n";
    std::cout << "  Cache Miss Rate: " << analysis.cacheMissRate * 100 << " %\n";
    std::cout << "  Theoretical Utilization: " << analysis.theoreticalBW * 100 << " %\n";

    // ==================== Hardware Limit Analysis ====================

    printSeparator();
    std::cout << "Hardware Limit Analysis\n";
    printSeparator();

    float peakBW = (WARP_SIZE == 64) ? 1843.0f : 1008.0f;
    float l2Cache = (WARP_SIZE == 64) ? 4.0f : 72.0f;  // MB

    std::cout << "Hardware Parameters:\n";
    std::cout << "  Warp Size: " << WARP_SIZE << "\n";
    std::cout << "  Peak Bandwidth: " << peakBW << " GB/s\n";
    std::cout << "  L2 Cache: ~" << l2Cache << " MB\n";

    float bestUtil = 0;
    std::string bestKernel;
    for (const auto& result : results) {
        if (result.correct && result.utilizationPercent > bestUtil) {
            bestUtil = result.utilizationPercent;
            bestKernel = result.kernelName;
        }
    }

    std::cout << "\nBest Achieved:\n";
    std::cout << "  Kernel: " << bestKernel << "\n";
    std::cout << "  Utilization: " << bestUtil << " %\n";
    std::cout << "  Gap from theoretical: " << (100 - bestUtil) << " %\n";

    // ==================== Write Report ====================

    std::ofstream report("deep_analysis_report.md");
    report << "# SpMV Deep Performance Analysis Report\n\n";
    report << "## Test Configuration\n\n";
    report << "- Rows: " << config.numRows << "\n";
    report << "- Columns: " << config.numCols << "\n";
    report << "- Avg NNZ/Row: " << config.avgNnzPerRow << "\n";
    report << "- Warp Size: " << WARP_SIZE << "\n";
    report << "- Iterations: " << config.numIterations << "\n\n";

    report << "## Performance Results\n\n";
    report << "| Kernel | Time (ms) | Bandwidth (GB/s) | Utilization (%) | Correctness |\n";
    report << "|--------|-----------|------------------|-----------------|-------------|\n";
    for (const auto& result : results) {
        report << "| " << result.kernelName << " | " << result.avgTimeMs
               << " | " << result.bandwidthGBs << " | " << result.utilizationPercent
               << " | " << (result.correct ? "PASS" : "FAIL") << " |\n";
    }

    report << "\n## Memory Analysis\n\n";
    report << "- Total Data Movement: " << analysis.totalBytes / 1024 << " KB\n";
    report << "- Random Access Factor: " << analysis.randomAccessFactor << "\n";
    report << "- Cache Miss Rate: " << analysis.cacheMissRate * 100 << " %\n";
    report << "- Theoretical Utilization: " << analysis.theoreticalBW * 100 << " %\n";

    report << "\n## Hardware Limits\n\n";
    report << "- Best Achieved Utilization: " << bestUtil << " %\n";
    report << "- Gap from 100%: " << (100 - bestUtil) << " %\n";
    report << "- Primary Limiting Factor: L2 Cache Size (" << l2Cache << " MB)\n";

    report.close();
    std::cout << "\nReport saved to: deep_analysis_report.md\n";

    // Cleanup
    delete[] h_x;
    delete[] h_reference;
    delete[] h_rowPtr;
    delete[] h_colIdx;
    delete[] h_values;
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_reference);
}

// ==================== Main ====================

int main(int argc, char** argv) {
    std::cout << "SpMV Deep Performance Analysis Tool\n";
    std::cout << "WARP_SIZE = " << WARP_SIZE << "\n";

    if (argc < 5) {
        std::cout << "\nUsage: " << argv[0] << " <rows> <cols> <avgNnz> <iterations>\n";
        std::cout << "\nExample:\n";
        std::cout << "  " << argv[0] << " 1000000 1000 10 100\n";
        return 1;
    }

    TestConfig config;
    config.numRows = std::atoi(argv[1]);
    config.numCols = std::atoi(argv[2]);
    config.avgNnzPerRow = std::atoi(argv[3]);
    config.numIterations = std::atoi(argv[4]);

    // Run analysis
    runComprehensiveAnalysis<float>(config);

    return 0;
}