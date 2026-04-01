/**
 * @file performance_benchmark.h
 * @brief Performance benchmarking utilities for SpMV
 */

#ifndef PERFORMANCE_BENCHMARK_H_
#define PERFORMANCE_BENCHMARK_H_

#include "utils/common.h"
#include "formats/sparse_formats.h"
#include "api/spmv_api.h"
#include <cuda_runtime.h>
#include <vector>

namespace muxi_spmv {
namespace benchmark {

/**
 * @brief Performance measurement result
 */
struct PerformanceResult {
    double timeMs;              ///< Execution time in milliseconds
    double gflops;              ///< GFLOPS (2 * nnz / time)
    double bandwidthGBs;        ///< Effective bandwidth in GB/s
    double bandwidthUtilization;///< Bandwidth utilization (effective / peak)
    int iterations;             ///< Number of iterations measured
    bool correctnessPassed;     ///< Correctness check result

    // Additional metrics
    double minTimeMs;           ///< Minimum time across iterations
    double maxTimeMs;           ///< Maximum time across iterations
    double avgTimeMs;           ///< Average time
    double stdDevMs;            ///< Standard deviation

    // Matrix info
    int numRows;
    int numCols;
    int nnz;
    double sparsity;

    // Device info
    int deviceId;
    int warpSize;
    double peakBandwidthGBs;    ///< Device peak bandwidth
};

/**
 * @brief Benchmark configuration
 */
struct BenchmarkConfig {
    int warmupIterations;       ///< Warmup iterations (not measured)
    int measureIterations;      ///< Measurement iterations
    bool checkCorrectness;      ///< Verify results after execution
    bool syncAfterEach;         ///< Synchronize after each iteration
    cudaStream_t stream;        ///< CUDA stream to use
    spmv_type_t floatType;      ///< Data type (SPMV_TYPE_FLOAT/DOUBLE)
    bool compareCusparse;       ///< Compare with cuSPARSE

    BenchmarkConfig() : warmupIterations(3), measureIterations(10),
                        checkCorrectness(true), syncAfterEach(true),
                        stream(0), floatType(SPMV_TYPE_FLOAT),
                        compareCusparse(false) {}
};

/**
 * @brief Calculate theoretical GFLOPS for SpMV
 * @param nnz Number of non-zeros
 * @param timeMs Execution time in milliseconds
 * @return GFLOPS value
 */
inline double calculateGFlops(int nnz, double timeMs) {
    // SpMV: 2 FLOPs per non-zero (multiply + add)
    // GFLOPS = (2 * nnz) / (timeMs * 1e-6) / 1e9
    return (2.0 * nnz) / (timeMs * 1e6) / 1e9;
}

/**
 * @brief Calculate effective bandwidth
 * @param nnz Number of non-zeros
 * @param numRows Number of rows
 * @param numCols Number of columns
 * @param timeMs Execution time in milliseconds
 * @param floatBytes Size of floating point type in bytes
 * @return Bandwidth in GB/s
 */
inline double calculateBandwidth(int nnz, int numRows, int numCols,
                                  double timeMs, int floatBytes = 4) {
    // Data transferred:
    // - Matrix values: nnz * floatBytes
    // - Matrix indices: nnz * 4 (int)
    // - Row pointers: (numRows + 1) * 4
    // - Input vector x: numCols * floatBytes
    // - Output vector y: numRows * floatBytes
    size_t dataBytes = nnz * floatBytes + nnz * 4 + (numRows + 1) * 4 +
                       numCols * floatBytes + numRows * floatBytes;

    // GB/s = dataBytes / (timeMs * 1e-6) / 1e9
    return static_cast<double>(dataBytes) / (timeMs * 1e6);
}

/**
 * @brief Measure SpMV performance for CSR matrix
 * @tparam FloatType Floating point type
 * @param matrix CSR matrix (must be on device)
 * @param d_x Input vector on device
 * @param d_y Output vector on device (will be overwritten)
 * @param config Benchmark configuration
 * @param handle SpMV handle
 * @param result Output performance result
 * @return Status code
 */
template<typename FloatType>
spmv_status_t measureCSRPerformance(
    const CSRMatrix<FloatType>& matrix,
    const FloatType* d_x,
    FloatType* d_y,
    const BenchmarkConfig& config,
    spmv_handle_t* handle,
    PerformanceResult& result);

/**
 * @brief Compare with cuSPARSE performance (NVIDIA only)
 * @tparam FloatType Floating point type
 * @param matrix CSR matrix (must be on device)
 * @param d_x Input vector on device
 * @param d_y Output vector on device
 * @param config Benchmark configuration
 * @param cusparseTime Output cuSPARSE time
 * @param cusparseGflops Output cuSPARSE GFLOPS
 * @return Status code
 * @note This function is only available on NVIDIA GPUs
 */
#ifdef ENABLE_CUSPARSE
template<typename FloatType>
spmv_status_t compareWithCusparse(
    const CSRMatrix<FloatType>& matrix,
    const FloatType* d_x,
    FloatType* d_y,
    const BenchmarkConfig& config,
    double& cusparseTime,
    double& cusparseGflops);
#else
template<typename FloatType>
inline spmv_status_t compareWithCusparse(
    const CSRMatrix<FloatType>&,
    const FloatType*,
    FloatType*,
    const BenchmarkConfig&,
    double&,
    double&) {
    // cuSPARSE not available on this platform
    return SPMV_ERROR_UNSUPPORTED_FORMAT;
}
#endif

/**
 * @brief Run comprehensive benchmark suite
 * @tparam FloatType Floating point type
 * @param matrix CSR matrix
 * @param config Benchmark configuration
 * @param result Output performance result
 * @return Status code
 */
template<typename FloatType>
spmv_status_t runBenchmark(
    CSRMatrix<FloatType>& matrix,
    const BenchmarkConfig& config,
    PerformanceResult& result);

/**
 * @brief Print performance result to stdout
 * @param result Performance result
 */
void printPerformanceResult(const PerformanceResult& result);

/**
 * @brief Write performance results to JSON file
 * @param results Vector of results
 * @param filename Output file path
 * @return Status code
 */
spmv_status_t writeResultsJSON(const std::vector<PerformanceResult>& results,
                                const char* filename);

} // namespace benchmark
} // namespace muxi_spmv

#endif // PERFORMANCE_BENCHMARK_H_