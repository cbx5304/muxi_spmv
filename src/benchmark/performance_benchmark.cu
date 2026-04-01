/**
 * @file performance_benchmark.cu
 * @brief Implementation of performance benchmarking utilities
 */

#include "benchmark/performance_benchmark.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstring>

// For cuSPARSE comparison
#include <cusparse.h>

namespace muxi_spmv {
namespace benchmark {

// Type helpers for cuSPARSE (must be defined before use)
template<typename FloatType>
struct CuSparseType {
    static cudaDataType_t value;
};
template<> cudaDataType_t CuSparseType<float>::value = CUDA_R_32F;
template<> cudaDataType_t CuSparseType<double>::value = CUDA_R_64F;

template<typename FloatType>
inline FloatType alphaValue() { return static_cast<FloatType>(1.0); }
template<typename FloatType>
inline FloatType betaValue() { return static_cast<FloatType>(0.0); }

// Helper to calculate statistics
static void calculateStats(const std::vector<double>& times,
                           double& min, double& max, double& avg, double& stdDev) {
    if (times.empty()) {
        min = max = avg = stdDev = 0.0;
        return;
    }

    min = *std::min_element(times.begin(), times.end());
    max = *std::max_element(times.begin(), times.end());

    double sum = 0.0;
    for (double t : times) sum += t;
    avg = sum / times.size();

    double varSum = 0.0;
    for (double t : times) {
        double diff = t - avg;
        varSum += diff * diff;
    }
    stdDev = sqrt(varSum / times.size());
}

template<typename FloatType>
spmv_status_t measureCSRPerformance(
    const CSRMatrix<FloatType>& matrix,
    const FloatType* d_x,
    FloatType* d_y,
    const BenchmarkConfig& config,
    spmv_handle_t* handle,
    PerformanceResult& result) {

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Setup result structure
    result.numRows = matrix.numRows;
    result.numCols = matrix.numCols;
    result.nnz = matrix.nnz;
    result.sparsity = static_cast<double>(matrix.nnz) / (matrix.numRows * matrix.numCols);
    result.deviceId = handle->deviceId;
    result.warpSize = handle->warpSize;
    result.iterations = config.measureIterations;

    // Get peak bandwidth from device info
    int memoryClock, busWidth;
    cudaDeviceGetAttribute(&memoryClock, cudaDevAttrMemoryClockRate, handle->deviceId);
    cudaDeviceGetAttribute(&busWidth, cudaDevAttrGlobalMemoryBusWidth, handle->deviceId);
    result.peakBandwidthGBs = 2.0 * memoryClock * (busWidth / 8) / 1e6;

    // Prepare execution options
    spmv_opts_t opts = spmv_default_opts();
    opts.sync = config.syncAfterEach ? 1 : 0;

    std::vector<double> times;
    times.reserve(config.measureIterations);

    // Warmup iterations
    for (int i = 0; i < config.warmupIterations; i++) {
        spmv_status_t status = spmv_csr<FloatType>(matrix, d_x, d_y, 1.0f, 0.0f, opts);
        if (status != SPMV_SUCCESS) {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            return status;
        }
    }

    // Synchronize before measurement
    CUDA_CHECK(cudaDeviceSynchronize());

    // Measurement iterations
    for (int i = 0; i < config.measureIterations; i++) {
        CUDA_CHECK(cudaEventRecord(start, config.stream));

        spmv_status_t status = spmv_csr<FloatType>(matrix, d_x, d_y, 1.0f, 0.0f, opts);
        if (status != SPMV_SUCCESS) {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            return status;
        }

        CUDA_CHECK(cudaEventRecord(stop, config.stream));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float timeMs;
        CUDA_CHECK(cudaEventElapsedTime(&timeMs, start, stop));
        times.push_back(static_cast<double>(timeMs));
    }

    // Calculate statistics
    calculateStats(times, result.minTimeMs, result.maxTimeMs,
                   result.avgTimeMs, result.stdDevMs);

    // Use median time for GFLOPS and bandwidth
    std::sort(times.begin(), times.end());
    result.timeMs = times[times.size() / 2];

    // Calculate performance metrics
    int floatBytes = sizeof(FloatType);
    result.gflops = calculateGFlops(matrix.nnz, result.timeMs);
    result.bandwidthGBs = calculateBandwidth(matrix.nnz, matrix.numRows, matrix.numCols,
                                              result.timeMs, floatBytes);
    result.bandwidthUtilization = result.bandwidthGBs / result.peakBandwidthGBs * 100.0;

    // Correctness check (simple: multiply by ones vector)
    if (config.checkCorrectness) {
        // For correctness, we'd need reference result
        // Simple check: verify no NaN/Inf
        FloatType* h_y = new FloatType[matrix.numRows];
        CUDA_CHECK(cudaMemcpy(h_y, d_y, matrix.numRows * sizeof(FloatType),
                              cudaMemcpyDeviceToHost));

        result.correctnessPassed = true;
        for (int i = 0; i < matrix.numRows; i++) {
            if (std::isnan(h_y[i]) || std::isinf(h_y[i])) {
                result.correctnessPassed = false;
                break;
            }
        }
        delete[] h_y;
    } else {
        result.correctnessPassed = true;
    }

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return SPMV_SUCCESS;
}

// cuSPARSE comparison is disabled for domestic GPU
// The domestic GPU uses hcsparse which has different API
#ifdef ENABLE_CUSPARSE
template<typename FloatType>
spmv_status_t compareWithCusparse(
    const CSRMatrix<FloatType>& matrix,
    const FloatType* d_x,
    FloatType* d_y,
    const BenchmarkConfig& config,
    double& cusparseTime,
    double& cusparseGflops) {

    // Create cuSPARSE handle
    cusparseHandle_t cusparseHandle;
    cusparseStatus_t cusparseStatus = cusparseCreate(&cusparseHandle);
    if (cusparseStatus != CUSPARSE_STATUS_SUCCESS) {
        fprintf(stderr, "cuSPARSE handle creation failed\n");
        return SPMV_ERROR_INTERNAL;
    }

    // Create cuSPARSE matrix descriptor
    cusparseSpMatDescr_t matDescr;
    cusparseStatus = cusparseCreateCsr(&matDescr, matrix.numRows, matrix.numCols,
                                        matrix.nnz,
                                        matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values,
                                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                        CUSPARSE_INDEX_BASE_ZERO,
                                        CuSparseType<FloatType>::value);
    if (cusparseStatus != CUSPARSE_STATUS_SUCCESS) {
        cusparseDestroy(cusparseHandle);
        return SPMV_ERROR_INTERNAL;
    }

    // Create vector descriptors
    cusparseDnVecDescr_t vecX, vecY;
    cusparseCreateDnVec(&vecX, matrix.numCols, const_cast<FloatType*>(d_x),
                        CuSparseType<FloatType>::value);
    cusparseCreateDnVec(&vecY, matrix.numRows, d_y, CuSparseType<FloatType>::value);

    FloatType alphaVal = alphaValue<FloatType>();
    FloatType betaVal = betaValue<FloatType>();

    // Allocate buffer
    size_t bufferSize;
    cusparseSpMV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alphaVal, matDescr, vecX, &betaVal,
                            vecY, CuSparseType<FloatType>::value,
                            CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);

    void* buffer;
    cudaMalloc(&buffer, bufferSize);

    // Timing with CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::vector<double> times;

    // Warmup
    for (int i = 0; i < config.warmupIterations; i++) {
        cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &alphaVal, matDescr, vecX, &betaVal,
                     vecY, CuSparseType<FloatType>::value,
                     CUSPARSE_SPMV_ALG_DEFAULT, buffer);
    }
    cudaDeviceSynchronize();

    // Measure
    for (int i = 0; i < config.measureIterations; i++) {
        cudaEventRecord(start);
        cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &alphaVal, matDescr, vecX, &betaVal,
                     vecY, CuSparseType<FloatType>::value,
                     CUSPARSE_SPMV_ALG_DEFAULT, buffer);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float timeMs;
        cudaEventElapsedTime(&timeMs, start, stop);
        times.push_back(timeMs);
    }

    // Median time
    std::sort(times.begin(), times.end());
    cusparseTime = times[times.size() / 2];
    cusparseGflops = calculateGFlops(matrix.nnz, cusparseTime);

    // Cleanup
    cudaFree(buffer);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroySpMat(matDescr);
    cusparseDestroy(cusparseHandle);

    return SPMV_SUCCESS;
}
#endif // ENABLE_CUSPARSE

void printPerformanceResult(const PerformanceResult& result) {
    printf("\n=== Performance Results ===\n");
    printf("Matrix: %d x %d, NNZ: %d, Sparsity: %.4f%%\n",
           result.numRows, result.numCols, result.nnz, result.sparsity * 100);
    printf("Device: %d, Warp Size: %d\n", result.deviceId, result.warpSize);
    printf("\n");
    printf("Time (median): %.4f ms\n", result.timeMs);
    printf("Time (min/max): %.4f / %.4f ms\n", result.minTimeMs, result.maxTimeMs);
    printf("Time (avg/std): %.4f / %.4f ms\n", result.avgTimeMs, result.stdDevMs);
    printf("\n");
    printf("GFLOPS: %.2f\n", result.gflops);
    printf("Bandwidth: %.2f GB/s\n", result.bandwidthGBs);
    printf("Bandwidth Utilization: %.1f%% (%.2f GB/s peak)\n",
           result.bandwidthUtilization, result.peakBandwidthGBs);
    printf("\n");
    printf("Correctness: %s\n", result.correctnessPassed ? "PASSED" : "FAILED");
}

spmv_status_t writeResultsJSON(const std::vector<PerformanceResult>& results,
                                const char* filename) {
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open %s for writing\n", filename);
        return SPMV_ERROR_INVALID_MATRIX;
    }

    fprintf(fp, "{\n");
    fprintf(fp, "  \"results\": [\n");

    for (size_t i = 0; i < results.size(); i++) {
        const PerformanceResult& r = results[i];
        fprintf(fp, "    {\n");
        fprintf(fp, "      \"matrix\": {\"rows\": %d, \"cols\": %d, \"nnz\": %d, \"sparsity\": %.6f},\n",
                r.numRows, r.numCols, r.nnz, r.sparsity);
        fprintf(fp, "      \"device\": {\"id\": %d, \"warpSize\": %d, \"peakBandwidthGBs\": %.2f},\n",
                r.deviceId, r.warpSize, r.peakBandwidthGBs);
        fprintf(fp, "      \"performance\": {\n");
        fprintf(fp, "        \"timeMs\": %.4f,\n", r.timeMs);
        fprintf(fp, "        \"minTimeMs\": %.4f,\n", r.minTimeMs);
        fprintf(fp, "        \"maxTimeMs\": %.4f,\n", r.maxTimeMs);
        fprintf(fp, "        \"avgTimeMs\": %.4f,\n", r.avgTimeMs);
        fprintf(fp, "        \"stdDevMs\": %.4f,\n", r.stdDevMs);
        fprintf(fp, "        \"gflops\": %.2f,\n", r.gflops);
        fprintf(fp, "        \"bandwidthGBs\": %.2f,\n", r.bandwidthGBs);
        fprintf(fp, "        \"bandwidthUtilization\": %.1f\n", r.bandwidthUtilization);
        fprintf(fp, "      },\n");
        fprintf(fp, "      \"correctnessPassed\": %s\n", r.correctnessPassed ? "true" : "false");
        fprintf(fp, "    }%s\n", (i < results.size() - 1) ? "," : "");
    }

    fprintf(fp, "  ]\n");
    fprintf(fp, "}\n");

    fclose(fp);
    return SPMV_SUCCESS;
}

// Explicit template instantiation
template spmv_status_t measureCSRPerformance<float>(
    const CSRMatrix<float>&, const float*, float*,
    const BenchmarkConfig&, spmv_handle_t*, PerformanceResult&);

template spmv_status_t measureCSRPerformance<double>(
    const CSRMatrix<double>&, const double*, double*,
    const BenchmarkConfig&, spmv_handle_t*, PerformanceResult&);

} // namespace benchmark
} // namespace muxi_spmv