/**
 * @file deep_analysis.cu
 * @brief Deep performance analysis tool for SpMV optimization
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>

#include "formats/sparse_formats.h"
#include "spmv/csr/spmv_csr.cuh"
#include "spmv/csr5/spmv_csr5.cuh"
#include "generators/matrix_generator.h"

using namespace muxi_spmv;
using namespace muxi_spmv::generators;

// GPU Timer using CUDA events
class GpuTimer {
public:
    GpuTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }
    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    void start(cudaStream_t stream = 0) {
        cudaEventRecord(start_, stream);
    }
    void stop(cudaStream_t stream = 0) {
        cudaEventRecord(stop_, stream);
        cudaEventSynchronize(stop_);
    }
    float elapsed_ms() {
        float ms;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }
private:
    cudaEvent_t start_, stop_;
};

void printSeparator() {
    std::cout << "========================================\n";
}

// Test partition strategies
void testPartitionStrategies(int numRows, int numCols, int avgNnz, int iterations) {
    printSeparator();
    std::cout << "Partition Strategy Analysis\n";
    printSeparator();

    std::cout << "Matrix: " << numRows << " rows, " << numCols << " cols, avgNnz=" << avgNnz << "\n";

    // Generate matrix
    CSRMatrix<float> matrix;
    int nnz = numRows * avgNnz;
    generateRandomMatrix<float>(numRows, numCols, nnz, matrix);
    matrix.copyToDevice();

    std::cout << "Actual NNZ: " << matrix.nnz << "\n";

    // Generate x vector
    float* h_x = new float[numCols];
    for (int i = 0; i < numCols; i++) h_x[i] = rand() / (float)RAND_MAX;

    float* d_x, *d_y;
    cudaMalloc(&d_x, numCols * sizeof(float));
    cudaMalloc(&d_y, numRows * sizeof(float));
    cudaMemcpy(d_x, h_x, numCols * sizeof(float), cudaMemcpyHostToDevice);

    // Test different partition configurations by modifying elementsPerPartition
    // We need to test through the API
    std::cout << "\nTesting with current merge-based kernel (ePP=16)\n";

    GpuTimer timer;
    std::vector<float> times;

    // Warmup
    cudaMemset(d_y, 0, numRows * sizeof(float));
    spmv_merge_based<float>(matrix, d_x, d_y, 0);
    cudaDeviceSynchronize();

    // Measure
    for (int i = 0; i < iterations; i++) {
        cudaMemset(d_y, 0, numRows * sizeof(float));
        timer.start();
        spmv_merge_based<float>(matrix, d_x, d_y, 0);
        timer.stop();
        times.push_back(timer.elapsed_ms());
    }

    float avgTime = 0;
    for (float t : times) avgTime += t;
    avgTime /= iterations;

    // Calculate bandwidth
    size_t dataBytes = numRows * sizeof(int) * 2 +  // rowPtr
                       matrix.nnz * sizeof(int) +     // colIdx
                       matrix.nnz * sizeof(float) +   // values
                       matrix.nnz * sizeof(float) +   // x
                       numRows * sizeof(float);       // y

    float peakBW = (WARP_SIZE == 64) ? 1843.0f : 1008.0f;
    float bw = (dataBytes / (avgTime * 1e-3)) / (1024 * 1024 * 1024);
    float util = bw / peakBW * 100;

    std::cout << "\nResults:\n";
    std::cout << "  Average time: " << avgTime << " ms\n";
    std::cout << "  Bandwidth: " << bw << " GB/s\n";
    std::cout << "  Utilization: " << util << " %\n";

    // Memory analysis
    std::cout << "\nMemory Analysis:\n";
    std::cout << "  Total data: " << dataBytes / 1024 << " KB\n";
    std::cout << "  rowPtr: " << (numRows * sizeof(int) * 2) / 1024 << " KB\n";
    std::cout << "  colIdx+values: " << (matrix.nnz * sizeof(int) + matrix.nnz * sizeof(float)) / 1024 << " KB\n";
    std::cout << "  x vector: " << (matrix.nnz * sizeof(float)) / 1024 << " KB (random access)\n";

    // Theoretical analysis
    float theoreticalTime = dataBytes / (peakBW * 1024 * 1024 * 1024);
    float randomPenalty = 2.5f;  // Random access penalty
    float expectedTime = theoreticalTime * randomPenalty;
    float expectedUtil = 1.0f / randomPenalty * 100;

    std::cout << "\nTheoretical Analysis:\n";
    std::cout << "  Ideal time: " << theoreticalTime * 1000 << " ms\n";
    std::cout << "  Random access penalty: ~" << randomPenalty << "x\n";
    std::cout << "  Expected utilization: ~" << expectedUtil << " %\n";
    std::cout << "  Actual vs Expected: " << (util / expectedUtil * 100) << " %\n";

    // Hardware limit
    float l2Cache = (WARP_SIZE == 64) ? 4.0f : 72.0f;
    float xVectorSize = numCols * sizeof(float) / (1024 * 1024);  // MB

    std::cout << "\nHardware Analysis:\n";
    std::cout << "  L2 Cache: ~" << l2Cache << " MB\n";
    std::cout << "  x vector size: " << xVectorSize << " MB\n";
    std::cout << "  Can fit in L2: " << (xVectorSize < l2Cache ? "Yes" : "No") << "\n";

    if (WARP_SIZE == 64) {
        std::cout << "\nMars X201 specific analysis:\n";
        std::cout << "  - L2 cache too small to hold x vector\n";
        std::cout << "  - Random x access causes high cache miss rate\n";
        std::cout << "  - Performance limited by memory bandwidth, not compute\n";
        std::cout << "  - Further optimization requires:\n";
        std::cout << "    1. Column reordering for better locality\n";
        std::cout << "    2. Hardware: larger L2 cache\n";
    } else {
        std::cout << "\nRTX 4090 specific analysis:\n";
        std::cout << "  - 72MB L2 can hold entire x vector for small matrices\n";
        std::cout << "  - High cache hit rate enables >100% theoretical utilization\n";
    }

    // Cleanup
    delete[] h_x;
    cudaFree(d_x);
    cudaFree(d_y);
}

// Compare kernels
void compareKernels(int numRows, int numCols, int avgNnz, int iterations) {
    printSeparator();
    std::cout << "Kernel Comparison Analysis\n";
    printSeparator();

    CSRMatrix<float> matrix;
    int nnz = numRows * avgNnz;
    generateRandomMatrix<float>(numRows, numCols, nnz, matrix);
    matrix.copyToDevice();

    float* h_x = new float[numCols];
    float* h_y_ref = new float[numRows];
    for (int i = 0; i < numCols; i++) h_x[i] = rand() / (float)RAND_MAX;

    // Compute reference on CPU
    int* h_rowPtr = new int[numRows + 1];
    int* h_colIdx = new int[matrix.nnz];
    float* h_values = new float[matrix.nnz];

    cudaMemcpy(h_rowPtr, matrix.d_rowPtr, (numRows + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_colIdx, matrix.d_colIdx, matrix.nnz * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_values, matrix.d_values, matrix.nnz * sizeof(float), cudaMemcpyDeviceToHost);

    for (int row = 0; row < numRows; row++) {
        float sum = 0;
        for (int idx = h_rowPtr[row]; idx < h_rowPtr[row + 1]; idx++) {
            sum += h_values[idx] * h_x[h_colIdx[idx]];
        }
        h_y_ref[row] = sum;
    }

    float* d_x, *d_y;
    cudaMalloc(&d_x, numCols * sizeof(float));
    cudaMalloc(&d_y, numRows * sizeof(float));
    cudaMemcpy(d_x, h_x, numCols * sizeof(float), cudaMemcpyHostToDevice);

    GpuTimer timer;

    struct KernelResult {
        std::string name;
        float time;
        float bw;
        float util;
        bool correct;
    };

    std::vector<KernelResult> results;

    size_t dataBytes = numRows * sizeof(int) * 2 +
                       matrix.nnz * sizeof(int) +
                       matrix.nnz * sizeof(float) +
                       matrix.nnz * sizeof(float) +
                       numRows * sizeof(float);
    float peakBW = (WARP_SIZE == 64) ? 1843.0f : 1008.0f;

    // Test Merge-based
    {
        KernelResult r;
        r.name = "Merge-based (ePP=16)";

        // Warmup
        cudaMemset(d_y, 0, numRows * sizeof(float));
        spmv_merge_based<float>(matrix, d_x, d_y, 0);
        cudaDeviceSynchronize();

        float totalTime = 0;
        for (int i = 0; i < iterations; i++) {
            cudaMemset(d_y, 0, numRows * sizeof(float));
            timer.start();
            spmv_merge_based<float>(matrix, d_x, d_y, 0);
            timer.stop();
            totalTime += timer.elapsed_ms();
        }

        r.time = totalTime / iterations;
        r.bw = (dataBytes / (r.time * 1e-3)) / (1024 * 1024 * 1024);
        r.util = r.bw / peakBW * 100;

        // Verify
        cudaMemset(d_y, 0, numRows * sizeof(float));
        spmv_merge_based<float>(matrix, d_x, d_y, 0);
        cudaDeviceSynchronize();

        float* h_y = new float[numRows];
        cudaMemcpy(h_y, d_y, numRows * sizeof(float), cudaMemcpyDeviceToHost);

        r.correct = true;
        for (int i = 0; i < numRows; i++) {
            float diff = fabs(h_y[i] - h_y_ref[i]);
            float tol = max(fabs(h_y_ref[i]) * 1e-4f, 1e-6f);
            if (diff > tol) r.correct = false;
        }
        delete[] h_y;

        results.push_back(r);
    }

    // Test Scalar
    {
        KernelResult r;
        r.name = "Scalar (1 thread/row)";

        spmv_opts_t opts;
        opts.sync = true;

        // Warmup
        cudaMemset(d_y, 0, numRows * sizeof(float));
        spmv_csr_scalar<float>(matrix, d_x, d_y, 1.0, 0.0, opts);
        cudaDeviceSynchronize();

        float totalTime = 0;
        for (int i = 0; i < iterations; i++) {
            cudaMemset(d_y, 0, numRows * sizeof(float));
            timer.start();
            spmv_csr_scalar<float>(matrix, d_x, d_y, 1.0, 0.0, opts);
            timer.stop();
            totalTime += timer.elapsed_ms();
        }

        r.time = totalTime / iterations;
        r.bw = (dataBytes / (r.time * 1e-3)) / (1024 * 1024 * 1024);
        r.util = r.bw / peakBW * 100;

        // Verify
        cudaMemset(d_y, 0, numRows * sizeof(float));
        spmv_csr_scalar<float>(matrix, d_x, d_y, 1.0, 0.0, opts);
        cudaDeviceSynchronize();

        float* h_y = new float[numRows];
        cudaMemcpy(h_y, d_y, numRows * sizeof(float), cudaMemcpyDeviceToHost);

        r.correct = true;
        for (int i = 0; i < numRows; i++) {
            float diff = fabs(h_y[i] - h_y_ref[i]);
            float tol = max(fabs(h_y_ref[i]) * 1e-4f, 1e-6f);
            if (diff > tol) r.correct = false;
        }
        delete[] h_y;

        results.push_back(r);
    }

    // Print results
    std::cout << "\n";
    std::cout << "| Kernel | Time (ms) | BW (GB/s) | Util (%) | Correct |\n";
    std::cout << "|--------|-----------|-----------|----------|---------|\n";
    for (const auto& r : results) {
        std::cout << "| " << r.name << " | " << r.time
                  << " | " << r.bw << " | " << r.util
                  << " | " << (r.correct ? "PASS" : "FAIL") << " |\n";
    }

    // Cleanup
    delete[] h_x;
    delete[] h_y_ref;
    delete[] h_rowPtr;
    delete[] h_colIdx;
    delete[] h_values;
    cudaFree(d_x);
    cudaFree(d_y);
}

int main(int argc, char** argv) {
    printSeparator();
    std::cout << "SpMV Deep Performance Analysis\n";
    std::cout << "WARP_SIZE = " << WARP_SIZE << "\n";
    printSeparator();

    if (argc < 5) {
        std::cout << "\nUsage: " << argv[0] << " <rows> <cols> <avgNnz> <iterations>\n";
        std::cout << "\nExample:\n";
        std::cout << "  " << argv[0] << " 1000000 1000 10 50\n";
        return 1;
    }

    int rows = atoi(argv[1]);
    int cols = atoi(argv[2]);
    int avgNnz = atoi(argv[3]);
    int iterations = atoi(argv[4]);

    std::cout << "\nConfiguration:\n";
    std::cout << "  Rows: " << rows << "\n";
    std::cout << "  Columns: " << cols << "\n";
    std::cout << "  Avg NNZ/Row: " << avgNnz << "\n";
    std::cout << "  Iterations: " << iterations << "\n";

    testPartitionStrategies(rows, cols, avgNnz, iterations);
    compareKernels(rows, cols, avgNnz, iterations);

    return 0;
}