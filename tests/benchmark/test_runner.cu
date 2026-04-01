/**
 * @file test_runner.cu
 * @brief Comprehensive test runner for SpMV benchmarking
 */

#include "utils/common.h"
#include "formats/sparse_formats.h"
#include "generators/matrix_generator.h"
#include "generators/mtx_io.h"
#include "benchmark/performance_benchmark.h"
#include "api/spmv_api.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace muxi_spmv;
using namespace muxi_spmv::generators;
using namespace muxi_spmv::benchmark;

// Command line options
struct TestOptions {
    // Matrix parameters
    int numRows;
    int numCols;
    double sparsity;
    MatrixType matrixType;
    int bandwidth;
    int blockSize;
    int numClusters;
    double concentrationFactor;
    double powerLawAlpha;

    // Test parameters
    int warmupIterations;
    int measureIterations;
    bool checkCorrectness;
    bool compareCusparse;

    // Output
    std::string outputFile;
    std::string mtxInputFile;
    std::string mtxOutputFile;

    // Data type
    bool useDouble;

    TestOptions() : numRows(1024), numCols(1024), sparsity(0.01),
                    matrixType(MatrixType::STRUCTURED_DIAGONAL),
                    bandwidth(1), blockSize(4), numClusters(10),
                    concentrationFactor(5.0), powerLawAlpha(2.5),
                    warmupIterations(3), measureIterations(10),
                    checkCorrectness(true), compareCusparse(false),
                    useDouble(false) {}
};

void printUsage(const char* progName) {
    printf("SpMV Test Runner - Comprehensive Benchmarking Tool\n\n");
    printf("Usage: %s [options]\n\n", progName);
    printf("Matrix Generation Options:\n");
    printf("  --rows <n>          Number of rows (default: 1024)\n");
    printf("  --cols <n>          Number of columns (default: 1024)\n");
    printf("  --sparsity <p>      Sparsity ratio 0.0-1.0 (default: 0.01)\n");
    printf("  --type <type>       Matrix type (default: diagonal):\n");
    printf("                        diagonal, banded, block_diag,\n");
    printf("                        random, concentrated, powerlaw\n");
    printf("  --bandwidth <w>     Bandwidth for banded matrix (default: 1)\n");
    printf("  --clusters <n>      Number of clusters for concentrated (default: 10)\n");
    printf("  --alpha <a>         Power-law alpha (default: 2.5)\n");
    printf("\n");
    printf("Test Options:\n");
    printf("  --warmup <n>        Warmup iterations (default: 3)\n");
    printf("  --measure <n>       Measurement iterations (default: 10)\n");
    printf("  --no-check          Skip correctness check\n");
    printf("  --cusparse          Compare with cuSPARSE\n");
    printf("  --double            Use double precision\n");
    printf("\n");
    printf("File Options:\n");
    printf("  --input <file>      Read matrix from MTX file\n");
    printf("  --output <file>     Write performance results to JSON\n");
    printf("  --save-mtx <file>   Save generated matrix to MTX file\n");
    printf("\n");
    printf("Examples:\n");
    printf("  %s --rows 10000 --type random --sparsity 0.001\n", progName);
    printf("  %s --input matrix.mtx --cusparse\n", progName);
    printf("  %s --rows 50000 --type concentrated --output results.json\n", progName);
}

MatrixType parseMatrixType(const char* str) {
    if (strcmp(str, "diagonal") == 0) return MatrixType::STRUCTURED_DIAGONAL;
    if (strcmp(str, "banded") == 0) return MatrixType::STRUCTURED_BANDED;
    if (strcmp(str, "block_diag") == 0) return MatrixType::STRUCTURED_BLOCK_DIAGONAL;
    if (strcmp(str, "random") == 0) return MatrixType::RANDOM_UNIFORM;
    if (strcmp(str, "concentrated") == 0) return MatrixType::CONCENTRATED_LOCAL;
    if (strcmp(str, "powerlaw") == 0) return MatrixType::REALWORLD_POWERLaw;
    fprintf(stderr, "Unknown matrix type: %s\n", str);
    return MatrixType::STRUCTURED_DIAGONAL;
}

bool parseArgs(int argc, char** argv, TestOptions& opts) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--rows") == 0 && i + 1 < argc) {
            opts.numRows = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--cols") == 0 && i + 1 < argc) {
            opts.numCols = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--sparsity") == 0 && i + 1 < argc) {
            opts.sparsity = atof(argv[++i]);
        } else if (strcmp(argv[i], "--type") == 0 && i + 1 < argc) {
            opts.matrixType = parseMatrixType(argv[++i]);
        } else if (strcmp(argv[i], "--bandwidth") == 0 && i + 1 < argc) {
            opts.bandwidth = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--clusters") == 0 && i + 1 < argc) {
            opts.numClusters = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--alpha") == 0 && i + 1 < argc) {
            opts.powerLawAlpha = atof(argv[++i]);
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            opts.warmupIterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--measure") == 0 && i + 1 < argc) {
            opts.measureIterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--no-check") == 0) {
            opts.checkCorrectness = false;
        } else if (strcmp(argv[i], "--cusparse") == 0) {
            opts.compareCusparse = true;
        } else if (strcmp(argv[i], "--double") == 0) {
            opts.useDouble = true;
        } else if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            opts.mtxInputFile = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            opts.outputFile = argv[++i];
        } else if (strcmp(argv[i], "--save-mtx") == 0 && i + 1 < argc) {
            opts.mtxOutputFile = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return false;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return false;
        }
    }
    return true;
}

template<typename FloatType>
int runTest(const TestOptions& opts) {
    printf("=== SpMV Test Runner ===\n\n");

    // Create handle
    spmv_handle_t* handle;
    spmv_status_t status = spmv_create_handle(&handle);
    if (status != SPMV_SUCCESS) {
        fprintf(stderr, "Failed to create handle: %d\n", status);
        return 1;
    }

    printf("Device: Warp size = %d\n", handle->warpSize);

    // Generate or load matrix
    CSRMatrix<FloatType> matrix;

    if (!opts.mtxInputFile.empty()) {
        printf("Loading matrix from: %s\n", opts.mtxInputFile.c_str());
        status = io::readMTXFile(opts.mtxInputFile.c_str(), matrix);
        if (status != SPMV_SUCCESS) {
            fprintf(stderr, "Failed to load MTX file: %d\n", status);
            spmv_destroy_handle(handle);
            return 1;
        }
    } else {
        printf("Generating matrix:\n");
        printf("  Rows: %d, Cols: %d\n", opts.numRows, opts.numCols);
        printf("  Type: %d, Sparsity: %.4f%%\n",
               static_cast<int>(opts.matrixType), opts.sparsity * 100);

        MatrixGenConfig config;
        config.numRows = opts.numRows;
        config.numCols = opts.numCols;
        config.sparsity = opts.sparsity;
        config.type = opts.matrixType;
        config.bandwidth = opts.bandwidth;
        config.numClusters = opts.numClusters;
        config.concentrationFactor = opts.concentrationFactor;
        config.powerLawAlpha = opts.powerLawAlpha;

        MatrixGenerator<FloatType>* generator = createGenerator<FloatType>(opts.matrixType);
        if (!generator) {
            fprintf(stderr, "Failed to create generator\n");
            spmv_destroy_handle(handle);
            return 1;
        }

        status = generator->generate(config, matrix);
        delete generator;

        if (status != SPMV_SUCCESS) {
            fprintf(stderr, "Matrix generation failed: %d\n", status);
            spmv_destroy_handle(handle);
            return 1;
        }
    }

    printf("Matrix info: %d x %d, NNZ: %d\n",
           matrix.numRows, matrix.numCols, matrix.nnz);

    // Save to MTX if requested
    if (!opts.mtxOutputFile.empty()) {
        printf("Saving matrix to: %s\n", opts.mtxOutputFile.c_str());
        io::writeMTXFile(opts.mtxOutputFile.c_str(), matrix);
    }

    // Allocate device memory
    matrix.allocateDevice();
    FloatType* d_x;
    FloatType* d_y;
    cudaMalloc(&d_x, matrix.numCols * sizeof(FloatType));
    cudaMalloc(&d_y, matrix.numRows * sizeof(FloatType));

    // Initialize vectors
    FloatType* h_x = new FloatType[matrix.numCols];
    for (int i = 0; i < matrix.numCols; i++) {
        h_x[i] = static_cast<FloatType>(i % 100 + 1) / static_cast<FloatType>(100);
    }

    cudaMemcpy(d_x, h_x, matrix.numCols * sizeof(FloatType), cudaMemcpyHostToDevice);
    cudaMemset(d_y, 0, matrix.numRows * sizeof(FloatType));

    // Copy matrix to device
    matrix.copyToDevice();

    // Setup benchmark config
    BenchmarkConfig benchConfig;
    benchConfig.warmupIterations = opts.warmupIterations;
    benchConfig.measureIterations = opts.measureIterations;
    benchConfig.checkCorrectness = opts.checkCorrectness;
    benchConfig.compareCusparse = opts.compareCusparse;

    // Run benchmark
    PerformanceResult result;
    status = measureCSRPerformance(matrix, d_x, d_y, benchConfig, handle, result);

    if (status != SPMV_SUCCESS) {
        fprintf(stderr, "Benchmark failed: %d\n", status);
        cudaFree(d_x);
        cudaFree(d_y);
        delete[] h_x;
        spmv_destroy_handle(handle);
        return 1;
    }

    // Print results
    printPerformanceResult(result);

    // cuSPARSE comparison
    if (opts.compareCusparse) {
        printf("\n=== cuSPARSE Comparison ===\n");
        double cusparseTime, cusparseGflops;
        status = compareWithCusparse(matrix, d_x, d_y, benchConfig,
                                      cusparseTime, cusparseGflops);

        if (status == SPMV_SUCCESS) {
            printf("cuSPARSE Time: %.4f ms\n", cusparseTime);
            printf("cuSPARSE GFLOPS: %.2f\n", cusparseGflops);
            printf("Speedup: %.2fx\n", cusparseTime / result.timeMs);
        } else {
            printf("cuSPARSE comparison failed (not available on this GPU)\n");
        }
    }

    // Write results to JSON
    if (!opts.outputFile.empty()) {
        std::vector<PerformanceResult> results;
        results.push_back(result);
        writeResultsJSON(results, opts.outputFile.c_str());
        printf("\nResults written to: %s\n", opts.outputFile.c_str());
    }

    // Cleanup
    cudaFree(d_x);
    cudaFree(d_y);
    delete[] h_x;
    spmv_destroy_handle(handle);

    printf("\nTest completed successfully.\n");
    return 0;
}

int main(int argc, char** argv) {
    TestOptions opts;

    if (!parseArgs(argc, argv, opts)) {
        return 1;
    }

    // Seed random number generator
    srand(12345);

    // Run with appropriate precision
    if (opts.useDouble) {
        return runTest<double>(opts);
    } else {
        return runTest<float>(opts);
    }
}