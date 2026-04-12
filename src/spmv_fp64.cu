/**
 * @file spmv_fp64.cu
 * @brief Main implementation of FP64 SpMV library
 */

#include "spmv_fp64.h"
#include "spmv_fp64_impl.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>

// ==================== Internal Matrix Structure ====================

struct spmv_fp64_matrix_t {
    int numRows;
    int numCols;
    int nnz;

    // Host data (pinned memory if use_pinned_memory)
    int* h_rowPtr;
    int* h_colIdx;
    double* h_values;

    // Device data
    int* d_rowPtr;
    int* d_colIdx;
    double* d_values;

    // Internal device vectors for execute
    double* d_x_internal;
    double* d_y_internal;
    int ownsInternalVectors;

    // Options
    spmv_fp64_opts_t opts;

    // GPU info
    int warpSize;
    char gpuName[256];
    double theoreticalBW;

    // Memory ownership
    int ownsHostMemory;
    int ownsDeviceMemory;
};

// ==================== Internal Helper Functions ====================

static int load_mtx_file(
    const char* filename,
    int** rowPtr,
    int** colIdx,
    double** values,
    int* numRows,
    int* numCols,
    int* nnz)
{
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "[SPMV_FP64] Error: Cannot open file %s\n", filename);
        return -1;
    }

    // Read header
    char line[256];
    if (!fgets(line, sizeof(line), fp)) {
        fclose(fp);
        return -1;
    }

    // Skip comment lines
    while (line[0] == '%') {
        if (!fgets(line, sizeof(line), fp)) {
            fclose(fp);
            return -1;
        }
    }

    // Parse dimensions
    int r, c, n;
    if (sscanf(line, "%d %d %d", &r, &c, &n) != 3) {
        fclose(fp);
        return -1;
    }

    *numRows = r;
    *numCols = c;
    *nnz = n;

    // Allocate COO arrays
    int* coo_row = (int*)malloc(n * sizeof(int));
    int* coo_col = (int*)malloc(n * sizeof(int));
    double* coo_val = (double*)malloc(n * sizeof(double));

    if (!coo_row || !coo_col || !coo_val) {
        free(coo_row);
        free(coo_col);
        free(coo_val);
        fclose(fp);
        return -1;
    }

    // Read COO data
    for (int i = 0; i < n; i++) {
        double v = 1.0;  // Default value if not provided
        int numRead = fscanf(fp, "%d %d %lf", &coo_row[i], &coo_col[i], &v);
        if (numRead < 2) {
            free(coo_row);
            free(coo_col);
            free(coo_val);
            fclose(fp);
            return -1;
        }
        coo_row[i]--;  // MTX is 1-indexed
        coo_col[i]--;
        if (numRead == 2) {
            coo_val[i] = 1.0;  // Pattern matrix
        } else {
            coo_val[i] = v;
        }
    }
    fclose(fp);

    // Convert COO to CSR
    *rowPtr = (int*)calloc(r + 1, sizeof(int));
    if (!*rowPtr) {
        free(coo_row);
        free(coo_col);
        free(coo_val);
        return -1;
    }

    // Count nnz per row
    for (int i = 0; i < n; i++) {
        (*rowPtr)[coo_row[i] + 1]++;
    }

    // Cumulative sum
    for (int i = 0; i < r; i++) {
        (*rowPtr)[i + 1] += (*rowPtr)[i];
    }

    // Allocate CSR arrays
    *colIdx = (int*)malloc(n * sizeof(int));
    *values = (double*)malloc(n * sizeof(double));

    if (!*colIdx || !*values) {
        free(*rowPtr);
        free(coo_row);
        free(coo_col);
        free(coo_val);
        return -1;
    }

    // Fill CSR arrays
    int* rowCounter = (int*)calloc(r, sizeof(int));
    for (int i = 0; i < n; i++) {
        int ridx = coo_row[i];
        int pos = (*rowPtr)[ridx] + rowCounter[ridx];
        (*colIdx)[pos] = coo_col[i];
        (*values)[pos] = coo_val[i];
        rowCounter[ridx]++;
    }

    free(coo_row);
    free(coo_col);
    free(coo_val);
    free(rowCounter);

    return 0;
}

static void get_gpu_info(spmv_fp64_matrix_handle_t handle) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    handle->warpSize = prop.warpSize;
    strncpy(handle->gpuName, prop.name, sizeof(handle->gpuName) - 1);
    handle->gpuName[sizeof(handle->gpuName) - 1] = '\0';

    // Theoretical bandwidth based on known GPU specifications
    handle->theoreticalBW = 500.0;  // Default estimate

    // Override with known values for common GPUs
    if (strstr(prop.name, "RTX 4090")) {
        handle->theoreticalBW = 1008.0;  // Known RTX 4090: 1008 GB/s
    } else if (strstr(prop.name, "RTX 4080")) {
        handle->theoreticalBW = 736.0;
    } else if (strstr(prop.name, "RTX 3080")) {
        handle->theoreticalBW = 760.0;
    } else if (strstr(prop.name, "Mars")) {
        handle->theoreticalBW = 1843.2;  // Known Mars X201: 1843 GB/s
    }
}

// ==================== Matrix Management API ====================

spmv_fp64_status_t spmv_fp64_create_matrix(
    spmv_fp64_matrix_handle_t* handle,
    int numRows,
    int numCols,
    int nnz,
    const int* rowPtr,
    const int* colIdx,
    const double* values,
    const spmv_fp64_opts_t* opts)
{
    if (!handle || !rowPtr || !colIdx || !values) {
        return SPMV_FP64_ERROR_INVALID_INPUT;
    }

    if (numRows <= 0 || numCols <= 0 || nnz < 0) {
        return SPMV_FP64_ERROR_INVALID_INPUT;
    }

    // Allocate handle
    spmv_fp64_matrix_handle_t mat = (spmv_fp64_matrix_handle_t)
        calloc(1, sizeof(struct spmv_fp64_matrix_t));

    if (!mat) {
        return SPMV_FP64_ERROR_MEMORY;
    }

    // Set options
    if (opts) {
        mat->opts = *opts;
    } else {
        mat->opts = SPMV_FP64_DEFAULT_OPTS;
    }

    mat->numRows = numRows;
    mat->numCols = numCols;
    mat->nnz = nnz;

    // Get GPU info
    get_gpu_info(mat);

    // Allocate pinned host memory for CSR data (always use pinned for best performance)
    cudaMallocHost(&mat->h_rowPtr, (numRows + 1) * sizeof(int));
    cudaMallocHost(&mat->h_colIdx, nnz * sizeof(int));
    cudaMallocHost(&mat->h_values, nnz * sizeof(double));

    if (!mat->h_rowPtr || !mat->h_colIdx || !mat->h_values) {
        spmv_fp64_destroy_matrix(mat);
        return SPMV_FP64_ERROR_MEMORY;
    }

    memcpy(mat->h_rowPtr, rowPtr, (numRows + 1) * sizeof(int));
    memcpy(mat->h_colIdx, colIdx, nnz * sizeof(int));
    memcpy(mat->h_values, values, nnz * sizeof(double));

    mat->ownsHostMemory = 1;

    // Allocate device memory for CSR data
    cudaError_t err = cudaMalloc(&mat->d_rowPtr, (numRows + 1) * sizeof(int));
    if (err != cudaSuccess) {
        spmv_fp64_destroy_matrix(mat);
        return SPMV_FP64_ERROR_CUDA;
    }

    err = cudaMalloc(&mat->d_colIdx, nnz * sizeof(int));
    if (err != cudaSuccess) {
        spmv_fp64_destroy_matrix(mat);
        return SPMV_FP64_ERROR_CUDA;
    }

    err = cudaMalloc(&mat->d_values, nnz * sizeof(double));
    if (err != cudaSuccess) {
        spmv_fp64_destroy_matrix(mat);
        return SPMV_FP64_ERROR_CUDA;
    }

    mat->ownsDeviceMemory = 1;

    // Allocate internal device vectors for execute
    err = cudaMalloc(&mat->d_x_internal, numCols * sizeof(double));
    if (err != cudaSuccess) {
        spmv_fp64_destroy_matrix(mat);
        return SPMV_FP64_ERROR_CUDA;
    }

    err = cudaMalloc(&mat->d_y_internal, numRows * sizeof(double));
    if (err != cudaSuccess) {
        spmv_fp64_destroy_matrix(mat);
        return SPMV_FP64_ERROR_CUDA;
    }

    mat->ownsInternalVectors = 1;

    // Copy CSR data to device
    cudaMemcpy(mat->d_rowPtr, mat->h_rowPtr, (numRows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(mat->d_colIdx, mat->h_colIdx, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(mat->d_values, mat->h_values, nnz * sizeof(double), cudaMemcpyHostToDevice);

    *handle = mat;
    return SPMV_FP64_SUCCESS;
}

spmv_fp64_status_t spmv_fp64_create_matrix_from_file(
    spmv_fp64_matrix_handle_t* handle,
    const char* filename,
    const spmv_fp64_opts_t* opts)
{
    if (!handle || !filename) {
        return SPMV_FP64_ERROR_INVALID_INPUT;
    }

    int* rowPtr = NULL;
    int* colIdx = NULL;
    double* values = NULL;
    int numRows, numCols, nnz;

    if (load_mtx_file(filename, &rowPtr, &colIdx, &values, &numRows, &numCols, &nnz) != 0) {
        return SPMV_FP64_ERROR_INVALID_INPUT;
    }

    spmv_fp64_status_t status = spmv_fp64_create_matrix(
        handle, numRows, numCols, nnz, rowPtr, colIdx, values, opts);

    // Free temporary arrays (create_matrix copied them)
    free(rowPtr);
    free(colIdx);
    free(values);

    return status;
}

spmv_fp64_status_t spmv_fp64_destroy_matrix(spmv_fp64_matrix_handle_t handle) {
    if (!handle) {
        return SPMV_FP64_ERROR_INVALID_INPUT;
    }

    if (handle->ownsInternalVectors) {
        if (handle->d_x_internal) cudaFree(handle->d_x_internal);
        if (handle->d_y_internal) cudaFree(handle->d_y_internal);
    }

    if (handle->ownsDeviceMemory) {
        if (handle->d_rowPtr) cudaFree(handle->d_rowPtr);
        if (handle->d_colIdx) cudaFree(handle->d_colIdx);
        if (handle->d_values) cudaFree(handle->d_values);
    }

    if (handle->ownsHostMemory) {
        if (handle->h_rowPtr) cudaFreeHost(handle->h_rowPtr);
        if (handle->h_colIdx) cudaFreeHost(handle->h_colIdx);
        if (handle->h_values) cudaFreeHost(handle->h_values);
    }

    free(handle);
    return SPMV_FP64_SUCCESS;
}

spmv_fp64_status_t spmv_fp64_get_matrix_info(
    spmv_fp64_matrix_handle_t handle,
    int* numRows,
    int* numCols,
    int* nnz)
{
    if (!handle) {
        return SPMV_FP64_ERROR_INVALID_INPUT;
    }

    if (numRows) *numRows = handle->numRows;
    if (numCols) *numCols = handle->numCols;
    if (nnz) *nnz = handle->nnz;

    return SPMV_FP64_SUCCESS;
}

// ==================== SpMV Execution API ====================

spmv_fp64_status_t spmv_fp64_execute(
    spmv_fp64_matrix_handle_t handle,
    const double* x,
    double* y,
    const spmv_fp64_opts_t* opts,
    spmv_fp64_stats_t* stats)
{
    return spmv_fp64_execute_general(handle, 1.0, 0.0, x, y, opts, stats);
}

spmv_fp64_status_t spmv_fp64_execute_scaled(
    spmv_fp64_matrix_handle_t handle,
    double alpha,
    const double* x,
    double* y,
    const spmv_fp64_opts_t* opts,
    spmv_fp64_stats_t* stats)
{
    return spmv_fp64_execute_general(handle, alpha, 0.0, x, y, opts, stats);
}

spmv_fp64_status_t spmv_fp64_execute_general(
    spmv_fp64_matrix_handle_t handle,
    double alpha,
    double beta,
    const double* x,
    double* y,
    const spmv_fp64_opts_t* opts,
    spmv_fp64_stats_t* stats)
{
    if (!handle || !x || !y) {
        return SPMV_FP64_ERROR_INVALID_INPUT;
    }

    // Use provided options or matrix default
    spmv_fp64_opts_t execOpts = opts ? *opts : handle->opts;
    cudaStream_t stream = execOpts.stream ? execOpts.stream : 0;

    // Copy x to device (using internal buffer)
    cudaMemcpyAsync(handle->d_x_internal, x, handle->numCols * sizeof(double),
                    cudaMemcpyHostToDevice, stream);

    // Launch optimal kernel based on warp size
    if (handle->warpSize == 64) {
        // Mars X201: Use TPR=8 kernel
        spmv_fp64_impl::launch_mars_optimal(
            handle->numRows,
            handle->d_rowPtr,
            handle->d_colIdx,
            handle->d_values,
            handle->d_x_internal,
            handle->d_y_internal,
            stream);
    } else {
        // NVIDIA: Use __ldg kernel
        spmv_fp64_impl::launch_nvidia_optimal(
            handle->numRows,
            handle->d_rowPtr,
            handle->d_colIdx,
            handle->d_values,
            handle->d_x_internal,
            handle->d_y_internal,
            stream);
    }

    // Copy y back to host
    cudaMemcpyAsync(y, handle->d_y_internal, handle->numRows * sizeof(double),
                    cudaMemcpyDeviceToHost, stream);

    // Synchronize if requested
    if (execOpts.sync_after_exec) {
        cudaStreamSynchronize(stream);
    }

    // Benchmark mode: measure performance
    if (execOpts.benchmark_mode && stats) {
        // Run multiple iterations for accurate timing
        const int iterations = 100;

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) {
            cudaMemcpyAsync(handle->d_x_internal, x, handle->numCols * sizeof(double),
                            cudaMemcpyHostToDevice, stream);

            if (handle->warpSize == 64) {
                spmv_fp64_impl::launch_mars_optimal(
                    handle->numRows,
                    handle->d_rowPtr,
                    handle->d_colIdx,
                    handle->d_values,
                    handle->d_x_internal,
                    handle->d_y_internal,
                    stream);
            } else {
                spmv_fp64_impl::launch_nvidia_optimal(
                    handle->numRows,
                    handle->d_rowPtr,
                    handle->d_colIdx,
                    handle->d_values,
                    handle->d_x_internal,
                    handle->d_y_internal,
                    stream);
            }
        }
        cudaStreamSynchronize(stream);
        auto end = std::chrono::high_resolution_clock::now();

        double timeMs = std::chrono::duration<double>(end - start).count() * 1000 / iterations;

        // Fill stats
        stats->kernel_time_ms = timeMs;
        stats->bandwidth_gbps = spmv_fp64_impl::calculate_bandwidth(
            handle->nnz, handle->numRows, timeMs);
        stats->theoretical_bw = handle->theoreticalBW;
        stats->utilization_pct = (stats->bandwidth_gbps / stats->theoretical_bw) * 100.0;
        stats->warp_size = handle->warpSize;
        stats->optimal_tpr = (handle->warpSize == 64) ? 8 : 1;
        stats->gpu_name = handle->gpuName;
    }

    return SPMV_FP64_SUCCESS;
}

// ==================== Utility Functions ====================

spmv_fp64_status_t spmv_fp64_alloc_pinned(void** ptr, size_t size) {
    cudaError_t err = cudaMallocHost(ptr, size);
    if (err != cudaSuccess) {
        return SPMV_FP64_ERROR_CUDA;
    }
    return SPMV_FP64_SUCCESS;
}

spmv_fp64_status_t spmv_fp64_free_pinned(void* ptr) {
    if (!ptr) {
        return SPMV_FP64_ERROR_INVALID_INPUT;
    }
    cudaError_t err = cudaFreeHost(ptr);
    if (err != cudaSuccess) {
        return SPMV_FP64_ERROR_CUDA;
    }
    return SPMV_FP64_SUCCESS;
}

spmv_fp64_status_t spmv_fp64_get_device_info(
    int* warpSize,
    const char** name,
    size_t* memory)
{
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        return SPMV_FP64_ERROR_CUDA;
    }

    if (warpSize) *warpSize = prop.warpSize;
    if (name) *name = prop.name;
    if (memory) *memory = prop.totalGlobalMem;

    return SPMV_FP64_SUCCESS;
}

spmv_fp64_status_t spmv_fp64_get_theoretical_bandwidth(double* bandwidth) {
    if (!bandwidth) {
        return SPMV_FP64_ERROR_INVALID_INPUT;
    }

    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        return SPMV_FP64_ERROR_CUDA;
    }

    // Theoretical bandwidth based on known GPU specifications
    *bandwidth = 500.0;  // Default estimate

    // Override with known values for common GPUs
    if (strstr(prop.name, "RTX 4090")) {
        *bandwidth = 1008.0;
    } else if (strstr(prop.name, "RTX 4080")) {
        *bandwidth = 736.0;
    } else if (strstr(prop.name, "RTX 3080")) {
        *bandwidth = 760.0;
    } else if (strstr(prop.name, "Mars")) {
        *bandwidth = 1843.2;
    }

    return SPMV_FP64_SUCCESS;
}

const char* spmv_fp64_get_error_string(spmv_fp64_status_t status) {
    switch (status) {
        case SPMV_FP64_SUCCESS:
            return "Success";
        case SPMV_FP64_ERROR_INVALID_INPUT:
            return "Invalid input parameters";
        case SPMV_FP64_ERROR_MEMORY:
            return "Memory allocation/deallocation error";
        case SPMV_FP64_ERROR_CUDA:
            return "CUDA runtime error";
        case SPMV_FP64_ERROR_NOT_SUPPORTED:
            return "Feature not supported on this GPU";
        case SPMV_FP64_ERROR_INTERNAL:
            return "Internal library error";
        default:
            return "Unknown error";
    }
}

const char* spmv_fp64_get_version(void) {
    return SPMV_FP64_VERSION_STRING;
}