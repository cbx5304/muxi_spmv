/**
 * @file common.h
 * @brief Common definitions and utilities for SpMV library
 */

#ifndef COMMON_H_
#define COMMON_H_

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

// Warp size configuration
#ifndef WARP_SIZE
#define WARP_SIZE 32  // Default to NVIDIA warp size
#endif

// Compile-time warp size check for domestic GPU
#if WARP_SIZE == 64
#define WARP_SIZE_64 1
#else
#define WARP_SIZE_32 1
#endif

// Error checking configuration
#ifndef SPMV_ENABLE_CHECKS
#define SPMV_ENABLE_CHECKS 0
#endif

// Status codes
typedef enum {
    SPMV_SUCCESS = 0,
    SPMV_ERROR_INVALID_HANDLE = -1,
    SPMV_ERROR_INVALID_MATRIX = -2,
    SPMV_ERROR_INVALID_VECTOR = -3,
    SPMV_ERROR_MEMORY_ALLOC = -4,
    SPMV_ERROR_MEMORY_COPY = -5,
    SPMV_ERROR_DEVICE = -6,
    SPMV_ERROR_UNSUPPORTED_FORMAT = -7,
    SPMV_ERROR_UNSUPPORTED_TYPE = -8,
    SPMV_ERROR_INTERNAL = -9
} spmv_status_t;

// Sparse matrix format types
typedef enum {
    SPMV_FORMAT_CSR = 0,
    SPMV_FORMAT_COO = 1,
    SPMV_FORMAT_CSR2 = 2,  // Tensor Core optimized format
    SPMV_FORMAT_BSR = 3,
    SPMV_FORMAT_CSR5 = 4,  // CSR5 load-balanced format
    SPMV_FORMAT_UNKNOWN = -1
} spmv_format_t;

// Floating point type tags
typedef enum {
    SPMV_TYPE_FLOAT = 0,
    SPMV_TYPE_DOUBLE = 1,
    SPMV_TYPE_HALF = 2,
    SPMV_TYPE_BF16 = 3,
    SPMV_TYPE_TF32 = 4,
    SPMV_TYPE_UNKNOWN = -1
} spmv_type_t;

// Operation types
typedef enum {
    SPMV_OP_NORMAL = 0,     // y = A * x
    SPMV_OP_TRANSPOSE = 1,  // y = A^T * x
    SPMV_OP_CONJUGATE = 2   // y = A^H * x (for complex types)
} spmv_operation_t;

// Execution options
typedef struct {
    spmv_operation_t operation;
    int sync;              // 0 = async, 1 = sync after operation
    int check_inputs;      // Override global check setting
    int use_tensor_core;   // Use Tensor Core if available
    int use_streaming;     // Use streaming for large matrices
    int stream_id;         // CUDA stream to use
    void* stream;          // Custom CUDA stream
} spmv_opts_t;

// Default options
inline spmv_opts_t spmv_default_opts() {
    spmv_opts_t opts;
    opts.operation = SPMV_OP_NORMAL;
    opts.sync = 0;
    opts.check_inputs = SPMV_ENABLE_CHECKS;
    opts.use_tensor_core = 0;
    opts.use_streaming = 0;
    opts.stream_id = 0;
    opts.stream = nullptr;
    return opts;
}

// Error checking macros
#define SPMV_CHECK(call)                                                       \
    do {                                                                       \
        spmv_status_t err = call;                                              \
        if (err != SPMV_SUCCESS) {                                             \
            fprintf(stderr, "SpMV Error at %s:%d - %d\n",                      \
                    __FILE__, __LINE__, err);                                  \
            return err;                                                        \
        }                                                                      \
    } while (0)

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            return SPMV_ERROR_DEVICE;                                          \
        }                                                                      \
    } while (0)

#define CUDA_CHECK_NO_RETURN(call)                                             \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
        }                                                                      \
    } while (0)

// Kernel launch helpers
#define SPMV_KERNEL_LAUNCH(kernel, grid, block, shared_mem, stream, ...)      \
    do {                                                                       \
        kernel<<<grid, block, shared_mem, stream>>>(__VA_ARGS__);              \
        cudaError_t err = cudaGetLastError();                                  \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "Kernel launch error at %s:%d - %s\n",             \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            return SPMV_ERROR_DEVICE;                                          \
        }                                                                      \
    } while (0)

// Utility functions
inline int div_round_up(int a, int b) {
    return (a + b - 1) / b;
}

inline int align_up(int a, int alignment) {
    return (a + alignment - 1) / alignment * alignment;
}

// Round up to nearest power of 2
inline int round_up_pow2(int n) {
    int p = 1;
    while (p < n) p *= 2;
    return p;
}

// Get next power of 2 (useful for shared memory sizing)
inline int next_pow2(int n) {
    if (n <= 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

#endif // COMMON_H_