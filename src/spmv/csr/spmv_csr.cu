/**
 * @file spmv_csr.cu
 * @brief CSR format SpMV kernel implementations
 */

#include "spmv_csr.cuh"
#include <cstdio>

namespace muxi_spmv {

// Get optimal kernel configuration
CSRKernelConfig getOptimalConfig(int numRows, int nnz, int avgNnzPerRow, int warpSize) {
    CSRKernelConfig config;

    // Determine block size based on warp size
    if (warpSize == 64) {
        // Domestic GPU with 64-thread warp
        config.blockSize = 256;  // 4 warps per block (increased for better occupancy)
        config.itemsPerThread = 4;
    } else {
        // NVIDIA GPU with 32-thread warp
        config.blockSize = 256;  // 8 warps per block
        config.itemsPerThread = 4;
    }

    // Decide whether to use shared memory
    config.useSharedMem = (avgNnzPerRow > 8) && (avgNnzPerRow < 1000);

    // Calculate shared memory size if needed
    if (config.useSharedMem) {
        int rowsPerBlock = config.blockSize / warpSize;
        config.rowsPerBlock = rowsPerBlock;
        config.sharedMemSize = (rowsPerBlock + 1) * sizeof(int);
    } else {
        config.rowsPerBlock = 0;
        config.sharedMemSize = 0;
    }

    return config;
}

// Helper function to get grid size
static int getGridSize(int numWorkItems, int blockSize) {
    int gridSize = (numWorkItems + blockSize - 1) / blockSize;
    // Limit grid size to avoid too many blocks
    int maxGridSize = 65535;
    return min(gridSize, maxGridSize);
}

// Template instantiation for float
template<>
spmv_status_t spmv_csr<float>(
    const CSRMatrix<float>& matrix,
    const float* x,
    float* y,
    float alpha,
    float beta,
    const spmv_opts_t& opts)
{
    if (matrix.nnz == 0) {
        return SPMV_SUCCESS;
    }

#if SPMV_ENABLE_CHECKS
    if (!matrix.d_rowPtr || !matrix.d_colIdx || !matrix.d_values) {
        return SPMV_ERROR_INVALID_MATRIX;
    }
    if (!x || !y) {
        return SPMV_ERROR_INVALID_VECTOR;
    }
#endif

    cudaStream_t stream = opts.stream ? (cudaStream_t)opts.stream : 0;

    // Scale y by beta if needed
    if (beta != 1.0f) {
        int blockSize = 256;
        int gridSize = getGridSize(matrix.numRows, blockSize);
        // TODO: Add scale kernel
    }

    // Get optimal configuration
    int avgNnzPerRow = matrix.nnz / max(matrix.numRows, 1);
    CSRKernelConfig config = getOptimalConfig(
        matrix.numRows, matrix.nnz, avgNnzPerRow, WARP_SIZE);

    if (WARP_SIZE == 64) {
        // Domestic GPU kernel - ALWAYS use vector kernel for warp=64
        // Scalar kernel is extremely inefficient on warp=64 architecture
        int blockSize = config.blockSize;
        // For vector kernel: each warp processes one row
        // Each block has (blockSize / warpSize) warps
        int warpsPerBlock = blockSize / WARP_SIZE;
        int gridSize = getGridSize(numRows, warpsPerBlock);

        if (opts.operation == SPMV_OP_TRANSPOSE) {
            spmv_csr_scalar_kernel<float, true><<<gridSize, blockSize, 0, stream>>>(
                matrix.numRows, matrix.numCols, matrix.nnz,
                matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values,
                x, y);
        } else {
            // Always use vector kernel for warp=64
            // Even for very sparse rows, vector kernel outperforms scalar
            // because it utilizes all 64 threads in a warp
            spmv_csr_vector_kernel<float, 64, false><<<gridSize, blockSize, 0, stream>>>(
                matrix.numRows, matrix.numCols, matrix.nnz,
                matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values,
                x, y);
        }
    } else {
        // NVIDIA GPU kernel
        int blockSize = config.blockSize;
        int gridSize = getGridSize(
            (matrix.numRows * 32 + blockSize - 1) / 32, blockSize);

        if (opts.operation == SPMV_OP_TRANSPOSE) {
            spmv_csr_scalar_kernel<float, true><<<gridSize, blockSize, 0, stream>>>(
                matrix.numRows, matrix.numCols, matrix.nnz,
                matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values,
                x, y);
        } else {
            if (avgNnzPerRow < 32) {
                spmv_csr_scalar_kernel<float, false><<<gridSize, blockSize, 0, stream>>>(
                    matrix.numRows, matrix.numCols, matrix.nnz,
                    matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values,
                    x, y);
            } else {
                spmv_csr_vector_kernel<float, 32, false><<<gridSize, blockSize, 0, stream>>>(
                    matrix.numRows, matrix.numCols, matrix.nnz,
                    matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values,
                    x, y);
            }
        }
    }

    if (opts.sync) {
        cudaStreamSynchronize(stream);
    }

    return SPMV_SUCCESS;
}

// Template instantiation for double
template<>
spmv_status_t spmv_csr<double>(
    const CSRMatrix<double>& matrix,
    const double* x,
    double* y,
    double alpha,
    double beta,
    const spmv_opts_t& opts)
{
    if (matrix.nnz == 0) {
        return SPMV_SUCCESS;
    }

#if SPMV_ENABLE_CHECKS
    if (!matrix.d_rowPtr || !matrix.d_colIdx || !matrix.d_values) {
        return SPMV_ERROR_INVALID_MATRIX;
    }
    if (!x || !y) {
        return SPMV_ERROR_INVALID_VECTOR;
    }
#endif

    cudaStream_t stream = opts.stream ? (cudaStream_t)opts.stream : 0;

    int avgNnzPerRow = matrix.nnz / max(matrix.numRows, 1);
    CSRKernelConfig config = getOptimalConfig(
        matrix.numRows, matrix.nnz, avgNnzPerRow, WARP_SIZE);

    if (WARP_SIZE == 64) {
        int blockSize = config.blockSize;
        // For vector kernel: each warp processes one row
        int warpsPerBlock = blockSize / WARP_SIZE;
        int gridSize = getGridSize(matrix.numRows, warpsPerBlock);

        if (opts.operation == SPMV_OP_TRANSPOSE) {
            spmv_csr_scalar_kernel<double, true><<<gridSize, blockSize, 0, stream>>>(
                matrix.numRows, matrix.numCols, matrix.nnz,
                matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values,
                x, y);
        } else {
            // Always use vector kernel for warp=64
            spmv_csr_vector_kernel<double, 64, false><<<gridSize, blockSize, 0, stream>>>(
                matrix.numRows, matrix.numCols, matrix.nnz,
                matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values,
                x, y);
        }
    } else {
        int blockSize = config.blockSize;
        int gridSize = getGridSize(
            (matrix.numRows * 32 + blockSize - 1) / 32, blockSize);

        if (opts.operation == SPMV_OP_TRANSPOSE) {
            spmv_csr_scalar_kernel<double, true><<<gridSize, blockSize, 0, stream>>>(
                matrix.numRows, matrix.numCols, matrix.nnz,
                matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values,
                x, y);
        } else {
            if (avgNnzPerRow < 32) {
                spmv_csr_scalar_kernel<double, false><<<gridSize, blockSize, 0, stream>>>(
                    matrix.numRows, matrix.numCols, matrix.nnz,
                    matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values,
                    x, y);
            } else {
                spmv_csr_vector_kernel<double, 32, false><<<gridSize, blockSize, 0, stream>>>(
                    matrix.numRows, matrix.numCols, matrix.nnz,
                    matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values,
                    x, y);
            }
        }
    }

    if (opts.sync) {
        cudaStreamSynchronize(stream);
    }

    return SPMV_SUCCESS;
}

} // namespace muxi_spmv