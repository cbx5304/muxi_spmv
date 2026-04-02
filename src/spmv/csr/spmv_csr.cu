/**
 * @file spmv_csr.cu
 * @brief CSR format SpMV kernel implementations
 */

#include "spmv_csr.cuh"
#include "../csr5/spmv_csr5.cuh"  // For merge-based kernel
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
        // Domestic GPU kernel - warp size = 64
        // Use adaptive strategy based on matrix sparsity
        int blockSize = config.blockSize;
        int warpsPerBlock = blockSize / WARP_SIZE;

        if (opts.operation == SPMV_OP_TRANSPOSE) {
            int gridSize = getGridSize(matrix.numRows, blockSize);
            spmv_csr_scalar_kernel<float, true><<<gridSize, blockSize, 0, stream>>>(
                matrix.numRows, matrix.numCols, matrix.nnz,
                matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values,
                x, y);
        } else {
            // Mars X201 optimization: Adaptive kernel selection based on sparsity
            //
            // Key findings from benchmark tests (1M rows, 1K cols):
            // - avgNnz=10: merge-based 14.9% vs scalar 9.8% (+52% improvement)
            // - avgNnz=16: merge-based 19.7% vs light-balanced 9.6% (+105% improvement)
            // - avgNnz=24: merge-based 24.7% vs light-balanced 10.7% (+131% improvement)
            // - avgNnz=28: merge-based 24.8% vs light-balanced 10.8% (+130% improvement)
            // - avgNnz=32: vector 37.0% vs merge-based 25.7% (+44% vector better)
            // - avgNnz=64: vector 52.2% vs merge-based 44.8% (+16% vector better)
            //
            // Strategy:
            // - avgNnz < 32: Use merge-based (better for sparse matrices on warp=64)
            // - avgNnz >= 32: Use vector kernel (optimal for denser matrices)

            if (avgNnzPerRow < 32) {
                // Sparse matrices: Use merge-based kernel
                // Merge-based divides work along merge path, providing better load balancing
                // for sparse matrices on warp=64 architecture
                spmv_merge_based<float>(matrix, x, y, stream);
            } else {
                // Denser matrices (avgNnz >= 32): vector kernel (1 warp per row)
                // Warp utilization is at least 32/64 = 50%
                int gridSize = getGridSize(matrix.numRows, warpsPerBlock);
                spmv_csr_vector_kernel<float, 64, false><<<gridSize, blockSize, 0, stream>>>(
                    matrix.numRows, matrix.numCols, matrix.nnz,
                    matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values,
                    x, y);
            }
        }
    } else {
        // NVIDIA GPU kernel - warp size = 32
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
        // Domestic GPU kernel - warp size = 64
        int blockSize = config.blockSize;
        int warpsPerBlock = blockSize / WARP_SIZE;

        if (opts.operation == SPMV_OP_TRANSPOSE) {
            int gridSize = getGridSize(matrix.numRows, blockSize);
            spmv_csr_scalar_kernel<double, true><<<gridSize, blockSize, 0, stream>>>(
                matrix.numRows, matrix.numCols, matrix.nnz,
                matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values,
                x, y);
        } else {
            // Mars X201 optimization: Adaptive kernel selection based on sparsity
            // Same strategy as float - use merge-based for sparse matrices
            if (avgNnzPerRow < 32) {
                // Sparse matrices: Use merge-based kernel
                spmv_merge_based<double>(matrix, x, y, stream);
            } else {
                // Denser matrices (avgNnz >= 32): vector kernel
                int gridSize = getGridSize(matrix.numRows, warpsPerBlock);
                spmv_csr_vector_kernel<double, 64, false><<<gridSize, blockSize, 0, stream>>>(
                    matrix.numRows, matrix.numCols, matrix.nnz,
                    matrix.d_rowPtr, matrix.d_colIdx, matrix.d_values,
                    x, y);
            }
        }
    } else {
        // NVIDIA GPU kernel - warp size = 32
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