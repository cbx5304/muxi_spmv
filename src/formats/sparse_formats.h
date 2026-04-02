/**
 * @file sparse_formats.h
 * @brief Sparse matrix format definitions for SpMV library
 *
 * Supports: CSR, COO, CSR2 (Tensor Core), BSR formats
 */

#ifndef SPARSE_FORMATS_H_
#define SPARSE_FORMATS_H_

#include "utils/common.h"
#include <cuda_runtime.h>

namespace muxi_spmv {

/**
 * @brief CSR (Compressed Sparse Row) format
 *
 * Memory layout:
 * - rowPtr: Array of size numRows + 1, pointing to start of each row in colIdx/values
 * - colIdx: Array of size nnz, column indices for each non-zero element
 * - values: Array of size nnz, values of each non-zero element
 */
template<typename FloatType>
struct CSRMatrix {
    int numRows;           ///< Number of rows
    int numCols;           ///< Number of columns
    int nnz;               ///< Number of non-zero elements
    int* rowPtr;           ///< Row pointers (size: numRows + 1)
    int* colIdx;           ///< Column indices (size: nnz)
    FloatType* values;     ///< Values (size: nnz)

    // Device pointers (for GPU execution)
    int* d_rowPtr;
    int* d_colIdx;
    FloatType* d_values;

    // Memory ownership flags
    bool ownsHostMemory;
    bool ownsDeviceMemory;

    // Default constructor
    CSRMatrix() : numRows(0), numCols(0), nnz(0),
                  rowPtr(nullptr), colIdx(nullptr), values(nullptr),
                  d_rowPtr(nullptr), d_colIdx(nullptr), d_values(nullptr),
                  ownsHostMemory(false), ownsDeviceMemory(false) {}

    // Destructor - cleanup memory if owned
    ~CSRMatrix() {
        if (ownsHostMemory) {
            free(rowPtr);
            free(colIdx);
            free(values);
        }
        if (ownsDeviceMemory) {
            CUDA_CHECK_NO_RETURN(cudaFree(d_rowPtr));
            CUDA_CHECK_NO_RETURN(cudaFree(d_colIdx));
            CUDA_CHECK_NO_RETURN(cudaFree(d_values));
        }
    }

    // Allocate host memory
    void allocateHost(int rows, int cols, int nonzeros) {
        numRows = rows;
        numCols = cols;
        nnz = nonzeros;

        rowPtr = (int*)malloc((numRows + 1) * sizeof(int));
        colIdx = (int*)malloc(nnz * sizeof(int));
        values = (FloatType*)malloc(nnz * sizeof(FloatType));

        ownsHostMemory = true;
    }

    // Allocate device memory
    void allocateDevice() {
        CUDA_CHECK_NO_RETURN(cudaMalloc(&d_rowPtr, (numRows + 1) * sizeof(int)));
        CUDA_CHECK_NO_RETURN(cudaMalloc(&d_colIdx, nnz * sizeof(int)));
        CUDA_CHECK_NO_RETURN(cudaMalloc(&d_values, nnz * sizeof(FloatType)));

        ownsDeviceMemory = true;
    }

    // Copy to device
    void copyToDevice(cudaStream_t stream = 0) {
        cudaMemcpyAsync(d_rowPtr, rowPtr, (numRows + 1) * sizeof(int),
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_colIdx, colIdx, nnz * sizeof(int),
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_values, values, nnz * sizeof(FloatType),
                        cudaMemcpyHostToDevice, stream);
    }

    // Copy from device
    void copyFromDevice(cudaStream_t stream = 0) {
        cudaMemcpyAsync(rowPtr, d_rowPtr, (numRows + 1) * sizeof(int),
                        cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(colIdx, d_colIdx, nnz * sizeof(int),
                        cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(values, d_values, nnz * sizeof(FloatType),
                        cudaMemcpyDeviceToHost, stream);
    }
};

/**
 * @brief COO (Coordinate) format
 *
 * Memory layout:
 * - rowIdx: Array of size nnz, row indices for each non-zero element
 * - colIdx: Array of size nnz, column indices for each non-zero element
 * - values: Array of size nnz, values of each non-zero element
 */
template<typename FloatType>
struct COOMatrix {
    int numRows;           ///< Number of rows
    int numCols;           ///< Number of columns
    int nnz;               ///< Number of non-zero elements
    int* rowIdx;           ///< Row indices (size: nnz)
    int* colIdx;           ///< Column indices (size: nnz)
    FloatType* values;     ///< Values (size: nnz)

    // Device pointers
    int* d_rowIdx;
    int* d_colIdx;
    FloatType* d_values;

    // Memory ownership flags
    bool ownsHostMemory;
    bool ownsDeviceMemory;

    COOMatrix() : numRows(0), numCols(0), nnz(0),
                  rowIdx(nullptr), colIdx(nullptr), values(nullptr),
                  d_rowIdx(nullptr), d_colIdx(nullptr), d_values(nullptr),
                  ownsHostMemory(false), ownsDeviceMemory(false) {}

    ~COOMatrix() {
        if (ownsHostMemory) {
            free(rowIdx);
            free(colIdx);
            free(values);
        }
        if (ownsDeviceMemory) {
            CUDA_CHECK_NO_RETURN(cudaFree(d_rowIdx));
            CUDA_CHECK_NO_RETURN(cudaFree(d_colIdx));
            CUDA_CHECK_NO_RETURN(cudaFree(d_values));
        }
    }

    // Allocate host memory
    void allocateHost(int rows, int cols, int nonzeros) {
        numRows = rows;
        numCols = cols;
        nnz = nonzeros;

        rowIdx = (int*)malloc(nnz * sizeof(int));
        colIdx = (int*)malloc(nnz * sizeof(int));
        values = (FloatType*)malloc(nnz * sizeof(FloatType));

        ownsHostMemory = true;
    }

    // Allocate device memory
    void allocateDevice() {
        CUDA_CHECK_NO_RETURN(cudaMalloc(&d_rowIdx, nnz * sizeof(int)));
        CUDA_CHECK_NO_RETURN(cudaMalloc(&d_colIdx, nnz * sizeof(int)));
        CUDA_CHECK_NO_RETURN(cudaMalloc(&d_values, nnz * sizeof(FloatType)));

        ownsDeviceMemory = true;
    }
};

/**
 * @brief CSR2 format for Tensor Core optimization
 *
 * Special format designed for sparse matrix multiplication using Tensor Cores.
 * Requires the matrix to have a 2:4 sparsity pattern (every 4 consecutive elements
 * have at least 2 zeros).
 *
 * Format details:
 * - Compressed row pointers similar to CSR
 * - Values stored in 2:4 sparse format
 * - Metadata for sparsity pattern
 */
template<typename FloatType>
struct CSR2Matrix {
    int numRows;           ///< Number of rows
    int numCols;           ///< Number of columns
    int nnz;               ///< Number of non-zero elements
    int* rowPtr;           ///< Row pointers
    FloatType* values;     ///< Compressed values in 2:4 format
    uint8_t* metadata;     ///< Sparsity pattern metadata

    // Device pointers
    int* d_rowPtr;
    FloatType* d_values;
    uint8_t* d_metadata;

    bool ownsHostMemory;
    bool ownsDeviceMemory;

    CSR2Matrix() : numRows(0), numCols(0), nnz(0),
                   rowPtr(nullptr), values(nullptr), metadata(nullptr),
                   d_rowPtr(nullptr), d_values(nullptr), d_metadata(nullptr),
                   ownsHostMemory(false), ownsDeviceMemory(false) {}

    ~CSR2Matrix() {
        if (ownsHostMemory) {
            free(rowPtr);
            free(values);
            free(metadata);
        }
        if (ownsDeviceMemory) {
            CUDA_CHECK_NO_RETURN(cudaFree(d_rowPtr));
            CUDA_CHECK_NO_RETURN(cudaFree(d_values));
            CUDA_CHECK_NO_RETURN(cudaFree(d_metadata));
        }
    }
};

/**
 * @brief BSR (Block Sparse Row) format
 *
 * Memory layout:
 * - rowPtr: Array of size numBlockRows + 1
 * - colIdx: Array of size numBlocks, block column indices
 * - values: Array of size numBlocks * blockSize^2, dense block values
 *
 * blockSize: Typically 2, 3, 4, 5, or larger for structured sparsity
 */
template<typename FloatType, int BlockSize>
struct BSRMatrix {
    int numRows;           ///< Number of rows (actual)
    int numCols;           ///< Number of columns (actual)
    int numBlockRows;      ///< Number of block rows
    int numBlockCols;      ///< Number of block columns
    int numBlocks;         ///< Number of non-zero blocks
    int blockSize;         ///< Block size (constant BlockSize)

    int* rowPtr;           ///< Block row pointers
    int* colIdx;           ///< Block column indices
    FloatType* values;     ///< Dense block values

    // Device pointers
    int* d_rowPtr;
    int* d_colIdx;
    FloatType* d_values;

    bool ownsHostMemory;
    bool ownsDeviceMemory;

    BSRMatrix() : numRows(0), numCols(0), numBlockRows(0), numBlockCols(0),
                  numBlocks(0), blockSize(BlockSize),
                  rowPtr(nullptr), colIdx(nullptr), values(nullptr),
                  d_rowPtr(nullptr), d_colIdx(nullptr), d_values(nullptr),
                  ownsHostMemory(false), ownsDeviceMemory(false) {}

    ~BSRMatrix() {
        if (ownsHostMemory) {
            free(rowPtr);
            free(colIdx);
            free(values);
        }
        if (ownsDeviceMemory) {
            CUDA_CHECK_NO_RETURN(cudaFree(d_rowPtr));
            CUDA_CHECK_NO_RETURN(cudaFree(d_colIdx));
            CUDA_CHECK_NO_RETURN(cudaFree(d_values));
        }
    }
};

/**
 * @brief CSR5 format for load-balanced SpMV
 *
 * CSR5 divides the matrix into tiles of fixed NNZ count (sigma).
 * Each tile is processed by one warp, eliminating thread idle time
 * for matrices with varying row lengths.
 *
 * Key parameters:
 * - sigma: Tile size (number of NNZ elements per tile, typically 256)
 * - numTiles: Number of tiles = ceil(nnz / sigma)
 * - tile_row_ptr: Precomputed row index for each tile start
 * - tile_nnz_offset: NNZ offset within start row for each tile
 */
template<typename FloatType>
struct CSR5Matrix {
    // Base CSR data (shared with original matrix)
    int numRows;           ///< Number of rows
    int numCols;           ///< Number of columns
    int nnz;               ///< Number of non-zero elements

    // CSR arrays (can reference external CSR matrix)
    int* rowPtr;           ///< Row pointers (size: numRows + 1)
    int* colIdx;           ///< Column indices (size: nnz)
    FloatType* values;     ///< Values (size: nnz)

    // CSR5-specific tile metadata
    int sigma;             ///< Tile size (NNZ per tile, typically 256)
    int numTiles;          ///< Number of tiles = ceil(nnz / sigma)
    int* tile_row_ptr;     ///< Row index for each tile start (size: numTiles)
    int* tile_nnz_offset;  ///< NNZ offset within start row (size: numTiles)

    // Device pointers
    int* d_rowPtr;
    int* d_colIdx;
    FloatType* d_values;
    int* d_tile_row_ptr;
    int* d_tile_nnz_offset;

    // Memory ownership flags
    bool ownsCsrData;      ///< Whether this struct owns CSR arrays
    bool ownsTileData;     ///< Whether this struct owns tile metadata
    bool ownsDeviceMemory;

    // Conversion timing info
    double conversionTimeMs; ///< Time spent in CSR->CSR5 conversion (ms)

    // Default constructor
    CSR5Matrix() : numRows(0), numCols(0), nnz(0),
                   rowPtr(nullptr), colIdx(nullptr), values(nullptr),
                   sigma(256), numTiles(0),
                   tile_row_ptr(nullptr), tile_nnz_offset(nullptr),
                   d_rowPtr(nullptr), d_colIdx(nullptr), d_values(nullptr),
                   d_tile_row_ptr(nullptr), d_tile_nnz_offset(nullptr),
                   ownsCsrData(false), ownsTileData(false),
                   ownsDeviceMemory(false),
                   conversionTimeMs(0.0) {}

    // Destructor
    ~CSR5Matrix() {
        if (ownsCsrData) {
            free(rowPtr);
            free(colIdx);
            free(values);
        }
        if (ownsTileData) {
            free(tile_row_ptr);
            free(tile_nnz_offset);
        }
        if (ownsDeviceMemory) {
            CUDA_CHECK_NO_RETURN(cudaFree(d_tile_row_ptr));
            CUDA_CHECK_NO_RETURN(cudaFree(d_tile_nnz_offset));
            if (ownsCsrData) {
                CUDA_CHECK_NO_RETURN(cudaFree(d_rowPtr));
                CUDA_CHECK_NO_RETURN(cudaFree(d_colIdx));
                CUDA_CHECK_NO_RETURN(cudaFree(d_values));
            }
        }
    }

    // Reference CSR data from existing CSRMatrix (no copy)
    void referenceFromCSR(const CSRMatrix<FloatType>& csr) {
        numRows = csr.numRows;
        numCols = csr.numCols;
        nnz = csr.nnz;
        rowPtr = csr.rowPtr;
        colIdx = csr.colIdx;
        values = csr.values;
        d_rowPtr = csr.d_rowPtr;
        d_colIdx = csr.d_colIdx;
        d_values = csr.d_values;
        ownsCsrData = false;
    }

    // Allocate tile metadata (host)
    void allocateTileMetadata(int tileCount) {
        numTiles = tileCount;
        tile_row_ptr = (int*)malloc(numTiles * sizeof(int));
        tile_nnz_offset = (int*)malloc(numTiles * sizeof(int));
        ownsTileData = true;
    }

    // Allocate tile metadata (device)
    void allocateDeviceTileMetadata() {
        CUDA_CHECK_NO_RETURN(cudaMalloc(&d_tile_row_ptr, numTiles * sizeof(int)));
        CUDA_CHECK_NO_RETURN(cudaMalloc(&d_tile_nnz_offset, numTiles * sizeof(int)));
        ownsDeviceMemory = true;
    }

    // Copy tile metadata to device
    void copyTileMetadataToDevice(cudaStream_t stream = 0) {
        cudaMemcpyAsync(d_tile_row_ptr, tile_row_ptr,
                        numTiles * sizeof(int), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_tile_nnz_offset, tile_nnz_offset,
                        numTiles * sizeof(int), cudaMemcpyHostToDevice, stream);
    }
};

// Format conversion utilities
template<typename FloatType>
CSRMatrix<FloatType> coo_to_csr(const COOMatrix<FloatType>& coo);

template<typename FloatType>
COOMatrix<FloatType> csr_to_coo(const CSRMatrix<FloatType>& csr);

template<typename FloatType, int BlockSize>
CSRMatrix<FloatType> bsr_to_csr(const BSRMatrix<FloatType, BlockSize>& bsr);

// CSR5 conversion
template<typename FloatType>
spmv_status_t convertCSRToCSR5(
    const CSRMatrix<FloatType>& csr,
    CSR5Matrix<FloatType>& csr5,
    int sigma = 256,
    cudaStream_t stream = 0);

} // namespace muxi_spmv

#endif // SPARSE_FORMATS_H_