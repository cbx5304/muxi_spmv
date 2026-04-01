/**
 * @file mtx_io.h
 * @brief Matrix Market (MTX) file format I/O utilities
 *
 * Matrix Market format is a standard format for sparse matrices
 * Used by the SuiteSparse matrix collection and many scientific applications
 */

#ifndef MTX_IO_H_
#define MTX_IO_H_

#include "utils/common.h"
#include "formats/sparse_formats.h"
#include <cstdio>
#include <cstring>

namespace muxi_spmv {
namespace io {

/**
 * @brief MTX matrix properties
 */
struct MTXHeader {
    bool isSymmetric;     ///< Symmetric matrix
    bool isComplex;       ///< Complex values (not supported yet)
    bool isPattern;       ///< Pattern matrix (no values)
    bool isInteger;       ///< Integer values
    int numRows;          ///< Number of rows
    int numCols;          ///< Number of columns
    int nnz;              ///< Number of non-zeros (or entries in file)
};

/**
 * @brief Read MTX file header
 * @param filename Path to MTX file
 * @param header Output header structure
 * @return Status code
 */
spmv_status_t readMTXHeader(const char* filename, MTXHeader& header);

/**
 * @brief Read MTX file into CSR matrix
 * @tparam FloatType Floating point type
 * @param filename Path to MTX file
 * @param matrix Output CSR matrix
 * @param expandSymmetric Whether to expand symmetric matrices to full storage
 * @return Status code
 */
template<typename FloatType>
spmv_status_t readMTXFile(const char* filename, CSRMatrix<FloatType>& matrix,
                           bool expandSymmetric = true);

/**
 * @brief Write CSR matrix to MTX file
 * @tparam FloatType Floating point type
 * @param filename Output file path
 * @param matrix CSR matrix to write
 * @param writeSymmetric Whether to write as symmetric (if matrix is symmetric)
 * @return Status code
 */
template<typename FloatType>
spmv_status_t writeMTXFile(const char* filename, const CSRMatrix<FloatType>& matrix,
                            bool writeSymmetric = false);

/**
 * @brief Read MTX file into COO matrix (simpler, no CSR conversion)
 * @tparam FloatType Floating point type
 * @param filename Path to MTX file
 * @param matrix Output COO matrix
 * @return Status code
 */
template<typename FloatType>
spmv_status_t readMTXToCOO(const char* filename, COOMatrix<FloatType>& matrix);

/**
 * @brief Write COO matrix to MTX file
 * @tparam FloatType Floating point type
 * @param filename Output file path
 * @param matrix COO matrix to write
 * @return Status code
 */
template<typename FloatType>
spmv_status_t writeCOOToMTX(const char* filename, const COOMatrix<FloatType>& matrix);

} // namespace io
} // namespace muxi_spmv

#endif // MTX_IO_H_