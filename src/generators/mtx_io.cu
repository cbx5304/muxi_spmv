/**
 * @file mtx_io.cu
 * @brief Implementation of Matrix Market file I/O
 */

#include "generators/mtx_io.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>

namespace muxi_spmv {
namespace io {

// Helper to parse MTX banner line
static bool parseBanner(const char* line, MTXHeader& header) {
    // MTX format: %%MatrixMarket matrix coordinate <type> <symmetry>
    // Example: %%MatrixMarket matrix coordinate real general

    if (strncmp(line, "%%MatrixMarket", 13) != 0) {
        return false;
    }

    header.isComplex = false;
    header.isPattern = false;
    header.isInteger = false;
    header.isSymmetric = false;

    // Parse tokens
    const char* ptr = line + 14;  // Skip %%MatrixMarket

    // Skip whitespace
    while (*ptr == ' ' || *ptr == '\t') ptr++;

    // Expect "matrix"
    if (strncmp(ptr, "matrix", 6) != 0) return false;
    ptr += 6;
    while (*ptr == ' ' || *ptr == '\t') ptr++;

    // Expect "coordinate" (for sparse matrices)
    if (strncmp(ptr, "coordinate", 10) != 0) return false;
    ptr += 10;
    while (*ptr == ' ' || *ptr == '\t') ptr++;

    // Parse type: real, complex, pattern, integer
    if (strncmp(ptr, "real", 4) == 0) {
        // Real matrix (default)
        ptr += 4;
    } else if (strncmp(ptr, "complex", 7) == 0) {
        header.isComplex = true;
        ptr += 7;
    } else if (strncmp(ptr, "pattern", 7) == 0) {
        header.isPattern = true;
        ptr += 7;
    } else if (strncmp(ptr, "integer", 7) == 0) {
        header.isInteger = true;
        ptr += 7;
    }

    while (*ptr == ' ' || *ptr == '\t') ptr++;

    // Parse symmetry: general, symmetric, skew-symmetric, hermitian
    if (strncmp(ptr, "symmetric", 9) == 0 || strncmp(ptr, "Symmetric", 9) == 0) {
        header.isSymmetric = true;
    }

    return true;
}

spmv_status_t readMTXHeader(const char* filename, MTXHeader& header) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        return SPMV_ERROR_INVALID_MATRIX;
    }

    char line[256];
    if (!fgets(line, sizeof(line), fp)) {
        fclose(fp);
        return SPMV_ERROR_INVALID_MATRIX;
    }

    if (!parseBanner(line, header)) {
        fclose(fp);
        fprintf(stderr, "Error: Invalid MTX banner in %s\n", filename);
        return SPMV_ERROR_INVALID_MATRIX;
    }

    // Skip comments
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] != '%') break;
    }

    // Parse dimensions line
    int rows, cols, nnz;
    if (sscanf(line, "%d %d %d", &rows, &cols, &nnz) != 3) {
        fclose(fp);
        fprintf(stderr, "Error: Invalid dimensions line in %s\n", filename);
        return SPMV_ERROR_INVALID_MATRIX;
    }

    header.numRows = rows;
    header.numCols = cols;
    header.nnz = nnz;

    fclose(fp);
    return SPMV_SUCCESS;
}

template<typename FloatType>
spmv_status_t readMTXFile(const char* filename, CSRMatrix<FloatType>& matrix,
                           bool expandSymmetric) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        return SPMV_ERROR_INVALID_MATRIX;
    }

    char line[256];
    if (!fgets(line, sizeof(line), fp)) {
        fclose(fp);
        return SPMV_ERROR_INVALID_MATRIX;
    }

    MTXHeader header;
    if (!parseBanner(line, header)) {
        fclose(fp);
        return SPMV_ERROR_INVALID_MATRIX;
    }

    // Skip comments
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] != '%') break;
    }

    // Parse dimensions
    int rows, cols, nnz_file;
    if (sscanf(line, "%d %d %d", &rows, &cols, &nnz_file) != 3) {
        fclose(fp);
        return SPMV_ERROR_INVALID_MATRIX;
    }

    // Read all entries
    std::vector<int> rowIdx;
    std::vector<int> colIdx;
    std::vector<FloatType> values;

    rowIdx.reserve(nnz_file);
    colIdx.reserve(nnz_file);
    values.reserve(nnz_file);

    for (int i = 0; i < nnz_file; i++) {
        int row, col;
        FloatType val;

        if (header.isPattern) {
            // Pattern matrix: only row and column
            if (fscanf(fp, "%d %d", &row, &col) != 2) {
                fclose(fp);
                return SPMV_ERROR_INVALID_MATRIX;
            }
            val = static_cast<FloatType>(1.0);
        } else {
            // Real matrix: row, column, value
            double v;
            if (fscanf(fp, "%d %d %lf", &row, &col, &v) != 3) {
                fclose(fp);
                return SPMV_ERROR_INVALID_MATRIX;
            }
            val = static_cast<FloatType>(v);
        }

        // MTX uses 1-based indexing, convert to 0-based
        row--;
        col--;

        rowIdx.push_back(row);
        colIdx.push_back(col);
        values.push_back(val);

        // Expand symmetric matrices
        if (header.isSymmetric && expandSymmetric && row != col) {
            rowIdx.push_back(col);
            colIdx.push_back(row);
            values.push_back(val);
        }
    }

    fclose(fp);

    int actual_nnz = static_cast<int>(values.size());

    // Sort by row for CSR conversion
    std::vector<int> perm(actual_nnz);
    for (int i = 0; i < actual_nnz; i++) perm[i] = i;

    std::sort(perm.begin(), perm.end(), [&](int a, int b) {
        if (rowIdx[a] != rowIdx[b]) return rowIdx[a] < rowIdx[b];
        return colIdx[a] < colIdx[b];
    });

    // Allocate CSR matrix
    matrix.numRows = rows;
    matrix.numCols = cols;
    matrix.nnz = actual_nnz;
    matrix.allocateHost(rows, cols, actual_nnz);

    // Build CSR structure
    int idx = 0;
    matrix.rowPtr[0] = 0;
    for (int r = 0; r < rows; r++) {
        while (idx < actual_nnz && rowIdx[perm[idx]] == r) {
            matrix.colIdx[idx] = colIdx[perm[idx]];
            matrix.values[idx] = values[perm[idx]];
            idx++;
        }
        matrix.rowPtr[r + 1] = idx;
    }

    return SPMV_SUCCESS;
}

template<typename FloatType>
spmv_status_t writeMTXFile(const char* filename, const CSRMatrix<FloatType>& matrix,
                            bool writeSymmetric) {
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file %s for writing\n", filename);
        return SPMV_ERROR_INVALID_MATRIX;
    }

    // Write banner
    fprintf(fp, "%%MatrixMarket matrix coordinate real %s\n",
            writeSymmetric ? "symmetric" : "general");

    // Write comment
    fprintf(fp, "%% Generated by muxi_spmv test framework\n");
    fprintf(fp, "%% Rows: %d, Columns: %d, NNZ: %d\n",
            matrix.numRows, matrix.numCols, matrix.nnz);

    // Write dimensions
    int nnz_to_write = matrix.nnz;
    if (writeSymmetric) {
        // For symmetric, only write upper triangle
        nnz_to_write = 0;
        for (int r = 0; r < matrix.numRows; r++) {
            for (int j = matrix.rowPtr[r]; j < matrix.rowPtr[r + 1]; j++) {
                if (matrix.colIdx[j] >= r) nnz_to_write++;
            }
        }
    }

    fprintf(fp, "%d %d %d\n", matrix.numRows, matrix.numCols, nnz_to_write);

    // Write entries
    for (int r = 0; r < matrix.numRows; r++) {
        for (int j = matrix.rowPtr[r]; j < matrix.rowPtr[r + 1]; j++) {
            int c = matrix.colIdx[j];
            FloatType v = matrix.values[j];

            if (writeSymmetric && c < r) continue;  // Skip lower triangle for symmetric

            // MTX uses 1-based indexing
            fprintf(fp, "%d %d %g\n", r + 1, c + 1, static_cast<double>(v));
        }
    }

    fclose(fp);
    return SPMV_SUCCESS;
}

template<typename FloatType>
spmv_status_t readMTXToCOO(const char* filename, COOMatrix<FloatType>& matrix) {
    MTXHeader header;
    spmv_status_t status = readMTXHeader(filename, header);
    if (status != SPMV_SUCCESS) return status;

    CSRMatrix<FloatType> csr;
    status = readMTXFile(filename, csr, false);
    if (status != SPMV_SUCCESS) return status;

    // Convert CSR to COO
    matrix.numRows = csr.numRows;
    matrix.numCols = csr.numCols;
    matrix.nnz = csr.nnz;

    matrix.allocateHost(csr.numRows, csr.numCols, csr.nnz);

    for (int r = 0; r < csr.numRows; r++) {
        for (int j = csr.rowPtr[r]; j < csr.rowPtr[r + 1]; j++) {
            matrix.rowIdx[j] = r;
            matrix.colIdx[j] = csr.colIdx[j];
            matrix.values[j] = csr.values[j];
        }
    }

    return SPMV_SUCCESS;
}

template<typename FloatType>
spmv_status_t writeCOOToMTX(const char* filename, const COOMatrix<FloatType>& matrix) {
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file %s for writing\n", filename);
        return SPMV_ERROR_INVALID_MATRIX;
    }

    fprintf(fp, "%%MatrixMarket matrix coordinate real general\n");
    fprintf(fp, "%% Generated by muxi_spmv\n");
    fprintf(fp, "%d %d %d\n", matrix.numRows, matrix.numCols, matrix.nnz);

    for (int i = 0; i < matrix.nnz; i++) {
        fprintf(fp, "%d %d %g\n",
                matrix.rowIdx[i] + 1,  // 1-based
                matrix.colIdx[i] + 1,
                static_cast<double>(matrix.values[i]));
    }

    fclose(fp);
    return SPMV_SUCCESS;
}

// Explicit template instantiation
template spmv_status_t readMTXFile<float>(const char* filename, CSRMatrix<float>& matrix, bool expandSymmetric);
template spmv_status_t readMTXFile<double>(const char* filename, CSRMatrix<double>& matrix, bool expandSymmetric);
template spmv_status_t writeMTXFile<float>(const char* filename, const CSRMatrix<float>& matrix, bool writeSymmetric);
template spmv_status_t writeMTXFile<double>(const char* filename, const CSRMatrix<double>& matrix, bool writeSymmetric);
template spmv_status_t readMTXToCOO<float>(const char* filename, COOMatrix<float>& matrix);
template spmv_status_t readMTXToCOO<double>(const char* filename, COOMatrix<double>& matrix);
template spmv_status_t writeCOOToMTX<float>(const char* filename, const COOMatrix<float>& matrix);
template spmv_status_t writeCOOToMTX<double>(const char* filename, const COOMatrix<double>& matrix);

} // namespace io
} // namespace muxi_spmv