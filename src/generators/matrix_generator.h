/**
 * @file matrix_generator.h
 * @brief Matrix generator interface for test matrix generation
 */

#ifndef MATRIX_GENERATOR_H_
#define MATRIX_GENERATOR_H_

#include "utils/common.h"
#include "formats/sparse_formats.h"
#include <cuda_runtime.h>

namespace muxi_spmv {
namespace generators {

/**
 * @brief Matrix type enumeration
 */
enum class MatrixType {
    STRUCTURED_DIAGONAL,      ///< Diagonal matrix (nnz = n)
    STRUCTURED_BANDED,        ///< Banded matrix with specified bandwidth
    STRUCTURED_BLOCK_DIAGONAL,///< Block diagonal matrix
    RANDOM_UNIFORM,           ///< Random sparse matrix (uniform distribution)
    RANDOM_NORMAL,            ///< Random sparse matrix (normal distribution)
    CONCENTRATED_LOCAL,       ///< Locally concentrated non-zero elements
    REALWORLD_POWERLaw        ///< Power-law distribution (simulates real graphs)
};

/**
 * @brief Matrix generation configuration
 */
struct MatrixGenConfig {
    int numRows;              ///< Number of rows
    int numCols;              ///< Number of columns (default: numRows for square)
    double sparsity;          ///< Sparsity ratio (nnz / (rows * cols))
    MatrixType type;          ///< Matrix type

    // Type-specific parameters
    int bandwidth;            ///< For banded matrices
    int blockSize;            ///< For block diagonal/BSR
    double concentrationFactor; ///< For concentrated matrices
    int numClusters;          ///< Number of clusters for concentrated matrices
    double powerLawAlpha;     ///< Power-law exponent for real-world-like matrices

    // Default constructor
    MatrixGenConfig() : numRows(1024), numCols(1024), sparsity(0.01),
                        type(MatrixType::STRUCTURED_DIAGONAL),
                        bandwidth(1), blockSize(4),
                        concentrationFactor(5.0), numClusters(10),
                        powerLawAlpha(2.5) {}
};

/**
 * @brief Base class for matrix generators
 */
template<typename FloatType>
class MatrixGenerator {
public:
    virtual ~MatrixGenerator() = default;

    /**
     * @brief Generate matrix in host memory
     * @param config Generation configuration
     * @param matrix Output matrix (CSR format)
     * @return Status code
     */
    virtual spmv_status_t generate(const MatrixGenConfig& config,
                                    CSRMatrix<FloatType>& matrix) = 0;

    /**
     * @brief Get matrix type this generator produces
     */
    virtual MatrixType getType() const = 0;

    /**
     * @brief Get generator name for logging
     */
    virtual const char* getName() const = 0;
};

// Factory function to create generator
template<typename FloatType>
MatrixGenerator<FloatType>* createGenerator(MatrixType type);

// Convenience functions for direct generation
template<typename FloatType>
spmv_status_t generateDiagonalMatrix(int n, CSRMatrix<FloatType>& matrix);

template<typename FloatType>
spmv_status_t generateBandedMatrix(int n, int bandwidth, CSRMatrix<FloatType>& matrix);

template<typename FloatType>
spmv_status_t generateRandomMatrix(int rows, int cols, int nnz, CSRMatrix<FloatType>& matrix);

template<typename FloatType>
spmv_status_t generateConcentratedMatrix(int rows, int cols, int nnz,
                                          int numClusters, double concentrationFactor,
                                          CSRMatrix<FloatType>& matrix);

template<typename FloatType>
spmv_status_t generatePowerLawMatrix(int rows, int cols, int nnz,
                                      double alpha, CSRMatrix<FloatType>& matrix);

} // namespace generators
} // namespace muxi_spmv

#endif // MATRIX_GENERATOR_H_