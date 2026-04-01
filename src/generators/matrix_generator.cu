/**
 * @file matrix_generator.cu
 * @brief Implementation of matrix generators
 */

#include "generators/matrix_generator.h"
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cmath>

namespace muxi_spmv {
namespace generators {

// Helper function to sort COO entries by row
template<typename FloatType>
struct COOEntry {
    int row;
    int col;
    FloatType val;
};

template<typename FloatType>
bool compareCOO(const COOEntry<FloatType>& a, const COOEntry<FloatType>& b) {
    if (a.row != b.row) return a.row < b.row;
    return a.col < b.col;
}

// Generate diagonal matrix
template<typename FloatType>
spmv_status_t generateDiagonalMatrix(int n, CSRMatrix<FloatType>& matrix) {
    matrix.numRows = n;
    matrix.numCols = n;
    matrix.nnz = n;

    matrix.allocateHost(n, n, n);

    for (int i = 0; i <= n; i++) {
        matrix.rowPtr[i] = i;
    }

    for (int i = 0; i < n; i++) {
        matrix.colIdx[i] = i;
        matrix.values[i] = static_cast<FloatType>(1.0);
    }

    return SPMV_SUCCESS;
}

// Generate banded matrix
template<typename FloatType>
spmv_status_t generateBandedMatrix(int n, int bandwidth, CSRMatrix<FloatType>& matrix) {
    // Calculate number of non-zeros
    int nnz = 0;
    for (int i = 0; i < n; i++) {
        int startCol = std::max(0, i - bandwidth);
        int endCol = std::min(n - 1, i + bandwidth);
        nnz += (endCol - startCol + 1);
    }

    matrix.numRows = n;
    matrix.numCols = n;
    matrix.nnz = nnz;

    matrix.allocateHost(n, n, nnz);

    int idx = 0;
    matrix.rowPtr[0] = 0;
    for (int i = 0; i < n; i++) {
        int startCol = std::max(0, i - bandwidth);
        int endCol = std::min(n - 1, i + bandwidth);

        for (int j = startCol; j <= endCol; j++) {
            matrix.colIdx[idx] = j;
            // Values based on distance from diagonal
            FloatType dist = static_cast<FloatType>(std::abs(j - i));
            matrix.values[idx] = static_cast<FloatType>(1.0) / (dist + static_cast<FloatType>(1.0));
            idx++;
        }
        matrix.rowPtr[i + 1] = idx;
    }

    return SPMV_SUCCESS;
}

// Generate random sparse matrix
template<typename FloatType>
spmv_status_t generateRandomMatrix(int rows, int cols, int nnz, CSRMatrix<FloatType>& matrix) {
    // Use COO format first, then convert to CSR
    COOEntry<FloatType>* coo = new COOEntry<FloatType>[nnz];

    // Generate random entries
    for (int i = 0; i < nnz; i++) {
        coo[i].row = rand() % rows;
        coo[i].col = rand() % cols;
        coo[i].val = static_cast<FloatType>(rand() % 100 + 1) / static_cast<FloatType>(100);
    }

    // Sort by row
    std::sort(coo, coo + nnz, compareCOO<FloatType>);

    matrix.numRows = rows;
    matrix.numCols = cols;
    matrix.nnz = nnz;

    matrix.allocateHost(rows, cols, nnz);

    // Convert to CSR
    int idx = 0;
    matrix.rowPtr[0] = 0;
    for (int r = 0; r < rows; r++) {
        while (idx < nnz && coo[idx].row == r) {
            matrix.colIdx[idx] = coo[idx].col;
            matrix.values[idx] = coo[idx].val;
            idx++;
        }
        matrix.rowPtr[r + 1] = idx;
    }

    delete[] coo;
    return SPMV_SUCCESS;
}

// Generate concentrated matrix (non-zeros cluster in specific regions)
template<typename FloatType>
spmv_status_t generateConcentratedMatrix(int rows, int cols, int nnz,
                                          int numClusters, double concentrationFactor,
                                          CSRMatrix<FloatType>& matrix) {
    // Calculate cluster positions
    int* clusterRowStart = new int[numClusters];
    int* clusterColStart = new int[numClusters];
    int* clusterSize = new int[numClusters];

    int clusterRows = rows / (numClusters + 1);
    int clusterCols = cols / (numClusters + 1);

    for (int c = 0; c < numClusters; c++) {
        clusterRowStart[c] = c * clusterRows + clusterRows / 2;
        clusterColStart[c] = c * clusterCols + clusterCols / 2;
        clusterSize[c] = static_cast<int>(nnz * concentrationFactor / numClusters);
    }

    // Generate entries with cluster bias
    COOEntry<FloatType>* coo = new COOEntry<FloatType>[nnz];

    for (int i = 0; i < nnz; i++) {
        // 80% chance to be in a cluster
        if (rand() % 100 < 80) {
            int cluster = rand() % numClusters;
            int rowOffset = rand() % clusterRows - clusterRows / 2;
            int colOffset = rand() % clusterCols - clusterCols / 2;
            coo[i].row = std::max(0, std::min(rows - 1, clusterRowStart[cluster] + rowOffset));
            coo[i].col = std::max(0, std::min(cols - 1, clusterColStart[cluster] + colOffset));
        } else {
            // Random position outside clusters
            coo[i].row = rand() % rows;
            coo[i].col = rand() % cols;
        }
        coo[i].val = static_cast<FloatType>(rand() % 100 + 1) / static_cast<FloatType>(100);
    }

    // Sort and convert to CSR
    std::sort(coo, coo + nnz, compareCOO<FloatType>);

    matrix.numRows = rows;
    matrix.numCols = cols;
    matrix.nnz = nnz;

    matrix.allocateHost(rows, cols, nnz);

    int idx = 0;
    matrix.rowPtr[0] = 0;
    for (int r = 0; r < rows; r++) {
        while (idx < nnz && coo[idx].row == r) {
            matrix.colIdx[idx] = coo[idx].col;
            matrix.values[idx] = coo[idx].val;
            idx++;
        }
        matrix.rowPtr[r + 1] = idx;
    }

    delete[] coo;
    delete[] clusterRowStart;
    delete[] clusterColStart;
    delete[] clusterSize;

    return SPMV_SUCCESS;
}

// Generate power-law distributed matrix (simulates real-world graphs)
template<typename FloatType>
spmv_status_t generatePowerLawMatrix(int rows, int cols, int nnz,
                                      double alpha, CSRMatrix<FloatType>& matrix) {
    // Power-law distribution for row selection
    // High-degree nodes get more edges

    COOEntry<FloatType>* coo = new COOEntry<FloatType>[nnz];

    for (int i = 0; i < nnz; i++) {
        // Power-law row selection
        double u = static_cast<double>(rand()) / RAND_MAX;
        double powerRow = pow(u, 1.0 / alpha) * rows;
        coo[i].row = std::min(rows - 1, static_cast<int>(powerRow));

        // Column can be random or also power-law
        coo[i].col = rand() % cols;

        coo[i].val = static_cast<FloatType>(rand() % 100 + 1) / static_cast<FloatType>(100);
    }

    // Sort and convert to CSR
    std::sort(coo, coo + nnz, compareCOO<FloatType>);

    matrix.numRows = rows;
    matrix.numCols = cols;
    matrix.nnz = nnz;

    matrix.allocateHost(rows, cols, nnz);

    int idx = 0;
    matrix.rowPtr[0] = 0;
    for (int r = 0; r < rows; r++) {
        while (idx < nnz && coo[idx].row == r) {
            matrix.colIdx[idx] = coo[idx].col;
            matrix.values[idx] = coo[idx].val;
            idx++;
        }
        matrix.rowPtr[r + 1] = idx;
    }

    delete[] coo;
    return SPMV_SUCCESS;
}

// Block diagonal matrix generator
template<typename FloatType>
spmv_status_t generateBlockDiagonalMatrix(int n, int blockSize, CSRMatrix<FloatType>& matrix) {
    int numBlocks = n / blockSize;
    int nnzPerBlock = blockSize * blockSize;
    int nnz = numBlocks * nnzPerBlock;

    matrix.numRows = n;
    matrix.numCols = n;
    matrix.nnz = nnz;

    matrix.allocateHost(n, n, nnz);

    int idx = 0;
    matrix.rowPtr[0] = 0;

    for (int block = 0; block < numBlocks; block++) {
        int blockStart = block * blockSize;
        for (int i = 0; i < blockSize; i++) {
            for (int j = 0; j < blockSize; j++) {
                matrix.colIdx[idx] = blockStart + j;
                matrix.values[idx] = static_cast<FloatType>((rand() % 100 + 1) / 100.0);
                idx++;
            }
            matrix.rowPtr[blockStart + i + 1] = idx;
        }
    }

    // Fill remaining rows if n is not divisible by blockSize
    for (int i = numBlocks * blockSize; i <= n; i++) {
        matrix.rowPtr[i] = idx;
    }

    return SPMV_SUCCESS;
}

// Generator class implementations

template<typename FloatType>
class DiagonalGenerator : public MatrixGenerator<FloatType> {
public:
    spmv_status_t generate(const MatrixGenConfig& config, CSRMatrix<FloatType>& matrix) override {
        return generateDiagonalMatrix(config.numRows, matrix);
    }
    MatrixType getType() const override { return MatrixType::STRUCTURED_DIAGONAL; }
    const char* getName() const override { return "DiagonalGenerator"; }
};

template<typename FloatType>
class BandedGenerator : public MatrixGenerator<FloatType> {
public:
    spmv_status_t generate(const MatrixGenConfig& config, CSRMatrix<FloatType>& matrix) override {
        return generateBandedMatrix(config.numRows, config.bandwidth, matrix);
    }
    MatrixType getType() const override { return MatrixType::STRUCTURED_BANDED; }
    const char* getName() const override { return "BandedGenerator"; }
};

template<typename FloatType>
class RandomGenerator : public MatrixGenerator<FloatType> {
public:
    spmv_status_t generate(const MatrixGenConfig& config, CSRMatrix<FloatType>& matrix) override {
        int nnz = static_cast<int>(config.numRows * config.numCols * config.sparsity);
        return generateRandomMatrix(config.numRows, config.numCols, nnz, matrix);
    }
    MatrixType getType() const override { return MatrixType::RANDOM_UNIFORM; }
    const char* getName() const override { return "RandomGenerator"; }
};

template<typename FloatType>
class ConcentratedGenerator : public MatrixGenerator<FloatType> {
public:
    spmv_status_t generate(const MatrixGenConfig& config, CSRMatrix<FloatType>& matrix) override {
        int nnz = static_cast<int>(config.numRows * config.numCols * config.sparsity);
        return generateConcentratedMatrix(config.numRows, config.numCols, nnz,
                                           config.numClusters, config.concentrationFactor, matrix);
    }
    MatrixType getType() const override { return MatrixType::CONCENTRATED_LOCAL; }
    const char* getName() const override { return "ConcentratedGenerator"; }
};

template<typename FloatType>
class PowerLawGenerator : public MatrixGenerator<FloatType> {
public:
    spmv_status_t generate(const MatrixGenConfig& config, CSRMatrix<FloatType>& matrix) override {
        int nnz = static_cast<int>(config.numRows * config.numCols * config.sparsity);
        return generatePowerLawMatrix(config.numRows, config.numCols, nnz,
                                       config.powerLawAlpha, matrix);
    }
    MatrixType getType() const override { return MatrixType::REALWORLD_POWERLaw; }
    const char* getName() const override { return "PowerLawGenerator"; }
};

// Factory function
template<typename FloatType>
MatrixGenerator<FloatType>* createGenerator(MatrixType type) {
    switch (type) {
        case MatrixType::STRUCTURED_DIAGONAL:
            return new DiagonalGenerator<FloatType>();
        case MatrixType::STRUCTURED_BANDED:
            return new BandedGenerator<FloatType>();
        case MatrixType::RANDOM_UNIFORM:
            return new RandomGenerator<FloatType>();
        case MatrixType::CONCENTRATED_LOCAL:
            return new ConcentratedGenerator<FloatType>();
        case MatrixType::REALWORLD_POWERLaw:
            return new PowerLawGenerator<FloatType>();
        default:
            return nullptr;
    }
}

// Explicit template instantiation
template spmv_status_t generateDiagonalMatrix<float>(int n, CSRMatrix<float>& matrix);
template spmv_status_t generateDiagonalMatrix<double>(int n, CSRMatrix<double>& matrix);
template spmv_status_t generateBandedMatrix<float>(int n, int bandwidth, CSRMatrix<float>& matrix);
template spmv_status_t generateBandedMatrix<double>(int n, int bandwidth, CSRMatrix<double>& matrix);
template spmv_status_t generateRandomMatrix<float>(int rows, int cols, int nnz, CSRMatrix<float>& matrix);
template spmv_status_t generateRandomMatrix<double>(int rows, int cols, int nnz, CSRMatrix<double>& matrix);
template spmv_status_t generateConcentratedMatrix<float>(int rows, int cols, int nnz, int numClusters, double concentrationFactor, CSRMatrix<float>& matrix);
template spmv_status_t generateConcentratedMatrix<double>(int rows, int cols, int nnz, int numClusters, double concentrationFactor, CSRMatrix<double>& matrix);
template spmv_status_t generatePowerLawMatrix<float>(int rows, int cols, int nnz, double alpha, CSRMatrix<float>& matrix);
template spmv_status_t generatePowerLawMatrix<double>(int rows, int cols, int nnz, double alpha, CSRMatrix<double>& matrix);

// Factory function explicit instantiation
template MatrixGenerator<float>* createGenerator<float>(MatrixType type);
template MatrixGenerator<double>* createGenerator<double>(MatrixType type);

} // namespace generators
} // namespace muxi_spmv