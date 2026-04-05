/**
 * @file test_real_matrix_pattern.cu
 * @brief Analyze real matrix patterns and their impact on performance
 *
 * Analyze the p0_A ~ p9_A matrices:
 * 1. Column index distribution
 * 2. Row length variance
 * 3. Memory access patterns
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <map>

struct CSRMatrix {
    int numRows, numCols, nnz;
    int* rowPtr;
    int* colIdx;
    float* values;
};

bool loadMatrixMarket(const std::string& filename, CSRMatrix& matrix) {
    std::ifstream file(filename);
    if (!file.is_open()) return false;

    std::string line;
    std::getline(file, line);
    while (std::getline(file, line) && line[0] == '%') {}

    std::istringstream iss(line);
    int rows, cols, nnz;
    iss >> rows >> cols >> nnz;

    matrix.numRows = rows;
    matrix.numCols = cols;
    matrix.nnz = nnz;

    std::vector<std::tuple<int, int, float>> entries;
    entries.reserve(nnz);

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '%') continue;
        std::istringstream iss2(line);
        int r, c;
        float v;
        iss2 >> r >> c >> v;
        entries.push_back({r - 1, c - 1, v});
    }

    std::sort(entries.begin(), entries.end());

    matrix.rowPtr = new int[rows + 1];
    matrix.colIdx = new int[nnz];
    matrix.values = new float[nnz];

    matrix.rowPtr[0] = 0;
    int currentRow = 0;
    for (int i = 0; i < nnz; i++) {
        int r = std::get<0>(entries[i]);
        while (currentRow < r) { currentRow++; matrix.rowPtr[currentRow] = i; }
        matrix.colIdx[i] = std::get<1>(entries[i]);
        matrix.values[i] = std::get<2>(entries[i]);
    }
    while (currentRow < rows) { currentRow++; matrix.rowPtr[currentRow] = nnz; }

    return true;
}

void analyzePattern(const CSRMatrix& matrix, const std::string& name) {
    printf("\n=== Pattern Analysis: %s ===\n", name.c_str());
    printf("Rows: %d, Cols: %d, NNZ: %d\n", matrix.numRows, matrix.numCols, matrix.nnz);

    // Row length statistics
    std::vector<int> rowLengths(matrix.numRows);
    int minLen = INT_MAX, maxLen = 0;
    double sumLen = 0;

    for (int i = 0; i < matrix.numRows; i++) {
        int len = matrix.rowPtr[i + 1] - matrix.rowPtr[i];
        rowLengths[i] = len;
        sumLen += len;
        if (len < minLen) minLen = len;
        if (len > maxLen) maxLen = len;
    }

    double avgLen = sumLen / matrix.numRows;
    double variance = 0;
    for (int i = 0; i < matrix.numRows; i++) {
        variance += (rowLengths[i] - avgLen) * (rowLengths[i] - avgLen);
    }
    variance /= matrix.numRows;
    double stdDev = sqrt(variance);

    printf("Row Length Statistics:\n");
    printf("  Min: %d, Max: %d, Avg: %.2f, StdDev: %.2f\n", minLen, maxLen, avgLen, stdDev);
    printf("  Coefficient of Variation: %.2f%%\n", (stdDev / avgLen) * 100);

    // Column index distribution
    std::map<int, int> colBuckets;
    int bucketSize = matrix.numCols / 100;
    for (int i = 0; i < matrix.nnz; i++) {
        int bucket = matrix.colIdx[i] / bucketSize;
        colBuckets[bucket]++;
    }

    // Calculate column access entropy (measure of randomness)
    double entropy = 0;
    for (auto& p : colBuckets) {
        double prob = (double)p.second / matrix.nnz;
        if (prob > 0) entropy -= prob * log2(prob);
    }
    double maxEntropy = log2(100);  // Max entropy for 100 buckets
    double normalizedEntropy = entropy / maxEntropy;

    printf("Column Access Pattern:\n");
    printf("  Entropy: %.2f / %.2f (%.1f%% random)\n", entropy, maxEntropy, normalizedEntropy * 100);

    // Band analysis
    int totalBandwidth = 0;
    for (int i = 0; i < matrix.numRows; i++) {
        int start = matrix.rowPtr[i];
        int end = matrix.rowPtr[i + 1];
        if (end > start) {
            int minCol = matrix.colIdx[start];
            int maxCol = matrix.colIdx[start];
            for (int j = start + 1; j < end; j++) {
                if (matrix.colIdx[j] < minCol) minCol = matrix.colIdx[j];
                if (matrix.colIdx[j] > maxCol) maxCol = matrix.colIdx[j];
            }
            totalBandwidth += (maxCol - minCol);
        }
    }
    double avgBandwidth = (double)totalBandwidth / matrix.numRows;
    printf("Average Row Bandwidth: %.2f columns\n", avgBandwidth);

    // Diagonal proximity
    int nearDiagonal = 0;
    int totalElements = 0;
    for (int i = 0; i < matrix.numRows; i++) {
        for (int j = matrix.rowPtr[i]; j < matrix.rowPtr[i + 1]; j++) {
            int col = matrix.colIdx[j];
            if (abs(col - i) < 100) nearDiagonal++;
            totalElements++;
        }
    }
    printf("Diagonal Proximity (<100 cols): %.2f%%\n", (double)nearDiagonal / totalElements * 100);

    // Estimate x-vector reuse potential
    printf("\nMemory Access Analysis:\n");
    double xVectorSize = matrix.numCols * 4.0 / (1024 * 1024);  // MB
    printf("  X-vector size: %.2f MB\n", xVectorSize);
    printf("  Expected utilization with L2=4MB: %.1f%% (cache limit)\n",
           std::min(100.0, 4.0 / xVectorSize * 100));

    delete[] matrix.rowPtr;
    delete[] matrix.colIdx;
    delete[] matrix.values;
}

int main(int argc, char** argv) {
    std::string baseDir = argc > 1 ? argv[1] : "./real_cases/mtx";

    printf("=== Real Matrix Pattern Analysis ===\n");

    for (int m = 0; m <= 9; m++) {
        std::string matrixFile = baseDir + "/p" + std::to_string(m) + "_A";
        CSRMatrix matrix;
        if (loadMatrixMarket(matrixFile, matrix)) {
            analyzePattern(matrix, "p" + std::to_string(m) + "_A");
        }
    }

    return 0;
}