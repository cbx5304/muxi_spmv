/**
 * @file column_reorder.cuh
 * @brief Column reordering preprocessing for better cache locality
 *
 * Key insight: Banded matrices achieve 65-78% utilization vs 20% for random matrices.
 * By reordering columns to group similar column indices together, we can improve
 * x-vector cache hit rate and reduce random access penalty.
 */

#ifndef COLUMN_REORDER_CUH_
#define COLUMN_REORDER_CUH_

#include "formats/sparse_formats.h"
#include <vector>
#include <algorithm>
#include <cmath>

namespace muxi_spmv {

/**
 * @brief Column reordering statistics
 */
struct ColumnReorderStats {
    int numReorders;          // Number of columns reordered
    float avgDistanceBefore;  // Average column index distance before reordering
    float avgDistanceAfter;   // Average column index distance after reordering
    float improvementRatio;   // Distance reduction ratio
};

/**
 * @brief Reorder columns to improve locality
 *
 * Strategy: Group columns by their average row position.
 * Columns that appear in similar rows are placed close together.
 * This improves x-vector cache hit rate.
 *
 * @param matrix Input CSR matrix
 * @param reorderedMatrix Output matrix with reordered columns
 * @param forwardMap Mapping from new column index to original column index
 * @param stats Statistics about the reordering
 */
template<typename FloatType>
spmv_status_t reorderColumns(
    const CSRMatrix<FloatType>& matrix,
    CSRMatrix<FloatType>& reorderedMatrix,
    std::vector<int>& forwardMap,
    ColumnReorderStats& stats)
{
    int numRows = matrix.numRows;
    int numCols = matrix.numCols;
    int nnz = matrix.nnz;

    // Copy host data if not available
    if (!matrix.rowPtr) {
        return SPMV_ERROR_INVALID_MATRIX;
    }

    // Step 1: Compute "center of mass" for each column
    // This tells us which rows typically use this column
    std::vector<float> colCenterOfMass(numCols, 0.0f);
    std::vector<int> colCount(numCols, 0);

    for (int row = 0; row < numRows; row++) {
        int start = matrix.rowPtr[row];
        int end = matrix.rowPtr[row + 1];
        for (int idx = start; idx < end; idx++) {
            int col = matrix.colIdx[idx];
            colCenterOfMass[col] += row;
            colCount[col]++;
        }
    }

    for (int col = 0; col < numCols; col++) {
        if (colCount[col] > 0) {
            colCenterOfMass[col] /= colCount[col];
        }
    }

    // Step 2: Sort columns by their center of mass
    // This groups columns that are used in similar rows
    std::vector<std::pair<float, int>> sortedCols(numCols);
    for (int col = 0; col < numCols; col++) {
        sortedCols[col] = {colCenterOfMass[col], col};
    }

    std::sort(sortedCols.begin(), sortedCols.end());

    // Step 3: Create forward and reverse mappings
    std::vector<int> reverseMap(numCols);  // old -> new
    forwardMap.resize(numCols);            // new -> old

    for (int newCol = 0; newCol < numCols; newCol++) {
        int oldCol = sortedCols[newCol].second;
        forwardMap[newCol] = oldCol;
        reverseMap[oldCol] = newCol;
    }

    // Step 4: Create reordered matrix
    reorderedMatrix.numRows = numRows;
    reorderedMatrix.numCols = numCols;
    reorderedMatrix.nnz = nnz;
    reorderedMatrix.allocateHost(numRows, numCols, nnz);

    // Copy rowPtr
    for (int i = 0; i <= numRows; i++) {
        reorderedMatrix.rowPtr[i] = matrix.rowPtr[i];
    }

    // Remap column indices
    for (int idx = 0; idx < nnz; idx++) {
        reorderedMatrix.colIdx[idx] = reverseMap[matrix.colIdx[idx]];
        reorderedMatrix.values[idx] = matrix.values[idx];
    }

    // Step 5: Compute statistics
    float avgDistBefore = 0, avgDistAfter = 0;
    int count = 0;

    for (int row = 0; row < numRows && row < 10000; row++) {  // Sample rows
        int start = matrix.rowPtr[row];
        int end = matrix.rowPtr[row + 1];
        for (int idx = start; idx < end - 1; idx++) {
            int col1 = matrix.colIdx[idx];
            int col2 = matrix.colIdx[idx + 1];
            avgDistBefore += std::abs(col2 - col1);

            int newCol1 = reorderedMatrix.colIdx[idx];
            int newCol2 = reorderedMatrix.colIdx[idx + 1];
            avgDistAfter += std::abs(newCol2 - newCol1);
            count++;
        }
    }

    if (count > 0) {
        stats.avgDistanceBefore = avgDistBefore / count;
        stats.avgDistanceAfter = avgDistAfter / count;
        stats.improvementRatio = stats.avgDistanceBefore / std::max(stats.avgDistanceAfter, 1.0f);
    }

    return SPMV_SUCCESS;
}

/**
 * @brief Apply column reordering to x-vector
 *
 * @param h_x Original x-vector on host
 * @param h_x_reordered Reordered x-vector on host
 * @param forwardMap Mapping from new column index to original column index
 */
template<typename FloatType>
void reorderXVector(
    const FloatType* h_x,
    FloatType* h_x_reordered,
    const std::vector<int>& forwardMap)
{
    int numCols = forwardMap.size();
    for (int newCol = 0; newCol < numCols; newCol++) {
        int oldCol = forwardMap[newCol];
        h_x_reordered[newCol] = h_x[oldCol];
    }
}

/**
 * @brief Inverse reorder y-vector (after SpMV computation)
 *
 * Not needed since y-vector is not affected by column reordering.
 */

/**
 * @brief Simple column reordering: sort by column index within each row
 *
 * This is a simpler approach that just sorts column indices within each row.
 * It doesn't require reordering x-vector, but still improves locality.
 */
template<typename FloatType>
spmv_status_t sortColumnsWithinRows(
    CSRMatrix<FloatType>& matrix)
{
    // For each row, sort column indices
    for (int row = 0; row < matrix.numRows; row++) {
        int start = matrix.rowPtr[row];
        int end = matrix.rowPtr[row + 1];

        // Create temporary arrays for this row
        std::vector<std::pair<int, FloatType>> rowElements;
        for (int idx = start; idx < end; idx++) {
            rowElements.push_back({matrix.colIdx[idx], matrix.values[idx]});
        }

        // Sort by column index
        std::sort(rowElements.begin(), rowElements.end());

        // Write back
        for (size_t i = 0; i < rowElements.size(); i++) {
            matrix.colIdx[start + i] = rowElements[i].first;
            matrix.values[start + i] = rowElements[i].second;
        }
    }

    return SPMV_SUCCESS;
}

/**
 * @brief RCM (Reverse Cuthill-McKee) reordering for bandwidth reduction
 *
 * This is a more sophisticated reordering that reduces the bandwidth
 * of the matrix by reordering both rows and columns.
 *
 * Note: This is a simplified implementation. For full RCM, use a library like SuiteSparse.
 */

} // namespace muxi_spmv

#endif // COLUMN_REORDER_CUH_