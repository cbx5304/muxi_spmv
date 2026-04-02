/**
 * @file spmv_ellpack.cu
 * @brief ELLPACK format SpMV implementation
 */

#include "spmv_ellpack.cuh"
#include <cstring>

namespace muxi_spmv {

template<typename FloatType>
spmv_status_t csr_to_ellpack(
    const CSRMatrix<FloatType>& csr,
    ELLPACKMatrix<FloatType>& ellpack,
    int maxElements)
{
    // Set number of elements per row
    if (maxElements <= 0) {
        // Auto-detect: find maximum row length
        maxElements = 0;
        for (int i = 0; i < csr.numRows; i++) {
            int rowLen = csr.rowPtr[i + 1] - csr.rowPtr[i];
            maxElements = std::max(maxElements, rowLen);
        }
    }

    ellpack.numRows = csr.numRows;
    ellpack.numCols = csr.numCols;
    ellpack.numElements = maxElements;
    ellpack.nnz = csr.nnz;

    // Allocate host memory
    int totalElements = csr.numRows * maxElements;
    int* h_colIdx = new int[totalElements];
    FloatType* h_values = new FloatType[totalElements];

    // Initialize with padding markers
    for (int i = 0; i < totalElements; i++) {
        h_colIdx[i] = -1;  // -1 indicates padding
        h_values[i] = static_cast<FloatType>(0);
    }

    // Convert CSR to ELLPACK (column-major storage)
    for (int row = 0; row < csr.numRows; row++) {
        int rowStart = csr.rowPtr[row];
        int rowEnd = csr.rowPtr[row + 1];
        int rowLen = rowEnd - rowStart;

        for (int i = 0; i < rowLen && i < maxElements; i++) {
            // Column-major index: row + elementIdx * numRows
            int idx = row + i * csr.numRows;
            h_colIdx[idx] = csr.colIdx[rowStart + i];
            h_values[idx] = csr.values[rowStart + i];
        }
    }

    // Allocate device memory
    cudaMalloc(&ellpack.d_colIdx, totalElements * sizeof(int));
    cudaMalloc(&ellpack.d_values, totalElements * sizeof(FloatType));

    // Copy to device
    cudaMemcpy(ellpack.d_colIdx, h_colIdx, totalElements * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ellpack.d_values, h_values, totalElements * sizeof(FloatType), cudaMemcpyHostToDevice);

    // Cleanup host memory
    delete[] h_colIdx;
    delete[] h_values;

    return SPMV_SUCCESS;
}

template<typename FloatType>
spmv_status_t spmv_ellpack(
    const ELLPACKMatrix<FloatType>& ellpack,
    const FloatType* d_x,
    FloatType* d_y,
    cudaStream_t stream)
{
    int blockSize = 256;
    int gridSize = (ellpack.numRows + blockSize - 1) / blockSize;

    spmv_ellpack_kernel<FloatType, 256><<<gridSize, blockSize, 0, stream>>>(
        ellpack.numRows,
        ellpack.numElements,
        ellpack.d_colIdx,
        ellpack.d_values,
        d_x,
        d_y);

    return SPMV_SUCCESS;
}

// Explicit template instantiation
template spmv_status_t csr_to_ellpack<float>(const CSRMatrix<float>&, ELLPACKMatrix<float>&, int);
template spmv_status_t csr_to_ellpack<double>(const CSRMatrix<double>&, ELLPACKMatrix<double>&, int);
template spmv_status_t spmv_ellpack<float>(const ELLPACKMatrix<float>&, const float*, float*, cudaStream_t);
template spmv_status_t spmv_ellpack<double>(const ELLPACKMatrix<double>&, const double*, double*, cudaStream_t);

} // namespace muxi_spmv