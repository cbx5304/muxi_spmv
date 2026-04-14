#!/bin/bash
# Build script for NVIDIA RTX 4090
#
# Standard CUDA build with architecture sm_89 (RTX 4090)

set -e

echo "=== Building SPMV FP64 for NVIDIA RTX 4090 ==="

# Create build directory
BUILD_DIR="build_rtx"
mkdir -p $BUILD_DIR

# Configure with CUDA architecture for RTX 4090 (sm_89)
cd $BUILD_DIR
cmake .. \
    -DCMAKE_CUDA_ARCHITECTURES=89 \
    -DCMAKE_BUILD_TYPE=Release \
    -DSPMV_FP64_BUILD_SHARED=ON \
    -DSPMV_FP64_BUILD_EXAMPLES=ON

# Build
make -j$(nproc)

echo ""
echo "=== Build Complete ==="
echo "Library: $BUILD_DIR/lib/libspmv_fp64.so"
echo "Example: $BUILD_DIR/bin/spmv_example"
echo ""
echo "To run the example:"
echo "  $BUILD_DIR/bin/spmv_example matrix.mtx"
echo ""