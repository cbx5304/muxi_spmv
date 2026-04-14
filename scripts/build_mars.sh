#!/bin/bash
# Build script for Mars X201 (domestic GPU)
#
# IMPORTANT:
# - Must use 'pre_make cmake' and 'pre_make make'
# - Do NOT set CUDA architectures (no sm_xx)
# - Set environment variables for cu-bridge

set -e

echo "=== Building SPMV FP64 for Mars X201 ==="

# Check environment
if [ ! -d "$HOME/cu-bridge" ]; then
    echo "ERROR: cu-bridge not found at $HOME/cu-bridge"
    echo "Please install cu-bridge first"
    exit 1
fi

# Set environment
export PATH=$PATH:$HOME/cu-bridge/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/cu-bridge/lib:$HOME/cu-bridge/lib64

echo "Environment configured:"
echo "  PATH includes: $HOME/cu-bridge/bin"
echo ""

# Create build directory
BUILD_DIR="build_mars"
mkdir -p $BUILD_DIR

# Configure WITHOUT CUDA architectures (domestic GPU doesn't support sm_xx)
cd $BUILD_DIR
pre_make cmake .. \
    -DCMAKE_CUDA_ARCHITECTURES="" \
    -DCMAKE_BUILD_TYPE=Release \
    -DSPMV_FP64_BUILD_SHARED=ON \
    -DSPMV_FP64_BUILD_EXAMPLES=ON

# Build
pre_make make -j$(nproc)

echo ""
echo "=== Build Complete ==="
echo "Library: $BUILD_DIR/lib/libspmv_fp64.so"
echo "Example: $BUILD_DIR/bin/spmv_example"
echo ""
echo "To run the example:"
echo "  CUDA_VISIBLE_DEVICES=7 $BUILD_DIR/bin/spmv_example matrix.mtx"
echo ""