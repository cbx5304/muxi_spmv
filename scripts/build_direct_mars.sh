#!/bin/bash
# Build script for Mars X201 (domestic GPU) using direct nvcc compilation
#
# This script compiles the library directly with nvcc without CMake,
# which is simpler and more reliable on the domestic GPU platform.

set -e

echo "=== Building SPMV FP64 for Mars X201 (direct nvcc) ==="

# Set environment for cu-bridge
source ~/.bashrc

# Build directory
BUILD_DIR="build_direct"
mkdir -p $BUILD_DIR

# Compile library object
echo "Compiling library..."
pre_make nvcc -O3 -I./include -I./src \
    -c src/spmv_fp64.cu -o $BUILD_DIR/spmv_fp64.o

# Create shared library
echo "Creating shared library..."
pre_make nvcc -shared -o $BUILD_DIR/libspmv_fp64.so $BUILD_DIR/spmv_fp64.o

# Compile example
echo "Compiling example..."
pre_make nvcc -O3 -I./include -I./src \
    examples/simple_example.cu \
    -L$BUILD_DIR -lspmv_fp64 \
    -o $BUILD_DIR/spmv_example \
    -Xlinker -rpath,$BUILD_DIR

echo ""
echo "=== Build Complete ==="
echo "Library: $BUILD_DIR/libspmv_fp64.so"
echo "Example: $BUILD_DIR/spmv_example"
echo ""
echo "To run:"
echo "  CUDA_VISIBLE_DEVICES=7 LD_LIBRARY_PATH=$BUILD_DIR:$LD_LIBRARY_PATH $BUILD_DIR/spmv_example matrix.mtx"
echo ""