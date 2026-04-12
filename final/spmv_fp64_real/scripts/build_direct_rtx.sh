#!/bin/bash
# Build script for NVIDIA RTX 4090 using direct nvcc compilation

set -e

echo "=== Building SPMV FP64 for RTX 4090 (direct nvcc) ==="

# CUDA path (may vary by system)
CUDA_PATH=${CUDA_PATH:-/usr/local/cuda-13.1}
export PATH=$CUDA_PATH/bin:$PATH

# Build directory
BUILD_DIR="build_direct"
mkdir -p $BUILD_DIR

# Compile library object
echo "Compiling library..."
nvcc -O3 -arch=sm_89 -Xcompiler -fPIC -I./include -I./src \
    -c src/spmv_fp64.cu -o $BUILD_DIR/spmv_fp64.o

# Create shared library
echo "Creating shared library..."
nvcc -shared -arch=sm_89 -Xcompiler -fPIC -o $BUILD_DIR/libspmv_fp64.so $BUILD_DIR/spmv_fp64.o

# Compile example
echo "Compiling example..."
nvcc -O3 -arch=sm_89 -I./include -I./src \
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
echo "  $BUILD_DIR/spmv_example matrix.mtx"
echo ""