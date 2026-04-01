#!/bin/bash
# 编译脚本 - 国产GPU (Mars X201)
# 使用cu-bridge的nvcc包装器直接编译

# 设置环境变量
export PATH=$HOME/cu-bridge/CUDA_DIR/bin:$PATH
export LIBRARY_PATH=$HOME/cu-bridge/CUDA_DIR/lib64:/opt/hpcc/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/cu-bridge/CUDA_DIR/lib64:/opt/hpcc/lib:$LD_LIBRARY_PATH

# 编译参数
NVCC=$HOME/cu-bridge/CUDA_DIR/bin/nvcc
OPT_FLAGS="-O3"
WARP_SIZE_FLAG="-DWARP_SIZE=64"
INCLUDE_FLAG="-Isrc"
SRC_DIR="src"

# 编译目标
build_device_info() {
    echo "Building device_info..."
    $NVCC $OPT_FLAGS $INCLUDE_FLAG tests/device_info.cu -o device_info
}

build_test_spmv() {
    echo "Building test_spmv..."
    $NVCC $OPT_FLAGS $WARP_SIZE_FLAG $INCLUDE_FLAG \
        $SRC_DIR/spmv/csr/spmv_csr.cu \
        $SRC_DIR/utils/device_info.cu \
        $SRC_DIR/api/spmv_api.cu \
        tests/correctness/test_spmv.cu \
        -o test_spmv
}

build_all() {
    build_device_info
    build_test_spmv
}

# 主入口
case "$1" in
    device_info)
        build_device_info
        ;;
    test_spmv)
        build_test_spmv
        ;;
    all)
        build_all
        ;;
    *)
        echo "Usage: $0 {device_info|test_spmv|all}"
        echo "Note: Run with CUDA_VISIBLE_DEVICES=7 for GPU execution"
        exit 1
        ;;
esac

echo "Build complete."