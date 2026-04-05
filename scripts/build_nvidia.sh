#!/bin/bash
# 编译脚本 - NVIDIA RTX 4090
# 使用标准CUDA nvcc编译

# 设置环境变量 (根据服务器配置调整)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 编译参数
NVCC=nvcc
OPT_FLAGS="-O3"
WARP_SIZE_FLAG="-DWARP_SIZE=32"
INCLUDE_FLAG="-Isrc"
SRC_DIR="src"

# 核心源文件
CORE_SRCS="$SRC_DIR/spmv/csr/spmv_csr.cu $SRC_DIR/spmv/csr5/spmv_csr5.cu $SRC_DIR/utils/device_info.cu $SRC_DIR/api/spmv_api.cu"

# 生成器源文件
GEN_SRCS="$SRC_DIR/generators/matrix_generator.cu $SRC_DIR/generators/mtx_io.cu"

# 性能测试源文件
BENCH_SRCS="$SRC_DIR/benchmark/performance_benchmark.cu"

# 编译目标
build_device_info() {
    echo "Building device_info..."
    $NVCC $OPT_FLAGS $INCLUDE_FLAG tests/device_info.cu -o device_info
}

build_test_spmv() {
    echo "Building test_spmv..."
    $NVCC $OPT_FLAGS $WARP_SIZE_FLAG $INCLUDE_FLAG \
        $CORE_SRCS \
        tests/correctness/test_spmv.cu \
        -o test_spmv
}

build_test_runner() {
    echo "Building test_runner..."
    # Compile all sources together - newer CUDA handles template instantiation
    $NVCC $OPT_FLAGS $WARP_SIZE_FLAG $INCLUDE_FLAG \
        $CORE_SRCS \
        $GEN_SRCS \
        $BENCH_SRCS \
        tests/benchmark/test_runner.cu \
        -o test_runner \
        -lcusparse
}

build_test_kernel_comparison() {
    echo "Building test_kernel_comparison..."
    $NVCC $OPT_FLAGS $WARP_SIZE_FLAG $INCLUDE_FLAG \
        $CORE_SRCS \
        $GEN_SRCS \
        tests/test_kernel_comparison.cu \
        -o test_kernel_comparison
}

build_all() {
    build_device_info
    build_test_spmv
    build_test_runner
    build_test_kernel_comparison
}

# 主入口
case "$1" in
    device_info)
        build_device_info
        ;;
    test_spmv)
        build_test_spmv
        ;;
    test_runner)
        build_test_runner
        ;;
    test_kernel_comparison)
        build_test_kernel_comparison
        ;;
    all)
        build_all
        ;;
    *)
        echo "Usage: $0 {device_info|test_spmv|test_runner|test_kernel_comparison|all}"
        exit 1
        ;;
esac

echo "Build complete."