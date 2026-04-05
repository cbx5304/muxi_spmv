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

# 核心源文件
CORE_SRCS="$SRC_DIR/spmv/csr/spmv_csr.cu $SRC_DIR/spmv/csr/spmv_sparse_optimized.cu $SRC_DIR/spmv/csr5/spmv_csr5.cu $SRC_DIR/spmv/ellpack/spmv_ellpack.cu $SRC_DIR/utils/device_info.cu $SRC_DIR/api/spmv_api.cu"

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
    # Use relocatable device code compilation for proper template instantiation
    # Step 1: Compile each source to device-code-only object

    # Compile core sources
    OBJ_FILES=""
    for src in $CORE_SRCS $GEN_SRCS $BENCH_SRCS; do
        obj=$(basename $src .cu).o
        echo "Compiling $src..."
        $NVCC $OPT_FLAGS $WARP_SIZE_FLAG $INCLUDE_FLAG -dc $src -o $obj
        OBJ_FILES="$OBJ_FILES $obj"
    done

    # Step 2: Compile test_runner and device-link all objects
    $NVCC $OPT_FLAGS $WARP_SIZE_FLAG $INCLUDE_FLAG \
        tests/benchmark/test_runner.cu \
        $OBJ_FILES \
        -o test_runner \
        -dl -lcudart

    # Cleanup object files
    rm -f $OBJ_FILES
}

build_test_ellpack() {
    echo "Building test_ellpack..."
    # Compile core sources
    OBJ_FILES=""
    for src in $CORE_SRCS $GEN_SRCS; do
        obj=$(basename $src .cu).o
        echo "Compiling $src..."
        $NVCC $OPT_FLAGS $WARP_SIZE_FLAG $INCLUDE_FLAG -dc $src -o $obj
        OBJ_FILES="$OBJ_FILES $obj"
    done

    # Step 2: Compile test_ellpack and device-link all objects
    $NVCC $OPT_FLAGS $WARP_SIZE_FLAG $INCLUDE_FLAG \
        tests/benchmark/test_ellpack.cu \
        $OBJ_FILES \
        -o test_ellpack \
        -dl -lcudart

    # Cleanup object files
    rm -f $OBJ_FILES
}

build_test_reorder() {
    echo "Building test_reorder..."
    # Compile core sources
    OBJ_FILES=""
    for src in $CORE_SRCS $GEN_SRCS; do
        obj=$(basename $src .cu).o
        echo "Compiling $src..."
        $NVCC $OPT_FLAGS $WARP_SIZE_FLAG $INCLUDE_FLAG -dc $src -o $obj
        OBJ_FILES="$OBJ_FILES $obj"
    done

    # Step 2: Compile test_reorder and device-link all objects
    $NVCC $OPT_FLAGS $WARP_SIZE_FLAG $INCLUDE_FLAG \
        tests/test_column_reorder.cu \
        $OBJ_FILES \
        -o test_reorder \
        -dl -lcudart

    # Cleanup object files
    rm -f $OBJ_FILES
}

build_test_global_reorder() {
    echo "Building test_global_reorder..."
    OBJ_FILES=""
    for src in $CORE_SRCS $GEN_SRCS; do
        obj=$(basename $src .cu).o
        echo "Compiling $src..."
        $NVCC $OPT_FLAGS $WARP_SIZE_FLAG $INCLUDE_FLAG -dc $src -o $obj
        OBJ_FILES="$OBJ_FILES $obj"
    done
    $NVCC $OPT_FLAGS $WARP_SIZE_FLAG $INCLUDE_FLAG \
        tests/test_global_reorder.cu \
        $OBJ_FILES \
        -o test_global_reorder \
        -dl -lcudart
    rm -f $OBJ_FILES
}

build_test_kernel_comparison() {
    echo "Building test_kernel_comparison..."
    OBJ_FILES=""
    for src in $CORE_SRCS $GEN_SRCS; do
        obj=$(basename $src .cu).o
        echo "Compiling $src..."
        $NVCC $OPT_FLAGS $WARP_SIZE_FLAG $INCLUDE_FLAG -dc $src -o $obj
        OBJ_FILES="$OBJ_FILES $obj"
    done
    $NVCC $OPT_FLAGS $WARP_SIZE_FLAG $INCLUDE_FLAG \
        tests/test_kernel_comparison.cu \
        $OBJ_FILES \
        -o test_kernel_comparison \
        -dl -lcudart
    rm -f $OBJ_FILES
}

build_test_sparse_optimized() {
    echo "Building test_sparse_optimized..."
    OBJ_FILES=""
    for src in $CORE_SRCS $GEN_SRCS; do
        obj=$(basename $src .cu).o
        echo "Compiling $src..."
        $NVCC $OPT_FLAGS $WARP_SIZE_FLAG $INCLUDE_FLAG -dc $src -o $obj
        OBJ_FILES="$OBJ_FILES $obj"
    done
    $NVCC $OPT_FLAGS $WARP_SIZE_FLAG $INCLUDE_FLAG \
        tests/test_sparse_optimized.cu \
        $OBJ_FILES \
        -o test_sparse_optimized \
        -dl -lcudart
    rm -f $OBJ_FILES
}

build_test_warp_optimization() {
    echo "Building test_warp_optimization..."
    OBJ_FILES=""
    for src in $GEN_SRCS; do
        obj=$(basename $src .cu).o
        echo "Compiling $src..."
        $NVCC $OPT_FLAGS $WARP_SIZE_FLAG $INCLUDE_FLAG -dc $src -o $obj
        OBJ_FILES="$OBJ_FILES $obj"
    done
    $NVCC $OPT_FLAGS $WARP_SIZE_FLAG $INCLUDE_FLAG \
        tests/test_warp_optimization.cu \
        $OBJ_FILES \
        -o test_warp_optimization \
        -dl -lcudart
    rm -f $OBJ_FILES
}

build_test_virtual_warp_size() {
    echo "Building test_virtual_warp_size..."
    OBJ_FILES=""
    for src in $GEN_SRCS; do
        obj=$(basename $src .cu).o
        echo "Compiling $src..."
        $NVCC $OPT_FLAGS $WARP_SIZE_FLAG $INCLUDE_FLAG -dc $src -o $obj
        OBJ_FILES="$OBJ_FILES $obj"
    done
    $NVCC $OPT_FLAGS $WARP_SIZE_FLAG $INCLUDE_FLAG \
        tests/test_virtual_warp_size.cu \
        $OBJ_FILES \
        -o test_virtual_warp_size \
        -dl -lcudart
    rm -f $OBJ_FILES
}

build_all() {
    build_device_info
    build_test_spmv
    build_test_runner
    build_test_ellpack
    build_test_reorder
    build_test_global_reorder
    build_test_kernel_comparison
    build_test_sparse_optimized
    build_test_warp_optimization
    build_test_virtual_warp_size
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
    test_ellpack)
        build_test_ellpack
        ;;
    test_reorder)
        build_test_reorder
        ;;
    test_global_reorder)
        build_test_global_reorder
        ;;
    test_kernel_comparison)
        build_test_kernel_comparison
        ;;
    test_sparse_optimized)
        build_test_sparse_optimized
        ;;
    test_warp_optimization)
        build_test_warp_optimization
        ;;
    test_virtual_warp_size)
        build_test_virtual_warp_size
        ;;
    all)
        build_all
        ;;
    *)
        echo "Usage: $0 {device_info|test_spmv|test_runner|test_ellpack|test_reorder|test_global_reorder|test_kernel_comparison|test_sparse_optimized|test_warp_optimization|test_virtual_warp_size|all}"
        echo "Note: Run with CUDA_VISIBLE_DEVICES=7 for GPU execution"
        exit 1
        ;;
esac

echo "Build complete."