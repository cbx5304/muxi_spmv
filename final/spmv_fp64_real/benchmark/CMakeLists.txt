cmake_minimum_required(VERSION 3.10)
project(spmv_benchmark)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)

# Include directories for both libraries
include_directories(/home/chenbinxiangc/spmv_comp/spmv_1/nctigpu_spmv/include)
include_directories(/home/chenbinxiangc/spmv_comp/spmv_2/myspmv/install_x201/include)

# Library paths
link_directories(/home/chenbinxiangc/spmv_comp/spmv_1/nctigpu_spmv/lib)
link_directories(/home/chenbinxiangc/spmv_comp/spmv_2/myspmv/install_x201/lib)

# CUDA flags for Mars X201
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -arch=sm_70")

# Add executable
cuda_add_executable(spmv_benchmark benchmark.cpp)

# Link libraries
target_link_libraries(spmv_benchmark
    nctigpu_spmv
    spmv_fp64
    cudart
)