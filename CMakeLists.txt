cmake_minimum_required(VERSION 3.18)

# Project name and version
project(muxi_spmv VERSION 1.0.0 LANGUAGES CXX CUDA)

# C++ and CUDA standards
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architecture settings (will be detected from GPU info)
set(CUDA_ARCH_FLAGS "" CACHE STRING "CUDA architecture flags")

# Enable position independent code for shared library
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Compiler options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# Warp size detection - default to 32 (NVIDIA)
# Will be overridden based on GPU detection
option(WARP_SIZE_32 "Use warp size 32 (NVIDIA GPUs)" ON)
option(WARP_SIZE_64 "Use warp size 64 (Domestic GPUs)" OFF)

if(WARP_SIZE_64)
    add_definitions(-DWARP_SIZE=64)
else()
    add_definitions(-DWARP_SIZE=32)
endif()

# Enable error checking (default OFF for performance)
option(SPMV_ENABLE_CHECKS "Enable input validation checks" OFF)
if(SPMV_ENABLE_CHECKS)
    add_definitions(-DSPMV_ENABLE_CHECKS=1)
else()
    add_definitions(-DSPMV_ENABLE_CHECKS=0)
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/src)

# Source files - SpMV core library
set(SPMV_CORE_SRCS
    src/spmv/csr/spmv_csr.cu
    src/spmv/coo/spmv_coo.cu
    src/spmv/csr2/spmv_csr2.cu
    src/spmv/bsr/spmv_bsr.cu
)

set(SPMV_FORMAT_SRCS
    src/formats/sparse_matrix.cu
    src/formats/csr_format.cu
    src/formats/coo_format.cu
    src/formats/csr2_format.cu
    src/formats/bsr_format.cu
)

set(SPMV_UTILS_SRCS
    src/utils/memory.cu
    src/utils/timer.cu
    src/utils/device_info.cu
)

set(SPMV_API_SRCS
    src/api/spmv_api.cpp
    src/api/spmv_handle.cpp
)

# Create SpMV library
add_library(muxi_spmv STATIC
    ${SPMV_CORE_SRCS}
    ${SPMV_FORMAT_SRCS}
    ${SPMV_UTILS_SRCS}
    ${SPMV_API_SRCS}
)

# Set CUDA architectures if specified
if(CUDA_ARCH_FLAGS)
    set_target_properties(muxi_spmv PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH_FLAGS}")
endif()

# Link CUDA libraries
find_library(CUDA_RUNTIME_LIB cudart REQUIRED)
target_link_libraries(muxi_spmv ${CUDA_RUNTIME_LIB})

# Test executables
option(BUILD_TESTS "Build test executables" ON)

if(BUILD_TESTS)
    # Matrix generator
    add_executable(matrix_generator tests/generator/matrix_generator.cpp)
    target_link_libraries(matrix_generator muxi_spmv)

    # Correctness tests
    add_executable(spmv_correctness tests/correctness/correctness_test.cu)
    target_link_libraries(spmv_correctness muxi_spmv)

    # Performance benchmarks
    add_executable(spmv_benchmark tests/performance/benchmark.cu)
    target_link_libraries(spmv_benchmark muxi_spmv)

    # Device info query
    add_executable(device_info tests/device_info.cu)
endif()

# Install targets
install(TARGETS muxi_spmv DESTINATION lib)
install(DIRECTORY src/ DESTINATION include FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")

# Custom target for clean
add_custom_target(clean_all
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_BINARY_DIR}/cmake_clean.cmake
    COMMENT "Cleaning all build artifacts"
)