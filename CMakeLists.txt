cmake_minimum_required(VERSION 3.18)

# Project name and version
project(muxi_spmv VERSION 1.0.0 LANGUAGES CXX CUDA)

# C++ and CUDA standards
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architecture settings (optional, not supported on domestic GPU)
# Set to empty string for domestic GPU (Mars X201)
set(CUDA_ARCH "" CACHE STRING "CUDA architecture (e.g., 89 for RTX 4090, empty for domestic GPU)")

# Only set CUDA architectures if specified (domestic GPU doesn't support sm_xx)
if(NOT CUDA_ARCH STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES "${CUDA_ARCH}")
    message(STATUS "CUDA architecture: ${CUDA_ARCH}")
else()
    message(STATUS "CUDA architecture not specified (for domestic GPU compatibility)")
endif()

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
    message(STATUS "Using warp size 64 (Domestic GPU)")
else()
    add_definitions(-DWARP_SIZE=32)
    message(STATUS "Using warp size 32 (NVIDIA GPU)")
endif()

# Enable error checking (default OFF for performance)
option(SPMV_ENABLE_CHECKS "Enable input validation checks" OFF)
if(SPMV_ENABLE_CHECKS)
    add_definitions(-DSPMV_ENABLE_CHECKS=1)
    message(STATUS "Input checks enabled")
else()
    add_definitions(-DSPMV_ENABLE_CHECKS=0)
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/src)

# Source files - SpMV core library
set(SPMV_CORE_SRCS
    src/spmv/csr/spmv_csr.cu
)

set(SPMV_UTILS_SRCS
    src/utils/device_info.cu
)

set(SPMV_API_SRCS
    src/api/spmv_api.cu
)

# Create SpMV library
add_library(muxi_spmv STATIC
    ${SPMV_CORE_SRCS}
    ${SPMV_UTILS_SRCS}
    ${SPMV_API_SRCS}
)

# Link CUDA runtime
find_package(CUDAToolkit REQUIRED)
target_link_libraries(muxi_spmv CUDA::cudart)

# Test executables
option(BUILD_TESTS "Build test executables" ON)

if(BUILD_TESTS)
    # Device info query
    add_executable(device_info tests/device_info.cu)

    # SpMV basic test
    add_executable(test_spmv tests/correctness/test_spmv.cu)
    target_link_libraries(test_spmv muxi_spmv)
endif()

# Install targets
install(TARGETS muxi_spmv DESTINATION lib)
install(DIRECTORY src/ DESTINATION include FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN "*.cuh")

# Custom target for clean
add_custom_target(clean_all
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_BINARY_DIR}/cmake_clean.cmake
    COMMENT "Cleaning all build artifacts"
)