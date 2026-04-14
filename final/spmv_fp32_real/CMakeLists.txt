cmake_minimum_required(VERSION 3.18)
project(spmv_fp32 VERSION 1.0.0 LANGUAGES CXX CUDA)

# ==================== Options ====================
option(SPMV_FP32_BUILD_SHARED "Build shared library" ON)
option(SPMV_FP32_BUILD_EXAMPLES "Build examples" OFF)
option(SPMV_FP32_ENABLE_BENCHMARK "Enable benchmark mode" ON)

# ==================== CUDA Configuration ====================
# NOTE: For Mars X201 (domestic GPU):
#   - Use 'pre_make cmake' instead of 'cmake'
#   - Do NOT set CUDA_ARCH (domestic GPU doesn't support sm_xx)
#   - The cu-bridge layer handles the translation

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

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ==================== Compiler Flags ====================
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# ==================== Library ====================
if(SPMV_FP32_BUILD_SHARED)
    add_library(spmv_fp32 SHARED
        src/spmv_fp32.cu
    )
else()
    add_library(spmv_fp32 STATIC
        src/spmv_fp32.cu
    )
endif()

# Set output directory
set_target_properties(spmv_fp32 PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# Include directories
target_include_directories(spmv_fp32
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

# Compile options
target_compile_options(spmv_fp32 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
)

# Cache configuration hint for kernels
# PreferL1 is critical for Mars X201 performance
target_compile_definitions(spmv_fp32 PRIVATE
    SPMV_ENABLE_CHECKS=1
)

# Link CUDA runtime
find_package(CUDAToolkit REQUIRED)
target_link_libraries(spmv_fp32 CUDA::cudart)

# ==================== Installation ====================
install(TARGETS spmv_fp32
    EXPORT spmv_fp32-targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include
)

install(EXPORT spmv_fp32-targets
    NAMESPACE spmv_fp32::
    DESTINATION lib/cmake/spmv_fp32
)

# ==================== Package Configuration ====================
include(CMakePackageConfigHelpers)

configure_package_config_file(
    "${CMAKE_SOURCE_DIR}/cmake/spmv_fp32Config.cmake.in"
    "${CMAKE_BINARY_DIR}/spmv_fp32Config.cmake"
    INSTALL_DESTINATION lib/cmake/spmv_fp32
)

write_basic_package_version_file(
    "${CMAKE_BINARY_DIR}/spmv_fp32ConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_BINARY_DIR}/spmv_fp32Config.cmake"
    "${CMAKE_BINARY_DIR}/spmv_fp32ConfigVersion.cmake"
    DESTINATION lib/cmake/spmv_fp32
)

# ==================== Print Configuration ====================
message(STATUS "")
message(STATUS "=== SPMV FP32 Library Configuration ===")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Library type: ${SPMV_FP32_BUILD_SHARED} (shared)")
message(STATUS "CUDA architecture: ${CUDA_ARCH}")
message(STATUS "Build examples: ${SPMV_FP32_BUILD_EXAMPLES}")
message(STATUS "")

# Mars X201 specific note
message(STATUS "NOTE: For Mars X201 (domestic GPU):")
message(STATUS "  - Run: pre_make cmake .. (instead of cmake ..)")
message(STATUS "  - Run: pre_make make (instead of make)")
message(STATUS "  - Do NOT set CUDA_ARCH (leave empty)")
message(STATUS "")