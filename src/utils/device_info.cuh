/**
 * @file device_info.cuh
 * @brief GPU device information query header
 *
 * Provides comprehensive GPU device information including:
 * - Compute capability
 * - Memory bandwidth
 * - Registers per SM
 * - Shared memory per SM
 * - Warp size
 * - Max threads per block/SM
 */

#ifndef DEVICE_INFO_CUH_
#define DEVICE_INFO_CUH_

#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

namespace muxi_spmv {

/**
 * @brief GPU device information structure
 */
struct DeviceInfo {
    int deviceId;                   ///< Device ID
    char name[256];                 ///< Device name
    int computeCapabilityMajor;     ///< Compute capability major version
    int computeCapabilityMinor;     ///< Compute capability minor version
    size_t totalGlobalMem;          ///< Total global memory (bytes)
    size_t totalConstMem;           ///< Total constant memory (bytes)
    int maxThreadsPerBlock;         ///< Maximum threads per block
    int maxThreadsPerSM;            ///< Maximum threads per SM
    int maxBlocksPerSM;             ///< Maximum blocks per SM
    int warpSize;                   ///< Warp size (32 for NVIDIA, 64 for some GPUs)
    int regsPerBlock;               ///< Registers per block
    int regsPerSM;                  ///< Registers per SM
    size_t sharedMemPerBlock;       ///< Shared memory per block (bytes)
    size_t sharedMemPerSM;          ///< Shared memory per SM (bytes)
    int maxSharedMemPerBlockOptin;  ///< Max shared memory with opt-in (bytes)
    int memoryBusWidth;             ///< Memory bus width (bits)
    int memoryClockRate;            ///< Memory clock rate (kHz)
    int gpuClockRate;               ///< GPU clock rate (kHz)
    int multiProcessorCount;        ///< Number of SMs
    int maxThreadsDim[3];           ///< Maximum threads in each dimension
    int maxGridSize[3];             ///< Maximum grid size in each dimension
    int concurrentKernels;          ///< Maximum concurrent kernels
    int asyncEngineCount;           ///< Number of async copy engines
    int l2CacheSize;                ///< L2 cache size (bytes)
    int persistingL2CacheMaxSize;   ///< Max persisting L2 cache size (bytes)
    int textureAlignment;           ///< Texture alignment requirement
    int unifiedAddressing;          ///< Unified addressing support
    int managedMemory;              ///< Managed memory support
    int computePreemption;          ///< Compute preemption support
    int canUseHostPointer;          ///< Can use host pointer directly
    int canAccessPeer;              ///< Can access peer devices
    int singleToDoublePrecisionPerfRatio; ///< SP/DP performance ratio

    // Derived metrics
    double peakMemoryBandwidthGBps; ///< Peak memory bandwidth (GB/s)
    double peakFlopsSP;             ///< Peak single-precision FLOPS
    double peakFlopsDP;             ///< Peak double-precision FLOPS
    double peakTensorCoreFlops;     ///< Peak Tensor Core FLOPS (if available)
    bool hasTensorCore;             ///< Has Tensor Core support
};

/**
 * @brief Get current device ID
 */
inline int getCurrentDevice() {
    int device = 0;
    cudaGetDevice(&device);
    return device;
}

/**
 * @brief Set device to use
 */
inline cudaError_t setDevice(int deviceId) {
    return cudaSetDevice(deviceId);
}

/**
 * @brief Get number of available devices
 */
inline int getDeviceCount() {
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
}

/**
 * @brief Get comprehensive device information
 * @param deviceId Device ID to query (default: current device)
 * @return DeviceInfo structure with all device properties
 */
DeviceInfo getDeviceInfo(int deviceId = -1);

/**
 * @brief Print device information to stdout
 */
void printDeviceInfo(const DeviceInfo& info);

/**
 * @brief Print device information in markdown format for documentation
 */
void printDeviceInfoMarkdown(const DeviceInfo& info);

/**
 * @brief Calculate peak memory bandwidth
 */
inline double calculatePeakMemoryBandwidth(int memoryBusWidth, int memoryClockRate) {
    // Bandwidth = BusWidth * ClockRate * 2 (DDR) / 8 (bits to bytes) / 1e9 (to GB/s)
    return (double)memoryBusWidth * (double)memoryClockRate * 2.0 / 8.0 / 1e6;
}

/**
 * @brief Check if device supports Tensor Core operations
 */
bool hasTensorCoreSupport(int computeMajor, int computeMinor);

/**
 * @brief Get Tensor Core peak FLOPS for device
 */
double getTensorCorePeakFlops(int computeMajor, int computeMinor, int gpuClockRate, int smCount);

/**
 * @brief Get warp size for device
 * @return 32 for NVIDIA GPUs, 64 for some domestic GPUs
 */
inline int getWarpSize(int deviceId = -1) {
    DeviceInfo info = getDeviceInfo(deviceId);
    return info.warpSize;
}

} // namespace muxi_spmv

#endif // DEVICE_INFO_CUH_