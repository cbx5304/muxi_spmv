/**
 * @file device_info.cu
 * @brief GPU device information implementation
 */

#include "device_info.cuh"
#include <cmath>

namespace muxi_spmv {

// FP32 cores per SM based on architecture
int getFP32CoresPerSM(int major, int minor) {
    if (major == 2) return minor == 0 ? 32 : 48;
    if (major == 3) return 192;
    if (major == 5) return 128;
    if (major == 6) return minor == 0 ? 64 : 128;
    if (major == 7) return 64;
    if (major == 8) return minor == 0 ? 64 : 128;
    if (major == 9) return 128;
    return 128;
}

bool hasTensorCoreSupport(int major, int minor) {
    return major >= 7;
}

double getTensorCorePeakFlops(int major, int minor, int clockRate, int smCount) {
    int tensorCoresPerSM = 8;
    double flopsPerTensorCorePerCycle = 256.0 * 2.0;

    if (major == 8 && minor == 0) {
        tensorCoresPerSM = 4;
        flopsPerTensorCorePerCycle = 256.0 * 2.0 * 4.0;
    } else if (major == 8) {
        tensorCoresPerSM = 4;
        flopsPerTensorCorePerCycle = 256.0 * 2.0 * 2.0;
    } else if (major == 9) {
        tensorCoresPerSM = 4;
        flopsPerTensorCorePerCycle = 256.0 * 2.0 * 4.0;
    }

    return (double)smCount * tensorCoresPerSM * flopsPerTensorCorePerCycle *
           (double)clockRate * 1e6;
}

DeviceInfo getDeviceInfo(int deviceId) {
    DeviceInfo info;
    memset(&info, 0, sizeof(info));

    if (deviceId < 0) {
        cudaGetDevice(&deviceId);
    }

    info.deviceId = deviceId;

    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, deviceId);

    if (err != cudaSuccess) {
        return info;
    }

    strncpy(info.name, prop.name, 255);
    info.name[255] = '\0';

    info.computeCapabilityMajor = prop.major;
    info.computeCapabilityMinor = prop.minor;
    info.totalGlobalMem = prop.totalGlobalMem;
    info.totalConstMem = prop.totalConstMem;
    info.maxThreadsPerBlock = prop.maxThreadsPerBlock;
    info.maxThreadsPerSM = prop.maxThreadsPerMultiProcessor;
    info.maxBlocksPerSM = prop.maxBlocksPerMultiProcessor;
    info.warpSize = prop.warpSize;
    info.regsPerBlock = prop.regsPerBlock;
    info.regsPerSM = prop.regsPerMultiprocessor;
    info.sharedMemPerBlock = prop.sharedMemPerBlock;
    info.sharedMemPerSM = prop.sharedMemPerMultiprocessor;
    info.maxSharedMemPerBlockOptin = prop.sharedMemPerBlockOptin;
    info.memoryBusWidth = prop.memoryBusWidth;
    info.multiProcessorCount = prop.multiProcessorCount;

    info.maxThreadsDim[0] = prop.maxThreadsDim[0];
    info.maxThreadsDim[1] = prop.maxThreadsDim[1];
    info.maxThreadsDim[2] = prop.maxThreadsDim[2];
    info.maxGridSize[0] = prop.maxGridSize[0];
    info.maxGridSize[1] = prop.maxGridSize[1];
    info.maxGridSize[2] = prop.maxGridSize[2];

    info.concurrentKernels = prop.concurrentKernels;
    info.asyncEngineCount = prop.asyncEngineCount;
    info.l2CacheSize = prop.l2CacheSize;
    info.persistingL2CacheMaxSize = prop.persistingL2CacheMaxSize;
    info.managedMemory = prop.managedMemory;

    // Get clock rates via device attributes (works across CUDA versions)
    int clockRate = 0, memClockRate = 0;
    cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, deviceId);
    cudaDeviceGetAttribute(&memClockRate, cudaDevAttrMemoryClockRate, deviceId);
    info.gpuClockRate = clockRate;
    info.memoryClockRate = memClockRate;

    // Calculate derived metrics
    info.peakMemoryBandwidthGBps = calculatePeakMemoryBandwidth(
        info.memoryBusWidth, info.memoryClockRate);

    int fp32CoresPerSM = getFP32CoresPerSM(info.computeCapabilityMajor, info.computeCapabilityMinor);
    info.peakFlopsSP = (double)info.multiProcessorCount * fp32CoresPerSM *
                       (double)info.gpuClockRate * 2.0 * 1e6;

    // Double precision ratio - default 1:32 for consumer GPUs
    info.peakFlopsDP = info.peakFlopsSP / 32.0;

    info.hasTensorCore = hasTensorCoreSupport(info.computeCapabilityMajor, info.computeCapabilityMinor);
    if (info.hasTensorCore) {
        info.peakTensorCoreFlops = getTensorCorePeakFlops(
            info.computeCapabilityMajor, info.computeCapabilityMinor,
            info.gpuClockRate, info.multiProcessorCount);
    } else {
        info.peakTensorCoreFlops = 0.0;
    }

    return info;
}

void printDeviceInfo(const DeviceInfo& info) {
    printf("========================================\n");
    printf("GPU Device Information\n");
    printf("========================================\n\n");

    printf("Device ID:                    %d\n", info.deviceId);
    printf("Device Name:                  %s\n", info.name);
    printf("Compute Capability:           %d.%d\n",
           info.computeCapabilityMajor, info.computeCapabilityMinor);

    printf("\n--- Memory ---\n");
    printf("Total Global Memory:          %.2f GB\n",
           (double)info.totalGlobalMem / 1e9);
    printf("Memory Bus Width:             %d bits\n", info.memoryBusWidth);
    printf("Peak Memory Bandwidth:        %.2f GB/s\n", info.peakMemoryBandwidthGBps);

    printf("\n--- Execution Resources ---\n");
    printf("Number of SMs:                %d\n", info.multiProcessorCount);
    printf("Warp Size:                    %d\n", info.warpSize);
    printf("Max Threads per Block:        %d\n", info.maxThreadsPerBlock);
    printf("Max Threads per SM:           %d\n", info.maxThreadsPerSM);

    printf("\n--- Registers & Shared Memory ---\n");
    printf("Registers per SM:             %d\n", info.regsPerSM);
    printf("Shared Memory per SM:         %.2f KB\n",
           (double)info.sharedMemPerSM / 1024);

    printf("\n--- Performance ---\n");
    printf("Peak FP32 FLOPS:              %.2f TFLOPS\n", info.peakFlopsSP / 1e12);
    printf("Tensor Core:                  %s\n", info.hasTensorCore ? "Yes" : "No");

    printf("\n========================================\n");
}

void printDeviceInfoMarkdown(const DeviceInfo& info) {
    printf("# GPU Device Information\n\n");
    printf("Device: %s\n\n", info.name);

    printf("## Execution Resources\n\n");
    printf("| Property | Value |\n");
    printf("|----------|-------|\n");
    printf("| Warp Size | %d |\n", info.warpSize);
    printf("| SM Count | %d |\n", info.multiProcessorCount);
    printf("| Max Threads/SM | %d |\n", info.maxThreadsPerSM);
    printf("| Registers/SM | %d |\n", info.regsPerSM);
    printf("| Shared Mem/SM | %.2f KB |\n", (double)info.sharedMemPerSM / 1024);
}

} // namespace muxi_spmv