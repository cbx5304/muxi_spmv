/**
 * @file device_info.cu
 * @brief GPU device information query implementation
 */

#include "device_info.cuh"
#include <cmath>

namespace muxi_spmv {

DeviceInfo getDeviceInfo(int deviceId) {
    DeviceInfo info;

    // Use current device if not specified
    if (deviceId < 0) {
        deviceId = getCurrentDevice();
    }

    info.deviceId = deviceId;

    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, deviceId);

    if (err != cudaSuccess) {
        // Return empty info on error
        memset(&info, 0, sizeof(info));
        info.deviceId = -1;
        return info;
    }

    // Basic properties
    strncpy(info.name, prop.name, 255);
    info.name[255] = '\0';

    info.computeCapabilityMajor = prop.major;
    info.computeCapabilityMinor = prop.minor;

    // Memory properties
    info.totalGlobalMem = prop.totalGlobalMem;
    info.totalConstMem = prop.totalConstMem;

    // Thread and block limits
    info.maxThreadsPerBlock = prop.maxThreadsPerBlock;
    info.maxThreadsPerSM = prop.maxThreadsPerMultiProcessor;
    info.maxBlocksPerSM = prop.maxBlocksPerMultiProcessor;

    // Warp size (critical for performance tuning)
    info.warpSize = prop.warpSize;

    // Register and shared memory
    info.regsPerBlock = prop.regsPerBlock;
    info.regsPerSM = prop.regsPerMultiprocessor;
    info.sharedMemPerBlock = prop.sharedMemPerBlock;
    info.sharedMemPerSM = prop.sharedMemPerMultiprocessor;
    info.maxSharedMemPerBlockOptin = prop.sharedMemPerBlockOptin;

    // Memory bandwidth info
    info.memoryBusWidth = prop.memoryBusWidth;
    info.memoryClockRate = prop.memoryClockRate;
    info.gpuClockRate = prop.clockRate;

    // SM count
    info.multiProcessorCount = prop.multiProcessorCount;

    // Grid and dimension limits
    info.maxThreadsDim[0] = prop.maxThreadsDim[0];
    info.maxThreadsDim[1] = prop.maxThreadsDim[1];
    info.maxThreadsDim[2] = prop.maxThreadsDim[2];
    info.maxGridSize[0] = prop.maxGridSize[0];
    info.maxGridSize[1] = prop.maxGridSize[1];
    info.maxGridSize[2] = prop.maxGridSize[2];

    // Execution capabilities
    info.concurrentKernels = prop.concurrentKernels;
    info.asyncEngineCount = prop.asyncEngineCount;

    // Cache info
    info.l2CacheSize = prop.l2CacheSize;
    info.persistingL2CacheMaxSize = prop.persistingL2CacheMaxSize;

    // Alignment and addressing
    info.textureAlignment = prop.textureAlignment;
    info.unifiedAddressing = prop.unifiedAddressing;
    info.managedMemory = prop.managedMemory;
    info.computePreemption = prop.computePreemption;
    info.canUseHostPointer = prop.canUseHostPointerRegisteredMem;
    info.canAccessPeer = prop.canAccessPeer;

    // Performance ratios
    info.singleToDoublePrecisionPerfRatio = prop.singleToDoublePrecisionPerfRatio;

    // Calculate derived metrics
    info.peakMemoryBandwidthGBps = calculatePeakMemoryBandwidth(
        info.memoryBusWidth, info.memoryClockRate);

    // Calculate peak FLOPS
    // FLOPS = SM_count * cores_per_SM * clock_frequency * ops_per_cycle
    // For CUDA cores: 128 FP32 cores per SM (modern architectures)
    int fp32CoresPerSM = getFP32CoresPerSM(info.computeCapabilityMajor, info.computeCapabilityMinor);
    info.peakFlopsSP = (double)info.multiProcessorCount * fp32CoresPerSM *
                       (double)info.gpuClockRate * 2.0 * 1e6; // 2 ops per cycle (FMA)

    // Double precision: typically 1/32 or 1/2 of SP throughput depending on architecture
    double dpRatio = info.singleToDoublePrecisionPerfRatio > 0 ?
                     1.0 / (double)info.singleToDoublePrecisionPerfRatio : 1.0 / 32.0;
    info.peakFlopsDP = info.peakFlopsSP * dpRatio;

    // Tensor Core support and FLOPS
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

int getFP32CoresPerSM(int major, int minor) {
    // CUDA core counts per SM based on architecture
    // Reference: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities
    if (major == 2) {
        // Fermi
        if (minor == 0) return 32;   // GF100
        if (minor == 1) return 48;   // GF10x
    } else if (major == 3) {
        // Kepler
        return 192;
    } else if (major == 5) {
        // Maxwell
        return 128;
    } else if (major == 6) {
        // Pascal
        if (minor == 0) return 64;   // GP100 (FP64 focused)
        if (minor == 1) return 128;  // GP10x
    } else if (major == 7) {
        // Volta, Turing
        if (minor == 0) return 64;   // Volta GV100 (FP64 + Tensor)
        if (minor == 2) return 64;   // Turing TU102
        if (minor == 5) return 64;   // Ampere GA100
    } else if (major == 8) {
        // Ampere
        if (minor == 0) return 64;   // GA100 (A100)
        if (minor == 6) return 128;  // GA10x (RTX 30 series)
        if (minor == 7) return 128;  // GA10x (RTX 30 series)
        if (minor == 9) return 128;  // Ada Lovelace
    } else if (major == 9) {
        // Hopper, Ada Lovelace
        if (minor == 0) return 128;  // Hopper H100
        if (minor == 1) return 128;  // Ada Lovelace RTX 40 series
    }
    // Default for unknown architectures
    return 128;
}

bool hasTensorCoreSupport(int major, int minor) {
    // Tensor Core support starts with Volta (SM 7.0)
    // SM 7.0: Volta (1st gen Tensor Core)
    // SM 7.2: Turing (2nd gen Tensor Core)
    // SM 7.5: Turing (2nd gen Tensor Core)
    // SM 8.0: Ampere A100 (3rd gen Tensor Core)
    // SM 8.6: Ampere RTX 30 (3rd gen Tensor Core)
    // SM 8.7: Ampere RTX 30 (3rd gen Tensor Core)
    // SM 8.9: Ada Lovelace (4th gen Tensor Core)
    // SM 9.0: Hopper H100 (4th gen Tensor Core)

    if (major >= 7) {
        return true;
    }
    return false;
}

double getTensorCorePeakFlops(int major, int minor, int clockRate, int smCount) {
    // Tensor Core FLOPS calculation
    // Each Tensor Core can perform 4x4x4 matrix multiply in one cycle
    // FP16: 256 FMA ops per Tensor Core per cycle = 512 FLOPS
    // Different architectures have different Tensor Core counts per SM

    int tensorCoresPerSM = 8; // Default for most architectures
    double flopsPerTensorCorePerCycle = 256.0 * 2.0; // FP16 FMA

    // Architecture-specific adjustments
    if (major == 7 && minor == 0) {
        // Volta GV100: 8 Tensor Cores per SM
        tensorCoresPerSM = 8;
    } else if (major == 7 && (minor == 2 || minor == 5)) {
        // Turing: 8 Tensor Cores per SM, supports INT8 and FP16
        tensorCoresPerSM = 8;
    } else if (major == 8 && minor == 0) {
        // Ampere A100: 4 Tensor Cores per SM, but much higher throughput
        tensorCoresPerSM = 4;
        flopsPerTensorCorePerCycle = 256.0 * 2.0 * 4.0; // Enhanced throughput
    } else if (major == 8 && (minor == 6 || minor == 7)) {
        // Ampere RTX 30 series
        tensorCoresPerSM = 4;
        flopsPerTensorCorePerCycle = 256.0 * 2.0;
    } else if (major == 8 && minor == 9) {
        // Ada Lovelace RTX 40 series: 4th gen Tensor Core
        tensorCoresPerSM = 4;
        flopsPerTensorCorePerCycle = 256.0 * 2.0 * 2.0; // Enhanced throughput
    } else if (major == 9) {
        // Hopper: 4th gen Tensor Core with FP8 support
        tensorCoresPerSM = 4;
        flopsPerTensorCorePerCycle = 256.0 * 2.0 * 4.0;
    }

    return (double)smCount * tensorCoresPerSM * flopsPerTensorCorePerCycle *
           (double)clockRate * 1e6;
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
    printf("Total Constant Memory:        %.2f KB\n",
           (double)info.totalConstMem / 1024);
    printf("Memory Bus Width:             %d bits\n", info.memoryBusWidth);
    printf("Memory Clock Rate:            %d MHz\n", info.memoryClockRate / 1000);
    printf("Peak Memory Bandwidth:        %.2f GB/s\n", info.peakMemoryBandwidthGBps);

    printf("\n--- Execution Resources ---\n");
    printf("Number of SMs:                %d\n", info.multiProcessorCount);
    printf("Warp Size:                    %d\n", info.warpSize);
    printf("Max Threads per Block:        %d\n", info.maxThreadsPerBlock);
    printf("Max Threads per SM:           %d\n", info.maxThreadsPerSM);
    printf("Max Blocks per SM:            %d\n", info.maxBlocksPerSM);

    printf("\n--- Registers & Shared Memory ---\n");
    printf("Registers per Block:          %d\n", info.regsPerBlock);
    printf("Registers per SM:             %d\n", info.regsPerSM);
    printf("Shared Memory per Block:      %.2f KB\n",
           (double)info.sharedMemPerBlock / 1024);
    printf("Shared Memory per SM:         %.2f KB\n",
           (double)info.sharedMemPerSM / 1024);
    printf("Max Shared Mem (Opt-in):      %.2f KB\n",
           (double)info.maxSharedMemPerBlockOptin / 1024);

    printf("\n--- Clock & Performance ---\n");
    printf("GPU Clock Rate:               %d MHz\n", info.gpuClockRate / 1000);
    printf("Peak FP32 FLOPS:              %.2f TFLOPS\n", info.peakFlopsSP / 1e12);
    printf("Peak FP64 FLOPS:              %.2f TFLOPS\n", info.peakFlopsDP / 1e12);
    printf("SP/DP Ratio:                  %d:1\n", info.singleToDoublePrecisionPerfRatio);

    printf("\n--- Tensor Core ---\n");
    printf("Tensor Core Support:          %s\n", info.hasTensorCore ? "Yes" : "No");
    if (info.hasTensorCore) {
        printf("Peak Tensor Core FLOPS:       %.2f TFLOPS\n",
               info.peakTensorCoreFlops / 1e12);
    }

    printf("\n--- Cache ---\n");
    printf("L2 Cache Size:                %.2f KB\n", (double)info.l2CacheSize / 1024);

    printf("\n--- Execution Capabilities ---\n");
    printf("Concurrent Kernels:           %d\n", info.concurrentKernels);
    printf("Async Copy Engines:           %d\n", info.asyncEngineCount);
    printf("Compute Preemption:           %s\n", info.computePreemption ? "Yes" : "No");

    printf("\n--- Grid Limits ---\n");
    printf("Max Threads Dimensions:       (%d, %d, %d)\n",
           info.maxThreadsDim[0], info.maxThreadsDim[1], info.maxThreadsDim[2]);
    printf("Max Grid Size:                (%d, %d, %d)\n",
           info.maxGridSize[0], info.maxGridSize[1], info.maxGridSize[2]);

    printf("\n========================================\n");
}

void printDeviceInfoMarkdown(const DeviceInfo& info) {
    printf("# GPU Device Information\n\n");
    printf("## Basic Information\n\n");
    printf("| Property | Value |\n");
    printf("|----------|-------|\n");
    printf("| Device ID | %d |\n", info.deviceId);
    printf("| Device Name | %s |\n", info.name);
    printf("| Compute Capability | %d.%d |\n",
           info.computeCapabilityMajor, info.computeCapabilityMinor);

    printf("\n## Memory Specifications\n\n");
    printf("| Property | Value |\n");
    printf("|----------|-------|\n");
    printf("| Total Global Memory | %.2f GB |\n",
           (double)info.totalGlobalMem / 1e9);
    printf("| Memory Bus Width | %d bits |\n", info.memoryBusWidth);
    printf("| Memory Clock Rate | %d MHz |\n", info.memoryClockRate / 1000);
    printf("| Peak Memory Bandwidth | %.2f GB/s |\n", info.peakMemoryBandwidthGBps);

    printf("\n## Execution Resources\n\n");
    printf("| Property | Value |\n");
    printf("|----------|-------|\n");
    printf("| Number of SMs | %d |\n", info.multiProcessorCount);
    printf("| Warp Size | %d |\n", info.warpSize);
    printf("| Max Threads per Block | %d |\n", info.maxThreadsPerBlock);
    printf("| Max Threads per SM | %d |\n", info.maxThreadsPerSM);
    printf("| Max Blocks per SM | %d |\n", info.maxBlocksPerSM);

    printf("\n## Registers & Shared Memory\n\n");
    printf("| Property | Value |\n");
    printf("|----------|-------|\n");
    printf("| Registers per Block | %d |\n", info.regsPerBlock);
    printf("| Registers per SM | %d |\n", info.regsPerSM);
    printf("| Shared Memory per Block | %.2f KB |\n",
           (double)info.sharedMemPerBlock / 1024);
    printf("| Shared Memory per SM | %.2f KB |\n",
           (double)info.sharedMemPerSM / 1024);
    printf("| Max Shared Mem (Opt-in) | %.2f KB |\n",
           (double)info.maxSharedMemPerBlockOptin / 1024);

    printf("\n## Performance Metrics\n\n");
    printf("| Property | Value |\n");
    printf("|----------|-------|\n");
    printf("| GPU Clock Rate | %d MHz |\n", info.gpuClockRate / 1000);
    printf("| Peak FP32 FLOPS | %.2f TFLOPS |\n", info.peakFlopsSP / 1e12);
    printf("| Peak FP64 FLOPS | %.2f TFLOPS |\n", info.peakFlopsDP / 1e12);
    printf("| SP/DP Ratio | %d:1 |\n", info.singleToDoublePrecisionPerfRatio);

    if (info.hasTensorCore) {
        printf("\n## Tensor Core\n\n");
        printf("| Property | Value |\n");
        printf("|----------|-------|\n");
        printf("| Tensor Core Support | Yes |\n");
        printf("| Peak Tensor Core FLOPS | %.2f TFLOPS |\n",
               info.peakTensorCoreFlops / 1e12);
    }

    printf("\n## Cache & Capabilities\n\n");
    printf("| Property | Value |\n");
    printf("|----------|-------|\n");
    printf("| L2 Cache Size | %.2f KB |\n", (double)info.l2CacheSize / 1024);
    printf("| Concurrent Kernels | %d |\n", info.concurrentKernels);
    printf("| Async Copy Engines | %d |\n", info.asyncEngineCount);
    printf("| Compute Preemption | %s |\n", info.computePreemption ? "Yes" : "No");
}

} // namespace muxi_spmv