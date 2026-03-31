/**
 * @file device_info.cu
 * @brief Standalone GPU device information query tool
 *
 * Run this on both remote servers to collect GPU hardware information
 * for documentation and performance optimization planning.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

// Structure to hold device properties
struct DeviceInfoExtended {
    int deviceId;
    char name[256];
    int major, minor;
    size_t totalGlobalMem;
    size_t totalConstMem;
    int maxThreadsPerBlock;
    int maxThreadsPerSM;
    int maxBlocksPerSM;
    int warpSize;
    int regsPerBlock;
    int regsPerSM;
    size_t sharedMemPerBlock;
    size_t sharedMemPerSM;
    int maxSharedMemPerBlockOptin;
    int memoryBusWidth;
    int memoryClockRate;
    int gpuClockRate;
    int multiProcessorCount;
    int maxThreadsDim[3];
    int maxGridSize[3];
    int concurrentKernels;
    int asyncEngineCount;
    int l2CacheSize;
    int persistingL2CacheMaxSize;
    int singleToDoublePrecisionPerfRatio;
    int managedMemory;
    int computePreemption;
    int canAccessPeer;
};

// Get FP32 cores per SM based on architecture
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

// Check Tensor Core support
bool hasTensorCore(int major, int minor) {
    return major >= 7;
}

// Calculate peak memory bandwidth (GB/s)
double calcPeakBandwidth(int busWidth, int memClock) {
    return (double)busWidth * (double)memClock * 2.0 / 8.0 / 1e6;
}

// Calculate peak FLOPS
double calcPeakFlops(int smCount, int coresPerSM, int gpuClock) {
    return (double)smCount * coresPerSM * (double)gpuClock * 2.0 * 1e6;
}

DeviceInfoExtended getDeviceInfo(int deviceId) {
    DeviceInfoExtended info;
    memset(&info, 0, sizeof(info));

    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, deviceId);

    if (err != cudaSuccess) {
        printf("Error getting device properties for device %d: %s\n",
               deviceId, cudaGetErrorString(err));
        return info;
    }

    info.deviceId = deviceId;
    strncpy(info.name, prop.name, 255);
    info.name[255] = '\0';
    info.major = prop.major;
    info.minor = prop.minor;
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
    info.memoryClockRate = prop.memoryClockRate;
    info.gpuClockRate = prop.clockRate;
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
    info.singleToDoublePrecisionPerfRatio = prop.singleToDoublePrecisionPerfRatio;
    info.managedMemory = prop.managedMemory;
    info.computePreemption = prop.computePreemption;
    info.canAccessPeer = prop.canAccessPeer;

    return info;
}

void printConsole(const DeviceInfoExtended& info) {
    printf("\n");
    printf("============================================\n");
    printf("  GPU Device: %s\n", info.name);
    printf("============================================\n\n");

    printf("[Basic Information]\n");
    printf("  Device ID:              %d\n", info.deviceId);
    printf("  Compute Capability:     %d.%d\n", info.major, info.minor);

    printf("\n[Memory]\n");
    printf("  Global Memory:          %.2f GB\n", (double)info.totalGlobalMem / 1e9);
    printf("  Const Memory:           %.2f KB\n", (double)info.totalConstMem / 1024);
    printf("  Memory Bus Width:       %d bits\n", info.memoryBusWidth);
    printf("  Memory Clock:           %d MHz\n", info.memoryClockRate / 1000);
    printf("  Peak Bandwidth:         %.2f GB/s\n",
           calcPeakBandwidth(info.memoryBusWidth, info.memoryClockRate));

    printf("\n[Execution Resources]\n");
    printf("  SM Count:               %d\n", info.multiProcessorCount);
    printf("  Warp Size:              %d  <- IMPORTANT for kernel design\n", info.warpSize);
    printf("  Max Threads/Block:      %d\n", info.maxThreadsPerBlock);
    printf("  Max Threads/SM:         %d\n", info.maxThreadsPerSM);
    printf("  Max Blocks/SM:          %d\n", info.maxBlocksPerSM);

    printf("\n[Registers & Shared Memory]\n");
    printf("  Registers/Block:        %d\n", info.regsPerBlock);
    printf("  Registers/SM:           %d\n", info.regsPerSM);
    printf("  Shared Mem/Block:       %.2f KB\n", (double)info.sharedMemPerBlock / 1024);
    printf("  Shared Mem/SM:          %.2f KB\n", (double)info.sharedMemPerSM / 1024);
    printf("  Max Shared (Opt-in):    %.2f KB\n", (double)info.maxSharedMemPerBlockOptin / 1024);

    printf("\n[Performance]\n");
    printf("  GPU Clock:              %d MHz\n", info.gpuClockRate / 1000);
    int coresPerSM = getFP32CoresPerSM(info.major, info.minor);
    double peakFlops = calcPeakFlops(info.multiProcessorCount, coresPerSM, info.gpuClockRate);
    printf("  FP32 Cores/SM:          %d\n", coresPerSM);
    printf("  Peak FP32 FLOPS:        %.2f TFLOPS\n", peakFlops / 1e12);
    double dpRatio = info.singleToDoublePrecisionPerfRatio > 0 ?
                     1.0 / info.singleToDoublePrecisionPerfRatio : 1.0 / 32.0;
    printf("  Peak FP64 FLOPS:        %.2f TFLOPS\n", peakFlops * dpRatio / 1e12);
    printf("  SP/DP Ratio:            %d:1\n", info.singleToDoublePrecisionPerfRatio);
    printf("  Tensor Core:            %s\n", hasTensorCore(info.major, info.minor) ? "Yes" : "No");

    printf("\n[Cache & Other]\n");
    printf("  L2 Cache:               %.2f KB\n", (double)info.l2CacheSize / 1024);
    printf("  Concurrent Kernels:     %d\n", info.concurrentKernels);
    printf("  Async Engines:          %d\n", info.asyncEngineCount);

    printf("\n[Grid Limits]\n");
    printf("  Max Threads Dims:       (%d, %d, %d)\n",
           info.maxThreadsDim[0], info.maxThreadsDim[1], info.maxThreadsDim[2]);
    printf("  Max Grid Size:          (%d, %d, %d)\n",
           info.maxGridSize[0], info.maxGridSize[1], info.maxGridSize[2]);

    printf("\n============================================\n");
}

void writeMarkdownFile(const DeviceInfoExtended& info, const char* filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        printf("Error: Cannot open file %s for writing\n", filename);
        return;
    }

    file << "# GPU Hardware Information\n\n";
    file << "Device: " << info.name << "\n\n";
    file << "Generated: " << __DATE__ << " " << __TIME__ << "\n\n";

    file << "## Basic Information\n\n";
    file << "| Property | Value |\n";
    file << "|----------|-------|\n";
    file << "| Device ID | " << info.deviceId << " |\n";
    file << "| Device Name | " << info.name << " |\n";
    file << "| Compute Capability | " << info.major << "." << info.minor << " |\n";

    file << "\n## Memory Specifications\n\n";
    file << "| Property | Value |\n";
    file << "|----------|-------|\n";
    file << "| Global Memory | " << (double)info.totalGlobalMem / 1e9 << " GB |\n";
    file << "| Constant Memory | " << (double)info.totalConstMem / 1024 << " KB |\n";
    file << "| Memory Bus Width | " << info.memoryBusWidth << " bits |\n";
    file << "| Memory Clock Rate | " << info.memoryClockRate / 1000 << " MHz |\n";
    double bandwidth = calcPeakBandwidth(info.memoryBusWidth, info.memoryClockRate);
    file << "| Peak Memory Bandwidth | " << bandwidth << " GB/s |\n";

    file << "\n## Execution Resources\n\n";
    file << "| Property | Value |\n";
    file << "|----------|-------|\n";
    file << "| SM Count | " << info.multiProcessorCount << " |\n";
    file << "| Warp Size | " << info.warpSize << " |\n";
    file << "| Max Threads/Block | " << info.maxThreadsPerBlock << " |\n";
    file << "| Max Threads/SM | " << info.maxThreadsPerSM << " |\n";
    file << "| Max Blocks/SM | " << info.maxBlocksPerSM << " |\n";

    file << "\n## Register & Shared Memory\n\n";
    file << "| Property | Value |\n";
    file << "|----------|-------|\n";
    file << "| Registers/Block | " << info.regsPerBlock << " |\n";
    file << "| Registers/SM | " << info.regsPerSM << " |\n";
    file << "| Shared Memory/Block | " << (double)info.sharedMemPerBlock / 1024 << " KB |\n";
    file << "| Shared Memory/SM | " << (double)info.sharedMemPerSM / 1024 << " KB |\n";
    file << "| Max Shared Memory (Opt-in) | " << (double)info.maxSharedMemPerBlockOptin / 1024 << " KB |\n";

    file << "\n## Performance Metrics\n\n";
    file << "| Property | Value |\n";
    file << "|----------|-------|\n";
    file << "| GPU Clock Rate | " << info.gpuClockRate / 1000 << " MHz |\n";
    int coresPerSM = getFP32CoresPerSM(info.major, info.minor);
    double peakFlops = calcPeakFlops(info.multiProcessorCount, coresPerSM, info.gpuClockRate);
    file << "| FP32 Cores/SM | " << coresPerSM << " |\n";
    file << "| Peak FP32 FLOPS | " << peakFlops / 1e12 << " TFLOPS |\n";
    double dpRatio = info.singleToDoublePrecisionPerfRatio > 0 ?
                     1.0 / info.singleToDoublePrecisionPerfRatio : 1.0 / 32.0;
    file << "| Peak FP64 FLOPS | " << peakFlops * dpRatio / 1e12 << " TFLOPS |\n";
    file << "| SP/DP Performance Ratio | " << info.singleToDoublePrecisionPerfRatio << ":1 |\n";
    file << "| Tensor Core Support | " << (hasTensorCore(info.major, info.minor) ? "Yes" : "No") << " |\n";

    file << "\n## Cache & Execution Capabilities\n\n";
    file << "| Property | Value |\n";
    file << "|----------|-------|\n";
    file << "| L2 Cache Size | " << (double)info.l2CacheSize / 1024 << " KB |\n";
    file << "| Concurrent Kernels | " << info.concurrentKernels << " |\n";
    file << "| Async Copy Engines | " << info.asyncEngineCount << " |\n";
    file << "| Compute Preemption | " << (info.computePreemption ? "Yes" : "No") << " |\n";
    file << "| Managed Memory | " << (info.managedMemory ? "Yes" : "No") << " |\n";

    file << "\n## Grid & Dimension Limits\n\n";
    file << "| Property | Value |\n";
    file << "|----------|-------|\n";
    file << "| Max Threads Dimensions | (" << info.maxThreadsDim[0] << ", "
         << info.maxThreadsDim[1] << ", " << info.maxThreadsDim[2] << ") |\n";
    file << "| Max Grid Size | (" << info.maxGridSize[0] << ", "
         << info.maxGridSize[1] << ", " << info.maxGridSize[2] << ") |\n";

    file << "\n## SpMV Optimization Implications\n\n";
    file << "### Warp Size Considerations\n";
    if (info.warpSize == 32) {
        file << "- Standard NVIDIA warp size (32 threads)\n";
        file << "- Use vector-based and merge-based SpMV kernels\n";
        file << "- Each warp processes multiple rows or uses merge-based load balancing\n";
    } else if (info.warpSize == 64) {
        file << "- **Extended warp size (64 threads) - Domestic GPU**\n";
        file << "- Critical: Kernel design must adapt to 64-thread warps\n";
        file << "- Register pressure higher per warp\n";
        file << "- Shared memory usage per warp doubled\n";
        file << "- Recommended: Use larger block sizes to match warp size\n";
    }

    file << "\n### Shared Memory Strategy\n";
    file << "- Available shared memory per SM: " << (double)info.sharedMemPerSM / 1024 << " KB\n";
    file << "- Max shared memory per block: " << (double)info.maxSharedMemPerBlockOptin / 1024 << " KB\n";
    file << "- For CSR SpMV: Consider using shared memory for row pointer caching\n";

    file << "\n### Register Usage\n";
    file << "- Registers per SM: " << info.regsPerSM << "\n";
    file << "- Registers per block: " << info.regsPerBlock << "\n";
    file << "- Max threads per SM: " << info.maxThreadsPerSM << "\n";
    file << "- Recommended register usage per thread: < " << info.regsPerSM / info.maxThreadsPerSM << "\n";

    file << "\n### Bandwidth Optimization\n";
    file << "- Peak bandwidth: " << bandwidth << " GB/s\n";
    file << "- SpMV is memory-bound; optimize for bandwidth utilization\n";
    file << "- Target: >80% bandwidth utilization for large matrices\n";

    file << "\n### SM Occupancy\n";
    file << "- Target occupancy: 50-100% for SpMV kernels\n";
    file << "- Block size recommendation: " << info.warpSize * 2 << " - " << info.warpSize * 4 << "\n";

    file.close();
    printf("Markdown file written to: %s\n", filename);
}

int main(int argc, char** argv) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess || deviceCount == 0) {
        printf("No CUDA devices found or error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("\nFound %d CUDA device(s)\n", deviceCount);

    // Parse command line arguments
    int targetDevice = -1;  // -1 means all devices
    bool writeFiles = false;
    char outputFile[512] = "gpu_info.md";

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
            targetDevice = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "-f") == 0 && i + 1 < argc) {
            writeFiles = true;
            strncpy(outputFile, argv[i + 1], 511);
            i++;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [-d device_id] [-f output_file]\n", argv[0]);
            printf("Options:\n");
            printf("  -d <device_id>  Query specific device (default: all)\n");
            printf("  -f <filename>   Write output to markdown file\n");
            printf("  -h, --help      Show this help\n");
            return 0;
        }
    }

    if (targetDevice >= 0 && targetDevice < deviceCount) {
        // Query single device
        DeviceInfoExtended info = getDeviceInfo(targetDevice);
        if (info.deviceId >= 0) {
            printConsole(info);
            if (writeFiles) {
                writeMarkdownFile(info, outputFile);
            }
        }
    } else {
        // Query all devices
        for (int dev = 0; dev < deviceCount; dev++) {
            DeviceInfoExtended info = getDeviceInfo(dev);
            if (info.deviceId >= 0) {
                printConsole(info);
                if (writeFiles) {
                    char filename[520];
                    sprintf(filename, "gpu_%d_info.md", dev);
                    writeMarkdownFile(info, filename);
                }
            }
        }
    }

    // Run bandwidth test
    printf("\n[Bandwidth Test]\n");
    printf("Running simple bandwidth test...\n\n");

    for (int dev = 0; dev < deviceCount; dev++) {
        if (targetDevice >= 0 && dev != targetDevice) continue;

        cudaSetDevice(dev);
        DeviceInfoExtended info = getDeviceInfo(dev);

        size_t size = 256 * 1024 * 1024;  // 256 MB
        float* d_data;
        cudaMalloc(&d_data, size);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Warmup
        cudaMemset(d_data, 0, size);

        // Measure read bandwidth (copy to dummy)
        cudaEventRecord(start);
        cudaMemset(d_data, 1, size);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float memsetTime = 0;
        cudaEventElapsedTime(&memsetTime, start, stop);
        double memsetBW = (double)size / memsetTime / 1e6;  // GB/s

        // Measure copy bandwidth
        float* h_data = (float*)malloc(size);
        cudaEventRecord(start);
        cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float d2hTime = 0;
        cudaEventElapsedTime(&d2hTime, start, stop);
        double d2hBW = (double)size / d2hTime / 1e6;

        cudaEventRecord(start);
        cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float h2dTime = 0;
        cudaEventElapsedTime(&h2dTime, start, stop);
        double h2dBW = (double)size / h2dTime / 1e6;

        printf("Device %d (%s) Bandwidth Test:\n", dev, info.name);
        printf("  Memset Bandwidth:     %.2f GB/s\n", memsetBW);
        printf("  Device->Host BW:      %.2f GB/s\n", d2hBW);
        printf("  Host->Device BW:      %.2f GB/s\n", h2dBW);
        printf("  Peak (Theoretical):   %.2f GB/s\n",
               calcPeakBandwidth(info.memoryBusWidth, info.memoryClockRate));
        printf("  Efficiency:           %.1f%%\n",
               memsetBW / calcPeakBandwidth(info.memoryBusWidth, info.memoryClockRate) * 100);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_data);
        free(h_data);

        printf("\n");
    }

    printf("Done.\n");
    return 0;
}