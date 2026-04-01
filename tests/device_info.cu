/**
 * @file device_info.cu
 * @brief Standalone GPU device information query tool
 *
 * Compatible with CUDA 11.x through CUDA 13.x and domestic GPUs
 * Note: Does not use CUDA_VERSION macro for domestic GPU compatibility
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Structure to hold device properties
struct DeviceInfo {
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
    size_t sharedMemPerBlockOptin;
    int memoryBusWidth;
    int memoryClockRate;    // in kHz
    int gpuClockRate;       // in kHz
    int multiProcessorCount;
    int maxThreadsDim[3];
    int maxGridSize[3];
    int concurrentKernels;
    int asyncEngineCount;
    int l2CacheSize;
    int persistingL2CacheMaxSize;
    int managedMemory;
};

// Get FP32 cores per SM based on architecture
int getFP32CoresPerSM(int major, int minor) {
    switch (major) {
        case 2: return minor == 0 ? 32 : 48;      // Fermi
        case 3: return 192;                        // Kepler
        case 5: return 128;                        // Maxwell
        case 6: return minor == 0 ? 64 : 128;     // Pascal
        case 7: return 64;                         // Volta, Turing
        case 8: return minor == 0 ? 64 : 128;     // Ampere
        case 9: return 128;                        // Hopper, Ada
        default: return 128;
    }
}

// Check Tensor Core support
bool hasTensorCore(int major, int minor) {
    return major >= 7;
}

// Calculate peak memory bandwidth (GB/s)
double calcPeakBandwidth(int busWidth, int memClockKHz) {
    return (double)busWidth * (double)memClockKHz * 2.0 / 8.0 / 1e6;
}

// Calculate peak FLOPS
double calcPeakFlops(int smCount, int coresPerSM, int gpuClockKHz) {
    return (double)smCount * coresPerSM * (double)gpuClockKHz * 2.0 * 1e6;
}

DeviceInfo getDeviceInfo(int deviceId) {
    DeviceInfo info;
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
    info.sharedMemPerBlockOptin = prop.sharedMemPerBlockOptin;
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

    return info;
}

void printConsole(const DeviceInfo& info) {
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

    printf("\n[Execution Resources - CRITICAL for SpMV]\n");
    printf("  SM Count:               %d\n", info.multiProcessorCount);
    printf("  Warp Size:              %d  <- IMPORTANT!\n", info.warpSize);
    printf("  Max Threads/Block:      %d\n", info.maxThreadsPerBlock);
    printf("  Max Threads/SM:         %d\n", info.maxThreadsPerSM);
    printf("  Max Blocks/SM:          %d\n", info.maxBlocksPerSM);

    printf("\n[Registers & Shared Memory]\n");
    printf("  Registers/Block:        %d\n", info.regsPerBlock);
    printf("  Registers/SM:           %d\n", info.regsPerSM);
    printf("  Shared Mem/Block:       %.2f KB\n", (double)info.sharedMemPerBlock / 1024);
    printf("  Shared Mem/SM:          %.2f KB\n", (double)info.sharedMemPerSM / 1024);
    printf("  Max Shared (Opt-in):    %.2f KB\n", (double)info.sharedMemPerBlockOptin / 1024);

    printf("\n[Performance]\n");
    printf("  GPU Clock:              %d MHz\n", info.gpuClockRate / 1000);
    int coresPerSM = getFP32CoresPerSM(info.major, info.minor);
    double peakFlops = calcPeakFlops(info.multiProcessorCount, coresPerSM, info.gpuClockRate);
    printf("  FP32 Cores/SM:          %d\n", coresPerSM);
    printf("  Peak FP32 FLOPS:        %.2f TFLOPS\n", peakFlops / 1e12);
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

void writeMarkdown(const DeviceInfo& info, const char* filename) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        printf("Error: Cannot open file %s for writing\n", filename);
        return;
    }

    fprintf(file, "# GPU Hardware Information\n\n");
    fprintf(file, "**Device:** %s\n\n", info.name);
    fprintf(file, "**Generated:** %s %s\n\n", __DATE__, __TIME__);

    fprintf(file, "## Basic Information\n\n");
    fprintf(file, "| Property | Value |\n");
    fprintf(file, "|----------|-------|\n");
    fprintf(file, "| Device ID | %d |\n", info.deviceId);
    fprintf(file, "| Device Name | %s |\n", info.name);
    fprintf(file, "| Compute Capability | %d.%d |\n", info.major, info.minor);

    fprintf(file, "\n## Memory Specifications\n\n");
    fprintf(file, "| Property | Value |\n");
    fprintf(file, "|----------|-------|\n");
    fprintf(file, "| Global Memory | %.2f GB |\n", (double)info.totalGlobalMem / 1e9);
    fprintf(file, "| Constant Memory | %.2f KB |\n", (double)info.totalConstMem / 1024);
    fprintf(file, "| Memory Bus Width | %d bits |\n", info.memoryBusWidth);
    fprintf(file, "| Memory Clock Rate | %d MHz |\n", info.memoryClockRate / 1000);
    double bandwidth = calcPeakBandwidth(info.memoryBusWidth, info.memoryClockRate);
    fprintf(file, "| Peak Memory Bandwidth | %.2f GB/s |\n", bandwidth);

    fprintf(file, "\n## Execution Resources\n\n");
    fprintf(file, "| Property | Value |\n");
    fprintf(file, "|----------|-------|\n");
    fprintf(file, "| SM Count | %d |\n", info.multiProcessorCount);
    fprintf(file, "| Warp Size | %d |\n", info.warpSize);
    fprintf(file, "| Max Threads/Block | %d |\n", info.maxThreadsPerBlock);
    fprintf(file, "| Max Threads/SM | %d |\n", info.maxThreadsPerSM);
    fprintf(file, "| Max Blocks/SM | %d |\n", info.maxBlocksPerSM);

    fprintf(file, "\n## Register & Shared Memory\n\n");
    fprintf(file, "| Property | Value |\n");
    fprintf(file, "|----------|-------|\n");
    fprintf(file, "| Registers/Block | %d |\n", info.regsPerBlock);
    fprintf(file, "| Registers/SM | %d |\n", info.regsPerSM);
    fprintf(file, "| Shared Memory/Block | %.2f KB |\n", (double)info.sharedMemPerBlock / 1024);
    fprintf(file, "| Shared Memory/SM | %.2f KB |\n", (double)info.sharedMemPerSM / 1024);
    fprintf(file, "| Max Shared Memory (Opt-in) | %.2f KB |\n", (double)info.sharedMemPerBlockOptin / 1024);

    fprintf(file, "\n## Performance Metrics\n\n");
    fprintf(file, "| Property | Value |\n");
    fprintf(file, "|----------|-------|\n");
    fprintf(file, "| GPU Clock Rate | %d MHz |\n", info.gpuClockRate / 1000);
    int coresPerSM = getFP32CoresPerSM(info.major, info.minor);
    double peakFlops = calcPeakFlops(info.multiProcessorCount, coresPerSM, info.gpuClockRate);
    fprintf(file, "| FP32 Cores/SM | %d |\n", coresPerSM);
    fprintf(file, "| Peak FP32 FLOPS | %.2f TFLOPS |\n", peakFlops / 1e12);
    fprintf(file, "| Tensor Core Support | %s |\n", hasTensorCore(info.major, info.minor) ? "Yes" : "No");

    fprintf(file, "\n## Cache & Execution Capabilities\n\n");
    fprintf(file, "| Property | Value |\n");
    fprintf(file, "|----------|-------|\n");
    fprintf(file, "| L2 Cache Size | %.2f KB |\n", (double)info.l2CacheSize / 1024);
    fprintf(file, "| Concurrent Kernels | %d |\n", info.concurrentKernels);
    fprintf(file, "| Async Copy Engines | %d |\n", info.asyncEngineCount);
    fprintf(file, "| Managed Memory | %s |\n", info.managedMemory ? "Yes" : "No");

    fprintf(file, "\n## Grid & Dimension Limits\n\n");
    fprintf(file, "| Property | Value |\n");
    fprintf(file, "|----------|-------|\n");
    fprintf(file, "| Max Threads Dimensions | (%d, %d, %d) |\n",
            info.maxThreadsDim[0], info.maxThreadsDim[1], info.maxThreadsDim[2]);
    fprintf(file, "| Max Grid Size | (%d, %d, %d) |\n",
            info.maxGridSize[0], info.maxGridSize[1], info.maxGridSize[2]);

    fprintf(file, "\n## SpMV Optimization Implications\n\n");
    fprintf(file, "### Warp Size Considerations\n");
    if (info.warpSize == 32) {
        fprintf(file, "- Standard NVIDIA warp size (32 threads)\n");
        fprintf(file, "- Use vector-based and merge-based SpMV kernels\n");
        fprintf(file, "- Each warp processes multiple rows or uses merge-based load balancing\n");
    } else if (info.warpSize == 64) {
        fprintf(file, "- **Extended warp size (64 threads) - Domestic GPU**\n");
        fprintf(file, "- Critical: Kernel design must adapt to 64-thread warps\n");
        fprintf(file, "- Register pressure higher per warp\n");
        fprintf(file, "- Shared memory usage per warp doubled\n");
        fprintf(file, "- Recommended: Use larger block sizes to match warp size\n");
    }

    fprintf(file, "\n### Shared Memory Strategy\n");
    fprintf(file, "- Available shared memory per SM: %.2f KB\n", (double)info.sharedMemPerSM / 1024);
    fprintf(file, "- Max shared memory per block: %.2f KB\n", (double)info.sharedMemPerBlockOptin / 1024);
    fprintf(file, "- For CSR SpMV: Consider using shared memory for row pointer caching\n");

    fprintf(file, "\n### Register Usage\n");
    fprintf(file, "- Registers per SM: %d\n", info.regsPerSM);
    fprintf(file, "- Registers per block: %d\n", info.regsPerBlock);
    fprintf(file, "- Max threads per SM: %d\n", info.maxThreadsPerSM);
    fprintf(file, "- Recommended register usage per thread: < %d\n", info.regsPerSM / info.maxThreadsPerSM);

    fprintf(file, "\n### Bandwidth Optimization\n");
    fprintf(file, "- Peak bandwidth: %.2f GB/s\n", bandwidth);
    fprintf(file, "- SpMV is memory-bound; optimize for bandwidth utilization\n");
    fprintf(file, "- Target: >80%% bandwidth utilization for large matrices\n");

    fprintf(file, "\n### SM Occupancy\n");
    fprintf(file, "- Target occupancy: 50-100%% for SpMV kernels\n");
    fprintf(file, "- Block size recommendation: %d - %d\n", info.warpSize * 2, info.warpSize * 4);

    fclose(file);
    printf("Markdown file written to: %s\n", filename);
}

int main(int argc, char** argv) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess || deviceCount == 0) {
        printf("No CUDA devices found or error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("\n============================================\n");
    printf("  SpMV GPU Device Information Query Tool\n");
    printf("============================================\n\n");

    printf("Found %d CUDA device(s)\n\n", deviceCount);

    // Parse command line arguments
    int targetDevice = -1;
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
        DeviceInfo info = getDeviceInfo(targetDevice);
        if (info.deviceId >= 0) {
            printConsole(info);
            if (writeFiles) {
                writeMarkdown(info, outputFile);
            }
        }
    } else {
        for (int dev = 0; dev < deviceCount; dev++) {
            DeviceInfo info = getDeviceInfo(dev);
            if (info.deviceId >= 0) {
                printConsole(info);
                if (writeFiles) {
                    char filename[520];
                    sprintf(filename, "gpu_%d_info.md", dev);
                    writeMarkdown(info, filename);
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
        DeviceInfo info = getDeviceInfo(dev);

        size_t size = 256 * 1024 * 1024;  // 256 MB
        float* d_data;
        cudaMalloc(&d_data, size);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Warmup
        cudaMemset(d_data, 0, size);

        // Measure memset bandwidth
        cudaEventRecord(start);
        cudaMemset(d_data, 1, size);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float memsetTime = 0;
        cudaEventElapsedTime(&memsetTime, start, stop);
        double memsetBW = (double)size / memsetTime / 1e6;

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

        double peakBW = calcPeakBandwidth(info.memoryBusWidth, info.memoryClockRate);

        printf("Device %d (%s) Bandwidth Test:\n", dev, info.name);
        printf("  Memset Bandwidth:     %.2f GB/s\n", memsetBW);
        printf("  Device->Host BW:      %.2f GB/s\n", d2hBW);
        printf("  Host->Device BW:      %.2f GB/s\n", h2dBW);
        printf("  Peak (Theoretical):   %.2f GB/s\n", peakBW);
        printf("  Efficiency:           %.1f%%\n", memsetBW / peakBW * 100);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_data);
        free(h_data);

        printf("\n");
    }

    printf("Done.\n");
    return 0;
}