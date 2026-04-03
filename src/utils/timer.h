/**
 * @file timer.h
 * @brief GPU timing utilities for performance measurement
 */

#ifndef TIMER_H_
#define TIMER_H_

#include <cuda_runtime.h>
#include <cstdio>

namespace muxi_spmv {

/**
 * @brief GPU timer using CUDA events for accurate timing
 */
class GpuTimer {
public:
    GpuTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }

    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start(cudaStream_t stream = 0) {
        cudaEventRecord(start_, stream);
    }

    void stop(cudaStream_t stream = 0) {
        cudaEventRecord(stop_, stream);
        cudaEventSynchronize(stop_);
    }

    float elapsed_ms() {
        float ms;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }

private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
};

/**
 * @brief CPU timer for host-side measurements
 */
class CpuTimer {
public:
    CpuTimer() : started_(false) {}

    void start() {
        start_ = std::chrono::high_resolution_clock::now();
        started_ = true;
    }

    void stop() {
        stop_ = std::chrono::high_resolution_clock::now();
    }

    float elapsed_ms() {
        if (!started_) return 0.0f;
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_ - start_);
        return duration.count() / 1000.0f;
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
    std::chrono::high_resolution_clock::time_point stop_;
    bool started_;
};

} // namespace muxi_spmv

#endif // TIMER_H_