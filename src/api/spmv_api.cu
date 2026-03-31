/**
 * @file spmv_api.cpp
 * @brief SpMV API implementation
 */

#include "spmv_api.h"
#include "utils/device_info.cuh"

namespace muxi_spmv {

spmv_status_t spmv_create_handle(spmv_handle_t** handlePtr) {
    if (!handlePtr) {
        return SPMV_ERROR_INVALID_HANDLE;
    }

    spmv_handle_t* handle = new spmv_handle_t();
    if (!handle) {
        return SPMV_ERROR_MEMORY_ALLOC;
    }

    // Get current device
    cudaGetDevice(&handle->deviceId);

    // Get device properties
    DeviceInfo info = getDeviceInfo(handle->deviceId);
    handle->warpSize = info.warpSize;
    handle->smCount = info.multiProcessorCount;
    handle->maxThreadsPerBlock = info.maxThreadsPerBlock;
    handle->sharedMemPerSM = info.sharedMemPerSM;

    // Default stream and options
    handle->stream = 0;
    handle->opts = spmv_default_opts();

    *handlePtr = handle;
    return SPMV_SUCCESS;
}

spmv_status_t spmv_destroy_handle(spmv_handle_t* handle) {
    if (!handle) {
        return SPMV_ERROR_INVALID_HANDLE;
    }

    delete handle;
    return SPMV_SUCCESS;
}

spmv_status_t spmv_set_stream(spmv_handle_t* handle, cudaStream_t stream) {
    if (!handle) {
        return SPMV_ERROR_INVALID_HANDLE;
    }

    handle->stream = stream;
    handle->opts.stream = &handle->stream;
    return SPMV_SUCCESS;
}

} // namespace muxi_spmv