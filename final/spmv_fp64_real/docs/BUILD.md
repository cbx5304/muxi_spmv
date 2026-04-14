# Build Instructions

## Prerequisites

### For NVIDIA RTX 4090

- CUDA Toolkit 11.6+ or 12.x/13.x
- nvcc compiler
- C++ compiler with C++11 support

### For Mars X201 (Domestic GPU)

- cu-bridge layer installed
- Access to GPU7 (`CUDA_VISIBLE_DEVICES=7`)
- `pre_make` wrapper for nvcc compilation

## Build Methods

### Method 1: Direct nvcc (Recommended)

#### NVIDIA RTX 4090

```bash
cd final/spmv_fp64_real
./scripts/build_direct_rtx.sh
```

Or manually:

```bash
# Set CUDA path if needed
export PATH=/usr/local/cuda-13.1/bin:$PATH

mkdir -p build_direct
nvcc -O3 -arch=sm_89 -Xcompiler -fPIC -I./include -I./src -c src/spmv_fp64.cu -o build_direct/spmv_fp64.o
nvcc -shared -arch=sm_89 -Xcompiler -fPIC -o build_direct/libspmv_fp64.so build_direct/spmv_fp64.o
nvcc -O3 -arch=sm_89 -I./include -I./src examples/simple_example.cu -Lbuild_direct -lspmv_fp64 -o build_direct/spmv_example -Xlinker -rpath,build_direct
```

Output:
- `build_direct/libspmv_fp64.so`
- `build_direct/spmv_example`

#### Mars X201

```bash
# SSH to Mars server
ssh -p 19936 chenbinxiangc@172.16.45.81

# Navigate to project
cd ~/cbx/spmv_muxi/final/spmv_fp64_real

# Build using pre_make wrapper
mkdir -p build_direct
pre_make nvcc -O3 -Xcompiler -fPIC -I./include -I./src -c src/spmv_fp64.cu -o build_direct/spmv_fp64.o
pre_make nvcc -shared -Xcompiler -fPIC -o build_direct/libspmv_fp64.so build_direct/spmv_fp64.o
pre_make nvcc -O3 -I./include -I./src examples/simple_example.cu -Lbuild_direct -lspmv_fp64 -o build_direct/spmv_example

# Run with GPU7 and library path
CUDA_VISIBLE_DEVICES=7 LD_LIBRARY_PATH=build_direct:$LD_LIBRARY_PATH ./build_direct/spmv_example matrix.mtx
```

## Tested Platforms

| Platform | Warp Size | Kernel | Status |
|----------|-----------|--------|--------|
| Mars X201 | 64 | TPR=8 | ✅ Verified |
| RTX 4090 | 32 | __ldg | ✅ Verified |

## Important Notes

### For Mars X201

1. **Do NOT set CUDA architectures**: Domestic GPU doesn't support `sm_xx` flags
2. **Use pre_make**: All nvcc commands must use `pre_make` wrapper
3. **GPU selection**: Always use GPU7 (`CUDA_VISIBLE_DEVICES=7`)
4. **Working directory**: Must work in `$HOME/cbx/spmv_muxi/` or `$HOME/spmv_muxi/`
5. **No printf**: Use logging library from `$HOME/muxi_print_bug` for debug

### For RTX 4090

1. **Set CUDA path**: If nvcc not in PATH, set `CUDA_PATH=/usr/local/cuda-13.1`
2. **Architecture**: Use `-arch=sm_89` for RTX 4090

## Troubleshooting

### CUDA Header Not Found

If you see:
```
fatal error: cuda_runtime.h: No such file or directory
```

Solution: Add CUDA include path:
```bash
export PATH=/usr/local/cuda/bin:$PATH
```

### cu-bridge Not Found

If you see:
```
pre_make: command not found
```

Solution: Check cu-bridge installation or use direct nvcc path.

### GPU Permission Error

If you see:
```
CUDA_ERROR_NO_DEVICE
```

Solution: Specify correct GPU:
```bash
CUDA_VISIBLE_DEVICES=7 ./build_direct/spmv_example matrix.mtx
```

### CUDA 13+ API Changes

If you see:
```
class "cudaDeviceProp" has no member "memoryClockRate"
```

This is expected - CUDA 13+ removed deprecated clock fields. The library uses GPU name detection instead.

## Installation

For system-wide installation:

```bash
sudo cp build_direct/libspmv_fp64.so /usr/local/lib/
sudo cp include/spmv_fp64.h /usr/local/include/
sudo ldconfig
```

## Testing New API Interfaces ⭐ NEW

The library provides three API modes:

1. **Host CSR Mode** - Library manages all memory (original API)
2. **Device CSR Mode** - User provides device CSR pointers (zero-copy)
3. **Direct Execution** - One-shot execution without handle

### Test All APIs

```bash
# Build test program
nvcc -O3 -arch=sm_89 -I./include -I./src -o build/test_new_api examples/test_new_api.cu build/libspmv_fp64.so

# Run test
LD_LIBRARY_PATH=build:$LD_LIBRARY_PATH ./build/test_new_api
```

Expected output:
```
=== Test 1: Host CSR API === PASSED
=== Test 2: Device CSR API === PASSED
=== Test 3: Direct Execution API === PASSED
=== Test 4: Error Handling === PASSED
=== All Tests Complete ===
```

### Mars X201 Testing

```bash
# SSH to Mars server
ssh -p 19936 chenbinxiangc@172.16.45.81
cd ~/spmv_muxi/final/spmv_fp64_real

# Build with pre_make
export PATH=/opt/hpcc/tools/cu-bridge/tools:$PATH
pre_make nvcc -O3 -I./include -I./src -o build/test_new_api examples/test_new_api.cu build/libspmv_fp64.so

# Run with GPU7
CUDA_VISIBLE_DEVICES=7 LD_LIBRARY_PATH=build:$LD_LIBRARY_PATH ./build/test_new_api
```

## Using in Your Project

After installation:

```cmake
include_directories(/usr/local/include)
link_directories(/usr/local/lib)
target_link_libraries(your_target PRIVATE spmv_fp64)
```

Or use directly in code:

```cpp
#include "spmv_fp64.h"

// Link with: -L/usr/local/lib -lspmv_fp64
```