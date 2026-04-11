# 沐曦工具链部分使用指令

## 一、反汇编与二进制分析
### 1. `htobjdump --print-code --source ****/aabbb.so > 1.txt`
+ **功能**：使用 `htobjdump`对共享库 `aabbb.so` 进行反汇编。
+ **参数说明**：
    - `--print-code`：打印机器码对应的汇编指令。
    - `--source`：尝试关联源代码（需有调试信息）。
+ **目的**：分析 `.so` 文件中 GPU 内核（kernel）的汇编实现，用于调试性能或正确性问题。
+ **输出重定向到 **`1.txt`：便于后续查看或比对。

> 类似于 NVIDIA 的 `cuobjdump`。
>

## 二、编译器与汇编器命令
### 1. `echo "" | llvm-mc --arch=htc --disassemble`
+ **功能**：向 `llvm-mc`（LLVM 汇编器/反汇编器）传入空输入，要求以 `htc` 架构进行反汇编。
+ **实际效果**：无输出（因输入为空），但可能用于测试工具链是否正常。
+ **常见用法**：通常配合十六进制机器码使用（见下一条）。

### 2.
```bash
echo "0x10,0x00,0x23,0x00,0x02,0x00,0x00,0x00,0xe0,0x23,0x00,0x00,0x40,0x10,0x00,0x00," \
  | /opt/x201-3.2.0.0/restricted/htgpu_llvm/bin/llvm-mc --arch=htc --disassemble
```

+ **功能**：将一段十六进制机器码（逗号分隔）通过管道传给 `llvm-mc`，要求反汇编成 `htc` 架构的汇编指令。
+ **目的**：验证某段二进制指令的含义，常用于调试内核崩溃或非法指令问题。

### 3. `htcc -hpcc-device-input thao.asm --device-bin -o t2.out`
+ **功能**：使用编译器 `htcc` 将汇编文件 `thao.asm` 编译为设备二进制（device binary）。
+ **参数**：
    - `-hpcc-device-input`：指定输入是设备端代码（非主机 C++）。
    - `--device-bin`：生成设备可执行二进制（而非 PTX 或中间表示）。
    - `-o t2.out`：输出文件。
+ **用途**：手写汇编内核 → 编译为可加载的 GPU 二进制。

## 三、驱动与内核模块管理
### 1. `modprobe -r mars && modprobe mars pri_mem_sz=36`
+ **功能**：
    - `modprobe -r mars`：卸载名为 `mars` 的 GPU 内核驱动模块。
    - `modprobe mars pri_mem_sz=36`：重新加载，并传入参数 `pri_mem_sz=36`。
+ **参数含义**：
    - `pri_mem_sz=36`：可能表示为每个 GPU 上下文分配 36KB的私有内存（private memory）。
+ **目的**：调整驱动内存配置，解决 OOM 或初始化失败问题。

## 四、编译选项与环境变量
### 1. `-aop --enable-device-O -O0`
+ **上下文**：这是传递给编译器`htcc`的选项。
+ **含义**：
    - `-aop`：可能是厂商自定义选项（如 “Advanced Optimization Pass” 或调试开关）。
    - `--enable-device-O`：启用设备端优化。
    - `-O0`：关闭CPU优化（用于调试）。

> 常见于调试阶段：保留符号、禁用优化以便单步调试。
>

### 2. -S -aop -lineinfo


+ **上下文**：这是传递给编译器`htcc`的选项。
+ **含义**：
    - `-S`：生成汇编。
    - `-aop`：“Advanced Optimization Pass” 调试开关。
    - `--lineinfo`：加入行号信息
+ **使用示例**：

premake nvcc ./test.cpp -S -aop -lineinfo

> 可以生成带行号的优化后的汇编文件
>

### 3. `HPCC_LAUNCH_BLOCKING=0 cmd`
+ **功能**：设置 `HPCC_LAUNCH_BLOCKING=0` 后运行 `cmd`。
+ **含义**：
    - 若为 `1`：GPU 内核启动同步（阻塞 CPU 直到 kernel 完成），便于调试。
    - `0`：异步启动（默认高性能模式）。
+ **此处设为 0**：可能为了性能测试，或避免死锁。

### 4. `-mllvm -marsgpu-inlinescope=200`
+ **功能**：向 LLVM 后端传递参数。
+ **含义**：设置 Mars GPU 的 inline scope 限制为 200（可能控制函数内联深度或作用域大小）。
+ **用途**：解决编译时内联爆炸或栈溢出问题。

---

### 5. 
```bash
-MetaXGPU-assume-external-call-stack-size=8000
-MarsGPU-assume-external-call-stack-size=8000
```

+ **功能**：告知编译器外部调用（如 host → device callback）所需栈空间为 8000 字节。
+ **目的**：防止 GPU 栈溢出，尤其在递归或深层调用场景。

---

### 6. `-mllvm -marsgpu-disable-promote-alloca-to-vector=1`
+ **功能**：禁止将局部数组（alloca）提升为向量寄存器。
+ **原因**：避免导致寄存器压力过大，引发性能下降或编译失败。



## 五、调试与日志环境变量
### 1.
```bash
export ISU_FASTMODEL=1
export DEBUG_ITRACE=1
export ITRACE_VERBOSE=2
export DEBUG_ITRACE_VERBOSE=1
```

+ **功能**：启用 GPU 指令级追踪（instruction trace）和调试日志。
+ **用途**：当内核行为异常时，记录每条指令执行情况，用于分析逻辑错误或硬件 bug。
