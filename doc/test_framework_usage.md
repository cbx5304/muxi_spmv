# 测试框架使用说明

## 概述

本项目包含完整的测试矩阵生成器和自动化性能测试框架，支持在NVIDIA RTX 4090和国产Mars X201 GPU上进行SpMV性能测试。

## 文件结构

```
src/
├── generators/              # 矩阵生成器
│   ├── matrix_generator.h   # 生成器接口
│   ├── matrix_generator.cu  # 实现（对角、带状、随机、集中、幂律）
│   ├── mtx_io.h             # Matrix Market格式I/O接口
│   └── mtx_io.cu            # MTX读写实现
├── benchmark/               # 性能测试
│   ├── performance_benchmark.h
│   └── performance_benchmark.cu
└── ...

tests/benchmark/
└── test_runner.cu           # 主测试程序

python/
└── analyze_results.py       # 结果分析和可视化
```

## 编译

### 国产GPU (Mars X201, warp=64)
```bash
cd ~/spmv_muxi
./scripts/build_domestic.sh test_runner
# 运行测试
CUDA_VISIBLE_DEVICES=7 ./test_runner [options]
```

### NVIDIA GPU (RTX 4090, warp=32)
```bash
cd ~/cbx/spmv_muxi
./scripts/build_nvidia.sh test_runner
# 运行测试
./test_runner [options]
```

## 使用方法

### 命令行参数
```
--rows <n>          矩阵行数
--cols <n>          矩阵列数
--sparsity <p>      稀疏度 (0.0-1.0)
--type <type>       矩阵类型: diagonal, banded, random, concentrated, powerlaw
--warmup <n>        预热迭代次数 (默认: 3)
--measure <n>       测量迭代次数 (默认: 10)
--output <file>     输出JSON结果文件
--double            使用双精度
```

### 示例
```bash
# 随机矩阵测试
./test_runner --rows 100000 --type random --sparsity 0.01

# 集中分布矩阵
./test_runner --rows 50000 --type concentrated --sparsity 0.001

# 输出结果到JSON
./test_runner --rows 100000 --type random --output results.json
```

## 性能指标

- **GFLOPS**: 每秒浮点运算次数 (2 * nnz / time)
- **带宽利用率**: 实际带宽 / 峰值带宽
- **时间统计**: 最小/最大/平均/标准差

## 已知限制

1. **国产GPU不支持cuSPARSE比较**: hcsparse API与cuSPARSE不兼容
2. **GFLOPS显示问题**: 当前版本GFLOPS计算可能显示为0（待修复）
3. **带宽利用率差异大**: 国产GPU当前kernel优化不足，利用率较低

## 测试结果示例

### RTX 4090 (100K x 1K, 1% sparsity)
- 时间: 0.0113 ms
- 带宽: 798.66 GB/s (79.2%利用率)

### Mars X201 (100K x 1K, 1% sparsity)
- 时间: 0.1733 ms
- 带宽: 51.91 GB/s (2.8%利用率)

**注意**: Mars X201性能较低的原因是当前scalar kernel未针对warp size=64优化。