# SpMV优化测试文件索引

## 文档信息

- 创建日期: 2026-04-09
- 目的: 索引所有优化测试文件，便于后续参考

---

## 测试文件分类

### 1. 基础性能测试

| 文件 | 目的 | 关键发现 |
|------|------|----------|
| `test_hctracer.cu` | hcTracer基本用法 | hcTracer可正确测量GPU时间 |
| `test_hctracer_comprehensive.cu` | 多avgNnz测试 | LDG kernel最优 |
| `test_real_matrix_hctracer.cu` | 真实矩阵测试 | 17.5%利用率 |

### 2. 根因分析测试

| 文件 | 目的 | 关键发现 |
|------|------|----------|
| `test_l2_cache_effect.cu` | L2缓存影响 | **随机访问是根因** |
| `test_col_locality.cu` | 列访问局部性 | 99.7%随机访问 |
| `test_cache_config_impact.cu` | 缓存配置测试 | L1配置+8%提升 |

### 3. 优化技术测试

| 文件 | 目的 | 效果 |
|------|------|------|
| `test_ldg_variants.cu` | `__ldg`预取测试 | 无效 |
| `test_shared_memory_opt.cu` | 共享内存优化 | **-1000x有害** |
| `test_rcm_reordering.cu` | RCM矩阵重排序 | +1-15% |
| `test_multi_matrix_rcm.cu` | 多矩阵RCM测试 | 效果有限 |
| `test_assembly_analysis.cu` | 汇编级分析 | ILP对Mars有害 |

### 4. 平台对比测试

| 文件 | 目的 | 关键发现 |
|------|------|----------|
| `test_rtx4090_baseline.cu` | RTX 4090基线 | RTX快12.6x |
| `test_strategy_comparison.cu` | 策略对比 | SCALAR最优 |

---

## 测试结果汇总

### Mars X201优化效果矩阵

| 技术 | 内核提升 | 端到端提升 | 推荐度 |
|------|----------|-----------|--------|
| Pinned Memory | - | **+140%** | ⭐⭐⭐ |
| L1缓存配置 | +8% | +8% | ⭐⭐⭐ |
| 4t/row配置 | +2% | +2% | ⭐⭐ |
| RCM重排序 | +1-15% | +1-15% | ⭐ |
| ILP双累加 | **-36%** | -36% | ❌ |
| 共享内存 | **-1000x** | -1000x | ❌ |
| `__ldg`预取 | 0% | 0% | - |
| CSR5格式 | **-44%** | -44% | ❌ |
| Merge-based | **-91%** | -91% | ❌ |

### RTX 4090优化效果矩阵

| 技术 | 内核提升 | 端到端提升 | 推荐度 |
|------|----------|-----------|--------|
| Pinned Memory | - | +20% | ⭐⭐ |
| ILP双累加 | +3-31% | +3-31% | ⭐⭐⭐ |
| 2t/row配置 | +5% | +5% | ⭐⭐ |
| RCM重排序 | +3% | +3% | ⭐ |
| L1缓存配置 | 0% | 0% | - |
| `__ldg`预取 | 0% | 0% | - |

---

## 编译运行命令

### Mars X201

```bash
# 编译
cd /home/chenbinxiangc/spmv_muxi/tests/benchmark
pre_make nvcc -o test_l2_cache_effect test_l2_cache_effect.cu

# 运行 (必须使用GPU7)
CUDA_VISIBLE_DEVICES=7 ./test_l2_cache_effect

# hcTracer分析
CUDA_VISIBLE_DEVICES=7 hcTracer --hctx --odname results ./test_hctracer
```

### RTX 4090

```bash
# 编译
cd /home/test/cbx/spmv_muxi/tests/benchmark
/usr/local/cuda/bin/nvcc -O3 -arch=sm_89 -o test_rtx4090_baseline test_rtx4090_baseline.cu

# 运行
./test_rtx4090_baseline
```

---

## hcTracer JSON解析脚本

```python
import json

def parse_tracer(json_file):
    with open(json_file) as f:
        data = json.load(f)
    
    events = data['traceEvents']
    kernel_events = [e for e in events if 'hcLaunchKernel' in e.get('name', '') and 'dur' in e]
    
    # 移除第一个预热kernel
    if kernel_events and kernel_events[0]['dur'] > 100000:
        kernel_events = kernel_events[1:]
    
    avg_dur = sum(e['dur'] for e in kernel_events) / len(kernel_events)
    return avg_dur

# 用法
avg_us = parse_tracer('tracer_out-xxx.json')
print(f'Average kernel time: {avg_us:.1f} us')
```

---

## 关键数据记录

### Mars X201真实矩阵性能 (FP64)

```
矩阵: p0_A (1.26M行, 13.5M NNZ, avgNnz=10.71)
内核时间: 3254 μs
有效带宽: 86 GB/s
带宽利用率: 4.7%
```

### RTX 4090真实矩阵性能 (FP64)

```
矩阵: p0_A (1.26M行, 13.5M NNZ, avgNnz=10.71)
内核时间: 258 μs
有效带宽: 1082 GB/s
带宽利用率: 107%
```

### 差距分析

```
内核时间: RTX快12.6x
带宽利用率: RTX高22.8x
端到端时间: Mars快1.57x (Pinned Memory)
```

---

## 文档更新日志

| 日期 | 更新内容 |
|------|----------|
| 2026-04-09 | 创建测试文件索引 |
| 2026-04-08 | 添加L2缓存分析、RCM重排序测试 |
| 2026-04-06 | 添加FP64端到端性能测试 |