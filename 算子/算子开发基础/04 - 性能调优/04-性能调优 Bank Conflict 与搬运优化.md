# 性能调优：Bank Conflict、Repeat/DataBlock 与搬运优化

> AI 辅助阅读官方代码生成的文档，纯理论分析，未上机验证

## 涉及用例清单

| 用例 | 说明 | 源码 |
|------|------|------|
| `matmul_high_performance` | Matmul 高阶 API 9 级递进优化（单核→多核→MDL→Cache→常量Tiling→UnitFlag） | `04_best_practices/01_matrix_compute_practices/matmul_high_performance/` |
| `data_copy` | GM↔UB、GM↔L1 搬运效率对比（分块大小、非对齐、L2Cache、同地址冲突） | `04_best_practices/04_memory_access_practices/data_copy/` |
| `bank_conflict_nd2nz` | ND→NZ 格式转换中 UB bank 冲突的 stride 规避 | `04_best_practices/04_memory_access_practices/bank_conflict_nd2nz/` |

> 用例路径均在 `asc-devkit/examples/01_simd_cpp_api/` 下。

---

## 1. 三个用例概览

### 1.1 matmul_high_performance — 9 级递进优化

矩阵规格 M=N=K=8192，NPU 架构 2201（24核），通过 `SCENARIO_NUM` 编译宏切换 9 个优化级别：

| Case | 优化策略 | 说明 |
|------|---------|------|
| 0 | 单核基础 | 单核完成全量计算，无 tiling |
| 1 | 单核 Tiling | 固定 baseM/baseN/baseK，tiling 分块 |
| 2 | 多核 M:N=2:12 切分 | 多核并行，M 分 2 份 N 分 12 份 |
| 3 | 多核 M:N=4:6 切分 | 平衡 M/N 方向的负载 |
| 4 | MDL (Matrix Data Layout) | 开启 MDL 模式，优化 L1→L0 数据搬运效率 |
| 5 | MDL + L1Cache | 增加 L1 缓存深度（depthA1=16, stepKa=8） |
| 6 | MDL + L1Cache + L2Cache | M 轴切分为 2 份循环计算，利用 L2 缓存复用 B 矩阵 |
| 7 | 常量 Tiling | 编译期生成 MatmulApiStaticTiling，消除运行时 Scalar 开销 |
| 8 | 常量 Tiling + UnitFlag | 使能 UnitFlag，减少 M/N 尾块处理的标量判断 |

**关键代码结构**（以 Case 6 L2Cache 为例）：

```cpp
// B 矩阵复用：A 矩阵沿 M 轴切 2 份，共享同一份 B
for (int i = 0; i < 2; i++) {
    matmulObj.SetTensorA(aGlobal[offsetA + i * (M >> 1) * K]);
    matmulObj.SetTensorB(bGlobal[offsetB]);          // B 复用
    matmulObj.IterateAll(cGlobal[offsetC + i * (M >> 1) * N]);
}
```

**核心结论**：Matmul 高阶 API 的调优不涉及手写 kernel 逻辑，而是通过调整 tiling 参数（split 比例、buffer 深度、cache 模式）和编译期常量化来优化。性能提升主要来自：减少 Scalar 计算开销（常量 Tiling）、提高数据复用率（L2 Cache 复用 B 矩阵）、减少尾块分支（UnitFlag）。

### 1.2 data_copy — 搬运效率对比

在 Vector Add 场景下对比 DataCopy 和 DataCopyPad 在不同参数下的效率：

内存搬运到 UB 的性能趋势（来自 README 的性能数据）：

- **分块粒度**：单次搬运量越大效率越高，16KB+ 达到带宽峰值
- **非对齐搬运**：首地址非 32B 对齐时性能显著下降
- **L2Cache 复用**：开启 L2 Cache 可提升 ~30%（GM↔L2 带宽远高于 GM↔HBM）
- **同地址冲突**：多个 DataCopy 同时访问相同 GM 地址时产生竞争，通过偏移起始地址规避

### 1.3 bank_conflict_nd2nz — ND→NZ 格式转换的 Bank 冲突规避

两个场景对比：默认 stride → 全冲突，调整 stride → 无冲突。详见第 3 节 bank conflict 原理分析。

---

## 2. Bank Conflict 深度解析

### 2.1 什么是 Bank Conflict

UB（Unified Buffer）不是一整块连续内存，而是被拆分成 **bank**——每个 bank 是独立的存储体，可以独立读写。

**910B (dav-2201) 的 UB 结构**：

```
UB 总大小 192KB
├── 16 个 bank group（每个 group 每拍最多 1 读或 1 写）
│   └── 每个 bank group 含 3 个 bank
│       └── 每个 bank = 4KB = 128 行 × 32B/行
总共 48 个 bank
```

Bank 地址映射规则：`bank ID = (address / 32B) % 48`
Bank group 划分：`group k = banks {k, k+16, k+32}`, k ∈ [0,15]

**3510 的差异**（用于对比理解）：
- UB 256KB，16 个 bank，8 个 bank group，每组 2 个 bank
- 每个 bank group 有**两组**读写口（可 2 读 0 写 或 1 读 1 写，2201 只能 1 读或 1 写）

### 2.2 三种冲突类型

| 类型 | 条件 | 后果 |
|------|------|------|
| **读写冲突** | 读和写同时访问同一个 bank | 排队串行 |
| **写写冲突** | 多个写同时访问同一个 bank group | 排队串行 |
| **读读冲突** | 多个读同时访问同一个 bank group | 排队串行（3510：2 读可并行，超过才冲突） |

### 2.3 冲突发生的原因

Vector 计算单元每拍**同时**读/写 8 个 DataBlock（每个 32B，共 256B）。如果这 8 个 DataBlock 因地址 stride 对齐而全部落在同一个 bank group 上，本来 1 拍完成的操作需要排队 8 拍——性能直接除 8。

**典型问题场景**：

```
Add(dst, src0, src1, repeatParams):
  src0 base=0x00000 (bank 0,  group 0)
  src1 base=0x04000 (bank 32, group 0)  ← 与 src0 同 group!
  dst  base=0x08000 (bank 0,  group 0)  ← 与 src0 同 group!

同一 Repeat 内：读src0 + 读src1 → 读读冲突（同bank group不同bank）
              读src0 + 写dst   → 读写冲突（相同bank）
```

### 2.4 如何检测

使用 `msprof op` 工具跑算子后，生成 `ResourceConflictRatio.csv`，其中展示 bank group conflict 在所有指令周期中的占比。

```bash
msprof op ./your_kernel_executable
# 输出目录下的 ResourceConflictRatio.csv 包含冲突占比数据
```

### 2.5 两种规避方法

#### 方法一：调整访问 Stride

把"跳着读"改成"连续读"，避免同一 Repeat 的 8 个 DataBlock 落入同 bank group：

```cpp
// 反例：blkStride=16 → 8 DataBlock 全落在 bank group 0
BinaryRepeatParams{1, 16, 16, 8, 8, 8};

// 正例：blkStride=1 → 8 DataBlock 均匀分散到 8 个不同 bank group
BinaryRepeatParams{1, 1, 1, 8, 8, 8};
```

#### 方法二：地址 Padding

多申请一点 UB 空间做偏移，让各 buffer 的起始地址落到不同 bank group：

```cpp
// 反例：x起始bank0, y起始bank32(同group0) → 读读冲突常量
pipe.InitBuffer(inQueueX, 1, 4096 * sizeof(float));
pipe.InitBuffer(inQueueY, 1, 4096 * sizeof(float));

// 正例：x 多申请 256B (8 DataBlock)，y 地址偏移到 bank 8 (group 8)
pipe.InitBuffer(inQueueX, 1, 4096 * sizeof(float) + 256);
pipe.InitBuffer(inQueueY, 1, 4096 * sizeof(float));
// 效果：x 和 y 起始 bank group 不同，无读读冲突
```

### 2.6 文档索引

| 文档 | 路径 | 关键行 |
|------|------|--------|
| Bank冲突概述 | `SIMD算子性能优化/内存访问/避免UB的bank冲突/概述.md` | 第 7-32 行（结构总览+dav对比） |
| 2201 规避指南 | `避免bank冲突（NPU架构版本2201）.md` | 第 10-17, 261-361 行 |
| 3510 规避指南 | `避免bank冲突（NPU架构版本3510）.md` | 第 3-10, 216-318 行 |
| SIMT Bank冲突 | `SIMT算子性能优化/内存访问/避免UB的Bank冲突.md` | 第 55-91 行（Padding 优化 25.7%） |
| msprof 检测 | `调试调优/性能调优.md` | 第 47, 100 行（ResourceConflictRatio.csv） |

---

## 3. Repeat / DataBlock 机制

### 3.1 硬件约束

Vector 计算单元每拍处理 **256 字节**（`NPU架构版本3510.md` 第 55 行）。一个 DataBlock = 32B，所以每拍 = 8 个 DataBlock。这是硬件固定宽度，不可配置。

### 3.2 Repeat = 一拍的硬件执行

一次 Vector API 调用（`Add`、`Mul`、`Exp` 等）的硬件行为等价于：

```
while (repeatTimes--) {
    读取 8 个 DataBlock（256B）
    执行计算
    写回 8 个 DataBlock（256B）
}
```

### 3.3 三个控制参数

```cpp
Repeat 0: [DB0] [DB1] [DB2] [DB3] [DB4] [DB5] [DB6] [DB7]
           ├── dataBlockStride ──┤  ← 同一 Repeat 内相邻 DB 的间隔（单位：DataBlock）

Repeat 0 DB0 地址: base
Repeat 1 DB0 地址: base + repeatStride × 32B  ← 相邻 Repeat 起始地址间隔
```

| 参数 | 范围 | 默认值 | 含义 |
|------|------|--------|------|
| `repeatTimes` | ≤255 | API 内部按 count 算 | 执行多少次 Repeat |
| `dataBlockStride` (blkStride) | ≥0 | 1（连续） | 同一 Repeat 内 8 个 DB 的间隔。0=始终复用第一个 DB |
| `repeatStride` (repStride) | ≥0 | 8（连续） | 相邻 Repeat 起始地址的间隔。0=反复计算同一组 DB |

### 3.4 与 Bank Conflict 的关系

```
dataBlockStride=1  →  8 DB 地址: 0,1,2,3,4,5,6,7  →  8 bank: 0,1,2,3,4,5,6,7
                    →  8 bank group: 0,1,2,3,4,5,6,7  →  全部不同 → 无冲突 ✓

dataBlockStride=16 →  8 DB 地址: 0,16,32,48,64,80,96,112
                    →  8 bank: 0,16,32,0,16,32,0,16
                    →  全部 bank group 0  →  全冲突 ✗ (8 拍完成)
```

**根因**：`dataBlockStride` 恰好匹配 bank group 映射周期时（每 16 个 bank 循环一次 bank group），所有 DB 落入同一 group。

### 3.5 文档索引

| 文档 | 关键行 | 内容 |
|------|--------|------|
| 术语表 | 第 148-156, 374-387 行 | DataBlock、Repeat/RepeatStride/RepeatTimes 定义 |
| 高维切分 | 第 23-84 行 | 8 DataBlock 机制、blkStride/repStride 四种场景详解+示意图 |
| UnaryRepeatParams | 全文 | 结构体字段定义与使用说明 |

---

## 4. 性能优化建议汇总

来自各文档的优化建议（`优化建议总览表.md`）：

| 优先级 | 优化项 | 说明 |
|--------|--------|------|
| 高 | 多核拆分 | 利用全部物理 core，确保负载均衡 |
| 高 | Double Buffer | 使能 double buffer 实现搬运与计算的流水重叠 |
| 高 | 减少 Bank 冲突 | 调整 stride 或地址 padding |
| 高 | 设置合理的 L2 CacheMode | 默认使能 L2Cache，数据量不大时无需干预 |
| 中 | L2 Cache 切分 | 当数据总量超过 L2 容量时，手动切分 L2 复用窗口 |
| 中 | 使能 L1 Cache | Matmul 场景中增加 L1 depth，提高数据复用 |
| 中 | 常量 Tiling | 编译期生成 tiling，消除运行时 Scalar 开销 |
| 中 | 数据搬运对齐 | 保证 GM↔UB 搬运大小 = DataBlock 整数倍且 ≥16KB |
| 中 | 避免同地址访问 | 多个 DataCopy 同时访问相同 GM 地址时产生竞争 |
| 低 | UnitFlag | 减少尾块处理标量判断（Matmul 高阶 API） |
| 低 | 纯 Cube 模式 | 跳过 KFC 消息框架开销（纯矩阵乘场景） |

---

## 5. 附录：关键术语

| 术语 | 含义 |
|------|------|
| **Repeat** | Vector 计算单元的一拍：同时读 8 个 DataBlock（256B）→ 计算 → 写回 |
| **DataBlock** | Vector 计算的基本单元，32B。UB 每行也是 32B |
| **dataBlockStride** | 同一 Repeat 内相邻 DataBlock 的间隔（单位=DataBlock） |
| **repeatStride** | 相邻 Repeat 起始 DataBlock 的间隔（单位=DataBlock） |
| **Bank** | UB 的独立存储体（2201: 4KB/个, 3510: 16KB/个） |
| **Bank Group** | 多个 bank 的逻辑组。2201: 3 bank/组, 3510: 2 bank/组 |
| **Bank Conflict** | 多个并发访问撞到同一 bank group，排队串行 |
| **MDL** | Matrix Data Layout 模式，优化 L1→L0 搬运（Matmul 调优） |
| **L2 Cache** | 硬件自动管理的 GM 缓存（~7TB/s vs GM ~1.6TB/s） |
| **常量 Tiling** | 编译期生成 tiling 数据为 `constexpr`，消除运行时标量计算 |
| **UnitFlag** | 硬件自动处理 M/N 尾块，无需 kernel 侧 `SetTail` |
