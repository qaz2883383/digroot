# 上机实验：两级流水 Vector Add + msprof 性能分析

> 在远程 910B 服务器上从头创建工程、编译、运行、采集性能数据。代码全AI生成，所以，让AI看完文档看完样例，完全可以满足编程需求，模型用的deepseek v4 pro。

## 实验环境

| 项目 | 值 |
|------|-----|
| 服务器 | 远程 910B 服务器 |
| 环境 | CANN 9.1.0-beta.1 |
| NPU 架构 | dav-2201 (910B) |
| 工程路径 | 用户自建工程目录 |
| 编译方式 | cmake + make, 容器内 source set_env.sh |

## 实验目标

实现 Vector Add 的**两级流水（Double Buffer）**版本，并通过数据证明 MTE2（GM→UB 搬运）与 MTE3（UB→GM 写回）在硬件上并行执行。

## 核心设计：depth=2 双缓冲流水线

### 为什么 depth=1 是串行的

```cpp
// depth=1 → 只有 1 块物理 buffer, 所有操作必须串行
for (i = 0; i < numTiles; i++) {
    auto x = AllocTensor();       // 拿 buffer
    DataCopy(x, xGm[i*tile], N);  // MTE2: GM→UB
    EnQue(x);                     // 标记 READY

    x = DeQue();                  // 等 READY
    auto z = AllocTensor();
    Add(z, x, y, N);              // V: 计算
    EnQue(z); FreeTensor(x);

    auto zo = DeQue();
    DataCopy(zGm[i*tile], zo, N); // MTE3: UB→GM
    FreeTensor(zo);
}
// MTE2 → V → MTE3 严格串行, 每个阶段等上一个完成
```

### depth=2 如何实现重叠

```cpp
// depth=2 → 2 块物理 buffer: [buf0] [buf1], 交替使用
pipe.InitBuffer(inQueueX, 2, TILE_LENGTH * sizeof(float));  // 2块, 每块16KB

// Prologue: 预加载 tile 0 → buf0
auto x = AllocTensor(); DataCopy(x, xGm[0], N); EnQue(x);

for (i = 0; i < numTiles - 1; i++) {
    // Stage1: CopyIn tile i+1 → buf1/buf0 交替 ∣∣
    auto xN = AllocTensor();                     // 拿空闲 buffer
    DataCopy(xN, xGm[(i+1)*tile], N);            // MTE2 异步搬运
    EnQue(xN);                                   // 标记 READY (不阻塞)

    // Stage2: Compute+CopyOut tile i              ∣∣ 与 Stage1 重叠
    auto xC = DeQue();                           // 等 tile i 的 EnQue → READY
    auto zC = AllocTensor();
    Add(zC, xC, yC, N);                          // V 计算
    EnQue(zC); FreeTensor(xC);

    auto zO = DeQue();
    DataCopy(zGm[i*tile], zO, N);                // MTE3 异步搬运
    FreeTensor(zO);
}
// Epilogue: 最后一个 tile
```

### 阻塞点标记

TQue 的四个操作中，只有两个会**阻塞等待**：

| 操作 | 阻塞? | 等待什么 |
|------|-------|---------|
| `AllocTensor()` | **可能** | 等有空闲 buffer（所有 buffer 被占用且未 Free） |
| `EnQue(tensor)` | 否 | 无等待，仅写标记位 |
| `DeQue()` | **是** | 等对应 buffer 被 EnQue 标记为 READY |
| `FreeTensor()` | 否 | 无等待，仅写标记位 |

**时间线视图**：

```
Scalar(控制)   │Alloc DCop EnQ│DeQ  Add  EnQ Free DeQ DCop Free│Alloc DCop EnQ│DeQ ...
               │  └──tile1──┘  │ └─────tile0──────────────┘     │  └──tile2──┘ │
               │     ↓        │    ↑            ↓               │     ↓       │
MTE2(硬件)     │     █████████  │                 │              │     ██████████
               │    GM→UB     │                 │              │    GM→UB
               │              │                 │              │
V(硬件)        │              │       ██         │              │
               │              │      UB Add      │              │
               │              │                  │              │
MTE3(硬件)     │              │           ████████              │
               │              │           UB→GM                 │
               │              │                                 │
               ├──tile1 CopyIn─┤── tile0 Compute+CopyOut ───────┤
               │              │                                 │
               │←─ MTE2 和 MTE3 在此重叠 ───→│
```

**核心原理**：`DataCopy` 和 `Add` 都是**异步发起**（Scarlar 调用后立即返回，硬件在后台干活）。`EnQue` 不阻塞。唯一阻塞点是 `DeQue` 等数据就绪。由于 Stage1 的 `EnQue` 在 Stage2 循环开始之前就已经发出，Stage2 的 `DeQue` 可以立即拿到数据——而此时 MTE2 还在后台搬 Stage1 的数据。

---

## 实验过程

### 1. 工程文件

| 文件 | 说明 |
|------|------|
| `CMakeLists.txt` | cmake 配置，`SCENARIO_NUM=0` 编译 depth=1，`=1` 编译 depth=2 |
| `add_pipeline.asc` | 核心代码：`add_serial` (depth=1) + `add_pipeline` (depth=2)，main 中 host 侧循环 100 次计时 |
| `data_utils.h` | ReadFile / WriteFile |
| `scripts/gen_data.py` | 生成 float 测试数据 |

### 2. 编译与运行

```bash
source /usr/local/Ascend/cann/set_env.sh

# depth=1 (串行)
mkdir serial && cd serial
cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 -DSCENARIO_NUM=0 ..
make -j

# depth=2 (流水)
cd ../pipeline
cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 -DSCENARIO_NUM=1 ..
make -j
```

### 3. Host 侧计时对比

```cpp
// main 中: REPEAT=100 次 kernel launch, 取平均值
double t0 = get_time_us();
for (int r = 0; r < 100; r++) {
    add_custom<<<numBlocks, nullptr, stream>>>(...);
}
aclrtSynchronizeStream(stream);
double t1 = get_time_us();
printf("Avg per launch: %.2f us\n", (t1 - t0) / 100.0);
```

**结果**（小 shape: totalLength=65536, TILE=2048, numBlocks=4）：

| 版本 | 平均每次 launch | 
|------|----------------|
| depth=1 (串行) | 10.35 us |
| depth=2 (流水) | 9.01 us |
| **提升** | **~15%** |

### 4. msprof 硬件性能采集

```bash
# 大 shape 版本 (totalLength=1M, TILE=4096)
cd pipeline2
msprof op ./add_pipeline
cat OPPROF_*/PipeUtilization.csv
```

**关键指标**（取 4 个 block 平均值）：

#### 小 shape (64KB/block, TILE=2048)

| 指标 | 值 |
|------|-----|
| aiv_time | ~7.4 us |
| aiv_mte2_ratio | ~86% |
| aiv_mte2_active_bw | ~19 GB/s |
| aiv_mte3_ratio | ~68% |
| aiv_mte3_active_bw | ~12 GB/s |

#### 大 shape (1MB/block, TILE=4096)

| 指标 | 值 |
|------|-----|
| aiv_time | ~34 us |
| aiv_mte2_ratio | **~97.7%** |
| aiv_mte2_active_bw | **~58.5 GB/s** |
| aiv_mte3_ratio | ~50% |
| aiv_mte3_active_bw | **~57 GB/s** |

**流水重叠证明**：大 shape 下 MTE2 ratio (97.7%) + MTE3 ratio (50%) + Vec ratio (17%) + Scalar ratio (43%) ≈ **208% >> 100%**，证明 MTE2、MTE3、Vector 三条硬件流水线同时工作。

### 5. 小 shape 带宽低的原因

小 shape 下 MTE2 带宽仅 19 GB/s，放大 16 倍后到 58.5 GB/s——因为单次 DataCopy 只有 8KB 时，DMA 启动开销（指令调度、地址计算、L2 Cache 行填充）占主导。增大到 16KB 后进入稳态搬运模式。

> 注：58.5 GB/s 是**单个 AIV 核**的 MTE2 搬运带宽。整卡 HBM 总带宽 1.6TB/s 是 40 个核共享的。

---

## 关键总结

| 要做什么 | 怎么做 |
|---------|--------|
| 实现流水 | `TQue<pos, 2>` + `InitBuffer(que, 2, size)` 开双缓冲，Stage1 和 Stage2 交替用 buffer |
| 证明重叠 | `msprof op` → PipeUtilization.csv → MTE2+V+MTE3 ratio 之和 > 100% |
| 查瓶颈 | PipeUtilization.csv 中 ratio 最高的那个流水线就是瓶颈 |
| MTE2 带宽低 | 检查单次 DataCopy 是否 ≥ 16KB，不够就增大 tile length |

---

## 文件清单

| 文件 | 说明 |
|------|------|
| `add_pipeline.asc` | 内核代码，含 `add_serial` (depth=1) 和 `add_pipeline` (depth=2) |
| `CMakeLists.txt` | 编译配置，`-DSCENARIO_NUM=0/1` 切换版本 |
| `serial/` | depth=1 编译运行目录 |
| `pipeline2/` | depth=2 编译运行目录 |
