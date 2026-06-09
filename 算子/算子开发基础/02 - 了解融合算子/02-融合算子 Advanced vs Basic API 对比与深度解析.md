# 融合算子 Advanced vs Basic API 对比与深度解析

> 用AI协助看完代码后生成的文档，纯理论分析，未上机验证

## 涉及用例清单

| 用例 | 说明 | 源码 |
|------|------|------|
| `matmul_leakyrelu_advanced_api` | Matmul+Bias+LeakyRelu 融合，高阶 API | `03_fusion_operation/matmul_leakyrelu_advanced_api/matmul_leakyrelu_advanced_api.asc` |
| `matmul_leakyrelu_basic_api` | Matmul+LeakyRelu 融合，基础 API / 静态 Tensor 编程 | `03_fusion_operation/matmul_leakyrelu_basic_api/matmul_leakyrelu_basic_api.asc` |
| `基本架构.md` | AI Core 架构概述（含存储层次、计算单元、数据通路） | `docs/guide/编程指南/高级编程/硬件实现/基本架构.md` |
| `NPU架构版本2201.md`（架构规格） | dav-2201 存储容量、数据格式、bank 结构、UB 布局 | `docs/guide/编程指南/高级编程/硬件实现/架构规格/NPU架构版本2201.md` |
| `NPU架构版本2201.md`（硬件约束） | dav-2201 对齐要求、搬运粒度限制、bank 冲突、容量上限 | `docs/guide/编程指南/高级编程/硬件实现/硬件约束/NPU架构版本2201.md` |

> 用例路径均在 `asc-devkit/examples/01_simd_cpp_api/00_introduction/` 下。

---

## 1. Advanced API vs Basic API 核心差异一览

### 1.1 结构对比（不重要）

| 维度 | Advanced API | Basic API |
|------|-------------|-----------|
| **split 方式** | N 轴切分，singleCoreN=64 | M 轴切分，singleCoreM=256 |
| **核部署** | 1 AIC + 2 AIV，1 组 (numBlocks=1) | 2 AIC + 4 AIV，2 组 (numBlocks=2) |
| **M/N 规格** | M=512, N=128, K=128 | M=512, N=128, K=128 |
| **Cube block** | baseM=128, baseN=64, baseK=128 | baseM=256, baseN=128, baseK=128 |
| **Bias** | 有 (Matmul+Bias+LeakyRelu) | 无 (Matmul+LeakyRelu) |
| **数据类型** | C/fBias 为 float；LeakyRelu 在 float 上做 | 全程 half；Fixpipe 做 F32→F16 精度转换 |

### 1.2 数据通路对比

> **关键前提**：910B (dav-2201) 上 AIC 和 AIV 之间**没有直接硬件通路**。核间数据只能通过 GM 中转。L0C→UB 直连硬通道是 3510 架构才新增的（见 `2201到3510架构变更.md` 第 24-26 行）。两个 API 使用的**硬件数据通路相同**，区别在于框架层封装程度和中间格式转换。

#### Advanced API：KFC 框架隐式中转
```
AIC:  GM→L1→L0A/L0B→Mmad→L0C ──[KFC框架 GM中转]──→  ██
                                                       UB(VECIN)
AIV:                                           ──→  ██
     UB(VECIN) → LeakyRelu(float) → DataCopy → GM(C)
```
- 结果仍经过 GM（KFC 隐式搬运），但免去 Fixpipe 的 Nz→ND 格式转换和 F32→F16 精度转换
- 对开发者呈现为"VECIN 上直接可用"，AIV 侧一次 DataCopy(UB→GM) 写回最终结果

#### Basic API：两步显式 GM 操作
```
AIC:  GM→L1→L0A/L0B→Mmad→L0C→Fixpipe(Nz→ND,F32→F16)→GM
AIV: GM→DataCopyPad→UB→LeakyRelu(half)→DataCopyPad→GM
```
- 多了一次 Fixpipe（含格式+精度转换）+ 一次 DataCopyPad(GM→UB) 显式读
- 多了 Nz→ND 格式转换和 F32→F16→F32 的精度损失

#### 总结
两个 API 在 910B 上**物理路径相同**（都经过 GM），差异在于：
1. Advanced API 用 KFC 框架隐式搬运代替显式 Fixpipe+DataCopyPad，省了一次 Nz↔ND/F32↔F16 转换
2. Advanced API 全程 float 精度（无 F32→F16 损失），Basic API 是 half 精度

### 1.3 AIC/AIV 协调方式对比

| | Advanced API | Basic API |
|---|---|---|
| **协调机制** | AIV 调用 `Iterate()` / `GetTensorC()` **驱动** AIC | AIC 完成 Mmad+Fixpipe 后 `CrossCoreSetFlag`；AIV `CrossCoreWaitFlag` |
| **同步粒度** | 隐式（Matmul 库 / TPipe / TQue 内部处理） | 7 个显式硬件事件 + 核间同步 |
| **AIV 分块** | 每个 AIV 处理当前 base 块全部 (128×64) | 1 个 Cube 的结果由 2 个 Vector 核各做一半 (`baseM/2`)；`GetBlockIdx()%2` 区分 |
| **AIC 驱动** | AIV 侧发消息，通过 KFC 消息框架中转后在 AIC 侧执行 | AIC 和 AIV 各自独立运行，仅通过 CrossCoreSetFlag/WaitFlag 同步 |

### 1.4 架构与 Tiling 方式对比

| | Advanced API | Basic API |
|---|---|---|
| **Tiling 生成** | 运行时 `MultiCoreMatmulTiling` + `SetFixSplit` | 全编译期模板参数，main 中硬编码 |
| **架构适配** | 库内部处理（`__NPU_ARCH__` 条件编译在库内部） | `#if __NPU_ARCH__ == 2201` / `3510` 显式分支 |
| **Cube 侧代码量** | ~20 行 | ~80 行 |

---

## 2. REGIST_MATMUL_OBJ 详解

> 对一个完全陌生的开发者，怎么知道为什么要调、如何传参？

### 2.1 是什么

`REGIST_MATMUL_OBJ` 是一个宏（定义于 `include/adv_api/matmul/matmul_intf.h:37`），等价于 `REGIST_CUBE_OBJ`。它的作用是**初始化 Matmul 对象**，将此对象与 TPipe（资源管理器）、系统 workspace 和 tiling 数据绑定。

**核心文档**：`docs/api/SIMD-API/高阶API/矩阵计算/Matmul-Kernel侧接口/REGIST_MATMUL_OBJ.md`

### 2.2 为什么要调用

在 MIX 模式（`__mix__`）下，AIC 核不会主动执行代码——它需要由 AIV 核通过消息机制驱动。`REGIST_MATMUL_OBJ` 内部完成：

1. **建立 AIV→AIC 通信通道**：通过 KFC（Kernel Function Communication，核间通信调用）框架，在 AIV 和 AIC 之间建立消息通道
2. **分配系统 workspace**：`GetSysWorkSpacePtr()` 提供 Matmul 库内部需要的临时存储空间。此 workspace 也包含 KFC 消息队列存放的数据区
3. **绑定 tiling 数据**：将 Host 侧生成的 `TCubeTiling` 传入，Matmul 据此完成分块搬运和计算调度
4. **分配 AIV 子核 ID**：在分离架构中自动初始化和赋值 AIV 核 ID

> **文档依据**：
> - "Matmul API默认使用MIX模式，即**用户从AIV侧发起消息，通过消息通信框架中转消息后，在AIC侧执行Matmul计算。**" — `Matmul高阶API使能纯Cube模式.md` 第 5 行
> - "调用本接口后，**AIC核不会主动执行接口，仅在AIV核执行到下述接口后，才会触发AIC核的执行。**" — `REGIST_MATMUL_OBJ.md` 第 62 行
> - "REGIST_MATMUL_OBJ**内部是由框架管理AIC和AIV的核间通信**。" — `KfcWorkspace/构造函数与析构函数.md` 第 91 行
> - "KFC全称为kernel function call，表示**核间通信调用**。" — `KfcWorkspace/构造函数与析构函数.md` 第 52 行
>
> **实现层次**（来自 `impl/adv_api/detail/kfc/` 源码）：
> - **AIC 侧**：`REGIST_CUBE_OBJ` 展开为 `KfcServer` 轮询循环——`while (server.isRun()) { server.Run() }`，不断接收消息并根据 service ID 分发（`SERVICE_ID_MATMUL` → 执行 Matmul，`SERVICE_QUIT` → 退出）
> - **AIV 侧**：展开为 `KfcCommClient`，`Iterate()`/`GetTensorC()` 等调用通过 Client 发消息给 AIC
> - **2201 上的传输介质**："NPU220架构中核间通信通过GM来完成" — `NPU架构版本3510.md` 第 245-247 行（3510 改为用 SSBuffer）

### 2.3 如何传参

```cpp
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), matmulObj, &tiling);
//                ↑       ↑                      ↑           ↑
//                TPipe   系统workspace指针       Matmul对象   Tiling数据
```

- **第1参 `&pipe`**: TPipe 对象的指针，所有 Matmul API 内部的 buffer 初始化和事件管理都通过它
- **第2参 `GetSysWorkSpacePtr()`**: 系统 workspace 指针。Host 侧通过 `GetLibApiWorkSpaceSize()` 查询所需大小并分配，传给 kernel。Matmul 库内部用此空间做临时存储
- **第3参 `matmulObj`**: Matmul 对象，类型为 `matmul::Matmul<...>`，已通过模板参数描述了 A/B/C/Bias 的位置、格式和数据类型
- **第4参 `&tiling`**: Host 侧 `GetTiling()` 生成的 `TCubeTiling` 结构体的指针。包含分核、分块、buffer 空间规划等所有 tiling 信息

### 2.4 约束

| 约束 | 说明 |
|------|------|
| **调用顺序** | 分离模式中必须在 `InitBuffer` **之前**调用（Matmul 先声明 workspace 需求，TPipe 再分配队列） |
| **最多4个对象** | 一个 kernel 中最多定义 4 个 Matmul 对象 |
| **flagId 占用** | N 个 Matmul 对象占用 flagId 范围 [0, 2N-1]，最多 [0, 7]。开发者不宜同时使用 `CrossCoreSetFlag` 以避免冲突 |
| **单对象可省略 tiling** | 只有一个 Matmul 对象时可不传 tiling 参数，后续单独调用 `mm.Init(&tiling)` |

### 2.5 文档入口

| 文档 | 内容 |
|------|------|
| `REGIST_MATMUL_OBJ.md` | **主 API 文档**：功能说明、参数说明、约束说明、调用示例 |
| `Matmul使用说明.md` | Matmul 完整开发流程（5 步），REGIST_MATMUL_OBJ 是第 3 步 |
| `Init-85.md` | 分离 tiling 传入方式的补充接口 |
| `SetSubBlockIdx.md` | 约束：使用 REGIST_MATMUL_OBJ 后不可再调用 SetSubBlockIdx（内部已处理） |
| `Matmul高阶API使能纯Cube模式.md` | 说明 MIX 模式下的消息通信开销，REGIST_MATMUL_OBJ 的作用是屏蔽 AIV/AIC 通信复杂度 |

---

## 3. Advanced API 代码逐行注释

### 3.1 Kernel 声明与参数

```cpp
// __mix__(1,2): 分离模式，1个AIC(Cube核)+2个AIV(Vector核)编为一组
// __gm__: 全局内存地址空间，数据存在HBM(GM)上
// __kfc_workspace__: 标记为KFC框架的workspace指针
__global__ __mix__(1, 2) void matmul_leakyrelu_custom(
    __gm__ uint8_t* a,                              // 输入矩阵A: M×K = 512×128, half类型
    __gm__ uint8_t* b,                              // 输入矩阵B: K×N = 128×128, half类型
    __gm__ uint8_t* bias,                           // Bias向量: 长度N=128, float类型
    __gm__ uint8_t* c,                              // 输出矩阵C: M×N = 512×128, float类型
    __kfc_workspace__ __gm__ uint8_t* workspace,    // Matmul库内部workspace
    AscendC::tiling::TCubeTiling tiling)            // Host侧生成的tiling数据(值传递)
```

### 3.2 资源声明

```cpp
{
    AscendC::TPipe pipe;                         // [1] TPipe: 片上队列资源管理器

    // [2-5] GlobalTensor: 描述GM上矩阵的"视图"(首地址+元素个数)
    AscendC::GlobalTensor<half> aGlobal;
    AscendC::GlobalTensor<half> bGlobal;
    AscendC::GlobalTensor<float> cGlobal;
    AscendC::GlobalTensor<float> biasGlobal;

    // [6] VECIN队列: depth=1, 存放Matmul输出中间结果(UB上)
    //     depth=1 → AIC/AIV串行: 写完→读完→才能再写
    AscendC::TQue<AscendC::TPosition::VECIN, 1> mmOutQueue;

    // [7] VECOUT队列: depth=1, 存放LeakyRelu输出结果(UB上)
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> reluOutQueue;

    // [8] Matmul模板参数: 依次描述A/B/C/Bias的位置、格式、数据类型
    //     C类型的关键: VECIN → Matmul结果写到Vector侧可读的UB位置，不写GM
    matmul::Matmul<
        matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>,      // A: GM, ND格式, half
        matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>,      // B: GM, ND格式, half
        matmul::MatmulType<AscendC::TPosition::VECIN, CubeFormat::ND, float>,  // C: VECIN(UB), ND格式, float
        matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>>     // Bias: GM, ND格式, float
        matmulObj;
```

### 3.3 初始化与数据绑定

```cpp
    // [9] REGIST_MATMUL_OBJ: 初始化Matmul对象
    //     绑定 pipe(资源管理) + workspace(临时空间) + tiling(分块规划)
    //     内部: 建立AIV↔AIC KFC通信通道，分配子核ID
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), matmulObj, &tiling);

    // [10-13] SetGlobalBuffer: 将GM裸指针绑定到GlobalTensor
    //        参数2 = 该Tensor可访问的元素总数
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(a), tiling.M * tiling.Ka);    // 512×128
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(b), tiling.Kb * tiling.N);    // 128×128
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c), tiling.M * tiling.N);    // 512×128
    biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bias), tiling.N);         // 128

    // [14] N轴切分偏移: singleCoreN=64, Core0偏移0, Core1偏移64
    uint32_t blockIdx = AscendC::GetBlockIdx();        // AIV核索引: 0或1
    uint32_t nCoreOffset = blockIdx * tiling.singleCoreN; // Core0=0, Core1=64
    bGlobal = bGlobal[nCoreOffset];       // B矩阵: 跳过前core已处理的N列
    biasGlobal = biasGlobal[nCoreOffset]; // Bias: 跳过前core已处理的N列
    cGlobal = cGlobal[nCoreOffset];       // C矩阵: 写回当前core负责的N列区域

    // [15-17] 绑定输入到Matmul对象
    matmulObj.SetTensorA(aGlobal);   // A不偏移(完整的512×128, N轴分片不影响A)
    matmulObj.SetTensorB(bGlobal);   // B已偏移(每核读K×64子矩阵)
    matmulObj.SetBias(biasGlobal);   // Bias已偏移(每核读64个元素)
```

### 3.4 队列初始化与迭代计算

```cpp
    // [18-19] InitBuffer: 通过TPipe为TQue分配片上内存
    //         num=1: 不开启double buffer(开则num=2)
    //         len = baseM * baseN * sizeof(float) = 128*64*4 = 32KB
    pipe.InitBuffer(mmOutQueue, 1, tiling.baseM * tiling.baseN * sizeof(float));
    pipe.InitBuffer(reluOutQueue, 1, tiling.baseM * tiling.baseN * sizeof(float));

    // [20] 迭代循环: 每核负责512×64输出, base块128×64, 512/128=4次Iterate
    uint32_t computeRound = 0;
    while (matmulObj.template Iterate<true>()) {  // Iterate<true> = 同步模式
        // === 每次Iterate: AIC计算128×64×128的Matmul → 结果入VECIN ===

        // [21] AllocTensor: 从mmOutQueue申请一块LocalTensor(VECIN位置, UB上)
        AscendC::LocalTensor<float> mmOutLocal = mmOutQueue.AllocTensor<float>();

        // [22] GetTensorC<true>: 同步模式, 将Matmul结果取到mmOutLocal
        //     第2参false: 不使能原子累加
        //     第3参true:  按连续地址顺序写入(便于Vector计算直接读取)
        matmulObj.template GetTensorC<true>(mmOutLocal, false, true);

        // [23] EnQue: 将mmOutLocal推入队列, 标记"写入完成", 唤醒下游DeQue
        mmOutQueue.EnQue(mmOutLocal);

        // [24] DeQue: 等待上游EnQue信号, 从队列取出已就绪的Matmul中间结果
        mmOutLocal = mmOutQueue.DeQue<float>();

        // [25] AllocTensor: 从reluOutQueue申请输出缓存
        AscendC::LocalTensor<float> reluOutLocal = reluOutQueue.AllocTensor<float>();

        // [26] LeakyRelu: 逐元素激活, 负值×0.001
        AscendC::LeakyRelu(reluOutLocal, mmOutLocal, static_cast<float>(0.001),
                           tiling.baseM * tiling.baseN);

        // [27] EnQue: 激活结果入队
        reluOutQueue.EnQue(reluOutLocal);

        // [28] DeQue: 取出激活结果, 准备写回GM
        reluOutLocal = reluOutQueue.DeQue<float>();

        // [29] FreeTensor: 释放VECIN缓存(mmOutLocal), 允许下一轮Iterate复用
        mmOutQueue.FreeTensor(mmOutLocal);

        // [30] 计算写回C矩阵的偏移: 第i轮Iterate写C[i*baseM : (i+1)*baseM][0:N]行
        uint64_t cOffset = static_cast<uint64_t>(computeRound) * tiling.baseM * tiling.N;

        // [31] DataCopy: UB→GM, 带stride的二维搬运
        //     DataCopyParams{count, len, srcStrideIn, dstStrideIn}
        //     count = baseM = 128行
        //     len   = baseN*sizeof(float)/32 = 64*4/32 = 8个32B块(每行)
        //     srcStrideIn = 0 (UB中行间连续存放)
        //     dstStrideIn = (N-baseN)*sizeof(float)/32 = 64*4/32 = 8个32B块
        //               → 写每行后跳过N-baseN=64列, 到达下一行对应位置
        AscendC::DataCopy(cGlobal[cOffset], reluOutLocal,
            AscendC::DataCopyParams{static_cast<uint16_t>(tiling.baseM),
                static_cast<uint16_t>(tiling.baseN * sizeof(float) / 32), 0,
                static_cast<uint16_t>((tiling.N - tiling.baseN) * sizeof(float) / 32)});

        // [32] FreeTensor: 释放VECOUT缓存
        reluOutQueue.FreeTensor(reluOutLocal);
        computeRound++;
    }

    // [33] End: 通知Matmul对象计算结束, 释放内部状态
    matmulObj.End();
}
```

### 3.5 数据流与执行流程图

> **注意**：910B (2201) 上 AIC↔AIV 无直接硬通道，KFC 消息和中间数据均通过 GM 中转。3510 上改用 SSBuffer 直连。

```
┌──────────────────────────────────────────────────────────────────┐
│                     KERNEL LAUNCH (Host)                          │
│  numBlocks=1, stream, 传入A/B/Bias/C/workspace/tiling            │
└──────────────────────────────────────────────────────────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
              ┌──────────┐  ┌──────────┐  ┌──────────┐
              │ AIV Core0│  │ AIV Core1│  │ AIC Core │
              │ (vec核)  │  │ (vec核)  │  │ (cube核) │
              └──────────┘  └──────────┘  └──────────┘
                    │              │              │
                    ▼              ▼              ▼
            ┌───────────────────────────────────────────┐
            │         REGIST_MATMUL_OBJ                 │
            │  AIV侧: 创建KfcCommClient(消息发送)       │
            │  AIC侧: 创建KfcServer, 进入消息轮询循环    │
            │  * 文档依据见 §2.2                        │
            └───────────────────────────────────────────┘
                    │              │              │
         ╔══════════╪══════════════╪══════════════╪══════╗
         ║  ITERATE LOOP (4次, 每次: computeRound=0,1,2,3)║
         ╠══════════╪══════════════╪══════════════╪══════╣
         ║          ▼              ▼              ▼      ║
         ║  ┌─────────────────────────────────────────┐  ║
         ║  │ Iterate<true>() → AIV通过KFC发消息→AIC  │  ║
         ║  │   AIC: GM→L1→L0A/L0B→Mmad→L0C          │  ║
         ║  │   → KFC通过GM中转→数据写入VECIN(UB)     │  ║
         ║  │   (2201上: L0C→GM→UB; 3510上: L0C→UB直连)│ ║
         ║  │   → KFC消息通知AIV: 计算完成             │  ║
         ║  └─────────────────────────────────────────┘  ║
         ║          │              │                     ║
         ║          ▼              ▼                     ║
         ║  ┌─────────────────────────────────────────┐  ║
         ║  │ GetTensorC → 从VECIN读取128×64结果      │  ║
         ║  │ EnQue(MM)→DeQue(MM) → LeakyRelu(float)  │  ║
         ║  │ EnQue(Relu)→DeQue(Relu) → Free(MM)     │  ║
         ║  │ DataCopy → 写回GM(C[i*128].*])          │  ║
         ║  │ Free(Relu)                               │  ║
         ║  └─────────────────────────────────────────┘  ║
         ╚══════════╪══════════════╪═════════════════════╝
                    │              │
                    ▼              ▼
          ┌──────────────────────────────────┐
          │          matmulObj.End()         │
          │  AIV发送SERVICE_QUIT → AIC退出   │
          │  消息轮询循环                      │
          └──────────────────────────────────┘
```

**流程要点**：
- AIC 和 AIV 在单次 Iterate 内**严格串行**（depth=1 队列深度保证）
- 两个 AIV 核**并行**执行：Core0 和 Core1 各自处理不同的 N 分片（列 [0,64) 和 [64,128)）
- 唯一的 AIC 通过 KFC 消息轮询被两个 AIV 轮流驱动
- **2201 上**：Matmul 中间结果经 KFC 框架通过 GM 中转写入 VECIN（对开发者透明）
- **3510 上**：L0C→UB 有直连硬通道，中间结果物理上不经过 GM

---

## 4. `.template` 语法解析

### 4.1 为什么必须写 `.template`

这是**标准 C++ 依赖名消歧义语法**，不是 Ascend C 特有。

```cpp
matmulObj.template Iterate<true>();
```

- `matmulObj` 的类型是 `matmul::Matmul<MatmulType<...>, MatmulType<...>, ...>`，这是一个**依赖名**（dependent name）——它的完整类型依赖于外围模板上下文的模板参数
- `Iterate` 是一个**成员函数模板**（member function template）
- 当对象是 dependent name 时，编译器无法确定 `<` 是"模板参数开始"还是"小于运算符"
- `.template` 关键字**明确告诉编译器**：后面跟的是一个模板成员，`<` 是模板参数开始

### 4.2 什么时候需要

| 场景 | 写法 | 是否需要 `.template` |
|------|------|---------------------|
| 省略模板参数（默认 sync=true） | `mm.Iterate()` | 不需要 |
| 省略模板参数 | `mm.GetTensorC(c)` | 不需要 |
| 显式 sync=true | `matmulObj.template Iterate<true>()` | **需要** |
| 显式 sync=false（异步） | `mm.template Iterate<false>()` | **需要** |
| 显式 sync=false | `mm.template GetTensorC<false>(c)` | **需要** |

**规则**：任何时候显式传递模板参数（`<true>` 或 `<false>`），在 dependent name 上下文中都需要 `.template`。

### 4.3 相关 API 文档

- `docs/api/SIMD-API/高阶API/矩阵计算/Matmul-Kernel侧接口/Iterate.md`（第74-186行）
- `docs/api/SIMD-API/高阶API/矩阵计算/Matmul-Kernel侧接口/GetTensorC.md`（第76-277行）
- `docs/guide/算子实践参考/SIMD算子实现/矩阵编程（高阶API）/特性场景/异步场景处理.md`（同步vs异步完整说明）

---

## 5. 融合的性能优化点分析

### 5.1 两个优化维度

#### 优化1：省去 Fixpipe 格式/精度转换 + 显式 DataCopyPad 读回

在 910B (2201) 上，AIC 的 L0C 和 AIV 的 UB 之间**没有直接硬通道**，数据物理上仍经过 GM。但 Advanced API 相比 Basic API 仍然更优：

**Basic API 的 GM 操作**：
1. Fixpipe(L0C→GM)：做 Nz→ND 格式转换 + F32→F16 精度转换
2. DataCopyPad(GM→UB)：读回 half 数据
3. DataCopyPad(UB→GM)：写回 half 最终结果

**Advanced API 的 GM 操作**：
1. KFC 框架内部 GM 中转（仅数据搬运，无格式转换）：一个搬运来回均走 GM
2. DataCopy(UB→GM)：写回 float 最终结果（一次）

**省去的是什么**：
- Fixpipe 的 Nz→ND 格式转换开销（矩阵在 L1 中已是 Nz 分形序，Fixpipe 需重组为行主序 ND）
- Fixpipe 的 F32→F16 精度转换 + AIV 侧的 F16→F32 精度损失
- 显式 DataCopyPad(GM→UB) 的启动延迟

**在 3510 上**：L0C→UB 有直连硬通道，中间结果物理上不经过 GM，真正实现了"免 GM 往返"。

#### 优化2：AIC/AIV 并行？（在当前例程中**未实现**）

当前例程中 `depth=1`，AIC 和 AIV 在单次 Iterate 内**严格串行**：
```
AIC计算(Matmul) → KFC中转 → AIV处理(LeakyRelu+写回) → 下一次Iterate
```

**如果要实现 AIC/AIV 并行**，需要：
- `depth=2`（双缓冲）：AIC 写第 N 次结果到 buffer[0] 的同时，AIV 可以处理第 N-1 次的结果（在 buffer[1] 中）
- 或使用 `Iterate<false>`（异步模式）+ `SetWorkspace`

当前例程不涉及 AIC/AIV 并行流水。

### 5.2 为什么没有启用并行

1. **矩阵太小**：M=512, N=128, K=128，总共只有 4 次 Iterate，pipeline 的启动/排空开销可能大于收益
2. **教学目的**：该例程集中展示 VECIN 融合机制，不引入 double buffer 复杂度
3. **depth=1 的编译器优化**：depth=1 时编译器有特殊优化路径，代码更精简

---

## 6. N轴切分的数学分析

### 6.1 分片策略

```
原始矩阵: C[M×N] = C[512×128]
单核计算: C[M×singleCoreN] = C[512×64]
base块:   128×64×128
每核Iterate次数: M / baseM = 512 / 128 = 4次
```

### 6.2 数学推导

第 `computeRound` 次 Iterate（0-indexed）产生的 base 块负责 C 矩阵的哪些行/列：

```
C[ i_start : i_end ][ j_start : j_end ]
  i_start = computeRound * baseM     = computeRound * 128
  i_end   = (computeRound+1) * baseM = (computeRound+1) * 128
  j_start = blockIdx * singleCoreN   = blockIdx * 64
  j_end   = (blockIdx+1) * singleCoreN = (blockIdx+1) * 64
```

**举例**：Core0 (blockIdx=0), computeRound=2:
- 处理行范围：C[256 : 384][0 : 64]
- cOffset = 2 × 128 × 128 = 32768（从 C[0][0] 起跳过的 float 元素数）
- DataCopy 从 cGlobal[cOffset] 开始，写 128 行，每行 64 个 float

### 6.3 B/Bias/C 三者偏移相同的原理

在 N 轴切分中，每个 core 负责 N 方向上连续的一"条"（64 列）：

| 矩阵 | 形状 | stride | 偏移后含义 |
|------|------|--------|-----------|
| **B** | K×N = 128×128 | N=128 | 从 B[0][64] 开始，stride 仍为 N，每行前 baseN 个元素是 B[i][64..127] |
| **Bias** | 长度 N=128 | — | 偏移 64 后得到 bias[64..127] |
| **C** | M×N = 512×128 | N=128 | 从 C[0][64] 开始写，dstStrideIn 跳过每行前 64 列 |
| **A** | M×K = 512×128 | K=128 | **不偏移**——N 轴分片不影响 A 矩阵 |

三者在 N 轴上的 stride 均为原 N（128），跳过前 core 已处理的 N 列所需的偏移量（`blockIdx × singleCoreN`）完全一致。这就是 `nCoreOffset` 相同的原因。

---

## 7. TPipe/TQue 框架深度解析

### 7.1 设计理念

Ascend C 的 TPipe/TQue 框架**承袭了经典 C/C++ 队列管道(Queue Pipeline)设计思想**，将复杂计算拆分为多个独立的 Stage，各 Stage 之间通过队列传递数据。

设计解决的问题：

| 问题 | 解决方案 |
|------|---------|
| **WAR (写后读)** — 生产者没写完，消费者就开始读 | **EnQue / DeQue**：EnQue 发 Set 信号（写完了），DeQue 发 Wait 信号（等待可读） |
| **RAW (读后写)** — 旧数据还有人读，新数据就覆盖了 | **AllocTensor / FreeTensor**：AllocTensor 等所有读操作完成才分配；FreeTensor 通知该内存可安全覆写 |

**核心分工**：
- **TPipe**：管**资源**（内存分配、事件 ID 管理、队列初始化）
- **TQue**：管**通信**（任务间数据传递 + 硬件同步信号）

### 7.2 TPipe 用法

```cpp
AscendC::TPipe pipe;  // 一个kernel必须且只能初始化一个TPipe对象

// 为TQue分配内存
pipe.InitBuffer(tque, num, len);
//   tque: TQue对象
//   num:  内存块数量, 1=不开启double buffer, 2=开启double buffer
//   len:  每块内存大小(字节), 内部自动32B对齐

// 为TBuf分配内存(临时变量)
pipe.InitBuffer(buf, len);
```

**关键约束**：
- 一个 kernel 中所有 Buffer 数量之和 ≤ 64
- 同一 TPosition 上最大 Buffer 数 = eventID 数量（910B 上为 4 个 TQue × depth 因子）
- TPipe 对象销毁时通过析构函数自动释放所有内存
- **不应在 kernel 类对象内创建 TPipe**（避免额外的构造/析构开销）

### 7.3 TQue 用法

```cpp
// TQue模板参数
template <TPosition pos, int32_t depth, auto mask = 0> class TQue;

// pos:   逻辑位置 → 物理内存映射
//   VECIN/VECCALC/VECOUT → Unified Buffer (UB)
//   A1/B1/C1             → L1 Buffer
//   CO1                  → L0C Buffer
//   CO2                  → GM (910B) 或 UB (其他)

// depth: 队列深度
//   1 = 单缓冲(编译器特殊优化路径), 不开double buffer
//   ≥2 = 开启对应深度的流水
//   0 = 原地操作(inplace), AllocTensor/DeQue需传已有Tensor

// 四步标准操作范式
auto tensor = tque.AllocTensor<T>();   // 1. 申请：等内存可覆写
// ... 计算/搬运 ...
tque.EnQue(tensor);                    // 2. 入队：通知下游"写完"
tensor = tque.DeQue<T>();              // 3. 出队：等待上游"写完"
// ... 计算/搬运 ...
tque.FreeTensor(tensor);               // 4. 释放：通知该内存可覆写
```

### 7.4 TQue 内部自动同步

TQue 的 EnQue/DeQue/AllocTensor/FreeTensor **自动插入硬件同步指令**（底层 `Set` / `Wait`）：

| TQue 操作 | 底层同步 |
|-----------|---------|
| `AllocTensor` | 发射 `Wait` — 等待该内存区域所有读操作完成 |
| `EnQue` | 发射 `Set` — 标记写入完成，唤醒等待的下游 |
| `DeQue` | 发射 `Wait` — 等待上游 `Set` 信号 |
| `FreeTensor` | 发射 `Set` — 标记该内存可安全覆写 |

### 7.5 TQue 简化写法（TQue 与 TQueBind 的映射）

```cpp
// TQue 是 TQueBind 的简化版本，隐式指定了源/目的位置:
TQue<VECIN, 1>   ≡  TQueBind<GM, VECIN, 1>     // 数据从GM来→到VECIN(UB)
TQue<VECOUT, 1>  ≡  TQueBind<VECOUT, GM, 1>     // 数据VECOUT(UB)→去GM
TQue<A1, 1>      ≡  TQueBind<GM, A1, 1>         // 数据从GM来→到A1(L1)
```

**关键文档**：
- `docs/guide/编程指南/编程模型/AI-Core-SIMD编程/基于TPipe-TQue框架编程/TPipe-TQue框架编程原理.md`（设计理念，70行）
- `docs/guide/编程指南/编程模型/AI-Core-SIMD编程/基于TPipe-TQue框架编程/TPipe-TQue框架编程范式.md`（工程范式，197行）
- `docs/api/SIMD-API/基础API/资源管理/Pipe和Que框架/`（完整API文档目录）

---

## 8. 算子执行的资源独占性

### 8.1 一次 kernel 调用的资源归属

**是的，一次算子执行（一次 kernel launch）中，被分配到的 AI Core 组合内所有资源归当前 kernel 独占。**

```
Host: mmad_vec_custom<<<numBlocks, nullptr, stream>>>(...)
                    ↑
              numBlocks = AI Core组合数量
```

- **numBlocks=1**：启动 1 组 AI Core（1 AIC + 2 AIV）。这组 core 在此期间**不执行其他 kernel**
- **numBlocks=2**：启动 2 组 AI Core（2 AIC + 4 AIV），并行执行但不共享内存
- 每组 AI Core 内部的资源独占：
  - **AIC 侧**：L1 Buffer (511KB)、L0A (64KB)、L0B (64KB)、L0C (128KB) — 全部归当前 kernel 的 AIC 代码使用
  - **AIV 侧**：UB (191KB 可用) — 全部归当前 kernel 的 AIV 代码使用
  - **共享**：L2 Cache (192MB) 和 HBM (64GB) — **多核共享**，通过硬件自动管理一致性

### 8.2 资源隔离边界

| 资源 | 粒度 | 机制 |
|------|------|------|
| UB / L1 / L0A / L0B / L0C | **per-core 独占** | 每个物理 core 有自己的独立片上内存，kernel 间不共享 |
| L2 Cache | **per-device 共享** | 硬件自动管理缓存一致性 |
| HBM (GM) | **per-device 共享** | 通过不同地址空间分段隔离，kernel 间通过 stream 串行化 |

---

## 9. AIV 两核区分与 AIC 角色

### 9.1 AIV 核区分

```cpp
uint32_t blockIdx = AscendC::GetBlockIdx();
```

- Core 0: `GetBlockIdx() = 0`
- Core 1: `GetBlockIdx() = 1`

在 Advanced API 中，AIV 通过 `GetBlockIdx()` 区分自己负责的 N 分片：
```cpp
bGlobal = bGlobal[blockIdx * singleCoreN];  // Core0: B[0:128][0:64], Core1: B[0:128][64:128]
```

在 Basic API 中，每个 Cube 的结果由 2 个 Vector 核各做一半：
```cpp
gmOffset = AscendC::GetBlockIdx() % 2 * (baseM / 2 * N);  // Core0: 上半块, Core1: 下半块
```

### 9.2 AIC 没有独立的 GetBlockIdx

在 `__mix__(1,2)` 模式下，**AIC 没有独立的 blockIdx**——它由 AIV 通过消息驱动运行。AIC 的 "身份" 由 AIV 侧的消息上下文隐式决定。

```cpp
if ASCEND_IS_AIC {
    // AIC侧: GetBlockIdx()返回值含义与AIV不同(取决于框架实现)
    //        在Basic API中用于M轴分片:
    aGM = aGM[AscendC::GetBlockIdx() * singleCoreM * K];
}
```

---

## 10. 910B 上的 "UB" 到底是不是真的 UB

### 10.1 物理内存映射

| TPosition | 物理内存 (910B/dav-2201) | 所在 Core |
|-----------|-------------------------|-----------|
| VECIN, VECCALC, VECOUT | **Unified Buffer (UB)** | AIV (Vector Core) |
| A1, B1, C1 | **L1 Buffer** | AIC (Cube Core) |
| CO1 | **L0C Buffer** | AIC (Cube Core) |
| CO2 | **Global Memory (GM)** | 片外 HBM |

**文档来源**：`docs/api/SIMD-API/通用说明和约束.md` 第 40-53 行（TPosition→物理内存映射表）；`TPipe-TQue框架编程范式.md` 第 100-110 行。

### 10.2 纯 Cube 模式下 Matmul 用的是 L1 不是 UB

在 910B 上，**纯 Cube 模式的 Matmul 完全不使用 UB 做数据存储**：

- **数据路径**：GM → L1 → L0A/L0B → Cube(Mmad) → L0C → FixPipe → GM
- **UB 用量 = 0**（tiling 输出可验证：`usedUBSize = 0`，`usedL1Size = 295424`）
- UB 仅用于 Matmul 库的控制 workspace

**文档来源**：
- `基本架构.md` 第 206-207 行：Cube 数据流不经过 UB
- `含有Matmul高阶API的算子精度问题.md` 第 115 行：`usedUBSize = 0`
- `概述.md` 第 21 行："Cube计算单元配套L1 Buffer...Vector计算单元配套UB"

### 10.3 Mix 模式下的 UB

在 Mix 模式下（融合算子），Advanced API 的 C 类型设为 VECIN：

```cpp
matmul::MatmulType<AscendC::TPosition::VECIN, ...>  // C → 写到 Vector 侧的 UB
```

此时 Matmul 结果**确实写入 UB**（VECIN = UB 上的一个逻辑位置），供 AIV 直接读取。但注意：

- 在 910B（2201）上，AIC 的 L0C 和 AIV 的 UB 之间**没有直接的硬件通路**
- 数据通路实际是：**L0C → GM → UB**（通过 KFC 框架在后台完成数据搬运）
- 3510 架构才新增了 L0C→UB 的直接硬通道

**文档来源**：
- `2201到3510架构变更.md` 第 24-26 行：UB↔L1 直连是 3510 新增特性
- `分离模式.md` 第 41 行：分离模式下数据无法经 VECIN/VECCALC/VECOUT 直接搬运到 A1/B1

### 10.4 架构文档定位

描述 910B（dav-2201）芯片架构的三份权威文档：

| 文档 | 路径 | 内容 |
|------|------|------|
| 基本架构 | `docs/guide/编程指南/高级编程/硬件实现/基本架构.md` | AI Core 框图总览、存储单元、计算单元、搬运单元、数据通路 |
| 架构规格 | `docs/guide/编程指南/高级编程/硬件实现/架构规格/NPU架构版本2201.md` | 存储容量 (UB 191KB, L0A 64KB, L0B 64KB, L0C 128KB, L1 511KB, L2 192MB, HBM 64GB)；数据格式；bank 结构；UB 排布方式 |
| 硬件约束 | `docs/guide/编程指南/高级编程/硬件实现/硬件约束/NPU架构版本2201.md` | 对齐要求 (32B)；搬运粒度下限 (16KB)；bank 冲突规则；容量上限 |

---

## 11. Basic API 深度解析

### 11.1 `ASCEND_IS_AIC` 与 `ASCEND_IS_AIV`

**本质**：**编译期 C++ 预处理器宏**（不是运行时判断），用于 `__aicore__` 函数中的条件编译。

- `ASCEND_IS_AIC`：Cube 核（AI Core Cube，矩阵计算核）上为 true
- `ASCEND_IS_AIV`：Vector 核（AI Core Vector，矢量计算核）上为 true

在分离模式（`__mix__`）下，编译工具链对同一核函数分别以 AIC 和 AIV 为目标编译两次，生成两份不同的二进制。`ASCEND_IS_AIC` / `ASCEND_IS_AIV` 的作用类似于 `#ifdef`，在预处理阶段决定当前目标核类型下哪些代码分支被保留。

**等价运行时替代**：`constexpr int32_t g_coreType`，取值 `AscendC::AIC` 或 `AscendC::AIV`。

**关键文档**：`docs/guide/编程指南/语言扩展层/SIMD-BuiltIn关键字.md`（"预定义宏"章节）

### 11.2 L2 Cache 为什么不需要管理

**L2 Cache 是硬件全自动管理的透明缓存层**，位于 MTE 搬运单元与 GM 之间。

- 所有通过 MTE2（搬入）、MTE3（搬出）、FixPipe（搬出）读写 GM 的数据**默认被 L2 Cache 缓存**
- 用户不能分配/释放/寻址 L2 空间，也不控制数据的换入换出
- 唯一干预手段：`SetL2CacheHint(CacheMode::DISABLE / NORMAL)` 按 per-GlobalTensor 开关缓存（默认使能）
- **带宽对比**：L2 Cache ~7TB/s vs GM ~1.6TB/s，约 3-4 倍差距

**关键文档**：`基本架构.md` 第 194 行："所有通过搬运单元读写GM的数据都缺省被缓存在L2Cache"

### 11.3 DataCopy 与 LoadData 的区别

#### DataCopy（通用数据搬运）

- **类别**：Memory 数据搬运 API
- **流水级**：MTE2（搬入，GM→L1/UB）、MTE3（搬出，UB→GM）
- **支持通路**：GM↔L1、GM↔UB、LM↔LM
- **格式转换**：支持随路 ND↔NZ 转换、量化/ReLU 激活
- **适用场景**：Stage 1 (GM→L1, ND→NZ)、Vector 侧 GM↔UB 搬运

#### LoadData（分形格式转换搬运）

- **类别**：矩阵计算 ISASI API（**不保证跨架构兼容**）
- **流水级**：MTE1（L1→L0A/L0B）
- **支持通路**：L1→L0A、L1→L0B
- **格式转换**：NZ→ZZ（左矩阵）、NZ→ZN（右矩阵）— Mmad 指令要求的格式
- **适用场景**：Stage 2 (L1→L0A/L0B，Cube 计算就绪格式化)

#### 典型 Matmul 流水线

```
Stage 1: CopyIn   DataCopy  (MTE2)    │ GM ──→ L1 Buffer  │ ND → NZ
Stage 2: Split    LoadData  (MTE1)    │ L1 ──→ L0A/L0B    │ NZ → ZZ/ZN
Stage 3: Compute  Mmad      (PIPE_M)  │ L0A × L0B → L0C
Stage 4: CopyOut  Fixpipe   (PIPE_FIX)│ L0C ──→ GM        │ NZ → ND + 精度转换
```

**关键文档**：`入门功能落地.md`、`矩阵计算流程.md`

### 11.4 HardEvent 硬件事件完整列表

定义文件：`impl/basic_api/kernel_event.h` 第 37-79 行。

命名规则：`<源流水级>_<目标流水级>`，表示"目标流水级等待源流水级完成"。

共 7 种流水级：`S` (Scalar)、`V` (Vector)、`M` (Matrix)、`MTE1`、`MTE2`、`MTE3`、`FIX`。

#### AIC 侧事件（24 个）

| 事件 | 含义 |
|------|------|
| `MTE2_MTE1` | MTE1 等待 MTE2（LoadData 等 DataCopy 完成） |
| `MTE1_MTE2` | MTE2 等待 MTE1 |
| `MTE1_M` | M 等待 MTE1（Mmad 等 LoadData 完成） |
| `M_MTE1` | MTE1 等待 M |
| `MTE3_MTE1` | MTE1 等待 MTE3 |
| `MTE1_MTE3` | MTE3 等待 MTE1 |
| `MTE2_M` | M 等待 MTE2 |
| `M_MTE2` | MTE2 等待 M（Mmad 完成后通知搬运侧） |
| `M_FIX` | FIX 等待 M（Fixpipe 等 Mmad 完成） |
| `FIX_M` | M 等待 FIX |
| `MTE3_MTE2` | MTE2 等待 MTE3 |
| `MTE2_MTE3` | MTE3 等待 MTE2 |
| `S_MTE2` | MTE2 等待 S |
| `MTE2_S` | S 等待 MTE2 |
| `S_MTE3` | MTE3 等待 S |
| `MTE3_S` | S 等待 MTE3 |
| `MTE2_FIX` | FIX 等待 MTE2 |
| `FIX_MTE2` | MTE2 等待 FIX |
| `FIX_S` | S 等待 FIX |
| `M_S` | S 等待 M |
| `FIX_MTE3` | MTE3 等待 FIX |
| `MTE1_FIX` | FIX 等待 MTE1 |
| `FIX_MTE1` | MTE1 等待 FIX |
| `FIX_FIX` | FIX→FIX（等效 PipeBarrier） |

#### AIV 侧事件（13 个）

| 事件 | 含义 |
|------|------|
| `MTE2_V` | V 等待 MTE2（Vector 计算等 DataCopy 搬入完成） |
| `V_MTE2` | MTE2 等待 V |
| `MTE3_V` | V 等待 MTE3 |
| `V_MTE3` | MTE3 等待 V（DataCopy 搬出等 Vector 计算完成） |
| `V_V` | V→V（等效 PipeBarrier） |
| `MTE3_MTE2` | MTE2 等待 MTE3 |
| `MTE2_MTE3` | MTE3 等待 MTE2 |
| `S_V` | V 等待 S |
| `V_S` | S 等待 V |
| `S_MTE2` | MTE2 等待 S |
| `MTE2_S` | S 等待 MTE2 |
| `S_MTE3` | MTE3 等待 S |
| `MTE3_S` | S 等待 MTE3 |

#### 架构条件事件（仅 5102/3003）

| 事件 | 含义 |
|------|------|
| `FIX_V` | V 等待 FIX |
| `V_FIX` | FIX 等待 V |

**关键文档**：`docs/api/SIMD-API/基础API/同步控制/核内同步/核内同步能力概述.md`（含完整合法同步组合表）

---

## 12. Basic API 代码中的事件序列

```
AIC 侧（第54-132行）:
  DataCopy GM→L1 (MTE2)
    └── SetFlag<MTE2_MTE1> / WaitFlag<MTE2_MTE1>    — MTE1 等 MTE2 搬完
  LoadData L1→L0A/L0B (MTE1)
    └── SetFlag<MTE1_M>    / WaitFlag<MTE1_M>         — M 等 LoadData 完成
  Mmad L0A×L0B→L0C (PIPE_M)
    └── SetFlag<M_MTE2>    / WaitFlag<M_MTE2>         — M 通知 MTE2 已完成
    └── SetFlag<M_FIX>     / WaitFlag<M_FIX>          — FIX 等 Mmad 完成
  Fixpipe L0C→GM (PIPE_FIX)
    └── CrossCoreSetFlag                            — 通知 AIV 可以读 GM

AIV 侧（第133-141行）:
  CrossCoreWaitFlag                                  — 等 AIC Fixpipe 写完
  DataCopyPad GM→UB (MTE2)
    └── SetFlag<MTE2_V> / WaitFlag<MTE2_V>           — V 等 MTE2 搬完
  LeakyRelu (PIPE_V)
    └── SetFlag<V_MTE3> / WaitFlag<V_MTE3>           — MTE3 等 V 计算完
  DataCopyPad UB→GM (MTE3)
```

---

## 13. 附录：关键术语速查

| 术语 | 含义 |
|------|------|
| `__mix__(1,2)` | 分离模式声明：1 个 AIC (Cube核) + 2 个 AIV (Vector核) 编为一组 |
| `ASCEND_IS_AIC` | 编译期宏，Cube 核上代码保留 |
| `ASCEND_IS_AIV` | 编译期宏，Vector 核上代码保留 |
| `REGIST_MATMUL_OBJ` | 初始化 Matmul 对象的宏——绑定 TPipe、workspace、tiling，建立 AIV↔AIC 通信 |
| **L2 Cache** | 硬件自动管理的 GM 透明缓存（~7TB/s） |
| **MTE1** | 搬运单元：L1 → L0A/L0B（走 LoadData） |
| **MTE2** | 搬运单元：GM → L1/UB/L0A/L0B（走 DataCopy 搬入） |
| **MTE3** | 搬运单元：UB → GM（走 DataCopy 搬出） |
| **FixPipe** | 搬运单元：L0C → GM/L1（可做随路量化/ReLU/格式转换） |
| **Nd2Nz** | ND（行列自然序）→ NZ（分形序）格式转换参数 |
| **VECIN** | TQue 位置：Matmul 写入、Vector 读取的 UB 中间缓存。2201上经KFC通过GM中转写入（对开发者透明），3510上L0C→UB直连 |
| **SetFixSplit** | 固定 baseM/baseN/baseK，供 kernel 侧 DataCopy 偏移逻辑与之对齐 |
| **ISASI** | 体系结构相关接口，不保证跨架构兼容（如 LoadData） |
| **`.template`** | C++ 依赖名消歧义语法：在 dependent name 上下文中调用成员函数模板时必须使用 |
| **KFC** | Kernel Function Communication：分离模式下 AIV→AIC 跨核消息通信框架 |
| **depth** | TQue 队列深度：1=单缓冲（编译器优化），≥2=多缓冲（pipeline 流水） |
