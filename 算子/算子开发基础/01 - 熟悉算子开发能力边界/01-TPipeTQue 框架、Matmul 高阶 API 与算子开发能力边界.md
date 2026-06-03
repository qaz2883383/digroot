# TPipe/TQue 框架、Matmul 高阶 API 与算子开发能力边界

## 涉及用例清单

| 用例 | 说明 | 源码 |
|------|------|------|
| `add` | Vector Add 基础 API 实现（昨日重点） | `01_vector/add/add.asc` |
| `add_tpipe_tque` | Vector Add 的 TPipe/TQue 实现 | `01_vector/add_tpipe_tque/add_tpipe_tque.asc` |
| `matmul_advanced_api` | Matmul 高阶 API 实现 | `02_matrix/matmul_advanced_api/matmul_advanced_api.asc` |
| `matmul_basic_api` | Matmul 基础 API 实现 | `02_matrix/matmul_basic_api/matmul_basic_api.asc` |
| `matmul_tensor_api` | Matmul Tensor API 实现（仅 950） | `02_matrix/matmul_tensor_api/` |

> 以上路径均在 `asc-devkit/examples/01_simd_cpp_api/00_introduction/` 下。

---

## 1. 能力边界认知（AI分析，辅助学习用，未验证）

在系统学习前先建立全局认知。遍历 `01_simd_cpp_api` 全部 ~141 个例程后的分析：

### 能做什么

| 能力 | 关键例程 |
|------|---------|
| Element-wise 向量计算 | `01_vector/` |
| 矩阵乘（FP16/BF16/FP8/INT4） | `02_matrix/` |
| Cube-Vector 融合 | `03_fusion_operation/matmul_leakyrelu_*` |
| SIMD+SIMT 混合 | `03_fusion_operation/gather_adds_simt_simd_hybrid/` |
| 高阶 API 全覆盖 | `03_libraries/`（Softmax/LayerNorm/GELU/RMSNorm 等） |
| PyTorch/TF/ONNX 框架集成 | `02_features/02_framework/` |

### 核心局限

| 局限 | 原因 |
|------|------|
| 单核 UB 仅 191KB | 数据量大必须多核 Tiling 拆分 |
| GM↔UB 必须显式搬运 | 不像 GPU 有透明 L1/L2 cache，不搬不算 |
| Vector 和 Cube 异步并行 | 必须手动 `PipeBarrier` 同步，顺序错结果错 |
| 不同架构二进制不兼容 | dav-2201 → 3510 需重新编译，无法复用 |
| 离散访存性能差 | Scatter/Gather 纯 SIMD 效率低，需 SIMT 辅助 |

### 硬件坑与规避

| 坑 | 规避 |
|----|------|
| Bank 冲突（同组 bank 排队降速） | 调整 stride，让访问均匀分散到不同 bank group |
| 单次搬运 < 16KB 带宽利用率低 | 尽量一次搬 16KB+ |
| 地址非 32B 对齐 | 首地址 32B 对齐，必要时 Pad |
| L0C 溢写（>128KB） | Tiling 控制单次块大小 ≤ 128KB |
| 核间同步 + numBlocks 超核数 → 死锁 | 同步场景 numBlocks ≤ 物理核数 |

> 硬件约束文档入口：`docs/guide/编程指南/高阶编程/硬件实现/硬件约束/NPU架构版本2201.md`

### 推荐阅读顺序

```
第一优先（必须跑通）：00_quickstart → 01_vector/add → 01_vector/add_tpipe_tque
第二优先（核心能力）：02_matrix/matmul_advanced_api → 03_fusion_operation/matmul_leakyrelu_advanced_api
                      → 03_libraries/01_activation/softmax → 03_libraries/02_normalization/layernorm
第三优先（调试必备）：01_utilities/printf → 01_utilities/cpudebug → 01_utilities/dump
第四优先（性能调优，按需）：04_best_practices/00_vector_compute/add_high_performance
                        → 04_best_practices/01_matrix_compute/matmul_high_performance
                        → 04_best_practices/04_memory_access/data_copy
                        → 04_best_practices/04_memory_access/bank_conflict_nd2nz

跳过（对入门无意义）：
  04_vector_reg/*（仅 950 平台）
  02_features/00_compilation/*（算子发布用）
  02_features/01_invocation/aclop_invocation（旧接口）
  02_features/04_aicpu/*、05_aclrtc/*（高级特性，短期不用）
  03_libraries/00_matrix/ 大部分 matmul_* 变体（只留 matmul/matmul_fused/matmul_quant 三个）
  03_libraries/05_sort 到 13_utils（特殊场景，碰到再查）
  05_compatibility_guide/*（架构迁移时再看）
```

---

## 2. 语法细节补充

### `static_cast` 在哪定义的

```cpp
x[i] = static_cast<float>(i);  // uint32_t → float
```

`static_cast` 不是函数、不是宏，是 **C++ 标准关键字**——和 `for`/`if`/`return` 一样由编译器内置识别，不需要任何 `#include`。四种 C++ cast（`static_cast`、`dynamic_cast`、`const_cast`、`reinterpret_cast`）全部是关键字。

### 核函数可以 return 吗

不可以。`__global__` kernel 函数返回值必须是 `void`。原因是 kernel 异步启动——`add_custom<<<8,0>>>(...)` 只是向 NPU 提交任务，host 代码立刻继续执行。kernel 的输出通过**写设备内存指针**实现，host 侧通过 `aclrtMemcpy(..., ACL_MEMCPY_DEVICE_TO_HOST)` 读回。

---

## 3. `<<<>>>` kernel 启动语法的两种形式

### 2 个参数（基础 API 模式）

```cpp
add_custom<2048><<<8, 0>>>(xDevice, yDevice, zDevice);
//                │   │
//           numBlocks  l2Ctrl
```

第二参数 `l2Ctrl`：L2 cache 控制标志，官方文档无详细说明，通常传 `0` 表示默认行为。

### 3 个参数（TPipe/TQue 框架模式）

```cpp
add_custom<<<8, nullptr, stream>>>(xDevice, yDevice, zDevice, totalLength);
//            │      │        │
//      numBlocks  workspace  aclrtStream
```

| 参数 | 含义 |
|------|------|
| `numBlocks` | block 数量 |
| `workspace` | 临时工作空间指针（`nullptr` 表示不额外分配） |
| `stream` | `aclrtStream`，支持异步流并发 |

stream 允许创建多个流，多个 kernel 可在不同流上并发执行——这是基础 API 直调做不到的。

### 参数文档

没有一篇完整的 `<<<>>>` 参数说明书。`numBlocks` 的约束散见在多处：

- `CalcTschNumBlocks.md`：融合模式下的 numBlocks 计算
- `InitDetermineComputeWorkspace.md` / `SyncAll.md`：使用核间同步时 **numBlocks 必须 ≤ 物理核数**，否则死锁

目前 `l2Ctrl` 和 `workspace` 无独立文档。

---

## 4. TQue/TPipe vs 基础 API（手写 PipeBarrier）

### 基础 API — 串行执行

```cpp
DataCopy(GM→UB)
PipeBarrier<PIPE_ALL>()     // 等 MTE 搬完
Add(zLocal, xLocal, yLocal) // Vector 计算
PipeBarrier<PIPE_ALL>()     // 等 Vector 算完
DataCopy(UB→GM)             // MTE 搬出
```

`PipeBarrier<PIPE_ALL>()` 强制等所有 pipe 都完成才继续。**搬运和计算不会并行**——MTE 搬运时 Vector 闲着，Vector 算时 MTE 闲着。

### TPipe/TQue — 流水线重叠

```cpp
// EnQue/DeQue 表达数据依赖：X 和 Y 搬完才能算 Add，Add 算完才能搬出
// 但 MTE（搬下一批）和 Vector（算这一批）可以在不同数据上同时进行

时间 →
MTE:    [搬 batch0] [搬 batch1] [搬 batch2] ...
Vector:            [算 batch0] [算 batch1] ...
                      ↑ 自动 overlap
```

**用 TQue 等于从"停-走-停-走"升级成"流水线不停工"。** 在简单 Add 上差异不大，在 Matmul+LeakyReLU 等多级流水线融合场景下收益显著。

### 两例关键差异对比

| | add.asc（基础 API） | add_tpipe_tque.asc（TPipe/TQue） |
|---|---|---|
| 内存分配 | `LocalMemAllocator::Alloc<T,N>()` | `pipe.InitBuffer(queue, 1, size)` |
| 同步方式 | `PipeBarrier<PIPE_ALL>()` | `EnQue`/`DeQue`，TPipe 自动插 barrier |
| blockLength | 模板参数（编译期常量） | `totalLength / GetBlockNum()`（运行时） |
| kernel 启动 | `<<<8, 0>>>` | `<<<8, nullptr, stream>>>` |
| InitSocState | 需要显式调用 | TPipe 内部处理 |
| 数据类型 | `float*` | `uint8_t*`，kernel 内 cast |

---

## 5. Matmul 高阶 API 逐行分析

### 头文件区

```cpp
#define ASCENDC_CUBE_ONLY          // 告诉编译器：只走 Cube 核（AIC），不启动 Vector（AIV）
#include "lib/matmul_intf.h"      // Matmul 高阶 API 头文件
```

### kernel 签名

```cpp
__global__ __cube__ void matmul_custom(   // __cube__：跑在 Cube 单元（不是 __vector__）
    __gm__ uint8_t* a, __gm__ uint8_t* b, __gm__ uint8_t* c,
    __gm__ uint8_t* workspace,            // 系统 workspace
    AscendC::tiling::TCubeTiling tiling)  // Host 侧算好的切分数据
```

### 配置 Matmul 对象

```cpp
Matmul<
    MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>,   // A：在 GM，ND 格式，half
    MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>,   // B：同上
    MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>>  // C：在 GM，ND，float
    mm;
```

三个 `MatmulType` 模板参数分别描述 A/B/C 的位置、数据格式、数据类型。**必须与 host 侧 `SetAType`/`SetBType`/`SetCType` 保持一致**。

> Matmul API 文档入口：`docs/api/SIMD-API/高阶API/矩阵计算/Matmul-Kernel侧接口/Matmul模板参数.md`

### REGIST_MATMUL_OBJ — 初始化

```cpp
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
```

`REGIST_MATMUL_OBJ` 只是 `REGIST_CUBE_OBJ` 的别名（定义于 `include/adv_api/matmul/matmul_intf.h:37`）。实际做三件事：

1. 将 Matmul 对象注册到 TPipe 流水线上，后续 SetTensorA/B、IterateAll 的 stage 切换由 TPipe 管理
2. 绑定系统 workspace（`GetSysWorkSpacePtr()` 返回编译器预留的 scratch buffer）
3. 注入 tiling 数据，使 IterateAll 知道按什么粒度搬运计算

**不调这一步，Matmul 对象不知道它属于哪个 TPipe、workspace 在哪、数据怎么切。**

关键约束：
- 融合模式下必须在 `InitBuffer` **之前**调用
- 最多支持 4 个 Matmul 对象（flagId 范围 [0, 2×N-1]，N≤4）
- 不能和手动 `CrossCoreSetFlag` 混用

> 文档：`docs/api/SIMD-API/高阶API/矩阵计算/Matmul-Kernel侧接口/REGIST_MATMUL_OBJ.md`

### SetOrgShape + SetTensorA/B + IterateAll 的切分逻辑

用例矩阵：M=512, N=512, K=128，M 轴分 2 核（`singleCoreM=256`）。

```
Core 0 (blockIdx=0):
  A：a[0] → A[row 0..255, col 0..127]    [256×128]
  B：b[0] → B[row 0..127, col 0..511]    [128×512]
  C：c[0] → 写 C[row 0..255, col 0..511]

Core 1 (blockIdx=1):
  A：a[32768] → A[row 256..511, col 0..127]  [256×128]
  B：b[0]     → B[row 0..127, col 0..511]    [128×512]
  C：c[131072] → 写 C[row 256..511, col 0..511]
```

**A 按 M 轴偏移（每个核算自己的行段），B 从头取（每个核都需要完整 B 做 K 轴累加）。**

不按 K 轴分核（SplitK）是因为需要额外累加归约——按 M 轴分（SplitM）每个核输出天然不重叠。

IterateAll 内部按 `baseM×baseN×baseK` 块大小循环，对每一块自动完成 `DataCopy(GM→L1)→LoadData(L1→L0A/L0B)→Mmad→Fixpipe(L0C→GM)` 的全流水线，K 方向多轮累加由 L0C 原地累加完成。

### GetLibApiWorkSpaceSize 为什么不随 tiling 变化

**不是算法上不需要，是调用顺序决定的。**

```cpp
size_t systemWorkspaceSize = ascendcPlatform->GetLibApiWorkSpaceSize();  // 先获取
auto tiling = GenerateTiling(ascendcPlatform);                           // 后算 Tiling
```

获取 workspace 大小时 Tiling 还没算出来。所以 API 返回的是**最坏情况下按硬件最大能力预留的值**（取决于核数、流水线深度、最大 Matmul 对象数 4），确保不管 Tiling 算什么结果都够用。

---

## 6. Basic / Advanced / Tensor 三套 API 对比

| | Basic API | Advanced API | Tensor API |
|---|---|---|---|
| **层级** | 指令级（最底层） | 算法封装层 | 布局抽象层 |
| **平台** | A2/A3/950 全系 | A2/A3/950 | **仅 950PR/DT** |
| **同步** | `SetFlag`/`WaitFlag` 手动精控 | `TPipe` + `REGIST_MATMUL_OBJ` 自动 | Layout 驱动 |
| **kernel 行数** | ~100 行 | ~7 行 | 布局声明式 |
| **Tiling** | 全部手动算，写死在模板参数 | `MultiCoreMatmulTiling` 自动算 | 类似 Advanced |

### Basic API kernel 流水线的 6 个阶段

每一步都需要手写对应硬件的同步 flag：

```
① LocalMemAllocator<L1/L0A/L0B/L0C>  分配 buffer
② DataCopy(GM→L1, Nd2NzParams{...})   搬运 + 格式转换
③ SetFlag<MTE2_MTE1> + WaitFlag       MTE2→MTE1 同步（搬完才能转 L1→L0）
④ LoadData(L1→L0A, LoadData2D{...})   指定 fractal 布局（2201: Nz→Zz/Zn）
⑤ SetFlag<MTE1_M> + WaitFlag           MTE1→M 同步（转完才能算）
⑥ Mmad + SetFlag<M_FIX> + Fixpipe      矩阵乘 + 写回 GM
```

### 使用建议

| 场景 | 推荐 |
|------|------|
| 日常开发，快速验证 | Advanced API |
| 需要极致控制每步搬运和同步 | Basic API |
| 950 平台，偏好 Layout 范式 | Tensor API |

---

## 7. `<<<>>>` 追加发现：numBlocks 的关键约束

使用核间同步 API（`NotifyNextBlock`、`WaitPreBlock`、`SyncAll`、`InitDetermineComputeWorkspace`）时：

> **numBlocks 必须 ≤ 物理 AI Core 核数，否则框架进行多轮调度时插入异常同步，导致 Kernel 卡死。**

不使用同步 API 时可随意超（实测 500 > 40 正常跑通）。
