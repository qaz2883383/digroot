# 调试工具链：printf / Dump / CPU Debug

> 三个官方 demo 都很薄——本质是 API 用法展示，无复杂逻辑。本文直接提炼结论。

## 涉及用例清单

| 用例 | 说明 | 源码 |
|------|------|------|
| `simple_printf` | Matmul kernel 中使用 `AscendC::printf` | `01_utilities/00_printf/simple_printf/printf.asc` |
| `simd_vf_printf` | `__simd_vf__` 中使用 `printf` + `asc_vf_call` | `01_utilities/00_printf/simd_vf_printf/simd_vf_printf.asc` |
| `simple_dump` | `asc_dump_gm/l1buf/cbuf` 打印各存储级数据 | `01_utilities/02_dump/simple_dump/dump.asc` |
| `simd_vf_dump` | `__simd_vf__` 中使用 `asc_dump_ubuf/reg` | `01_utilities/02_dump/simd_vf_dump/simd_vf_dump.asc` |
| `cpu_debug` | CPU 域编译 + GDB 断点单步调试 Add kernel | `01_utilities/03_cpudebug/cpu_debug.asc` |

> 用例路径均在 `asc-devkit/examples/01_simd_cpp_api/` 下。

---

## 1. 三种工具横向对比

| | printf | Dump | CPU Debug |
|---|---|---|---|
| **用途** | 打印标量/指针/字符串 | 打印片上各存储级**张量数据** | GDB 断点/单步/查看变量 |
| **数据量** | 少量（格式串每行一条） | dumpSize 个元素（可大批量） | GDB 交互式查任意变量 |
| **跑在** | 真实 NPU | 真实 NPU | CPU（无需 NPU） |
| **编译差异** | 正常 NPU 编译 | 正常 NPU 编译 | `-DCMAKE_ASC_RUN_MODE=cpu` |
| **适用阶段** | 快速确认变量值/分支走哪 | 张量中间结果可视化、精度排查 | 复杂逻辑逐行追踪 |
| **头文件** | 不需要额外 include | `#include "utils/debug/asc_dump.h"` | `#ifdef ASCENDC_CPU_DEBUG`<br>`#include "cpu_debug_launch.h"` |

---

## 2. printf — 核函数内部打印

### 2.1 aicore 侧用法

直接在 `__aicore__` 函数中调用，与标准 C `printf` 语法一致：

```cpp
#include "kernel_operator.h"  // AscendC::printf 随此头文件引入

AscendC::printf("printf pointer %p\n", a);
AscendC::printf("mIterIdx is %u, blockIdx is %u.\n", mIterIdx, AscendC::GetBlockIdx());
AscendC::printf("aGM[0] = %f\n", aGM.GetValue(0));
AscendC::printf("baseM=%d, baseK=%d, baseN=%d.\n", baseM, baseK, baseN);
AscendC::printf("M is %x, N is %x.\n", M, N);
```

支持的格式符：

| 格式符 | 说明 |
|--------|------|
| `%p` | 指针地址 |
| `%d` | 有符号整型 / bool |
| `%u` | 无符号整型 |
| `%x` | 十六进制 |
| `%f` | half / float 浮点 |
| `%s` | 字符串 |

### 2.2 simd_vf 侧用法

`__simd_vf__` 中使用**小写** `printf`（非 `AscendC::printf`），且格式串必须标注 `__ubuf__`：

```cpp
__simd_vf__ inline void SimdVfPrintAdd(
    __ubuf__ float* x, __ubuf__ float* y, __ubuf__ float* z, uint32_t count)
{
    __ubuf__ const char* fmt = "[simd_vf] add[%u]: %f + %f = %f\n";
    for (uint32_t i = 0; i < count; i++) {
        printf(fmt, i, x[i], y[i], z[i]);  // 小写 printf, 直接用 UB 指针访问元素
    }
}

// aicore 侧通过 asc_vf_call 调用 simd_vf
asc_vf_call<SimdVfPrintAdd>((__ubuf__ float*)xLocal.GetPhyAddr(),
    (__ubuf__ float*)yLocal.GetPhyAddr(), (__ubuf__ float*)zLocal.GetPhyAddr(), PRINT_COUNT);
```

### 2.3 输出去哪里

printf 输出打印到宿主机的**标准输出（stdout）**，和普通 C 程序一样。跑在 NPU 上时通过设备→主机通道回传。

### 2.4 实际使用建议

- ✅ 查分支走向（`printf("entered AIC branch\n")`）
- ✅ 看 tiling 参数是否正确（`printf("singleCoreN=%u\n", singleCoreN)`）
- ✅ 确认某元素值（`printf("c[0][0]=%f\n", cLocal.GetValue(0))`）
- ❌ 大批量打印（用 Dump 代替）
- ❌ `__simd_vf__` 中的 printf **仅限于 CPU 调测模式**，设备侧不可用

---

## 3. Dump — 张量数据可视化

### 3.1 API 速查表

| API | 内存层级 | 地址空间 | 适用架构 |
|-----|----------|----------|---------|
| `asc_dump_gm<T>(addr, id, size)` | GM (HBM) | `__gm__` | 全部 |
| `asc_dump_l1buf<T>(addr, id, size)` | L1 Buffer | `__cbuf__` | A2/A3 (2201) |
| `asc_dump_cbuf<T>(addr, id, size)` | L0C Buffer | `__cc__` | 全部 |
| `asc_dump_ubuf<T>(addr, id, size)` | UB | `__ubuf__` | 全部 |
| `asc_dump_reg<T>(reg, id, size)` | Reg 寄存器 | — | 全部 |
| `asc_dump<T>(addr, id, size)` | 泛型（自动识别） | — | 全部 |

参数含义：
- **模板参数 `<T>`**：元素类型（half / float / int32_t 等）
- **id**：自定义整数标识符，用于在 dump 输出中区分不同 dump 点
- **dumpSize**：dump 的元素个数（不是字节数）

### 3.2 aicore 侧用法示例

```cpp
#include "utils/debug/asc_dump.h"

// dump GM 数据（A 矩阵的前 32 个元素）
asc_dump_gm<half>((__gm__ half*)aGM.GetPhyAddr(), 1, 32);

// dump L1 数据（DataCopy 完成后检查 NZ 格式是否正确）
asc_dump_l1buf<half>((__cbuf__ half*)a1Local.GetPhyAddr(), 3, 32);

// dump L0C 结果（Mmad 完成后检查中间结果）
asc_dump_cbuf<float>((__cc__ float*)cLocal.GetPhyAddr(), 5, 32);
```

### 3.3 simd_vf 侧用法

```cpp
// 在 __simd_vf__ 中 dump UB 数据
__simd_vf__ inline void DumpUbufData(__ubuf__ float* x)
{
    asc_dump_ubuf<float>(x, 5, 32);    // dump x 的前 32 个 float
    asc_dump<float>(x, 5, 16);         // 泛型 dump，自动识别为 UB
}

// dump Reg 数据
__simd_vf__ inline void DumpRegData(__ubuf__ float* x)
{
    AscendC::Reg::RegTensor<float> srcReg;
    AscendC::Reg::LoadAlign(srcReg, x);        // UB → Reg
    AscendC::Reg::Duplicate<float>(srcReg, (float)3); // 修改寄存器
    asc_dump_reg<float>(srcReg, 5, 32);        // dump 寄存器
}
```

### 3.4 Dump 输出与查看

执行后会在当前目录生成 `dump_data/` 文件夹，按 `id` 组织输出文件。可以用 Python 解析或直接 hexdump 查看。

### 3.5 核心用途

Dump 最大的价值是**排查精度问题**：想知道 Matmul 中间结果是哪里算错的？在每一级存储（GM→L1→L0A→L0C→GM）分别 dump，比对即可定位。

---

## 4. CPU Debug — 不依赖 NPU 的 GDB 调试

### 4.1 独特性

printf 和 Dump 都需要跑在真实 NPU 上——编译慢、排队慢、输出受限。CPU Debug 让 kernel 在 **x86 CPU 上以子进程方式运行**，支持完整的 GDB 交互式调试。

### 4.2 启用方式

**代码侧**：在 kernel 文件顶部加条件编译：

```cpp
#ifdef ASCENDC_CPU_DEBUG
#include "cpu_debug_launch.h"   // CPU 模式的核函数启动支持
#endif
```

**编译侧**：加 `CMAKE_ASC_RUN_MODE=cpu`：

```bash
cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..
make -j
```

**调试侧**：

```bash
gdb --args ./cpu_debug

# 必须设置：kernel 跑在子进程中，GDB 默认不跟踪子进程
(gdb) set follow-fork-mode child

# 在 kernel 入口设断点
(gdb) break add_custom

# 运行
(gdb) run

# 单步
(gdb) next

# 打印变量
(gdb) print blockLength
(gdb) print xLocal.GetValue(0)

# 继续到下一个断点
(gdb) continue
```

### 4.3 实现细节

demo 的 kernel 是标准的 Vector Add，与我们在 Day 1 学习的版本结构完全一致，唯一区别是：
- main 中**不用 `ReadFile` 读 bin 文件**——CPU 调试模式直接在 Host 内存里生成随机输入
- 不用 `aclrtMalloc` / `aclrtMemcpy` 做设备内存管理——CPU 模式下的 `cpu_debug_launch.h` 会 mock 这些调用
- 用 `std::vector` 在 Host 侧直接验证结果

### 4.4 适用场景与局限

| 适用 | 不适用 |
|------|--------|
| 逻辑正确性验证（Add/Mul/Relu 组合） | 真实 NPU 性能测量 |
| 边界条件排查（blockLength=0? 溢出?） | Cube 计算（Mmad 仅真实 NPU 支持） |
| 快速迭代开发（编译 1 秒 vs NPU 编译 30 秒） | 多核交互/核间同步行为 |
| 学习 Ascend C 语法 | 精度问题（CPU 是 FP32 全精度，NPU 是 FP16/BF16） |

---

## 5. 实战建议

```
开发流程推荐:
  Step 1: CPU Debug → 快速验证算子逻辑（不涉及 Cube 时）
  Step 2: printf   → 上 NPU 后确认 tiling 参数、核 ID、分支走向
  Step 3: Dump     → 精度不对时，逐级 dump GM/L1/L0C 定位哪一步算错
```

**一句话总结**：CPU Debug 用于**开发阶段逻辑验证**，printf 用于**快速排查配置/分支问题**，Dump 用于**深度精度定位**。三者递进使用，不是替代关系。
