# 刨根问底计划 · digroot

> 2026 儿童节启动。像孩子一样追问「为什么」，不跳过任何一个细节，直到触及本质。
> 回归底层认知，反哺上层业务。

---

## 主旨

以「深度优先」为原则的 AI 学习仓库。核心信条：**不放过任何一个「差不多懂了」的细节**。

---

## 学习计划

### 算子 — Ascend C（C++ SIMD API）

| 日期 | 内容 | 状态 | 产出 |
|:---:|------|:----:|------|
| 06-01 | **环境搭建 + 最简工程跑通**：hello_world → Vector Add | ✅ | [00 - 环境搭建与基础demo](算子/算子开发基础/00%20-%20环境搭建与基础demo/) |
| 06-02 | **能力边界 + TPipe/Matmul 入门**：全局能力分析、TQue 流水线、Matmul 高阶 API 逐行拆解 | ✅ | [01 - 熟悉算子开发能力边界](算子/算子开发基础/01%20-%20熟悉算子开发能力边界/) |
| 06-03 | **融合算子深度解析**：Advanced vs Basic API 对比、REGIST_MATMUL_OBJ 详解、KFC 核间通信框架、TPipe/TQue 设计理念、`.template` 语法、910B 架构参数 | ✅ | [02 - 了解融合算子](算子/算子开发基础/02%20-%20了解融合算子/) |
| 06-10 | **调试工具链**：printf 核内打印、Dump 张量可视化、CPU Debug + GDB 断点调试 | ✅ | [03 - 调试工具链](算子/算子开发基础/03%20-%20调试工具链/) |
| 06-11 | **性能调优入门**：bank conflict 原理与规避、Repeat/DataBlock 机制、搬运效率对比、Matmul 9 级递进优化 | ✅ | [04 - 性能调优](算子/算子开发基础/04%20-%20性能调优/) |
| 06-12 | **上机实验**：两级流水 Vector Add、DoubleBuffer 阻塞点分析、msprof 性能采集与 PipeUtilization 分析 | ✅ | [05 - 上机实验](算子/算子开发基础/05%20-%20上机实验/) |

### 模型结构

#### DeepSeek V4

| 日期 | 内容 | 状态 | 产出 |
|:---:|------|:----:|------|
| 06-29 | **DSpark 论文研读**：投机推理发展史（自回归 → 并行 → 半自回归）、Confidence-Scheduled 调度机制、与 MTP/DFlash/Eagle3 的横向对比 | ✅ | [DSpark 论文原稿](模型/deepseek_v4/DSpark_paper.pdf) |
| 06-30 | **DeepSpec 代码仓通读**：6 个核心 Trick 的逐行注释 — Markov 头、置信度头、L1 主+CE 辅损失、Teacher Forcing、位置衰减权重、STS 校准/异步调度原理 | ✅ | [DeepSpec 核心代码注释](模型/deepseek_v4/DeepSpec_核心代码注释.md) |
| 07-02 | **DeepSeek V4 模型结构学习**：MHC 多超连接残差流、MLA 多头潜在注意力、MoE 路由与负载均衡、MTP 多 Token 预测 | ✅ | [DeepSeek V4 模型结构学习](模型/deepseek_v4/DeepSeek_V4_模型结构学习.md) |

---

## 链接

| 平台 | 地址 |
|------|------|
| CSDN 专栏 | [刨根问底计划](https://blog.csdn.net/qaz2883383/category_13173041.html) |
| GitCode（主仓） | [qaz2883383/digroot](https://gitcode.com/qaz2883383/digroot) |
| GitHub（镜像） | [qaz2883383/digroot](https://github.com/qaz2883383/digroot) |

---

## 约定

1. **深度优先**：宁可少写一篇，也要把一篇写透
2. **代码可运行**：涉及代码的笔记必须附带可复现的脚本或 Notebook
3. **持续迭代**：笔记初稿只是起点，持续修订
4. **诚实标注**：不懂的地方标出来，这正是深挖的入口

---

> *Dig to the root.*
