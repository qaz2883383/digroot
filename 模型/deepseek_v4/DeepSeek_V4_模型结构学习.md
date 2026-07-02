
# DeepSeek V4 模型结构

> 本文档基于 DeepSeek V3/V4 论文和 Hugging Face transformers 源码编写。预备篇覆盖标准 Transformer 基础知识，后续章节在此基础上展开。未标注文件的代码路径均相对于 `src/transformers/models/`。

---

## 目录

- [架构纵览：DeepSeek V4 是什么](#架构纵览deepseek-v4-是什么)
- [预备篇：标准 Transformer（复习）](#预备篇标准-transformer复习)
  - [0.1 MHA 全流程](#01-mha-全流程)
  - [0.2 SwiGLU FFN](#02-swiglu-ffn)
  - [0.3 RoPE（旋转位置编码）](#03-rope旋转位置编码)
  - [0.4 KV Cache](#04-kv-cache)
- [第一章：V3 遗产 —— V4 继承了什么、丢弃了什么](#第一章v3-遗产--v4-继承了什么丢弃了什么)
  - [1.1 MLA：Multi-head Latent Attention（V4 丢弃）](#11-mlamulti-head-latent-attentionv4-丢弃)
  - [1.2 DeepSeekMoE](#12-deepseekmoe)
  - [1.3 负载均衡：e_score_correction_bias](#13-负载均衡e_score_correction_bias)
  - [1.4 MTP：Multi-Token Prediction](#14-mtpmulti-token-prediction)
- [第二章：V4 注意力 —— 三层混合 + 序列维度压缩](#第二章v4-注意力--三层混合--序列维度压缩)
  - [2.1 Shared K=V MQA](#21-shared-kv-mqa)
  - [2.2 Sliding Window Attention](#22-sliding-window-attention)
  - [2.3 HCA：Heavily Compressed Attention](#23-hcaheavily-compressed-attention)
  - [2.4 CSA：Compressed Sparse Attention](#24-csacompressed-sparse-attention)
    - [2.4.1 CSA 缓存层（CSACache）](#241-csa-缓存层csacache)
    - [2.4.2 CSA 压缩器（CSACompressor）——重叠窗口](#242-csa-压缩器csacompressor重叠窗口)
    - [2.4.3 Lightning Indexer —— 稀疏选择](#243-lightning-indexer--稀疏选择)
    - [2.4.4 block_bias —— 从索引到 mask](#244-block_bias--从索引到-mask)
    - [2.4.5 CSA 层完整 KV 尺寸与 prefill/decode 行为](#245-csa-层完整-kv-尺寸与-prefilldecode-行为)
  - [2.5 Partial RoPE + 输出反转](#25-partial-rope--输出反转)
    - [2.5.1 RoPE 的核心：用旋转获得相对位置](#251-rope-的核心用旋转获得相对位置)
    - [2.5.2 为什么只加 64 维 RoPE](#252-为什么只加-64-维-rope)
    - [2.5.3 输出反转：消除 V 上的绝对位置](#253-输出反转消除-v-上的绝对位置)
    - [2.5.4 YaRN 外推](#254-yarn-外推)
  - [2.6 Attention Sink](#26-attention-sink)
  - [2.7 Grouped Output Projection](#27-grouped-output-projection)
- [第三篇：V4 连接与聚合](#第三篇v4-连接与聚合)
  - [3.1 mHC：Manifold-Constrained Hyper-Connections](#31-mhcmanifold-constrained-hyper-connections)
  - [3.2 Sinkhorn-Knopp 算法](#32-sinkhorn-knopp-算法)
  - [3.3 HyperHead](#33-hyperhead)
- [第四篇：V4 的 MoE 路由](#第四篇v4-的-moe-路由)
  - [4.1 Hash-MoE（前 3 层）](#41-hash-moe前-3-层)
  - [4.2 Top-K MoE（后续层）](#42-top-k-moe后续层)
  - [4.3 SwiGLU Clamping](#43-swiglu-clamping)
  - [4.4 SparseMoeBlock 整体](#44-sparsemoeblock-整体)
- [第五篇：端到端串联](#第五篇端到端串联)
  - [5.1 DeepseekV4DecoderLayer](#51-deepseekv4decoderlayer)
  - [5.2 DeepseekV4Model](#52-deepseekv4model)
  - [5.3 DeepseekV4ForCausalLM](#53-deepseekv4forcausallm)
  - [5.4 KV Cache 结构](#54-kv-cache-结构)
- [附录：对照阅读索引](#附录对照阅读索引)

---

## 架构纵览：DeepSeek V4 是什么

DeepSeek V4 是一个支持 **百万 token 上下文** 的 MoE 语言模型。有两个规格：V4-Flash（284B 总参数，13B 激活）和 V4-Pro（1.6T 总参数，49B 激活）。本文档以 V4-Flash 为主线展开。

下面是一个 token 从输入到输出的完整旅程：

```
Input tokens [B, S]
      │
      ▼
Embedding [B, S, 4096]
      │
      │  拷贝 4 份（mHC 的 4 条并行残差流）
      ▼
hidden_states [B, S, 4, 4096]
      │
      ├── Layer 0 ──┐
      ├── Layer 1   │
      │   ...       │  共 43 层，每层内部结构相同：
      ├── Layer 42 ─┘
      │
      ▼
HyperHead: 4 条流 → 1 条 [B, S, 4096]
      │
      ▼
RMSNorm → lm_head → logits [B, S, 129280]
```

**每一层（DecoderLayer）内部**，数据依次经过两个子层——Attention 和 FFN。每个子层由 mHC（Manifold-Constrained Hyper-Connections）管理 4 条残差流的流入流出：

```
输入 [B, S, 4, 4096]
      │
      ├── mHC (attn_hc)    —— 4 条流塌缩 → Attention 处理 → 注入 + 流间混合
      │
      ├── mHC (ffn_hc)     —— 4 条流塌缩 → MoE 处理    → 注入 + 流间混合
      │
      ▼
输出 [B, S, 4, 4096]
```

**Attention 子层**有三种类型，按层号分配：

| 层 | 类型 | 核心机制 |
|----|------|---------|
| 0-1 | Sliding Window | 只看最近 128 个 token，不压缩 |
| 2,4,6... | **CSA** (Compressed Sparse Attention) | m=4 压缩 + Lightning Indexer 选 top-512 |
| 3,5,7... | **HCA** (Heavily Compressed Attention) | m'=128 极端压缩，全量 dense attention |

所有类型共享同一个 backbone：**Shared K=V MQA**（单 KV 头，K 和 V 是同一个张量，head_dim=512），配合 Partial RoPE（只对最后 64 维旋转）+ 输出反转 + Attention Sink + Grouped Output Projection。

**FFN 子层**全是 MoE（SparseMoeBlock），但前 3 层和后 40 层的路由方式不同：

| 层 | 路由 | 机制 |
|----|------|------|
| 0-2 | **Hash-MoE** | token_id → expert_id 查表，路由冻结 |
| 3-42 | **Top-K MoE** | sqrt(softplus) 打分 → 全局 top-6，路由可学习 |

每层 MoE 都有 256 个路由专家 + 1 个共享专家，配合 SwiGLU Clamping 和 e_score_correction_bias 负载均衡。

**V4 相对于标准 Transformer 最核心的变化**：不再让每个 token 看所有历史 token（O(S) attention），而是把长历史压缩成少量"摘要块"——CSA 压到 1/4、HCA 压到 1/128、Indexer 再从中精选。配合 mHC 的多流残差和 Hash-MoE 的路由引导，在百万 token 下同时做到高效和高性能。

下面的章节依次拆解每个模块的具体实现。

---

## 预备篇：标准 Transformer（复习）

本预备篇覆盖标准 Transformer 的核心组件。如果对这些内容已经熟悉，可以直接跳到第一章。

---

### 0.1 MHA 全流程

Multi-Head Attention 是 Transformer 的核心计算单元。每一个 decoder layer 的输入是形状为 `[batch, seq_len, d_model]` 的 hidden states。

#### 三个投影矩阵

```python
# d_model = 4096, n_heads = 64, head_dim = 64  (标准配置举例)

Q = hidden @ W_Q    # [B, S, 4096] → [B, S, 64, 64]
K = hidden @ W_K    # [B, S, 4096] → [B, S, 64, 64]
V = hidden @ W_V    # [B, S, 4096] → [B, S, 64, 64]
```

**Q（Query）**：代表"我想查什么"。当前 token 想知道的内容。

**K（Key）**：代表"我是什么"。历史 token 的"标签"，供 query 匹配。

**V（Value）**：代表"我有什么"。历史 token 携带的实际信息，被加权后送给当前 token。

直觉：Q 和 K 做匹配（"这个历史 token 跟我有关吗？"），匹配结果（attention weights）对 V 做加权求和（"有关的话，你说了什么？"）。

#### Scaled Dot-Product Attention

拿到 Q、K、V 之后，attention 的计算分四步：

**Step 1：算分数 (dot-product)**
```python
scores = Q @ K.transpose(-2, -1)
# [B, 64, S, S]  ← 每个 query 和每个 key 做点积
# scores[b, h, i, j] = query_token_i 和 key_token_j 的"相关度"分数
```
这一步就是矩阵乘法——64 个 head 各算各的。S 个 query × S 个 key，结果是 S×S 的方阵。

**Step 2：缩放 (scaled)** — 为什么除以 √d？
```python
scores = scores / sqrt(head_dim)   # head_dim = 64, 所以除以 8
```
如果不除以 √d，当 head_dim 很大时，点积结果的方差会随维度线性增长。举个例子：两个 512 维的向量做点积，结果的取值范围可能从 -500 到 +500——这时候 softmax 的每个输入不是 0 就是 ±∞，梯度直接饱和。除以 √d 把方差"压"回 1 附近，softmax 的输入不会太极端。

**Step 3：Causal Mask**
```python
mask = torch.triu(torch.full(S, S, -inf), diagonal=1)
# 上三角全是 -inf，下三角和对角线是 0
scores = scores + mask
```
自回归生成时，token t 只能看到 <t 的 token——"不能偷看未来"。mask 把 `scores[t, j]` 在 j ≥ t 的地方设为 -inf。

`softmax(-inf) = 0`，数学上：
```
exp(-inf) = 0
分母包含了这些 0 → 不影响其他位置的归一化
被 mask 掉的权重 = 0 → 对 V 的加权求和贡献为 0
```

**Step 4：Softmax + 加权 V**
```python
weights = softmax(scores, dim=-1)  # 每行归一化，权重和为 1
output = weights @ V
# [B, 64, S, S] @ [B, 64, S, 64] → [B, 64, S, 64]
```
每一行 weights[t] 是 token t 给所有历史 token 的"注意力预算"。预算总数为 1.0，向哪些 token 投多少，由 scores 的相对大小决定。

**Step 5：拼回头**
```python
# 64 个 head 的输出拼成 4096 维
output = output.transpose(1,2).reshape(B, S, 4096)
# 再过一个输出投影 (W_O)，混合所有 head 的信息
output = W_O @ output  # [B, S, 4096]
```

#### Causal Mask 为什么重要

自回归生成时，token t 只能依赖于 token 0..t-1，不能"偷看"未来。mask 把 score[t, j] (j ≥ t) 设为 -inf → softmax(-inf)=0 → 权重为 0。

---

### 0.2 SwiGLU FFN

标准 ReLU FFN 对每个 token 独立做两层全连接：

```python
# 标准 FFN (ReLU):
output = W_2 @ relu(W_1 @ x)
#         ↑ hidden → 4×hidden  ↑ 4×hidden → hidden
```

**SwiGLU 的改进**——用门控机制替代简单 ReLU：

```python
# SwiGLU FFN:
gate = W_gate @ x    # 门控信号: 决定"要不要通过"
up   = W_up @ x      # 内容信号: 实际信息
output = W_down @ (silu(gate) * up)
#                   ↑ SiLU: x·σ(x)，平滑版 ReLU
#                   ↑ 门控乘内容: gate 控制了 up 的哪些维度通过
```

对比：

| | ReLU FFN | SwiGLU FFN |
|---|---|---|
| 激活函数 | relu(x) | x·σ(x)（平滑无死区） |
| 结构 | 简单两层 | 门控（gate·up）+ 下投影 |
| 可学习性 | 死区梯度为 0 | 处处可微 |
| 表现 | 基线 | 几乎全面优于 ReLU |

SwiGLU 需要三个投影（gate/up/down）而非两个（W_1/W_2），所以 intermediate_size 通常设为 2/3 以保持总参数量不变。LLaMA、Mixtral、DeepSeek 全部使用 SwiGLU。

---

### 0.3 RoPE（旋转位置编码）

#### 动机

Transformer 的 attention（Q·K）天然对 token 顺序不敏感——token 0 在位置 3 还是在位置 500 做 attention，Q·K 的值完全一样。需要位置编码告诉模型"谁在哪"。

#### 核心直觉：用旋转编码位置

在一个二维平面上：

```
向量 (x₀, x₁) 逆时针旋转 θ 弧度:
  x₀' = x₀·cos(θ) - x₁·sin(θ)
  x₁' = x₁·cos(θ) + x₀·sin(θ)
```

如果让每个 token 的 Q 和 K 都用它的位置 p 作为旋转角：

```
Q' = R(θ·p_q) · q      ← q 旋转了 p_q 弧度
K' = R(θ·p_k) · k      ← k 旋转了 p_k 弧度

Q' · K' = q · R(θ·(p_k - p_q)) · k
        ↑ 只依赖 Δ = p_k - p_q（相对位置！）
```

**旋转矩阵 R 的正交性（R(α)⁻¹ = R(-α)）保证了绝对位置被消去。**

#### 多维 RoPE

上面讲的是单个二维平面上的旋转。但 transformer 的 head 有 d 维（比如 64 维或 512 维），不是一个二维向量。RoPE 的做法是把这 d 维两两配对，拆成 d/2 个独立的二维平面，**每个平面对用自己的旋转角，各转各的**。

举个例子就懂了。假设 head 是 8 维向量 `[x₀, x₁, x₂, x₃, x₄, x₅, x₆, x₇]`。RoPE 把它看成 4 个二维对：

```
对 0: (x₀, x₁)  — 用旋转角 θ₀ × position
对 1: (x₂, x₃)  — 用旋转角 θ₁ × position
对 2: (x₄, x₅)  — 用旋转角 θ₂ × position
对 3: (x₆, x₇)  — 用旋转角 θ₃ × position
```

每个对独立旋转，和其他对互不干扰。为什么不同对要用不同的 θ？因为如果 4 个对全部用同一个旋转角，它们看到的"位置信息"就完全一样，等于浪费了 4 份算力只做了一份工作。不同频率让每个对编码不同尺度上的位置差异。

频率怎么定？按指数衰减：

```
θ_i = base^{-2i/d}    base 取 10000（Llama 风格）或 160000（V4 压缩层风格）
```

d=8, base=10000 时：

```
i=0: θ₀ = 10000^(  0/8) = 1.0        → 每步旋转 1 弧度，转得非常快
i=1: θ₁ = 10000^(-2/8) ≈ 0.316      → 中等转速
i=2: θ₂ = 10000^(-4/8) ≈ 0.1        → 比较慢
i=3: θ₃ = 10000^(-6/8) ≈ 0.032      → 非常慢
```

**θ 越大（i 越小），旋转越快 → 对短距离（Δ=1, Δ=3）敏感。θ 越小（i 越大），旋转越慢 → 几百步之后才有明显变化 → 对长距离敏感。**

所以 RoPE 的多维设计本质就是：**高频通道负责"相邻 token 的区分"（"the"和"cat"谁在前谁在后），低频通道负责"远程 token 的区分"（第 3 段的"苹果"和第 3000 段的"苹果"谁近谁远）。** 所有频率的加权点积合在一起，给每个相对距离 Δ 一个独一无二的衰减模式。

#### 代码实现核心

```python
# cos/sin 预计算
inv_freq = 1.0 / (base ** (arange(0, d, 2) / d))
freqs = position[:, None] * inv_freq[None, :]
cos, sin = cos(freqs), sin(freqs)

# 旋转公式（interleaved 格式: (x0,x1) 为一对）
rope = x[..., -rope_dim:]        # 只转最后 rope_dim 维
rotated = rope * cos + rotate_half(rope) * sin
# rotate_half: [x0,x1,x2,x3] → [-x1,x0,-x3,x2]  即每对旋转 90°
```

---

### 0.4 KV Cache

#### 问题

自回归生成时，每个 decode step 只产出一个新 token。如果不缓存，每步都要对**全部历史 token** 重新算 Q/K/V → 计算量 O(S²)：

```python
# 无缓存: 第 t 步
K_all = W_K @ [h_0, h_1, ..., h_t]   # 全部重算！
V_all = W_V @ [h_0, h_1, ..., h_t]   # 全部重算！
# t=1M 时: 1M 个 token 的 K/V 投影 → 天文数字
```

#### 解法

**K 和 V 只与历史 token 有关，不依赖当前 query。** 每个 decode step 只需算当前 token 的 K/V 并追加到缓存：

```python
# decode step t: 只算当前
K_new = W_K @ h_t      # [1, head_dim]
V_new = W_V @ h_t

# 追加到缓存
K_cache = cat([K_cache, K_new], dim=0)  # [t, head_dim]
V_cache = cat([V_cache, V_new], dim=0)

# Attention: 当前 Q 对缓存中的历史 K/V 做 dot product
Q_new = W_Q @ h_t      # [1, head_dim]
scores = Q_new @ K_cache.T / sqrt(d)     # [1, t]
output = softmax(scores) @ V_cache       # [1, head_dim]
```

**每步只新增 1 个 K/V 投影的计算量，但 attention 的 dot product 仍要遍历全部历史 → O(S) 依然存在。** 这就是为什么长上下文需要 MLA、滑动窗口、压缩注意力等技术——它们各自从不同角度砍这个 O(S)。

```
缓存的内容: K 和 V（每个 token、每层一份）
不缓存的内容: Q（当前 token 当场算）
大小: 每 token 每层 = num_heads × head_dim × 2（K+V）× bytes_per_elem
     V3: 128 × (192+128) × 2 = 82 KB（HF 全秩）
     V4: 1 × 512 × 2 = 1 KB（K=V MQA）
```

---

## 第一章：V3 遗产 —— V4 继承了什么、丢弃了什么

### 1.1 MLA：Multi-head Latent Attention（V4 丢弃）

**代码位置**：`deepseek_v3/modular_deepseek_v3.py:169-284`

#### 1.1.1 问题：标准 MHA 的 KV cache 太大

标准 Multi-Head Attention 在推理时需要缓存所有历史 token 的 K 和 V。以 V3 671B 模型为例：

```
1 个 token 的 KV cache（单层）：
  K: 128 heads × 192 dim = 24,576 dim
  V: 128 heads × 128 dim = 16,384 dim
  合计: 40,960 dim ≈ 82 KB（BF16）

61 层 × 1M tokens × 82 KB ≈ 5 TB —— 完全不可行
```

#### 1.1.2 MLA 的解法：低速缓存，用时展开

MLA 把 KV 的下投影和上投影拆成两步——缓存压缩后的低秩潜变量，使用时再展开回全秩 K/V：

```
标准 MHA:
  hidden(7168) ── W_kv ──→ K(128×192) + V(128×128)   ← 40,960 dim 存入 cache

MLA:
  hidden(7168) ── kv_a_proj ──→ k_pass(512) + k_rot(64)   ← 576 dim 存入 cache
                                          ↓ 使用时展开
       k_pass(512) ── kv_b_proj ──→ K_nope(128×128) + V(128×128)
```

代码实现：

```python
# modular_deepseek_v3.py L195-205

# 第一步：下投影（压缩），输出 512 + 64 = 576
self.kv_a_proj_with_mqa = nn.Linear(
    config.hidden_size,                          # 7168
    self.kv_lora_rank + self.qk_rope_head_dim,   # 512 + 64 = 576
)

# 第二步：上投影（展开），把 512 维潜变量恢复到完整 K(128) + V(128)
self.kv_a_layernorm = DeepseekV3RMSNorm(self.kv_lora_rank)   # RMSNorm 512
self.kv_b_proj = nn.Linear(
    self.kv_lora_rank,                                      # 512
    self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), # 128×(128+128) = 32,768
)
```

压缩比：

```
cache_mha = 128 × (192 + 128) = 40,960 dim
cache_mla = 512 + 64           =    576 dim
压缩比: 576 / 40,960 ≈ 1.4%，缩减 98.6%
```

**Q 端也做了低秩压缩**（省参数，不省 cache）：

```python
# L191-193：Q 的低秩压缩，和 KV 同思路
self.q_a_proj = nn.Linear(7168, 1536)        # 下压缩
self.q_a_layernorm = DeepseekV3RMSNorm(1536) # 归一化
self.q_b_proj = nn.Linear(1536, 128×192)     # 上展开

# 参数对比：
# 标准: 7168 × 24576 = 176M
# MLA:  7168×1536 + 1536×24576 = 48.7M  节省 72%
```

#### 1.1.3 RoPE 拆分为 nope + rope

Q/K 的 192 维被拆成两块：

```
qk_head_dim = 192
├── qk_nope_head_dim = 128  (不加 RoPE，可压缩)
└── qk_rope_head_dim = 64   (加 RoPE，不可压缩)
```

先看 Q 和 KV 各自的数据流，然后解释为什么 KV 的 rope 要提前分离。

**Q 的数据流**（RoPE 在展开之后加）：

```
hidden [7168]
  │
  ▼
q_a_proj (7168 → 1536)       ─── 低秩压缩
  │
  ▼
RMSNorm(1536)
  │
  ▼
q_b_proj (1536 → 128×192)    ─── 低秩展开
  │
  ▼
reshape → [128 heads, 192 dim]
  │
  ├── split ──→ q_nope [128]  ──────────────────────┐
  │                                                   │
  └── split ──→ q_rope [64]  → RoPE(pos) → [64]  ──┤
                                                      │
                                          concat → Q [192]
```
**Q 的 rope 在 q_b_proj 展开之后才施加 RoPE**——RoPE 作用在完整的 64 维 rope 部分上，不受低秩压缩影响。

**KV 的数据流**（rope 必须在压缩前分离）：

```
hidden [7168]
  │
  ▼
kv_a_proj_with_mqa (7168 → 576)
  │
  ├── split ──→ k_pass [512]               k_rot [64] ←── split ──┤
  │                │                              │                │
  │                ▼                              ▼                │
  │          RMSNorm(512)                   view [1 head, 64]      │
  │                │                              │                │
  │                ▼                              ▼                │
  │          kv_b_proj (512 → 128×256)      RoPE(pos)              │
  │                │                              │                │
  │                ▼                              ▼                │
  │          reshape → [128 heads, 256]     expand → [128, 64]     │
  │                │                              │                │
  │    split ──→ K_nope [128]                    │                │
  │    split ──→ V       [128]                   │                │
  │      │                                        │                │
  └──────┼────────────────────────────────────────┘                │
         │                                                         │
         ▼                                                         │
   concat → K [128 nope + 64 rope = 192]                          │
   V = [128]  (不加 RoPE)                                          │
```

**关键差异一目了然**：Q 的 rope 在 q_b_proj **展开之后**加 RoPE，KV 的 rope 在 kv_a_proj **压缩之前**就被剥离出来单独处理。下面是原因。

**为什么 k_rot 的 64 维不参与压缩？** 因为 RoPE 是位置相关的旋转变换，不能与矩阵乘法交换：

```
W @ R(x) ≠ R(W @ x)    除非 W 是正交旋转矩阵

如果 k_rot 被混进 512 维潜变量再解压，位置信息就丢失了。
所以 k_rot 必须独立存储，不走 kv_a → kv_b 路径。
```

Q 为什么可以全走压缩？因为 RoPE 在**展开之后**才施加：

```python
# Q 路径: 全部压缩再展开，然后 split，然后在 rope 部分加 RoPE
q_states = q_b_proj(RMSNorm(q_a_proj(hidden)))  # 压缩再展开
q_pass, q_rot = split(q_states, [128, 64])       # 展开后才拆分
q_rot = RoPE(q_rot)                              # ✅ 没问题

# KV 路径: k_rot 必须在压缩前就分离出来
k_pass, k_rot = split(kv_a_proj(hidden), [512, 64]) # k_rot 在压缩前分离
k_pass = kv_b_proj(RMSNorm(k_pass))                  # 只有 nope 部分走压缩
k_rot = RoPE(k_rot)                                  # k_rot 独立旋转
```

#### 1.1.4 主流推理引擎缓存的是什么

论文 V3 §2.1.1 原文（蓝色高亮部分）：

> "only the blue-boxed vectors (i.e., **c_t^KV** and **k_t^R**) need to be cached during generation"

**生产环境（vLLM、SGLang、DeepSeek 自有引擎）缓存的是压缩态：**

```
cache: c_t^KV (512 dim) + k_t^R (64 dim) = 576 dim / token / layer
```

**Hugging Face 参考实现缓存的是展开后的全秩 K/V：**

```python
# modeling_deepseek_v3.py L455
key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
# key_states: [B, 128, S, 192]  ← 全秩 K
# value_states: [B, 128, S, 128]  ← 全秩 V
```

HF 存全秩 K/V 是为了兼容通用 `past_key_values.update()` 接口。这不是 MLA 设计的问题，是 HF 的通用性代价。生产环境通过吸收优化跳过了解压——将 `W^UK` 吸收进 `W^Q`，直接用 Q 的压缩态和 K 的压缩态做 dot product，根本不展开 K。

#### 1.1.5 为什么 V4 不用 MLA 了

MLA 的设计在 V3 的 128K 训练上下文下运行良好。但 V4 要支持 1M 上下文——差了 8 倍。MLA 的两个计算瓶颈在百万 token 下暴露：

**瓶颈 1：解压开销。** MLA 在 decode 时存入 512 维压缩潜变量，但做 attention 时需要展开回全秩的 32,768 维。生产环境通过吸收优化（W^UK 吸收进 W^Q）跳过了很多展开，但 RoPE 部分（k_rot 的 64 维）无法吸收——必须对全部历史 token 的 k_rot 做独立的旋转变换。1M token 下，即使只对 k_rot(64) 做旋转，每层每步也要 64M FLOPs，43 层合计 2.8G FLOPs，加上 attention 本身的 QK 点积，总计算量堆到了不可接受的程度。

**瓶颈 2：Attention 的 O(S) 项仍然存在。** MLA 省的是 KV cache 的空间（576 维 vs 40,960 维），但 decode 的 attention 计算仍然是 Q(1 token) × K(全部 S token) ——这个 O(S) 的矩阵向量乘法在 1M 时每层约 33M FLOPs，43 层约 1.4G FLOPs。MLA 在空间维度省了 80 倍，但在时间维度（计算量）一分没省。

V4 换了一条路：不在"维度空间"压缩，直接在"序列维度"压缩。CSA 把 S 个 token 压成 S/4 个条目，HCA 压成 S/128 个条目——**条目数的减少直接砍掉了 attention 的 O(S) 项。** 1M 上下文下，CSA 只 attend 640 个条目（128 滑动 + 512 选中压缩块），不是 1M 个。

---

### 1.2 DeepSeekMoE

**代码位置**：`deepseek_v3/modular_deepseek_v3.py:91-166`

#### 1.2.1 基本结构

普通 MoE（如 Mixtral）用 8 个大的专家，每个 token 选 2 个。DeepSeekMoE 把这个设计往两个方向推：

**(1) 细粒度专家。** 把 8 个大专家拆成 256 个小专家，每个的中间维度从 14336 压到 2048，同时每 token 激活数从 2 提高到 8（V3）或 6（V4）。总计算量基本不变，但专家的"分工精度"提升了。

打个比方：Mixtral 的 8 个专家就像 8 个全科医生——每个都得学内科、外科、儿科，知识分散在每个医生脑子里。DeepSeekMoE 的 256 个专家就是 256 个专科医生——血液科、骨科、神经科各司其职。病人来的时候不是找 2 个全科医生会诊，而是找 6 个最对口的专科医生。

**(2) 共享专家。** 256 个路由专家之外，再加 1 个"全量专家"——对所有 token 始终在线，不经过路由选择。它的输出直接加到路由专家的结果上。

这个共享专家学的是"通用知识"——语法结构、基本词汇、常见句式。这些知识如果每个路由专家都学一遍，256 个专家的参数量有大量冗余。抽出来集中学习，路由专家就可以专注于自己的专业领域，不用重复学基础的东西。

```
Mixtral 标准 MoE:   8 个大专家，每个 intermediate = 14336
DeepSeekMoE (V3):   256 个小专家，每个 intermediate = 2048 + 1 个共享专家
DeepSeekMoE (V4):   256 个小专家，每个 intermediate = 2048 + 1 个共享专家
                          (V3: 激活 8 个, V4: 激活 6 个)
```

```python
# V3 DeepseekV3MoE L113-166
class DeepseekV3MoE(nn.Module):
    def __init__(self, config):
        self.experts = DeepseekV3NaiveMoe(config)     # 256 个细粒度路由专家
        self.gate = DeepseekV3TopkRouter(config)       # 路由打分
        self.shared_experts = DeepseekV3MLP(           # 1 个共享专家
            config, intermediate_size=moe_intermediate * n_shared_experts
        )
        self.top_k = config.num_experts_per_tok        # V3: 8, V4: 6

    def forward(self, hidden_states):
        residual = hidden_states                       # 保存残差
        indices, weights = self.gate(hidden_states)    # 路由选择
        routed_out = self.experts(hidden_states, indices, weights)
        return routed_out + self.shared_experts(residual)  # 共享专家始终运行
```

每个专家的内部结构就是标准的两层全连接 SwiGLU MLP：

```python
# MixtralExperts (modeling_mixtral.py:70-71)
self.gate_up_proj = nn.Parameter([num_experts, 2*intermediate, hidden])  # Linear 1
self.down_proj    = nn.Parameter([num_experts, hidden, intermediate])    # Linear 2

def forward(self, x, indices, weights):
    for expert_i in active_experts:
        gate, up = (x @ gate_up_proj[i]).chunk(2)   # SwiGLU
        out = silu(gate) * up
        final += (out @ down_proj[i]) * weights       # 加权累加
```

#### 1.2.2 V3 的分组约束路由

V3 在 256 个专家中加了分组约束——先选 4 组再在组内选 top-8：

```python
# V3 route_tokens_to_experts L133-156
def route_tokens_to_experts(self, router_logits):
    router_logits = router_logits.sigmoid()                # sigmoid 打分
    scores = router_logits + e_score_correction_bias       # 加负载均衡偏置

    # 步骤 1: 256 专家 → reshape 成 8 组，每组 32 个
    #        每组取 top-2 的分求和 → 组得分 → 选 top-4 组
    group_scores = scores.view(-1, 8, 32)         # [S, 8, 32]
                         .topk(2, dim=-1)[0]       # 每组取 top-2 分
                         .sum(dim=-1)              # 求每组的得分和
    top_groups = topk(group_scores, k=4)           # 选 4 个最高分组

    # 步骤 2: 把未被选中的 4 组全部 mask 掉（设 -inf）
    group_mask = scatter_zeros(1, top_groups)      # 在 8 个位置中标记选中的 4 个
    score_mask = group_mask.unsqueeze(-1)          # 扩展到 32 个专家
                  .expand(-1, 8, 32)               # [S, 8, 32]
                  .reshape(-1, 256)                # 展回 [S, 256]，True=可见
    scores_for_choice = scores.masked_fill(~score_mask, -inf)

    # 步骤 3: 在剩下的 4×32=128 个专家中做 top-8
    indices = topk(scores_for_choice, k=8)

    # 权重: 用原始 sigmoid 分数（不加 bias）
    weights = router_logits.gather(indices)            # 选原始分数
    weights /= weights.sum()                           # 归一化
    weights *= routed_scaling_factor                   # × 2.5
```

分组约束是**通信优化**——256 个专家分布在 8 台设备上，如果不加约束，top-8 可能全落在 1 台设备上，通信爆炸。限制在 4 组（≤4 台设备）内，通信被分散。

#### 1.2.3 V4 的改动

| 方面 | V3 | V4 |
|------|-----|-----|
| 打分函数 | sigmoid | `sqrt(softplus(x))` |
| 分组约束 | 8 组选 4 组 | **无**（去掉 n_group/topk_group） |
| 激活专家数 | 8 | 6 |
| Hash-MoE | 无 | 前 3 层 |
| SwiGLU Clamping | 无 | gate≤10, up∈[-10,10] |

`sqrt(softplus(x))` 的好处：sigmoid 在 x≥5 时趋近饱和（梯度→0），且 sigmoid 的上限 1.0 无法区分"好"和"极好"的专家。softplus 无上界、梯度永不为零、sqrt 压缩数值范围防止极端高分主导路由。

V4 去掉分组约束是因为 V4 的 wave-based 专家并行（论文 §3.1）把通信藏到了计算背后——通信和计算流水线并行，哪个专家被选中已经不影响通信瓶颈。

---

### 1.3 负载均衡：e_score_correction_bias

**代码位置**：`deepseek_v3/modular_deepseek_v3.py:91-103` / `deepseek_v4/modular_deepseek_v4.py:913-927`

论文 V3 §2.1.2 "Auxiliary-Loss-Free Load Balancing" 原文：

> "We introduce a bias term **b_i** for each expert and add it to the corresponding affinity scores to determine the top-K routing. Note that the bias term is **only used for routing**. The gating value, which will be multiplied with the FFN output, is still derived from the original affinity score s_i,t."

```python
# V4 DeepseekV4TopKRouter.forward() L920-927
logits = F.linear(flat, self.weight)                     # [S, 256]
scores = self.score_fn(logits)                           # sqrt(softplus(logits))

indices = topk(scores + e_score_correction_bias, k=6)    # ← bias 加在 argmax 上
weights = scores.gather(1, indices)                      # ← 权重用原始 scores，不含 bias
weights = weights / weights.sum() * routed_scaling_factor
```

**bias 只影响"选谁"，不影响"给多少权重"。**

更新规则（训练框架中，不在 modeling 代码里，论文 §4.2.2 bias_update_speed=0.001）：

```python
# 每个 training step 后
for expert i:
    actual = count_tokens_assigned_to_i()
    expected = total_tokens × top_k / n_experts   # = S × 6 / 256

    if actual > expected:
        bias[i] -= 0.001 × (actual - expected) / expected   # 太受欢迎 → 降
    else:
        bias[i] += 0.001 × (expected - actual) / expected   # 没人选 → 升
```

**为什么这比传统辅助 loss 好？** 传统方法在 loss 上加惩罚项 `λ × Σ(f_i × P_i)`，梯度会干扰主任务。bias 不产生梯度——主任务 loss 不受影响，argmax 算子被"外部推力"调节。

---

### 1.4 MTP：Multi-Token Prediction

**代码位置**：V3/V4 的 HF 代码中无实现，仅在 `configuration_deepseek_v4.py:173` 保留字段 `num_nextn_predict_layers=1`，加载时跳过 MTP 权重。

MTP 在每一步预测 t+1 之外，用一个额外的 Transformer 层预测 t+2：

```
标准 LM:  [t1, t2, t3] → 预测 t4

MTP (depth=1):
  [t1, t2, t3] → 预测 t4         ← 主模型
               → 预测 t5         ← MTP 模块
```

MTP 模块是怎么预测 t5 的？它的输入是两部分：主模型最后一层的 hidden state（含有位置 3 及之前的所有信息），加上已经预测出的 t4 的 token embedding。两个输入各自过 RMSNorm，拼接，过一个线性投影把 2d 压回 d，然后送入一个完整的 Transformer 层（self_attn + MoE）。这个 Transformer 层和主模型的层结构完全一样，但参数独立。输出过共享的 lm_head 得到 t5 的预测。

MTP 模块结构（来自官方 `README_WEIGHTS.md`）：

```
model.layers.61                     ← 额外 Transformer 层
    ├── enorm  (RMSNorm)            ← 对主模型输出做归一化
    ├── hnorm  (RMSNorm)            ← 对 Emb(t_{i+1}) 做归一化
    ├── eh_proj (Linear 2d→d)       ← 拼接后的降维投影
    ├── self_attn (MLA)             ← 和主模型层完全一致
    └── mlp (DeepSeekMoE)           ← 和主模型层完全一致

model.embed_tokens                  ← 共享（和主模型同一个）
lm_head                             ← 共享（和主模型同一个）

总参数: 11.5B 独立 + 1.8B 共享 = 13.3B
```

训练时，总 loss 是主任务 loss 和 MTP loss 的加权和：

```
total_loss = LM_loss + λ × MTP_loss
```

其中 LM_loss 是主模型预测 token t+1 的交叉熵，MTP_loss 是 MTP 模块预测 token t+2 的交叉熵。λ 控制 MTP 辅助任务在总 loss 中的占比。V3 和 V4 都用 λ=0.3（大部分训练），最后降到 0.1。V3 和 V4 的 MTP 结构完全一致。

---

## 第二章：V4 注意力 —— 三层混合 + 序列维度压缩

### 2.1 Shared K=V MQA

**代码位置**：`deepseek_v4/modular_deepseek_v4.py:648-766`

#### 2.1.1 三个概念叠加

Shared K=V MQA 是三个独立设计的叠加：

**(1) MQA（Multi-Query Attention）：** 所有 query head 共享 1 个 KV head。

```python
# L672
self.num_key_value_groups = config.num_attention_heads  # = 64
# 含义: 64 个 query head 全共享 1 个 KV head
```

**(2) K=V：** K 和 V 是同一个张量，只用一个投影矩阵。

常规 attention 的 K 和 V 分别投影——K 用来匹配 Q 做注意力打分，V 用来做加权求和产出实际信息。但在 V4 里，KV 条目的数量已经被压缩器大幅削减了——1M token 下 HCA 只有 ~8K 个压缩条目，每个条目是一段 token 的"摘要"。对于摘要来说，"它是关于什么的"（K 的作用）和"它包含什么信息"（V 的作用）高度重叠——一个压缩块的内容本身就可以拿来当标注关键词用。

K=V 意味着一行代码覆盖两者的工作：
```python
self.kv_proj = nn.Linear(config.hidden_size, self.head_dim)   # 4096 → 512，一个投影干两份活
```
K 和 V 各用一个投影是 8192 维的输出维度。K=V 把输出压到 512 维——**投影参数量减半，对 KV cache 和计算都有直接好处。**

```python
# L683: 只有一个投影矩阵，产出 512 维 → 既是 K 也是 V
self.kv_proj = nn.Linear(config.hidden_size, self.head_dim)   # 4096 → 512

# forward L714-715
kv = self.kv_norm(self.kv_proj(hidden_states))   # [B, S, 512]
kv = kv.view(B, S, 1, 512).transpose(1, 2)       # [B, 1, S, 512]

# L742-746: 同一个 kv 同时当 key 和 value 用
attn_output = attention_interface(
    q,   # [B, 64, S, 512]
    kv,  # [B,  1, S, 512]  ← key
    kv,  # [B,  1, S, 512]  ← value（同一个！）
    ...
)
```

**(3) head_dim=512：** 远超常规的 128，信息密度更高。

#### 2.1.2 对比 V3 MLA 和标准 MHA

| 机制 | KV 投影参数 | KV cache 尺寸 (dim/token/layer) |
|------|------------|------|
| 标准 MHA (128 heads, 128 dim) | W_k + W_v | 40,960 |
| V3 MLA (压缩态) | kv_a_proj(7168→576) + kv_b_proj(512→32768) | 576（需解压） |
| **V4 K=V MQA** | **kv_proj(4096→512)** | **512（不需解压）** |

V4 的 512 比 V3 压缩态的 576 还略小，且不需要 kv_b_proj 解压。比 V3 的全秩展开态 40,960 小了 **80 倍**。

#### 2.1.3 收益

1. **KV cache 急剧缩小。** 1M 上下文下，标准 causal attention 每层 1GB，V4 K=V MQA 每层仅 1MB。
2. **投影参数极少。** V3 的 KV 侧 21M 参数，V4 仅 2.1M。
3. **decode 时无需解压。** 不存在 V3 的 kv_b_proj 展开开销。
4. **代价**：head_dim=512 超出 FlashAttention 的 256 上限，只能用 eager attention。

---

### 2.2 Sliding Window Attention

**代码位置**：`deepseek_v4/modular_deepseek_v4.py:648-718`

#### 2.2.1 滑的是 token，不是 layer

滑动窗口就是一个固定容量 128 的 FIFO 队列——每次新 token 进、最旧 token 出。它不做任何压缩、聚合或权重变换，每个 KV 条目就是一个标准的 512 维向量，和普通 attention 里每个 token 的 K/V 完全一样。

```python
# DeepseekV4HCACache.update() L167-179
def update(self, key_states, value_states, *args, **kwargs):
    # key_states: [B, 1, seqlen_new, 512] —— 本轮新 token 的共享 K=V（单头 MQA）
    if not self.is_initialized:
        self.lazy_initialization(key_states, value_states)    # 分配 [B, 128, 512] 的 buffer
        self.values = self.keys                                # K=V，value 缓存复用 key 缓存

    full = torch.cat([self.keys, key_states], dim=-2)          # 拼接旧缓存 + 新 token 的 KV
    self.keys = full[:, :, -127:, :]                           # 只保留最后 127 个
    # 为什么是 127 不是 128？因为当前 token 自己也在序列里，
    # 加上当前 token 自己的 KV，总共 128 个条目对 query 可见
    self.values = self.keys
    return full, full                                          # 返回拼接后的完整 KV 给 attention 用
```

**第 0 层的滑动窗口和第 3 层完全独立，互不影响。** 每层有自己专属的 `self.keys` 缓冲区，各自维护 128 token 的窗口。

在 attention 中，滑动窗口 KV 和压缩 KV 通过一次 `torch.cat` 拼接：

```python
# Attention.forward() L714-725
kv_sliding = self.kv_norm(self.kv_proj(hidden_states))     # 当前 token 的原始 KV
kv_sliding = cache.update(kv_sliding)                       # 加入 FIFO 并返回全量窗口

kv_compressed = self.compressor(...)                        # 压缩器产出压缩块 KV
kv = torch.cat([kv_sliding, kv_compressed], dim=2)          # 拼在一起
# [B, 1, 128, 512] + [B, 1, T, 512] → [B, 1, 128+T, 512]

# attention: Q(64 heads) 对全部 KV 条目加权
attn_output = attention_interface(q, kv, kv, attention_mask, sliding_window=128)
```

滑动窗口的 128 个条目是**原始精度、零压缩**的。压缩块的条目可能有信息损失（softmax 加权聚合），而滑动窗口保留了每个 token 的完整 KV 表示——这里是提供给 query 的"高保真局部上下文"。

#### 2.2.2 V4 中滑动窗口的双重角色

滑动窗口在 V4 里干了两个不同的活——理解这个区别很重要。前 2 层和 CSA/HCA 层里都用到了滑动窗口，但目的完全不同。

**(1) 前 2 层独立注意力——给压缩器暖场。**

第 0 和第 1 层没有 compressor（`self.compressor = None`）。这两层只有滑动窗口 KV，query 只看最近 128 个 token，不对历史做任何压缩。

```python
# L690-691
self.compressor = (
    COMPRESSOR_CLASSES[self.layer_type](config)
    if self.layer_type != "sliding_attention" else None   # 前 2 层: None
)
```

为什么前 2 层不直接用压缩？因为 CSA 需要攒满 4 个 token 才能产出第一个压缩块，HCA 需要攒满 128 个。序列开头的"冷启动"阶段窗口还没闭合，压缩器什么都产不出来。前 2 层用纯滑动窗口拖过这个冷启动期。

**(2) CSA/HCA 层的辅助分支——修补压缩盲区。**

CSA/HCA 层压缩当前窗口的时候，最后一个未完成的窗口的 token 还没有被压缩进任何条目，在压缩块中不出现。滑动窗口的 128 个原始 token 覆盖这个盲区——当前窗口的末尾 token 虽然不在压缩块里，但一定在滑动窗口里。

例如 HCA 层，compress_rate=128。当前缓存了 t0-t127 的压缩块 C0 已产出。t128 到 t254 的还在积累中，window 2 还没闭合。t=200 的 query 想看书 t=180 的内容——t180 在当前的未完成窗口中，压缩块里不存在。但滑动窗口覆盖了最近 128 个 token（t73-t200），t180 在窗口内——直接可访问。

#### 2.2.3 数字对比（1M 上下文，单层）

下表总结三种模式下每层的 KV cache 尺寸、decode 每步计算量和信息覆盖。表格中的数字是这样得出的：

- **无滑动窗口 (standard causal)**：KV cache = 512 dim × 1M token = 512M 个元素。decode 每步 attention = Q(64 heads, 1 token, 512 dim) 对 K(1 head, 1M token, 512 dim) 做点积。
- **有滑动窗口 + HCA**：KV cache = 滑动窗口 (128) + buffer (最多 127) + 压缩条目 (1M/128=7812)。buffer 的 127 是 compress_rate-1——攒满 128 才压缩一次。decode attention = 128 个滑动 + 全量压缩条目 (HCA 不做 sparse selection)。
- **有滑动窗口 + CSA**：KV cache 同 HCA 但压缩条目数 = 1M/4=250K。buffer 同理最多 m-1=3。decode attention = 128 个滑动 + indexer 选中的 top-512 个压缩条目（CSA 只对选中的做 attention）。

| | 无滑动窗口 (standard causal) | 有滑动窗口 + HCA | 有滑动窗口 + CSA |
|---|---|---|---|
| KV cache 尺寸 | 512×1M = 512M dim = **1 GB** | 512×(128+127+7812) ≈ 4.1M dim ≈ **8 MB** | 512×(128+127+250000) ≈ 128M dim ≈ **256 MB** |
| decode 每步计算量 | Q @ K(1M) → **~33G FLOPs** | Q @ K(128+7812) → **~260M FLOPs** | Q @ K(128+512) → **~21M FLOPs** |
| 信息覆盖 | 全量 1M token | 128 个原始 + 7812 个压缩块 | 128 个原始 + 512 个选中块 |

---

### 2.3 HCA：Heavily Compressed Attention

**代码位置**：`deepseek_v4/modular_deepseek_v4.py:298-379` & `L134-215`

#### 2.3.1 核心思想

**每 128 个 token 压缩成 1 个条目，全部压缩块对 query 可见。**

V4 Flash 43 层中，第 3,5,7...层是 HCA（共 21 层）。每层独立压缩，压缩条目不跨层传递。

#### 2.3.2 压缩模块

```python
# DeepseekV4HCACompressor L320-328
class DeepseekV4HCACompressor(nn.Module):
    def __init__(self, config):
        self.compress_rate = 128              # m' = 128

        self.kv_proj   = nn.Linear(4096, 512)       # 每个 token 的"内容"
        self.gate_proj  = nn.Linear(4096, 512)       # 每个 token 的"重要性"
        self.position_bias = nn.Parameter([128, 512]) # 窗口内位置偏置
        self.kv_norm    = RMSNorm(512)           # 压缩后归一化
        self.rotary_emb = RotaryEmbedding()      # YaRN θ=160000
```

注意 `kv_proj` 和 `gate_proj` 是 compressor 自己的投影，跟 attention 主干的 `kv_proj`（L683）**不是同一个**——compressor 这组投影专门服务于压缩融合。

#### 2.3.3 压缩过程（逐行注释）

```python
def forward(self, hidden_states, q_residual, position_ids, past_key_values, layer_idx):
    batch, _, _ = hidden_states.shape
    cache_layer = past_key_values.layers[layer_idx] if past_key_values else None

    # ① 投影：每个 token 产出"内容"和"重要性"
    kv   = self.kv_proj(hidden_states)    # [B, S, 512]
    gate = self.gate_proj(hidden_states)  # [B, S, 512]

    # ② 攒窗口材料
    #    prefill (cache_layer=None): 直接截断到 128 的整数倍
    #    decode (cache_layer 存在): buffer 追加，满 128 返回 chunk
    if cache_layer is None:
        usable = (kv.shape[1] // 128) * 128
        chunk_kv, chunk_gate, first_window_position = kv[:, :usable], gate[:, :usable], 0
    else:
        chunk_kv, chunk_gate, first_window_position = \
            cache_layer.store_compression_weights("compressor", kv, gate)

    # ③ 窗口内 softmax 加权聚合
    if chunk_kv.shape[1] > 0:
        n_windows = chunk_kv.shape[1] // 128
        chunk_kv   = chunk_kv.view(batch, n_windows, 128, 512)
        chunk_gate = chunk_gate.view(batch, n_windows, 128, 512) + self.position_bias

        # softmax 沿 128 个 token 维度 → 每个通道有独立权重
        weights = chunk_gate.softmax(dim=2, dtype=torch.float32).to(chunk_kv.dtype)
        #   [B, N, 128, 512] × [B, N, 128, 512] → 沿 dim=2 求和
        compressed = (chunk_kv * weights).sum(dim=2)   # [B, N, 512]

        compressed = self.kv_norm(compressed)           # RMSNorm

        # 给压缩条目施加 YaRN RoPE（位置 = 窗口起始 token 的绝对位置）
        positions = (torch.arange(n_windows) * 128 + first_window_position)
        cos, sin = self.rotary_emb(compressed, position_ids=positions)
        compressed = apply_rotary_pos_emb(compressed, cos, sin)

    else:
        compressed = chunk_kv.new_zeros((batch, 0, 512))

    # ④ 追加到历史缓存
    if cache_layer is not None:
        compressed = cache_layer.update_compressor_states("compressor", compressed)
        # compressed_kv = [C0, C1, C2, ...]  → 长度不断增长，永久保留

    compressed_kv = compressed.unsqueeze(1)  # [B, 1, T, 512]

    # ⑤ 因果 mask：压缩条目 w 代表 token [w×128, (w+1)×128-1]
    #    query t 只有当窗口 w 闭合（t ≥ (w+1)×128）时才能看到它
    causal_threshold = (position_ids + 1) // 128       # [B, S]
    entry_indices = torch.arange(compressed_len)
    block_bias = zeros((batch, 1, seq_len, compressed_len))
    block_bias.masked_fill_(entry_indices >= causal_threshold, -inf)

    return compressed_kv, block_bias
```

#### 2.3.4 为什么用 softmax 加权求和而不是直接接一个 Linear

先回看压缩器的核心代码——窗口内的 128 个 token 是怎么被捏成 1 个压缩条目的：

```python
# L349-353
chunk_kv   = chunk_kv.view(batch, n_windows, 128, 512)
chunk_gate = chunk_gate.view(batch, n_windows, 128, 512) + self.position_bias
#                             ↑ 门的投影（通过 gate_proj）    ↑ 可学习的位置偏置

weights = chunk_gate.softmax(dim=2, dtype=torch.float32).to(chunk_kv.dtype)
# softmax 沿 128 个 token 维度 → 每个 token 在每个通道上有一个权重（和为 1）

compressed = (chunk_kv * weights).sum(dim=2)
# 128 个 token 逐通道加权求和 → 1 个压缩条目
```

这里的 `softmax` 不是 attention 里的 softmax——它是**横向**对 128 个 token 做的，作用是把 gate_proj 产出的"重要性分数"变成概率分布。注意 `chunk_gate` 和 `chunk_kv` 形状完全相同（都是 `[128, 512]`）——**每个通道有独立的 128 个权重**，不是 128 个 token 共享一个标量权重。

那为什么选 softmax 加权求和而不是直接用一个 Linear 把 128 个 token 映射成 1 个？把 128×512=65536 维的展平向量喂给一个 Linear 是最直观的替代方案。下面是比较：

| 方案 | 每层参数量 | 归纳偏置 | 数值稳定 |
|------|----------|---------|---------|
| **Softmax 加权**（当前） | 65K (position_bias) | convex combination，天然收缩 | ✅ 输出在原始值域 |
| Linear(65536→512) | 33.5M × 21 层 = 704M | 无约束 | 可能爆炸 |
| Mean Pooling | 0 | 等权平均 | ✅ 但丢区分度 |
| 单标量 Attention | ~512K | 512 通道共享一个权重 | ✅ 但无通道级选择 |

softmax 加权方案同时满足了三个条件：省参数、逐通道独立选择、数值稳定。

#### 2.3.5 缓存结构及 prefill/decode 行为

HCA 的压缩器不是一次性把 1M token 全压完的——decode 阶段是一步一个 token 增量地积累、压缩、缓存。这个过程中需要维护两类 KV 状态：

- **滑动窗口 KV**（从父类 `DynamicSlidingWindowLayer` 继承）：始终保留最近 128 个 token 的原始 KV，query 可直接访问。
- **压缩器状态**（`DeepseekV4HCACache` 增加）：管理压缩器所需的 buffer、已压缩条目列表和计数。

具体的状态字段：

```
buffer_kv["compressor"]:   ≤127 个未压实的 kv (等待下一个窗口闭合)
buffer_gate["compressor"]: ≤127 个未压实的 gate

compressed_kv["compressor"]: [C0, C1, ..., C_T-1]  已压实的条目列表, 永久保留
entry_count["compressor"]: T                        已压实条目计数

滑动窗口 (继承): [B, 128, 512] 最近 128 个 token 的高精度 KV
```

**buffer** 就像一个蓄水池——每次 decode 流入 1 个新 token 的 kv/gate，蓄到 128 个时压缩一次（产出 1 个压缩条目，清空蓄水池），然后继续积累。

**compressed_kv** 是历史所有压缩条目的列表——C0 在 t=127 时产生，到 t=1M 时仍然保留在列表中随时被 query 访问。

下面是 prefill 和 decode 两个阶段的行为对比：

**Prefill vs Decode 对比：**

| | Prefill (cache_layer=None) | Decode (cache_layer 存在) |
|---|---|---|
| 输入形状 | [B, S, 4096], S 可能很大 | [B, 1, 4096] |
| 窗口切分 | 直接截断到 128 整数倍 | buffer 累积，每 128 步压缩一次 |
| 条目缓存 | 不缓存（返回单次产出） | 追加到 compressed_kv 历史列表 |
| 返回 mask | block_bias（每 query 可见哪些块） | seq_len=1 → None |

---

### 2.4 CSA：Compressed Sparse Attention

**代码位置**：`deepseek_v4/modular_deepseek_v4.py:218-263` (CSACache) & `L398-522` (Indexer) & `L525-638` (CSACompressor)

#### CSA 与 HCA 的核心区别

| | HCA | CSA |
|---|---|---|
| 压缩率 | m'=128 | m=4 |
| 窗口是否重叠 | 不重叠 | 重叠（Ca/Cb 双序列，感受野 8 token） |
| 选择机制 | 全量 dense attention | Lightning Indexer 做 top-512 稀疏选择 |
| 投影维度 (kv_proj) | head_dim (512) | 2×head_dim (1024) |
| 1M 时压缩条目数 | ~7.8K | ~250K |
| 每 query 实际 attend | 128 + 7.8K | 128 + 512 = 640 |

---

#### 2.4.1 CSA 缓存层（CSACache）

继承 HCACache，额外增加两项：

```python
class DeepseekV4CSACache(DeepseekV4HCACache):
    def __init__(self, config):
        super().__init__(config)
        self.compress_rate = 4  # m=4

        # 新增 1: indexer 的独立 buffer/compressed/计数
        self.buffer_kv["indexer"]   = None
        self.buffer_gate["indexer"] = None
        self.compressed_kv["indexer"] = None
        self.entry_count["indexer"] = 0

        # 新增 2: 重叠状态 —— 上一个窗口的 Ca 切片
        self.overlap_kv   = {"compressor": None, "indexer": None}
        self.overlap_gate = {"compressor": None, "indexer": None}
```

`overlap_kv` 存上一个窗口的 Ca 切片（`chunk[:, -1, :, :512]`），供下一个窗口的重叠压缩使用。这是跨 decode step 的状态传递。

---

#### 2.4.2 CSA 压缩器（CSACompressor）——重叠窗口

m=4 的压缩窗口面临一个困境：窗口太小，每个压缩条目只看 4 个 token。HCA 的 m'=128 窗口大到感觉像"摘要"，CSA 的 m=4 窗口小到几乎等于原始 token——但 CSA 需要从 250K 个条目中选 512 个做 attention，每个条目要有充足的信息量才能被 indexer 有效筛选。

CSA 用一种巧妙的方式扩大感受野而不降低压缩率：**让相邻窗口重叠。** 核心技巧是利用 Ca/Cb 双投影系：

投影到 1024 维 = [512:Ca, 512:Cb]，两个投影系各管一半。Ca 是 token 给"下一个窗口"的贡献，Cb 是给"当前窗口"的贡献。

走一个具体的例子看懂了。

假设序列 `t0 t1 t2 t3 t4 t5 t6 t7 t8 t9 t10 t11`，m=4，分割成 3 个窗口：

```
窗口 0: t0-t3   窗口 1: t4-t7   窗口 2: t8-t11
```

每个 token 都经过了 `kv_proj` 和 `gate_proj` 投影，产出 1024 维 = Ca 前 512 + Cb 后 512。

压缩条目 C1（窗口 1 的产出）：

```
C1 的组合来源:
  - 前一窗口 (窗口 0) 的 Ca 切片（4 个 token 的 Ca，512 维）→ 填在 8 个 slot 的前 4 个
  - 当前窗口 (窗口 1) 的 Cb 切片（4 个 token 的 Cb，512 维）→ 填在 8 个 slot 的后 4 个

8 个 slot 的 softmax 加权求和 → 窗口 1 的 1 个压缩条目
```

所以 C1 看的是 8 个 token：t0-t3 的 Ca 部分 + t4-t7 的 Cb 部分。窗口 0 的 Ca 不受窗口 0 的压缩本身影响——Ca 为下一个窗口提供服务，Cb 给当前窗口用。

图示：

```
token:  t0 t1 t2 t3 | t4 t5 t6 t7 | t8 t9 t10 t11
Cb:     ├──窗口0───┤├──窗口1───┤├──窗口2───┤
Ca:        ├──窗口0───┤├──窗口1───┤├──窗口2───┤

C0:  无前序(填0) + Cb(t0..t3)             → 只有 4 个
C1:  Ca(t0..t3) + Cb(t4..t7)              → 8 个 token 重叠
C2:  Ca(t4..t7) + Cb(t8..t11)             → 8 个
```

**最终结果：压缩率仍是 m=4（每 4 个 token 产一个条目），但每个条目的感受野是 2m=8 token。** 这个 trick 跨 decode step 还需要额外维护——上次 forward 的最后一个窗口的 Ca 必须保存到 `overlap_kv` 状态中，供下次调用使用。

**代码实现：**

```python
# CSACompressor.forward() L559-616

kv   = self.kv_proj(hidden_states)      # [B, S, 1024]  Ca 前 512, Cb 后 512
gate = self.gate_proj(hidden_states)    # [B, S, 1024]

# 攒窗口 (m=4) —— 和 HCA 一样的 store_compression_weights

# 重叠组装：
chunk_kv   = chunk_kv.view(B, n_windows, 4, 1024)
chunk_gate = chunk_gate.view(B, n_windows, 4, 1024) + self.position_bias

# 双倍宽度暂存区 [B, n_windows, 8, 512]
#   前 4 slot = Ca (前一窗口)
#   后 4 slot = Cb (当前窗口)
new_kv   = zeros((B, n_windows, 8, 512))
new_gate = full((B, n_windows, 8, 512), -inf)

# 当前窗口的 Cb 填到后 4 个 slot
new_kv[:,   :, 4:] = chunk_kv[..., 512:]    # Cb
new_gate[:, :, 4:] = chunk_gate[..., 512:]

# 前一窗口的 Ca 填到前 4 个 slot
if n_windows > 1:
    new_kv[:,   1:, :4] = chunk_kv[:, :-1, :, :512]   # 同批内的前窗口
if cache_layer is not None:
    prior_kv, prior_gate = cache_layer.update_overlap_state(...)
    if prior_kv is not None:
        new_kv[:,   0, :4] = prior_kv   # 跨调用的前窗口（上次存的）

# softmax 聚合: [B, n_win, 8, 512] → 沿 8 个 slot 加权求和 → [B, n_win, 512]
compressed = self.kv_norm(
    (new_kv * new_gate.softmax(dim=2)).sum(dim=2)
)
```

---

#### 2.4.3 Lightning Indexer —— 稀疏选择

CSA 在 1M 上下文下有 250K 个压缩条目——全量 dense attention 不现实。需要预筛选。

**Indexer 是一个独立的微型检索系统，有自己的压缩器（在 index_head_dim=128 维操作）、自己的 query 投影、和打分器。**

**Step 1：Indexer 自己的微型压缩器**

```python
# DeepseekV4Indexer L429-441
class DeepseekV4Indexer(nn.Module):
    def __init__(self, config):
        self.compress_rate = 4
        self.num_heads = 64         # index_n_heads: 64 个 indexer query head
        self.head_dim = 128          # index_head_dim
        self.index_topk = 512        # 每 query 选 top-512

        # 独立的 kv/gate 投影，产出 256 维 = 2×128 (Ca+Cb)
        self.kv_proj   = nn.Linear(4096, 256)
        self.gate_proj = nn.Linear(4096, 256)
        self.position_bias = nn.Parameter([4, 256])
        self.kv_norm = RMSNorm(128)

        # indexer 的 query 投影: 从 Q 的压缩潜变量 (1024 维) 出 64×128
        self.q_b_proj = nn.Linear(1024, 64 * 128)

        # 打分器
        self.scorer = DeepseekV4IndexerScorer(config)
```

压缩过程完全复用 CSA 的 Ca/Cb 重叠逻辑，但维度是 128（不是 512）。**Q 来自 attention 主干的 `q_residual`（q_a_proj 的 1024 维输出）——Indexer 的 Q 和核心 attention 的 Q 共享同一个语义源头，但上面的投影独立。**

**Step 2：打分配方 `Σ_h w_h · ReLU(q_h · K^IComp_s)`**

```python
# DeepseekV4IndexerScorer.forward() L391-395

# K^IComp: [B, T, 128]  — Indexer 版压缩 key
# q:       [B, 64, S, 128] — Indexer 版 query

# 子步骤 A: 每个 head 对每个压缩块做点积
scores = q.float() @ K^IComp.T.float()      # [B, 64, S, T]

# 子步骤 B: ReLU — 负相关直接丢弃
scores = F.relu(scores) * (128 ** -0.5)      # [B, 64, S, T]

# 子步骤 C: 每个 head 的动态权重（从 hidden states 学习）
weights = self.weights_proj(hidden_states) * (64 ** -0.5)  # [B, S, 64]

# 子步骤 D: 加权合并
return (scores * weights.unsqueeze(-1)).sum(dim=2)  # [B, S, T]
```

论文 eq 15-16 的形式：

```
I_{t,s} = Σ_{h=1}^{64} w_{t,h} × ReLU(q_{t,h} · K^{IComp}_s)
```

**Q 包含当前 token 的信息，K^IComp 不含当前 token（当前窗口还没闭合）。Indexer 做的事情就是：用当前 token 的语义特征去检索那些已经闭合的历史压缩块。**

**Step 3：因果过滤 + Top-k**

```python
# 因果: query t 只能看到已闭合的压缩块
causal_threshold = (position_ids + 1) // 4
future_mask = entry_indices >= causal_threshold
index_scores.masked_fill_(future_mask, -inf)

# top-k 选 512 个最高分
top_k_indices = index_scores.topk(512).indices  # [B, S, 512]

# 特殊情况: 有效块 < 512 个 → -1 sentinel
invalid = top_k_indices >= causal_threshold
return torch.where(invalid, -1, top_k_indices)
```

---

#### 2.4.4 block_bias —— 从索引到 mask

Indexer 的索引通过 `scatter_` 操作转化为 attention mask：

```python
# L629-638
top_k_indices = self.indexer(...)  # [B, S, 512]

valid = top_k_indices >= 0
safe_indices = torch.where(valid, top_k_indices, compressed_len)  # -1→额外列

# 初始化为全 -inf
block_bias = new_full((B, 1, S, compressed_len + 1), -inf)

# scatter_: 沿着 dim=-1，在 safe_indices 指定位置写入 0.0
block_bias.scatter_(-1, safe_indices.unsqueeze(1), 0.0)

# 返回前 T 列
return compressed_kv, block_bias[..., :compressed_len]
```

**`block_bias` 每行只有 512 个位置是 0（可见），其余全部 -inf（不可见）。** 拼接进入 attention_mask 后，只有被 indexer 选中的压缩条目参与实际的注意力计算。`softmax(-inf) = 0`，所以未选中的 249488 个块对 attention 输出的贡献为 0，等价于不存在。

#### 2.4.5 CSA 层完整 KV 尺寸与 prefill/decode 行为

```
query 可见的 KV 条目:
  滑动窗口 (128 个原始 token) + 选中压缩块 (512 个 Indexer 选择)
  = 640 个条目

未选中的 249488 个压缩块: 被 -inf 屏蔽，不进注意力计算

decode 每步计算量:
  Q(64,512) @ K(640,512).T → 64 × 512 × 640 ≈ 21M FLOPs
```

**prefill/decode 逻辑和 HCA 一致——都是 `cache_layer is None` 走一次性分支、否则走累积分支。** 区别在于 CSA 有 overlap 状态的跨调用传递。

---

### 2.5 Partial RoPE + 输出反转

**代码位置**：`deepseek_v4/modular_deepseek_v4.py:46-131` & `L648-766`

**关键数值**：`partial_rotary_factor = 64/512 = 1/8`，每个 head 只有最后 64 维加 RoPE，前 448 维不做任何旋转。

代码实现：

```python
# L58-63
def apply_rotary_pos_emb(x, cos, sin):
    # cos/sin shape: [..., rope_head_dim//2] = [..., 32]
    # interleaved 格式: 每对 (x0,x1) 共用一个频率
    cos = cos.repeat_interleave(2, dim=-1)    # [32] → [64]
    sin = sin.repeat_interleave(2, dim=-1)

    rope_dim = 64
    nope = x[..., :-64]       # [..., 448]  不变
    rope = x[..., -64:]       # [..., 64]   要旋转

    # 旋转公式: x*cos + rotate_half(x)*sin
    rotated = rope.float() * cos + rotate_half(rope).float() * sin

    return cat([nope, rotated], dim=-1)
```

#### 2.5.1 RoPE 的核心：用旋转获得相对位置

RoPE 对 Q 和 K 施加位置 p 的旋转 R(θp)：

```
Q' = R(θ·p_q) · q
K' = R(θ·p_k) · k

Q' · K' = q · R(θ·p_k - θ·p_q) · k
        = q · R(θ·Δ) · k         ← 只依赖 Δ = p_k - p_q
```

**旋转矩阵的正交性保证了这一步是恒等式，不是近似。**

多维 RoPE 把 d 维空间拆为 d/2 个独立二维平面，每个有独立的频率 θ_i。不同频率管不同距离尺度：

```
θ₀=1.0 (最低频):  cos(1.0×Δ) → Δ=3 时负相关 → 管短距离
θ₈=0.1:           cos(0.1×Δ) → Δ=31 时负相关 → 管中距离
θ₃₁=7×10⁻⁶:       cos(0.000007×Δ) → Δ=1M 时 cos≈0.7 → 管极远距离
```

多频率组合在一起，每个 Δ 在 32 个频率上产生独特的 cos 衰减模式——就像指纹一样唯一。

**为什么 cos 和 sin 只存 `rope_head_dim // 2 = 32` 个？** 因为 interleaved 格式下每对 (x₀,x₁) 为一个二维平面，共用一个频率。64/2 = 32 个频率。

#### 2.5.2 为什么只加 64 维 RoPE

以 Δ=1（相邻 token）为例，不同频率的 cos(θ_i×1)：

```
V4 64 维 (32 个频率):
  i=0:  cos(1.0) = 0.540      强区分
  i=7:  cos(0.33) = 0.946     中等区分
  i=15: cos(0.1) = 0.995      弱区分，仍有用
  i=23: cos(0.03) = 0.9996    接近满，帮助极小
  i=31: cos(0.008) = 0.99997  几乎无效

如果 512 维 (256 个频率):
  i=128: cos(0.01) = 0.99995  完全无效
  i=255: cos(0.0001) ≈ 1.0    恒等变换
```

**θ_i 按指数衰减。i 越大，旋转角越小 → cos(θ_i×Δ) 越接近 1.0 → 对位置的分辨没有任何贡献。** 64 维已经覆盖了有用频率的绝大部分，512 维的后 448 维都是白算。

#### 2.5.3 输出反转：消除 V 上的绝对位置

由于 K=V，V 在 L715 施加了 RoPE——V 的所有 512 维中最后 64 维也带上了绝对位置信息。attention 输出是对 V 的加权求和，自然含了绝对位置污染。

L761 的反转消除它：

```python
# L761
attn_output = apply_rotary_pos_emb(attn_output.transpose(1,2), cos, -sin)
#                                                               ↑ sin 取负 = 反向旋转

# 数学上: o' = Σ_j a_j · v_j · R(pos_j) × R(-pos_q)
#            = Σ_j a_j · v_j · R(pos_j - pos_q)    ← 相对化了！
```

**为什么这一步对 K=V 至关重要？** 如果 K≠V（标准 MHA），V 不带 RoPE，attention 输出天然是位置无关的。V4 的 K=V 使 V 被迫带上了 RoPE——不反转的话，位置 50000 的 token 和位置 100 的 token 即使语义完全相同，attention 输出也不同。反转确保输出只取决于"距 query 有多远"，而非绝对位置。

**反转只影响最后 64 维（rope 部分），前 448 维（nope 部分）完全不受干扰。**

#### 2.5.4 YaRN 外推

**问题**：模型训练时上下文 4096，推理时要 1M。RoPE 的每个频率适应了短上下文，长上下文下低频（管远程距离的频率）无法区分 100K vs 1M 的位置——cos 值太接近。

**YaRN 的做法**：把频率分成三段，每段不同策略处理：

```
高频段 → 不变（保持短距离敏感度）
中频段 → 频率除以 factor（拉伸到新上下文长度）
低频段 → 频率除以 factor（拉伸到新上下文长度）

V4 的 CSA/HCA 层: θ=160000, YaRN factor=16
  最低频 θ₃₁ = 160000^(-62/64) ≈ 9.1×10⁻⁶
  YaRN 后: θ₃₁' = 9.1×10⁻⁶ / 16 ≈ 5.7×10⁻⁷
  绕一圈: 2π / 5.7×10⁻⁷ ≈ 11,000,000 步 → 1M 轻松驾驭
```

**V4 使用了两种 RoPE 配置：**

| 层类型 | θ | YaRN | 理由 |
|--------|-----|------|------|
| 滑动层 (前 2 层) | 10000 | 无 | window=128，永远不超过训练长度 |
| CSA/HCA 层 | 160000 | factor=16 | 需要支持 1M 上下文 |

**θ=160000 和 YaRN 是两个独立的旋钮。** θ=160000 提供了基础余量（有效到 690K），YaRN 在此基础上进一步拉伸中低频（扩展到 11M），同时保持高频不动以保证短距离的区分度。

---

### 2.6 Attention Sink

**代码位置**：`deepseek_v4/modular_deepseek_v4.py:689` & `modeling_deepseek_v4.py:733-741`

**标准 attention 的隐含约束**：softmax 后权重和 = 1。即使当前 query 和所有历史 token 都不相关，也必须把 1 分配给某些 token → 噪音注入。

**Attention Sink 增加一个"全局废纸桶"列**，让多余的注意力被 sink 吸走：

```python
# L689: 每 head 一个可学习标量，初始化为 0
self.sinks = nn.Parameter(torch.empty(self.num_heads))  # [64]

# modeling_deepseek_v4.py L733-741
sinks = module.sinks.reshape(1, -1, 1, 1).expand(B, -1, S, -1)
# [64] → [B, 64, S, 1]

combined_logits = torch.cat([attn_weights, sinks], dim=-1)
# [B, 64, S, kv_len] + [B, 64, S, 1] → [B, 64, S, kv_len+1]

probs = softmax(combined_logits)  # 分母 = Σ exp(score_i) + exp(sink)
scores = probs[..., :-1]          # 扔掉 sink 列，只保留原始 KV 的权重
# 每行权重和 < 1，差额被 sink 吸收
```

**数值例子：**

```python
scores = [0.1, 0.15, 0.05, 0.08, 0.12]
sink = 2.0

combined = [0.1, 0.15, 0.05, 0.08, 0.12, 2.0]
exp_combined = [1.11, 1.16, 1.05, 1.08, 1.13, 7.39]
softmax = [0.087, 0.091, 0.083, 0.085, 0.089, 0.582]

scores = [0.087, 0.091, 0.083, 0.085, 0.089]  和=0.435
# 只有 43.5% 的注意力给了真实 token，其余被 sink 吸收
```

**sink 越大，head "越沉默"。** sink→+∞ 时所有真实 token 权重→0（这个 head 关闭）。sink→-∞ 时退化为标准 attention。每个 head 通过反向传播学习自己的 sink 值。

**和输出反转不冲突。** Sink 在 softmax 层面工作（控制权重分配），反转在 V 的值域层面工作（控制位置编码）——管线上下游完全独立。

---

### 2.7 Grouped Output Projection

**代码位置**：`deepseek_v4/modular_deepseek_v4.py:266-295` & `L685-688`

#### 2.7.1 问题

V4 Flash attention 的输出 = 64 heads × 512 dim = 32,768 维。直接投影回 hidden_size(4096)：

```python
# 直接投影:
output = nn.Linear(32768, 4096)(attn_output)
# 参数量: 32768 × 4096 = 134M
# 每 token 计算量: 268M FLOPs —— 比整个 MLP 层还贵！
```

#### 2.7.2 分组低秩方案

把 64 个 head 分成 g=8 组，每组 8 个 head（4096 维），各组独立投影到低秩瓶颈（1024 维），最后 8 组拼接再全局投影回 4096：

```python
# L685-688
self.o_a_proj = DeepseekV4GroupedLinear(
    in_features_per_group=4096,  # 每组输入: 8 heads × 512 = 4096
    out_features=1024,           # 每组输出: o_lora_rank = 1024
    n_groups=8                   # g=8
)
self.o_b_proj = nn.Linear(8 × 1024, 4096)  # 8192 → 4096

# L763-765 forward
grouped = attn_output.reshape(B, S, 8, 4096)   # 分成 8 组
grouped = self.o_a_proj(grouped)                # 组内独立投影: [B, S, 8, 1024]
grouped = grouped.flatten(2)                    # [B, S, 8192]
output = self.o_b_proj(grouped)                  # [B, S, 4096]
```

**GroupedLinear 实现为 batched matmul——等效于 8 个并行的 Linear(4096→1024)：**

```python
# L289-295
def forward(self, x):  # x: [B, S, 8, 4096]
    x = x.reshape(-1, 8, 4096).transpose(0, 1)  # [8, B×S, 4096]
    w = self.weight.view(8, -1, 4096).transpose(1, 2)  # [8, 4096, 1024]
    y = torch.bmm(x, w)  # 每组独立乘
    y = y.transpose(0, 1).reshape(B, S, 8, 1024)
    return y
```

#### 2.7.3 数字对比

| | 直接投影 | 分组投影 |
|---|---|---|
| 参数量 | 32768×4096 = 134M | 8×4096×1024 + 8192×4096 = **67M** |
| 每 token FLOPs | 268M | **134M** |
| 节省 | — | **50%** |

组内 8 个 head 的语义先独立压缩，最后全局融合——不需要跨组交互的第一时间就做全连接，等到 `o_b_proj` 再做全局混合。

对比 V3：输出维度只有 128×128=16,384，单层 Linear(16384→7168) 约 117M 参数，不需要分组。**V4 的 head_dim=512 把输出炸到了 V3 的两倍宽，分组降维是不得不做的优化。**

---

## 第三篇：V4 连接与聚合

### 3.1 mHC：Manifold-Constrained Hyper-Connections

**代码位置**：`deepseek_v4/modular_deepseek_v4.py:769-845`

#### 3.1.1 标准残差连接的局限

标准 Transformer 每层的残差 `x = x + F(x)` 在**梯度反传**上是可靠的（恒等映射 I 保证梯度不消失），但**信息容量**上只有单条流——所有语义都在一条 4096 维管道里跑。F 的输出无条件全量加回去，没有选择性。

#### 3.1.2 HC 的核心思想

HC（Hyper-Connections）把残差流从 1 条扩到 4 条并行：

```
标准残差:     x [B, S, 4096]

HC 残差:      X [B, S, 4, 4096]
              ↑ 4 条独立的残差流，每条都是完整的 4096 维
```

更新规则（论文 eq 1）：

```
X_{l+1} = B_l · X_l + C_l · F_l(A_l · X_l)

A_l: [B, S, H]      4 条流塌缩 → 1 条，送入子层 (pre)
F_l: Attention/FFN   子层计算
C_l: [B, S, H]      子层输出 → 注入回每条流 (post)
B_l: [B, S, H, H]   旧流之间的混合矩阵 (comb)
```

HC 把残差的无条件全量传递拆成三个可控旋钮：

- **pre** — "F 应该从 4 条流中各取多少信息？" 模型可以只把关键流送入 attention，其他流保持静默
- **post** — "F 的结果应该注入回每条流多少？" 一条流可以高速更新，另一条可以保持不变
- **comb** — "不经过 F，流之间如何共享信息？" 流 A 可以自己不跑 attention，通过 comb 从流 B 搬运处理过的信息

#### 3.1.3 mHC 的 "m"：B 被约束到双随机流形

标准 HC 的 B 矩阵无约束——某元素 >1 时，深层叠会导致信息指数增长，出现 loss spike。mHC 增加约束：

```
B ∈ M = { M ∈ R^{4×4} | M·1=1, 1^T·M=1^T, M_{ij} ≥ 0 }

即：所有元素非负，每行和=1，每列和=1
→ ||B||₂ ≤ 1 → 信号传播永不膨胀
```

通过 Sinkhorn-Knopp 算法强制执行此约束（详见 §3.2）。

#### 3.1.4 代码逐行

```python
class DeepseekV4HyperConnection(nn.Module):
    def __init__(self, config):
        self.hc_mult = 4
        self.hc_sinkhorn_iters = 20
        self.input_norm = UnweightedRMSNorm()
        self.fn = nn.Parameter(torch.empty(24, 16384))  # (2+H)×H = 6×4 = 24
        self.base = nn.Parameter(torch.empty(24))
        self.scale = nn.Parameter(torch.empty(3))

    def forward(self, hidden_streams):
        # hidden_streams: [B, S, 4, 4096]

        # ① 展平 4 条流 → 16384 维 → RMSNorm → Linear 映射 → 拆三段
        flat = self.input_norm(hidden_streams.flatten(2).float())
        pre_w, post_w, comb_w = F.linear(flat, self.fn).split([4, 4, 16], dim=-1)
        # pre_w: [B,S,4]  post_w: [B,S,4]  comb_w: [B,S,16]

        # ② pre: sigmoid → 塌缩权重，每条流一个标量
        pre = sigmoid(pre_w * scale + base) + eps  # [B, S, 4]

        # ③ post: 2×sigmoid → 注入权重，范围 [0, 2]
        post = 2 * sigmoid(post_w * scale + base)  # [B, S, 4]

        # ④ comb: softmax → Sinkhorn → 双随机 4×4 混合矩阵
        comb_logits = comb_w.view(B, S, 4, 4) * scale + base.view(4, 4)
        comb = softmax(comb_logits, dim=-1) + eps       # 行=1
        comb = comb / comb.sum(dim=-2) + eps             # 列归一
        for _ in range(19):                              # 交替 19 轮
            comb = comb / comb.sum(dim=-1) + eps         # 行归一
            comb = comb / comb.sum(dim=-2) + eps         # 列归一
        # → 收敛到双随机: 行和=1, 列和=1, 所有元素非负

        # ⑤ 用 pre 把 4 条流塌缩为 1 条
        collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2)
        # [B,S,4096] — 送入子层

        return post, comb, collapsed
```

#### 3.1.5 在 DecoderLayer 中的使用

每层有**两个** HyperConnection 实例——一个管 attention、一个管 FFN：

```python
# Attention 子层:
post, comb, collapsed = self.attn_hc(hidden_states)
attn_output = self.self_attn(layernorm(collapsed))
hidden_states = post * attn_output + comb.T @ hidden_states
#               ↑ 子层输出注入每条流     ↑ 4 条流间混合

# FFN 子层:
post, comb, collapsed = self.ffn_hc(hidden_states)
mlp_output = self.mlp(layernorm(collapsed))
hidden_states = post * mlp_output + comb.T @ hidden_states
```

**4 条流在入口（embedding）初始化为完全相同，经过 43 层 mHC 的差异化 pre/post/comb 控制，自然分化出不同的语义分工。**

---

### 3.2 Sinkhorn-Knopp 算法

**代码位置**：`deepseek_v4/modular_deepseek_v4.py:835-840`

#### 3.2.1 问题

任意正矩阵 M 如何变成双随机矩阵（每行和=1，每列和=1，所有元素非负）？

#### 3.2.2 算法

**交替行列归一化，必然收敛到双随机。**

```python
# 从 softmax 后的正矩阵开始（已行归一化）
comb = softmax(comb_logits) + eps                # 行=1, 正 ✓

# 第一轮
comb = comb / comb.sum(dim=-2) + eps             # 列归一 → 列=1✓, 行≈1
for _ in range(19):
    comb = comb / comb.sum(dim=-1) + eps         # 行归一
    comb = comb / comb.sum(dim=-2) + eps         # 列归一
# 20 轮后: 行=1✓, 列=1✓, 正✓ → 双随机
```

示例（2×2 矩阵 6 轮收敛）：

```
初始: [[2.0, 1.0], [0.5, 1.0]]
轮 1: [[0.741, 0.259], [0.294, 0.706]]    行✓列✗
轮 3: [[0.725, 0.275], [0.277, 0.723]]    行≈列
轮 6: [[0.723, 0.277], [0.277, 0.723]]    行✓列✓ → 双随机
```

**Sinkhorn-Knopp 定理（1967）** 保证此过程必然收敛。

#### 3.2.3 为什么对 mHC 至关重要

comb 的 4×4 矩阵在 43 层间不断相乘。如果不约束为双随机（||B||₂ ≤ 1），某层 B 值大于 1 → 43 层累计放大 → 数值爆炸。双随机保证重分配信号不增不减。

---

### 3.3 HyperHead

**代码位置**：`deepseek_v4/modular_deepseek_v4.py:848-864`

#### 3.3.1 功能

43 层 decoder 跑完后，hidden_states 仍是 `[B, S, 4, 4096]`。需要塌缩回 `[B, S, 4096]` 才能送入 lm_head。

HyperHead = 砍掉 post 和 comb 的 HyperConnection——只保留 pre 的 4 条流加权塌缩功能：

```python
class DeepseekV4HyperHead(nn.Module):
    def __init__(self, config):
        self.hc_fn = nn.Parameter(torch.empty(4, 16384))
        self.hc_base = nn.Parameter(torch.empty(4))
        self.hc_scale = nn.Parameter(torch.empty(1))

    def forward(self, x):
        # x: [B, S, 4, 4096]
        flat = self.input_norm(x.flatten(2).float())     # [B,S,16384] → RMSNorm
        mixes = F.linear(flat, self.hc_fn.float())       # [B,S,4]
        pre = sigmoid(mixes * scale + base) + eps         # [B,S,4]
        # 4 条流加权求和 → 1 条
        return (pre.unsqueeze(-1) * x).sum(dim=2)         # [B,S,4096]
```

在 Model.forward 中：

```python
hidden_states = inputs_embeds.unsqueeze(2).expand(-1,-1,4,-1)  # [B,S,4,4096] 入口扩4流
for layer in self.layers:                                       # 43 层 mHC 保持 4 流
    hidden_states = layer(hidden_states)
hidden_states = self.norm(self.hc_head(hidden_states))          # 出口塌为 1 流
```

**只有入口 (embedding 拷贝) 和出口 (HyperHead) 是多/单流转换——中间 43 层始终维护 4 条流。**

---

## 第四篇：V4 的 MoE 路由

### 4.1 Hash-MoE（前 3 层）

**代码位置**：`deepseek_v4/modular_deepseek_v4.py:930-954`

#### 4.1.1 功能

V4 Flash 的前 3 层（`mlp_layer_types[0:3] = "hash_moe"`），路由选择不靠学习——靠预训练的 `token_id → expert_id` 查表：

```python
class DeepseekV4HashRouter(MixtralTopKRouter):
    def __init__(self, config):
        self.score_fn = sqrt_softplus      # 和 Top-K 一样的打分
        # 核心: 冻结的 token→expert 映射表
        self.register_buffer("tid2eid",
            torch.zeros(vocab_size=129280, top_k=6, dtype=torch.long))

    def forward(self, hidden_states, input_ids):
        scores = sqrt_softplus(hidden @ W_gate)       # [S, 256] 打分仍学
        indices = self.tid2eid[input_ids].long()       # [S, 6]  选谁不学
        weights = scores.gather(1, indices)            # 用原始 scores 算权重
        weights = weights / weights.sum() * 2.5
        return logits, weights, indices
```

**Hash-MoE 的两部分独立运作：** `tid2eid` 决定选谁（冻结），`W_gate` 决定权重（可训练）。

#### 4.1.2 为什么需要

对比 V3 前 3 层用 Dense FFN（所有 token 同一个变换），V4 换成 Hash-MoE（256 个专家中对不同 token 做不同变换）。路由冻结不动 → 训练稳定，容量大幅提升。

---

### 4.2 Top-K MoE（后续层）

**代码位置**：`deepseek_v4/modular_deepseek_v4.py:913-927`

```python
class DeepseekV4TopKRouter(MixtralTopKRouter):
    def forward(self, hidden_states):
        logits = F.linear(flat, self.weight)                     # [S, 256]
        scores = sqrt(softplus(logits))                          # ← 新打分函数
        indices = topk(scores + e_score_correction_bias, 6)      # ← k=6, 无分组
        weights = scores.gather(1, indices)                      # 原始分不含 bias
        weights = weights / weights.sum() * 2.5
        return logits, weights, indices
```

| | Mixtral (标准 MoE) | V3 | V4 |
|---|---|---|---|
| 打分 | softmax | sigmoid | sqrt(softplus) |
| 分组约束 | 无 | 8 组→4 组→top-8 | **无** |
| 激活数 k | 2 | 8 | **6** |
| bias | 无 | e_score_correction_bias ✓ | ✓ |
| SwiGLU Clamping | 无 | 无 | **gate≤10, up∈[-10,10]** |

**V4 的简化逻辑：** sqrt(softplus) 替代 sigmoid 解决梯度饱和 + 上界自限；去掉分组约束因为 wave-based EP 掩盖了通信瓶颈；k=8→6 减少 25% 通信量；Clamping 消灭 outlier 防 loss spike。

---

### 4.3 SwiGLU Clamping

**代码位置**：`deepseek_v4/modular_deepseek_v4.py:867-875` & `L879-893`

共享专家和路由专家都限定 SwiGLU 的各维度：

```python
# 共享专家
def forward(self, x):
    gate = self.gate_proj(x).clamp(max=10)            # gate 上界 10
    up   = self.up_proj(x).clamp(min=-10, max=10)     # up 范围 [-10, 10]
    return self.down_proj(silu(gate) * up)

# 路由专家
def _apply_gate(self, gate_up):
    gate, up = gate_up.chunk(2, dim=-1)
    gate = gate.clamp(max=10)
    up   = up.clamp(min=-10, max=10)
    return silu(gate) * up
```

`silu(gate) × up` 的最大值 = silu(10) × 10 ≈ 100。无论原始投影多偏，送入下一层的值不超过 100。

**为什么 gate 只 clamp 上限，而 up 需要双向 clamp？**

gate 经过 silu 后，负的 gate 值会让 silu(gate)≈0（因为 silu(x) = x·σ(x)，负 x 会让 sigmoid 压低），自然抑制了 gate 的负面影响。所以 gate 的负方向本身是安全的，只需要限定正方向的上界。

up 是值信号——up 直接和 silu(gate) 相乘。如果 up 取了很大的负值，silu(gate)×up 会变成很大的负数，经过 down_proj 后放大到后续层。up 的双向 clamp 就是为了堵住这个方向的漏洞。

**为什么是 10？** 10 是经验值，来自 OpenAI gpt-oss 和 V4 团队的训练实验。论文 §4.2.3 原文：

> "Throughout the training of both DeepSeek-V4-Flash and DeepSeek-V4-Pro, we clamped the linear component of SwiGLU to the range of [-10, 10], while capping the upper bound of the gate component at 10."

数值消融表明 10 对性能无损且有效。太小（如 1.0）会限制模型容量，太大（如 100）接近没有 clamp，在 Muon 优化器下仍可能产生 outlier 导致 loss spike。

---

### 4.4 SparseMoeBlock 整体

**代码位置**：`deepseek_v4/modular_deepseek_v4.py:957-974`

SparseMoeBlock 是每层 FFN 子层的实际执行体——把路由选择、专家计算和共享专家融合打包在一个 Module 内。调用方 (DecoderLayer) 只需传 hidden states，内部自动根据层号选择 Hash 或 Top-K 路由。

```python
class DeepseekV4SparseMoeBlock(nn.Module):
    def __init__(self, config, layer_idx):
        self.is_hash = config.mlp_layer_types[layer_idx] == "hash_moe"
        self.gate = DeepseekV4HashRouter(config) if self.is_hash \
                    else DeepseekV4TopKRouter(config)
        self.experts = DeepseekV4Experts(config)             # 256 个专家
        self.shared_experts = DeepseekV4MLP(config)           # 1 个共享专家

    def forward(self, hidden_states, input_ids=None):
        residual = hidden_states
        flat = hidden_states.view(-1, hidden_dim)
        if self.is_hash:
            _, weights, indices = self.gate(hidden_states, input_ids)
        else:
            _, weights, indices = self.gate(hidden_states)
        routed = self.experts(flat, indices, weights)         # 路由专家加权
        return routed + self.shared_experts(residual)          # + 共享专家
```

**SparseMoeBlock 是一个薄抽象——把 Hash/Top-K 路由切换、专家加权计算、共享专家融合封装在一个 Module 内。DecoderLayer 只需调 `self.mlp(collapsed)`，不需感知层号和路由类型。**

**V4 所有 43 层都用 SparseMoeBlock——没有 V3 前 3 层的 Dense FFN。前 3 层走 Hash-MoE（路由冻结+容量提升），后续层走 Top-K MoE（自由路由）。**

---

## 第五篇：端到端串联

### 5.1 DeepseekV4DecoderLayer

**代码位置**：`deepseek_v4/modular_deepseek_v4.py:977-1025`

单层 forward 的执行顺序：

1. **mHC 塌缩 4 条流 → 1 条混合流**：`attn_hc.forward()` 通过 pre 权重把 4 条残差流混合成 1 条 collapsed，同时对 4 条流的残差做 comb 混合（不经过 attention）
2. **LayerNorm + Attention**：collapsed 过 `input_layernorm` → `self_attn`（CSA/HCA/sliding 自动分发）
3. **mHC 注入 + 残差融合**：attention 输出通过 post 注入回 4 条流，comb 做流间混合，两项相加 = 新的残差状态
4. **重复 1-3，但走 FFN**：`ffn_hc` 再做一次塌缩 4→1 → LayerNorm → `self.mlp`（Hash/Top-K 自动分发）→ post 注入 + comb 混合



```python
class DeepseekV4DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        self.self_attn = DeepseekV4Attention(config, layer_idx)   # CSA/HCA/sliding
        self.mlp       = DeepseekV4SparseMoeBlock(config, layer_idx)  # Hash/TopK
        self.input_layernorm = RMSNorm(4096)
        self.post_attention_layernorm = RMSNorm(4096)
        self.attn_hc = DeepseekV4HyperConnection(config)    # attention 的 mHC
        self.ffn_hc  = DeepseekV4HyperConnection(config)    # FFN 的 mHC

    def forward(self, hidden_states, **kwargs):
        # hidden_states: [B, S, 4, 4096] — mHC 4 条流

        # Attention 子层
        post, comb, collapsed = self.attn_hc(hidden_states)
        attn_output = self.self_attn(self.input_layernorm(collapsed), **kwargs)
        hidden_states = post * attn_output + comb.T @ hidden_states

        # FFN 子层
        post, comb, collapsed = self.ffn_hc(hidden_states)
        mlp_output = self.mlp(self.post_attention_layernorm(collapsed), input_ids=...)
        return post * mlp_output + comb.T @ hidden_states
```

**相对于标准 Transformer，V4 的 DecoderLayer 加了两个维度：mHC 的 4 流管理 + 混合注意力类型自动分发。IO 形状一致对调用方透明。**

### 5.2 DeepseekV4Model

**代码位置**：`deepseek_v4/modular_deepseek_v4.py:1132-1197`

```python
class DeepseekV4Model:
    def __init__(self, config):
        self.embed_tokens = Embedding(129280, 4096)
        self.layers = [DecoderLayer × 43]
        self.rotary_emb = RotaryEmbedding()  # main(θ=10000) + compress(θ=160000+YaRN)
        self.hc_head = HyperHead()           # 4 流 → 1 流
        self.norm = RMSNorm(4096)

    def forward(self, input_ids, ...):
        # ① Embedding → 拷贝 4 份
        inputs_embeds = self.embed_tokens(input_ids)           # [B, S, 4096]
        hidden_states = inputs_embeds.unsqueeze(2).expand(-1, -1, 4, -1)

        # ② 预计算两种 RoPE
        position_embeddings = {
            "main":     RoPE(θ=10000),
            "compress": RoPE(θ=160000, YaRN factor=16),
        }

        # ③ 43 层循环
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings, ...)

        # ④ HyperHead + RMSNorm
        return self.norm(self.hc_head(hidden_states))          # [B, S, 4096]
```

**入口扩展 4 流、43 层保持 4 流、出口塌缩→具体化→预测。**

### 5.3 DeepseekV4ForCausalLM

**代码位置**：`deepseek_v4/modular_deepseek_v4.py:1200`

```python
class DeepseekV4ForCausalLM(MixtralForCausalLM):
    pass
```

完全继承 MixtralForCausalLM——标准的 `model.forward → lm_head → cross_entropy(labels)`。`lm_head.weight` 和 `embed_tokens.weight` 共享。

**推理时只用主模型，MTP 权重被 `_keys_to_ignore_on_load_unexpected` 跳过。**

### 5.4 KV Cache 结构

V4 的 `DynamicCache(config=config)` 读取 `config.layer_types`，为每一层按类型创建匹配的 cache：

| 层类型 | Cache 类 | 管理状态 |
|--------|---------|---------|
| sliding_attention | DynamicSlidingWindowLayer | 滑动窗口 KV (128 token) |
| heavily_compressed_attention | DeepseekV4HCACache | + buffer_kv/gate + compressed_kv + entry_count |
| compressed_sparse_attention | DeepseekV4CSACache | + indexer buffer/compressed + overlap_kv/gate |

自动注册——靠类的 `layer_type` 属性。`DeepseekV4HCACache.layer_type = "heavily_compressed_attention"` → DynamicCache 查找时自动实例化对应类型。

**V4 的关键约束：**

- `_is_stateful = True` — cache 不可回退，禁用 prompt lookup 等需要 cache rollback 的生成模式
- `_supports_flash_attn = False` — head_dim=512 超 FlashAttention 256 上限
- `_supports_sdpa = False` — torch SDPA 不支持 per-head learnable sink
- `_can_compile_fullgraph = False` — compressor 的 rolling-window 状态不兼容 StaticCache

---

## 附录：对照阅读索引

| 论文节 | 代码位置 |
|--------|---------|
| §2.1 MoE + Hash 路由 | `DeepseekV4SparseMoeBlock` + `DeepseekV4HashRouter` |
| §2.2 mHC（eq 1-8） | `DeepseekV4HyperConnection` |
| §2.3.1 CSA（eq 9-17） | `DeepseekV4CSACompressor` + `DeepseekV4Indexer` |
| §2.3.2 HCA（eq 20-23） | `DeepseekV4HCACompressor` |
| §2.3.3 Partial RoPE + Sink + Sliding Window | `DeepseekV4RotaryEmbedding` + `DeepseekV4Attention` |
| §2.4 Muon 优化器 | Muon 不在 modeling 代码中，在训练框架 |
| §3.1 Wave-based EP | 不在 modeling 代码中，在 CUDA kernel (MegaMoE) |
| §4.2.3 SwiGLU Clamping | `DeepseekV4MLP` + `DeepseekV4Experts._apply_gate` |
