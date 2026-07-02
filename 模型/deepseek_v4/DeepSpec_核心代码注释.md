# DSpark 核心 Trick 代码逐行注释

---

## Trick 1: Markov 头 — 给并行 drafter 加 token 间依赖

文件: `deepspec/modeling/dspark/markov_head.py`

### 1.1 网络结构

```python
class VanillaMarkov(nn.Module):
    def __init__(self, *, vocab_size: int, markov_rank: int):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.markov_rank = int(markov_rank)        # 低秩维度 r，默认 256

        # W1: 词表 → 低维隐空间
        #    [vocab_size, markov_rank]  约 V×256 个参数
        #    作用: 把任意 token 压缩成一个 256 维向量
        self.markov_w1 = nn.Embedding(self.vocab_size, self.markov_rank)

        # W2: 低维隐空间 → 词表
        #    [markov_rank, vocab_size]  约 256×V 个参数
        #    作用: 把修正向量映射回 vocab 空间，变成对 logit 的偏置
        self.markov_w2 = nn.Linear(self.markov_rank, self.vocab_size, bias=False)
```

### 1.2 训练时: Teacher Forcing（一次性并行修正整个 block）

```python
    def compute_step_bias(self, token_ids, hidden_states):
        # token_ids: [B, num_blocks, block_size]
        #  训练时这里是 ground truth 的前一个 token，不是模型自己猜的！

        del hidden_states  # VanillaMarkov 不需要 backbone hidden，只依赖上一个 token

        # Step 1: W1 查表 → [B, num_blocks, block_size, 256]
        # Step 2: W2 投影 → [B, num_blocks, block_size, vocab_size]
        #   这就是论文 Eq.(5): B(x_{k-1}, · ) = W1[x_{k-1}] · W2
        return self.project_bias(self.get_prev_embeddings(token_ids))

    def apply_block_logits(self, base_logits, *, token_ids, hidden_states):
        # 训练用: 并行 backbone 产出的 logits + Markov 偏置
        markov_bias = self.compute_step_bias(token_ids, hidden_states)
        return base_logits + markov_bias
        # ↑ 这里的 token_ids 是 ground truth，所以可以一次并行算完 7 个位置的修正
        #   因为: 位置 2 的修正用的是位置 1 的真值，位置 3 用位置 2 的真值...
        #   全部已知 → 全并行
```

### 1.3 推理时: Autoregressive Sampling（串行采样 + 串行修正）

```python
    def sample_block_tokens(self, base_logits, *, first_prev_token_ids, hidden_states, temperature=0.0):
        # base_logits: [batch, proposal_len, vocab]  并行 backbone 产出的基础 logits
        # first_prev_token_ids: [batch]  锚点 token（上一轮 target 产出的 bonus token）

        batch_size, proposal_len = base_logits.shape[:2]
        sampled_tokens = []      # 收集采样出的 token
        corrected_logits = []    # 收集修正后的完整 logits（用于给 target 做拒绝采样）
        prev_token_ids = first_prev_token_ids   # 第 1 个位置的"前一个 token" = 锚点

        for step_idx in range(proposal_len):          # 逐个位置串行处理

            # Step 1: 算出修正偏置
            #   位置 0: prev = 锚点 token        → 偏置基于锚点
            #   位置 1: prev = 位置 0 采样结果   → 偏置基于"刚才猜的那个字"
            #   ...
            step_logits = self.apply_step_logits(
                base_logits[:, step_idx, :],           # 当前位置的 base logits
                token_ids=prev_token_ids,              # 上一个位置的实际 token（自己采的！）
                hidden_states=hidden_states[:, step_idx],
            )
            corrected_logits.append(step_logits.unsqueeze(1))

            # Step 2: 从修正后的 logits 中采样出当前位置的 token
            next_token_ids = sample_tokens(step_logits.unsqueeze(1), temperature).squeeze(1)
            sampled_tokens.append(next_token_ids)

            # Step 3: 这个采样结果变成下一步的"prev_token"
            prev_token_ids = next_token_ids

        return torch.stack(sampled_tokens, dim=1), torch.cat(corrected_logits, dim=1)
```

### 1.4 为什么训练和推理代码不同？

```
训练: apply_block_logits
  token_ids = ground truth → 7 个位置并行修正，一条 GPU kernel 搞定

推理: sample_block_tokens
  token_ids = 自己采的 → 必须串行，因为位置 2 依赖位置 1 的采样结果
  （但串行部分只用了两个极小的矩阵乘法，延迟 < 总延迟的 1%）

这就是论文说的: "Parallel backbone + lightweight sequential correction"
背骨并行（快），修正串行（准），修正极轻（不慢）
```

---

## Trick 2: 置信度头 — 预测每个 token 的接受概率

文件: `deepspec/modeling/dspark/common.py`

### 2.1 网络结构

```python
class AcceptRatePredictor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        # 就是一个线性层: input_dim → 1，然后外面套 sigmoid
        # input_dim = hidden_size                    (如果不吃 Markov 特征)
        #           = hidden_size + markov_rank       (如果吃 Markov 特征)
        self.proj = nn.Linear(int(input_dim), 1)

    def forward(self, features):
        # features: [B, blocks, 7, input_dim]
        # 输出: [B, blocks, 7] — 每个位置一个标量 logit
        return self.proj(features).squeeze(-1)       # 后面接 sigmoid → 0~1 的置信度
```

### 2.2 在模型 forward 中如何调用

文件: `deepspec/modeling/dspark/qwen3/modeling.py`（model forward 的最后一段）

```python
        # ====== 置信度头预测 ======
        confidence_pred = None
        if self.confidence_head is not None:
            if self.confidence_head_with_markov:
                # 模式 A: 置信度头吃 [backbone_hidden + Markov_embedding]
                #   这样置信度头能"感知"到 Markov 头对当前 token 的修正程度

                # 先查 W1 表，把前一个 token 转成 Markov embedding
                prev_embeddings = self.markov_head.get_prev_embeddings(prev_token_ids)

                # 拼接: [hidden | markov_embedding]
                confidence_features = torch.cat(
                    [output_hidden_4d, prev_embeddings], dim=-1
                )
                # 过线性层 → 出 logit
                confidence_pred = self.confidence_head(confidence_features).float()
            else:
                # 模式 B: 只用 backbone hidden
                confidence_pred = self.confidence_head(output_hidden_4d).float()
```

---

## Trick 3: L1 损失为主 — "分布像不像"比"猜没猜对"更重要

文件: `deepspec/modeling/dspark/loss.py`

### 3.1 计算每个位置的接受概率 label

```python
def _compute_accept_rate_3d(*, outputs, aligned_target_logits):
    # 1. 把 logits 转成概率分布
    draft_probs = torch.softmax(outputs.draft_logits.float(), dim=-1)
    #   draft_logits: [B, blocks, 7, vocab] → softmax → draft_probs

    target_probs = torch.softmax(aligned_target_logits.float(), dim=-1)
    #   aligned_target_logits: target 模型在相同位置的概率分布（从训练 cache 读的）

    # 2. 算 L1 距离的逐 token 求和 ||p_d - p_t||₁
    #    (draft_probs - target_probs).abs().sum(dim=-1)
    #    每个位置得到一个标量，范围 [0, 2]

    # 3. 转成接受率 c_k* = 1 - 0.5 × L1
    #    论文 Eq.(8): 投机推理中，第 k 个 token 被接受的概率精确等于这个值
    accept_rate_3d = 1.0 - 0.5 * (draft_probs - target_probs).abs().sum(dim=-1)

    # 4. clamp 到 [0, 1] 区间（数值安全）
    return accept_rate_3d.clamp_(0.0, 1.0)
```

### 3.2 计算 L1 loss

```python
def _compute_local_l1_term(*, outputs, aligned_target_logits, loss_weight_mask):
    draft_probs = torch.softmax(outputs.draft_logits.float(), dim=-1)
    target_probs = torch.softmax(aligned_target_logits.float(), dim=-1)

    # ||p_d - p_t||₁ 逐 token 的距离
    l1_dist_per_token = (draft_probs - target_probs).abs().sum(dim=-1)

    # 乘以位置衰减权重 loss_weight_mask
    l1_loss_num = (l1_dist_per_token * loss_weight_mask).sum()
    l1_loss_den = loss_weight_mask.sum()

    return l1_loss_num, l1_loss_den
```

### 3.3 置信度 loss（BCE with soft label）

```python
        # 在 _collect_local_terms 中:
        if has_confidence:
            # label 是连续值，不是 0/1
            confidence_targets = accept_rate_3d.detach()
            # accept_rate_3d 在 0~1 之间，例如 0.73

            # F.binary_cross_entropy_with_logits 天生支持 soft label
            # 公式: -[y·log(σ(x)) + (1-y)·log(1-σ(x))]
            #       y = confidence_targets (可以是 0.73)
            #       x = confidence_pred (网络输出的 logit)
            confidence_errors = F.binary_cross_entropy_with_logits(
                outputs.confidence_pred.float(),      # logits
                confidence_targets,                   # 0~1 的连续 label
                reduction="none",
            ) * loss_weight_mask

            confidence_loss_num = confidence_errors.sum()
            confidence_loss_den = loss_weight_mask.sum()
```

### 3.4 位置衰减权重

```python
def _build_loss_weight_mask(*, eval_mask, block_size, device, loss_decay_gamma):
    # eval_mask: [B, blocks, 7] — 哪些位置有有效的监督信号
    loss_weight_mask = eval_mask.to(torch.float32)

    if loss_decay_gamma is not None and loss_decay_gamma > 0:
        # 生成 [0, 1, 2, 3, 4, 5, 6] 位置索引
        positions = torch.arange(block_size, device=device).view(1, 1, -1)

        # w_k = exp(-k / gamma)
        #   位置 0: exp(-0/4.0) = 1.000  ← 权重最高
        #   位置 3: exp(-3/4.0) = 0.472
        #   位置 6: exp(-6/4.0) = 0.223  ← 权重最低
        decay_weights = torch.exp(-positions.float() / float(loss_decay_gamma))

        # eval_mask 已经过滤了无效位置，乘上衰减后有效位置的权重自然递减
        loss_weight_mask = loss_weight_mask * decay_weights

    return loss_weight_mask
```

### 3.5 总 loss 组合

```python
# 默认权重: ce_loss_alpha=0.1, l1_loss_alpha=0.9, confidence_head_alpha=1.0
backward_loss = (
    ce_loss_alpha * ce_loss           # 0.1 × CE: 猜对字也是好的，但不是主要目标
    + l1_loss_alpha * l1_loss         # 0.9 × L1: 核心！让分布逼近 target
    + confidence_head_alpha * confidence_loss  # 1.0 × BCE: 让置信度头学会预测接受率
) * world_size                         # 多卡时缩放
```

---

## Trick 4: 置信度校准的评估代码 (STS 的"评估"部分在开源库中)

文件: `deepspec/eval/dspark/confidence_head.py`

### 4.1 推理时怎么用置信度截断

文件: `deepspec/eval/dspark/draft_ops.py`

```python
def _confident_prefix_length(confidence_logits, *, block_size, threshold):
    # 如果 threshold=0，就是不截断，全验
    if threshold <= 0.0:
        return int(block_size)

    # sigmoid 把 logit 转成 0~1 的概率
    # below_threshold: bool tensor，标记哪些位置低于阈值
    below_threshold = confidence_logits.sigmoid() < threshold

    # 找第一个低于阈值的位置 —— 因为投机解码是前缀匹配，第一个不自信的就该裁了
    if not bool(below_threshold[0].any().item()):
        return int(block_size)                            # 全过 → 全验

    return int(torch.nonzero(below_threshold[0], as_tuple=False)[0].item())
    # 返回第一个低于阈值的位置索引，裁剪点在此处
```

### 4.2 置信度校准评估（ECE / AUC / Brier）

这里的代码不训练、不做校准（开源的 STS 校准是 DeepSeek 内部的），但提供了完整的**评估校准效果**的工具链。

```python
class PerPositionConfidenceMetrics:
    def update(self, *, probs, targets):
        # probs: 置信度头预测的前缀联合概率 ∏c_i，例如 [0.94, 0.80, 0.58, 0.26]
        # targets: 实际是否被接受（1/0）

        # 按预测概率分到粗粒度 bin（20 个 bin）用于算 ECE
        coarse_idx = (probs * num_coarse_bins).long().clamp_(0, num_coarse_bins - 1)
        # 累加每个 bin 的: 样本数、预测概率和、实际接受率和

        # 按预测概率分到细粒度 bin（1000 个 bin）用于算 AUC
        fine_idx = (probs * num_fine_bins).long().clamp_(0, num_fine_bins - 1)
        # 分别记录正例和负例的直方图
```

```python
    def compute(self):
        # 对每个位置输出:
        #  ECE  = Σ bin_weight × |avg_pred - avg_target| / total_weight
        #  AUC  = 梯形法从 fine_pos / fine_neg 直方图算
        #  Brier = Σ (pred - target)² / n
        #
        # 这三个指标分别衡量: 校准准确度 / 排序能力 / 整体概率误差
```

---

## Trick 5: 训练时的完整 forward 流程（串联所有 trick）

文件: `deepspec/modeling/dspark/qwen3/modeling.py`

```python
    def forward(self, input_ids, target_hidden_states, loss_mask,
                target_last_hidden_states=None):

        # ====== Step 1: 随机采样 512 个 anchor 位置 ======
        # 从序列中随机选位置作为 block 的起点
        anchor_positions, block_keep_mask = sample_anchor_positions(
            seq_len=seq_len, loss_mask=loss_mask,
            num_anchors=self.num_anchors,         # 512
            device=device,
        )

        # ====== Step 2: 构造 block 输入 ======
        # 每个 block = [anchor_token, MASK, MASK, ..., MASK]
        # 7 个位置: 第 1 个是真实 token，后 6 个是 mask
        noise_embedding = create_noise_embed(
            self.embed_tokens, input_ids, anchor_positions, block_keep_mask,
            mask_token_id=self.mask_token_id, block_size=self.block_size,
        )

        # ====== Step 3: 构造 DSpark 特殊的注意力掩码 ======
        # 每个 block 只能看到: anchor 之前的 context + 自己 block 内的 token
        dspark_attn_mask = create_dspark_attention_mask(...)

        # ====== Step 4: Draft backbone 前向 ======
        # target_hidden_states 注入 + 5 层 Transformer
        output_hidden = self._forward_backbone(
            position_ids=full_position_ids,
            noise_embedding=noise_embedding,
            target_hidden_states=target_hidden_states,    # KV injection
            attention_mask=dspark_attn_mask,
        )
        output_hidden_4d = output_hidden.reshape(B, num_blocks, 7, -1)

        # ====== Step 5: 算 base logits ======
        draft_logits = self.compute_logits(output_hidden).reshape(B, num_blocks, 7, -1)

        # ====== Step 6: Markov 头修正 (Teacher Forcing) ======
        # 用 ground truth 的 prev_token 作为 Markov 头的输入
        if self.markov_head is not None:
            draft_logits = self.markov_head.apply_block_logits(
                draft_logits,
                token_ids=prev_token_ids,       # ← ground truth，不是采样值！
                hidden_states=output_hidden_4d,
            )

        # ====== Step 7: 置信度头预测（可选） ======
        confidence_pred = None
        if self.confidence_head is not None:
            # 拼接 [backbone_hidden | Markov_embedding]
            confidence_pred = self.confidence_head(features).float()

        # ====== Step 8: 对齐 target logits（做 L1 loss 的 label） ======
        aligned_target_logits = None
        if target_last_hidden_states is not None:
            aligned_target_hidden = ...  # 从 target cache 读取对应位置的 hidden
            aligned_target_logits = self.compute_logits(aligned_target_hidden)

        return DSparkForwardOutput(
            draft_logits=draft_logits,                    # 修正后的 logits
            target_ids=target_ids,                        # ground truth token ids
            eval_mask=eval_mask,                          # 前缀 cumprod mask
            block_keep_mask=block_keep_mask,
            confidence_pred=confidence_pred,              # 置信度 logits
            aligned_target_logits=aligned_target_logits,  # target 的 logits
        )
```

---

## 全局数据流总结

```
训练一个 batch（config: local_batch_size=1, global_batch_size=512, 梯度累积 128 步）:

target_cache  ← 离线预生成，存了每条序列的 target hidden states
     ↓
CacheDataset  ← 读取二进制缓存，返回 {input_ids, target_hidden_states, loss_mask, target_last_hidden_states}
     ↓
Qwen3DSparkModel.forward()
     ├── sample_anchor_positions()         # 随机选 512 个 anchor
     ├── create_noise_embed()              # 构造 [anchor, MASK×6]
     ├── _forward_backbone()              # 并行 backbone（5 层 Transformer + KV injection）
     ├── markov_head.apply_block_logits()  # Markov 头修正（Teacher Forcing）
     ├── confidence_head()                 # 置信度预测
     └── aligned_target_logits            # target 模型对应位置的 logits
     ↓
compute_dspark_loss()
     ├── cross_entropy × 0.1              # CE: 猜对字
     ├── L1 distance × 0.9                # L1: 分布匹配（主损失！）
     └── BCE(confidence_pred, accept_rate) × 1.0  # 置信度头的校准
     ↓
backward() → optimizer.step()

推理（开源版，不含生产调度器）:

build_dspark_proposal()
     ├── forward_dspark_draft_block()      # backbone 前向
     ├── model.sample_draft_tokens()       # Markov 头串行采样 + 修正
     ├── _predict_confidence_logits()      # 置信度头预测
     └── _confident_prefix_length()        # 静态阈值截断
         ↓
target.verify()
         ↓
accepted_tokens, bonus_token → 下一轮
```
