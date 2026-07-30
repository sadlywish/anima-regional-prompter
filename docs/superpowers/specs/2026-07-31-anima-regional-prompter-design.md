# ANIMA Regional Prompter — 设计文档

日期：2026-07-31
目标宿主：sd-webui-forge-neo
目标模型：ANIMA（`circlestone-labs/Anima`）

## 1. 目标与范围

为 Forge Neo 提供一个类似 [sd-webui-regional-prompter](https://github.com/hako-mikan/sd-webui-regional-prompter)
的分区提示词扩展，适配 ANIMA 模型。

### v1 范围

| 功能 | 状态 |
|------|------|
| Attention 模式（遮罩 cross-attention） | 做 |
| 按比例二维分区（`1,2;1,1,1` 式嵌套） | 做 |
| Base prompt（全局提示词按权重混入各区） | 做 |
| 实时布局预览 | 做 |
| PNG info 元数据写入 | 做 |
| Common prompt（公共前缀文本拼接） | 不做，留待后续 |
| 分区负面提示词 | 不做 |
| Latent couple 模式 | 不做 |
| Prompt 模式（按关键词自适应分区） | 不做 |
| 遮罩模式（用户上传遮罩图） | 不做 |

### 非目标

- 不支持 ANIMA 以外的模型。检测到非 ANIMA 模型时，扩展完全旁路（零开销），控制台警告一次。
- 不支持区域提示词内部的步数调度（`[a:b:5]`）与 AND 组合。这是「旁路注入」方案的已知代价，见 §5。

## 2. 宿主环境事实

以下事实来自阅读 Forge Neo 源码，是设计的前提。实现时若发现不符，需回到本文档修订。

### 2.1 ANIMA 是 cross-attention DiT

`backend/nn/anima.py` 中的 `Anima` 是 Cosmos-Predict2 系架构，**不是 MMDiT**。每个 `Block` 含：

- `self_attn`：`SelfCrossAttention(x_dim, None, ...)`，图像 token 之间，带 3D RoPE
- `cross_attn`：`SelfCrossAttention(x_dim, context_dim, ...)`，图像 token 作 query，文本 `crossattn_emb` 作 K/V
- `mlp`

关键性质：

1. **文本只通过 cross-attn 进入模型。** 没有 pooled embedding，没有文本参与的 adaLN 调制
   （`adaln_modulation_*` 的输入是 timestep embedding，与文本无关）。因此遮罩 cross-attn
   足以完整控制分区文本影响——这比 SDXL 干净，SDXL 的 pooled embed 是全局的、无法分区。
2. **cross-attn 的 context 不施加位置编码。** `compute_qkv` 中
   `if self.is_SelfAttn and rope_emb is not None` 才应用 RoPE。位置信息已在 `LLMAdapter` 中烘焙。
   因此多段提示词沿 token 维拼接、或替换整个 context，都不会破坏位置语义。
3. **`output_proj` 是 `nn.Linear(..., bias=False)`，`output_dropout` 在推理期是 `Identity`。**
   因此 `Σ mask_i · output_proj(a_i) == output_proj(Σ mask_i · a_i)`，投影可以只做一次。
4. **`SelfCrossAttention.forward` 直接接收 `transformer_options`**（由 `Block.forward` 透传），
   这是天然的 hook 面。

### 2.2 token 布局

- `PatchEmbed` 把 `(B, C, T, H_lat, W_lat)` 变成 `(B, T', H', W', D)`，
  其中 `T' = T / patch_temporal`，`H' = H_lat / patch_spatial`，`W' = W_lat / patch_spatial`。
- `Anima.forward` 先调 `pad_to_patch_size`，所以实际是向上取整。
- 进入 attention 前 `rearrange(x, "b t h w d -> b (t h w) d")`，**token 展平顺序是 `(t h w)`**。
- VAE 是 Wan VAE（`VAE(model=..., is_wan=True)`），空间下采样 8 倍。
- `patch_spatial` / `patch_temporal` 从 checkpoint 配置读取，**实现中必须运行时从
  `diffusion_model` 对象读，不得硬编码**。

### 2.3 文本编码

`backend/text_processing/anima_engine.py` 的 `AnimaTextProcessingEngine`：

- Qwen3-0.6B + T5 双 tokenizer，经 `LLMAdapter` 输出 `(1, L, 1024)`，`L ≥ 512`（不足补零）。
- `tokenize_line` 里有 `assert len(chunks) == 1`，**不支持 `BREAK`**。分段必须由本扩展自己完成。
- 各段长度可能不同（长提示词超过 512）。

`Anima.get_learned_conditioning(prompts: list[str])` 返回 list，每个元素 `(1, L_i, 1024)`。

### 2.4 采样期可用信息

`backend/sampling/sampling_function.py` 的 `calc_cond_uncond_batch` 会往
`transformer_options` 里写：

| 键 | 含义 |
|----|------|
| `cond_or_uncond` | 每个 chunk 是 cond(0) 还是 uncond(1) |
| `cond_mark` | 展开到每个 batch 行的 0/1 张量 |
| `cond_indices` / `uncond_indices` | 行索引列表 |
| `sigmas` | 当前 timestep |

`model_options["transformer_options"]` 会被 `.copy()` 进来，所以扩展通过
`UnetPatcher.set_transformer_option(k, v)` 写入的内容能到达 attention 层。

`set_model_unet_function_wrapper` 设置的 wrapper 收到
`{"input", "timestep", "c", "cond_or_uncond"}`，其中 `c["transformer_options"]` 是可变 dict。

### 2.5 conditioning 张量形状

`reconstruct_multicond_batch` → `stack_conds` 会把 list of `(1, L, 1024)` 堆成
`(B, 1, L, 1024)`。这个多出来的维度是无害的：`torch_attention_op` 用
`rearrange(t, "b ... h v -> b h ... v").view(shape[0], shape[-2], -1, shape[-1])`
处理任意前导维，最终都塌成 `(B, heads, S, head_dim)`。

## 3. 模块划分

```
F:\anima-regional-prompter\
├─ scripts\
│   └─ anima_regional_prompter.py   Script 生命周期 + Gradio UI（唯一碰 Forge API 的文件）
├─ lib_anima_rp\
│   ├─ __init__.py
│   ├─ layout.py        比例字符串 → 归一化矩形列表
│   ├─ masks.py         矩形 → token 掩码张量（带缓存）
│   ├─ prompts.py       主提示词按 ADDBASE/BREAK 切段
│   ├─ conditioning.py  各段 → (N, L, 1024) cond 张量
│   ├─ attention.py     SelfCrossAttention.forward 的 patch + 分区注意力
│   └─ state.py         模型识别、运行期配置对象 RegionalConfig
├─ tests\
│   ├─ test_layout.py
│   ├─ test_masks.py
│   ├─ test_prompts.py
│   └─ test_attention_math.py
├─ docs\superpowers\specs\
└─ README.md
```

`layout` / `masks` / `prompts` 是纯函数，零 Forge 依赖。`attention` 只依赖 torch。
因此除 `scripts/` 外全部可以在不启动 Forge 的情况下 pytest。

## 4. 分区几何（layout.py）

### 4.1 语法

```
Divide ratio: 1,2;1,1,1
Main split:   Horizontal
```

- `;` 分段。**第一段是顶层比例**，决定沿主轴的切分。
- 第 `i+1` 段细分顶层的第 `i` 个切片（沿垂直于主轴的方向）。
- 段数少于顶层切片数时，剩余切片整块作为一个区域。
- 某段为空、或只有一个数时，该切片不再细分。

### 4.2 方向语义

以**分割线的方向**定义，避免 RP 里常见的记反问题：

- **Horizontal**：分割线水平 → 顶层切成**行**（上下堆叠），子比例把每行切成**列**
- **Vertical**：分割线垂直 → 顶层切成**列**（左右排列），子比例把每列切成**行**

`1,2;1,1,1` + Horizontal 的结果：

```
┌───────────────────────┐
│        区域 1          │  ← 高度占 1/3
├───────┬───────┬───────┤
│  区域2 │  区域3 │  区域4 │  ← 高度占 2/3，横向 1:1:1
└───────┴───────┴───────┘
```

UI 上提供**实时布局预览**（CSS grid 渲染 + 区域编号），使方向语义无需记忆。

### 4.3 接口

```python
@dataclass(frozen=True)
class Rect:
    x0: float; y0: float; x1: float; y1: float   # 归一化 [0, 1]

def parse_layout(ratio_str: str, main_split: str) -> list[Rect]
```

### 4.4 错误处理

`parse_layout` 自身须对以下输入抛 `ValueError`：非数字、负数、全零比例、
`;` 段数超过顶层切片数。

**区域数与提示词段数不匹配同样抛 `ValueError`，不静默降级。** 这是跨模块检查
（`parse_layout` 的输出 vs `split_prompt` 的输出），放在 `state.build_config()` 里，
在注入 `transformer_options` 之前完成。RP 在这里静默截断，导致「少写一个 BREAK
后画面莫名其妙」这类难以排查的问题。错误信息须包含实际区域数、实际段数、
以及解析出的比例结构。

## 5. 条件注入（conditioning.py）

采用**旁路注入**：不改动 Forge 的提示词解析与批处理管道。

```
Forge 管道（原样不动）：
    正向 prompt → cond   (B, 1, L, 1024)
    负向 prompt → uncond (B, 1, L, 1024)     形状一致 → 仍能合批一次前向

本扩展的旁路：
    transformer_options["anima_rp"].region_conds = (N, L, 1024)
    transformer_options["anima_rp"].base_cond    = (L, 1024) | None

attention hook 内按 cond_mark 分流：
    cond   行 → 用 region_conds 做分区注意力
    uncond 行 → 用传入的 context，走原逻辑
```

### 5.1 编码时机

在 `Script.process()` 中调用 `p.sd_model.get_learned_conditioning(segments)`。
此时文本编码器本来就要加载到显存（Forge 紧接着会编码主提示词），不会引发额外的模型换入换出。
若放到 `process_before_every_sampling`，此时 UNet 已在显存，调用文本编码器会触发换页。

### 5.2 变长对齐

各段 `L_i` 可能不同。pad 到 `max(L_i)`，**重复最后一个向量**——与 Forge 自己的
`prompt_parser.stack_conds` 行为一致。

### 5.3 缓存

按 `(tuple(segments), id(p.sd_model))` 缓存，避免 hires 二阶段重复编码。
缓存挂在 `p` 上，随生成结束自然释放。

### 5.4 已知代价

区域提示词内部的步数调度 `[a:b:5]`、交替 `[a|b]`、AND 组合不生效，因为这些由
`prompt_parser` 在 Forge 管道内实现，而我们绕开了它。区域提示词内的
**权重语法 `(word:1.2)` 仍然生效**，因为它由 `AnimaTextProcessingEngine` 自身的
`parse_prompt_attention` 处理。

## 6. 提示词切分（prompts.py）

```
背景描述 ADDBASE 左边人物 BREAK 右边人物
└──────┬──────┘ └───┬───┘        └───┬───┘
    base prompt   region 1         region 2
```

- `ADDBASE`：最多出现一次，之前的内容是 base prompt。缺失则无 base。
- `BREAK`：区域分隔符。`ADDCOL` / `ADDROW` 作为等价别名接受（RP 兼容，
  本扩展的分区由 divide ratio 显式决定，三者语义相同）。
- 分隔符必须大写、且是独立词（正则 `\b(?:BREAK|ADDCOL|ADDROW)\b`），避免误伤单词。
- 每段去首尾空白与多余逗号。空段抛错。

## 7. 注意力 patch（attention.py）

### 7.1 patch 时机与旁路

扩展加载时对 `backend.nn.anima.SelfCrossAttention.forward` 做一次 monkeypatch，
原函数存入 `_ORIG_FORWARD`：

```python
def patched_forward(self, x, context=None, rope_emb=None, transformer_options=None):
    rp = (transformer_options or {}).get("anima_rp")
    if rp is None or self.is_SelfAttn or context is None:
        return _ORIG_FORWARD(self, x, context, rope_emb, transformer_options)
    return regional_forward(self, x, context, rope_emb, transformer_options, rp)
```

未启用时是一次 dict 查找 + 委托，开销可忽略。self-attn、非 ANIMA 模型、
其他扩展的生成全部不受影响。

### 7.2 分区注意力

设 `rc = cond_indices`（正向行），`ru = uncond_indices`（负向行）。

```python
# --- 正向行：分区路径 ---
xc = x[rc]                                             # (nc, TOK, x_dim)
q  = q_norm(rearrange(q_proj(xc)))                     # 只算一次（4096 图像 token）

acc = torch.zeros(nc, TOK, inner_dim)                  # 未经 output_proj 的累加器
for i, ctx_i in enumerate(region_conds):               # 512 文本 token，投影很便宜
    k_i = k_norm(rearrange(k_proj(ctx_i)))
    v_i = rearrange(v_proj(ctx_i))
    a_i = torch_attention_op(q, k_i, v_i)              # (nc, TOK, inner_dim)
    acc += masks[i][None, :, None] * a_i

if base_cond is not None:
    a_b = torch_attention_op(q, k_base, v_base)
    acc = (1 - base_ratio) * acc + base_ratio * a_b

out = torch.empty(b, TOK, x_dim, ...)
out[rc] = output_proj(acc)                             # 正向只投影一次（见 §2.1 性质 3）

# --- 负向行：原路径，自带 output_proj ---
if ru:
    out[ru] = _ORIG_FORWARD(self, x[ru], context[ru], rope_emb, transformer_options)
```

`output_proj` 在两条路径上各作用一次，不重复也不遗漏。

若 `cond_mark` 全为 1（纯 uncond 批），跳过分区路径，直接委托 `_ORIG_FORWARD` 处理整个 batch。

**开销估计**：cross-attn 的 attention op 重复 N 次，q 投影与 output 投影各一次。
cross-attn 只占 Block 总开销的一小部分，N=3 时预期整体 1.3–1.6×。

### 7.3 dtype / device

`region_conds` 在使用前 `.to(device=x.device, dtype=context.dtype)`。
注意 `Anima.forward` 把残差流保持在 fp32 但 attention 输入被 `.to(compute_dtype)`，
所以对齐目标是**传入的 `context` 的 dtype**，不是 `x` 的。

### 7.4 latent 网格尺寸的获取

`SelfCrossAttention.forward` 只能看到 `(b, T*H*W, d)`，无法反推 H、W。
用一个 `unet_function_wrapper` 在每次前向前把网格尺寸写进 `transformer_options`：

```python
prev = unet.model_options.get("model_function_wrapper")

def wrapper(model_function, kwargs):
    x = kwargs["input"]                      # (B, C, T, H_lat, W_lat)
    ps, pt = diffusion_model.patch_spatial, diffusion_model.patch_temporal
    grid = (ceil(x.shape[2] / pt), ceil(x.shape[3] / ps), ceil(x.shape[4] / ps))
    kwargs["c"]["transformer_options"]["anima_rp_grid"] = grid
    if prev is not None:
        return prev(model_function, kwargs)
    return model_function(kwargs["input"], kwargs["timestep"], **kwargs["c"])

unet.set_model_unet_function_wrapper(wrapper)
```

**必须链式保留已有 wrapper**（`set_model_unet_function_wrapper` 是覆盖式的），
否则会静默禁用其他扩展。

向上取整与 `pad_to_patch_size` 的行为一致。

## 8. 掩码（masks.py）

```python
def build_masks(rects: list[Rect], grid: tuple[int, int, int],
                device, dtype) -> torch.Tensor   # (N, T*H*W)
```

- 判定规则：**token 中心点落在哪个矩形，就属于哪个区域**。保证每个 token 恰好属于一个区域。
- **不变量：`masks.sum(0)` 处处为 1。** 写成断言 + 测试。若 Σmask ≠ 1，区域边界会出现
  亮度漂移（attention 输出被整体缩放）。
- 展平顺序 `(t h w)`，与 §2.2 一致。
- 某区域在当前分辨率下分不到任何 token（区域过窄）时：合法，但记录警告。

缓存 key：`(ratio_str, main_split, grid, device, dtype)`。hires 二阶段分辨率变化时自动重建。

## 9. UI（scripts/anima_regional_prompter.py）

`InputAccordion`，`scripts.AlwaysVisible`（`show()` 时还不知道当前模型）。

| 控件 | 类型 | 默认 |
|------|------|------|
| Enable | InputAccordion 开关 | 关 |
| Main split | Radio [Horizontal, Vertical] | Horizontal |
| Divide ratio | Textbox | `1,1` |
| Base ratio | Slider 0–1，step 0.05 | 0.2 |
| 布局预览 | gr.HTML（随前两项实时更新） | — |

Base ratio 仅在提示词含 `ADDBASE` 时生效。

### 9.1 模型检测

`p.sd_model.forge_objects.unet.model.diffusion_model.__class__.__name__ == "Anima"`
（与 `sd_forge_radial` 检测 `WanModel` 的方式一致）。非 ANIMA 时不注入
`transformer_options`，控制台警告一次。

### 9.2 元数据

写入 `p.extra_generation_params`，使 PNG info 可复现：

```
RP Active:       True
RP Divide ratio: 1,2;1,1,1
RP Main split:   Horizontal
RP Base ratio:   0.2
```

## 10. 测试

用 forge-neo 的 venv Python（3.13）跑 pytest，**不需要启动 Forge**。

| 文件 | 覆盖 |
|------|------|
| `test_layout.py` | 一维/二维比例解析；段数不足；畸形输入抛错并含可读信息 |
| `test_masks.py` | `Σmask ≡ 1`；已知比例下各区 token 数符合预期；极窄区域退化；`(t h w)` 展平顺序 |
| `test_prompts.py` | `BREAK`/`ADDCOL`/`ADDROW` 等价；`ADDBASE` 一次；空段抛错；不误伤含 BREAK 子串的单词 |
| `test_attention_math.py` | **正确性核心**，见下 |

### 10.1 注意力等价性测试

用纯 torch 构造一个 `SelfCrossAttention`（随机权重），验证：

1. **单区域全覆盖**：`regional_forward` 输出 == 原 `forward(x, ctx)`（数值容差内）。
2. **两区域隔离**：区域 A 内 token 的输出 == 只喂 `ctx_A` 的原 `forward` 对应行输出。
   这验证了硬掩码下「分别 softmax」与「联合 masked softmax」的等价性。
3. **Base 混合**：`base_ratio=0` 时等同无 base；`base_ratio=1` 时等同全局 base。
4. **投影外提**：验证 `Σ m_i·proj(a_i) == proj(Σ m_i·a_i)`（即 §2.1 性质 3 成立）。

### 10.2 端到端验证

跑一张实图：`1girl, red dress BREAK 1boy, blue suit`，ratio `1,1`，Vertical。
预期左半红裙女性、右半蓝西装男性。作为交付验收标准。

## 11. 安装

代码住在 `F:\anima-regional-prompter`（独立 git 仓库），
在 `F:\sd-webui-forge-neo\extensions\anima-regional-prompter` 建 Windows 目录联接（junction）指过去。
junction 在 Windows 上不需要管理员权限，改代码即时生效。

## 12. 分阶段交付

| 阶段 | 内容 | 验收 |
|------|------|------|
| 1 | `layout` / `masks` / `prompts` | 三个测试文件全绿 |
| 2 | `attention` patch | `test_attention_math` 全绿 |
| 3 | `scripts/` 接线 + UI + junction | Forge 能加载，UI 出现，非 ANIMA 模型不受影响 |
| 4 | 端到端验证 | §10.2 的实图符合预期 |
