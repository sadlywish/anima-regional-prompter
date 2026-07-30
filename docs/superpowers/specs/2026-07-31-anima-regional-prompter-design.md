# ANIMA Regional Prompter — 设计文档

日期：2026-07-31
目标宿主：sd-webui-forge-neo
目标模型：ANIMA（`circlestone-labs/Anima`）
参照实现：[sd-webui-regional-prompter](https://github.com/hako-mikan/sd-webui-regional-prompter)（下称 RP）

## 1. 目标与范围

为 Forge Neo 提供分区提示词扩展，适配 ANIMA 模型，功能对标 RP 的 Attention 模式。
**API 调用契约与 RP 保持一致**，使现有 RP 的 `alwayson_scripts` payload 只需改动 key 一行即可迁移。

### v1 范围

| 功能 | 状态 |
|------|------|
| Attention 模式（遮罩 cross-attention） | 做 |
| RP 原版 Divide ratio 语法（二维分区） | 做 |
| Base prompt（`ADDBASE`，逐区域权重） | 做 |
| Common prompt（`ADDCOMM`，文本前缀） | 做 — 见下方说明 |
| RP 兼容的 20 位置参数 API 契约 | 做 |
| 实时布局预览 | 做 |
| PNG info 元数据写入 | 做 |
| 分区负面提示词 | 不做 |
| Latent couple 模式（`calcmode="Latent"`） | 不做 |
| Mask 模式 / Prompt 模式 | 不做 |
| `Random` 分区子模式 | 不做 |
| LoRA 负面比例相关参数 | 不做 |

**Common prompt 为何在范围内**：它原本被划出 v1，但 API 兼容要求把它变成了必需项。
RP 的 `usecom=true` 提示词以 `ADDCOMM` 分隔出公共段；若我们忽略该参数，这段文本会被当成
一个区域提示词，直接触发区域数不匹配错误。任何降级方案都劣于直接实现，而实现成本是一次字符串拼接。

### 非目标

- 不支持 ANIMA 以外的模型。检测到非 ANIMA 模型时扩展完全旁路（零开销），控制台警告一次。
- 不支持区域提示词内部的步数调度（`[a:b:5]`）与 AND 组合。这是「旁路注入」方案的已知代价，见 §5.4。

## 2. 宿主环境事实

以下事实来自阅读 Forge Neo 源码，是设计的前提。实现时若发现不符，需回到本文档修订，
而不是在代码里硬凑。

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
   `if self.is_SelfAttn and rope_emb is not None` 才应用 RoPE，位置信息已在 `LLMAdapter` 中烘焙。
   因此替换整个 context、或沿 token 维拼接多段提示词，都不会破坏位置语义。
3. **`output_proj` 是 `nn.Linear(..., bias=False)`，`output_dropout` 在推理期是 `Identity`。**
   因此 `Σ mask_i · output_proj(a_i) == output_proj(Σ mask_i · a_i)`，投影可以只做一次。
4. **`SelfCrossAttention.forward` 直接接收 `transformer_options`**（由 `Block.forward` 透传），
   这是天然的 hook 面。

### 2.2 token 布局

- `PatchEmbed` 把 `(B, C, T, H_lat, W_lat)` 变成 `(B, T', H', W', D)`，
  `T' = T / patch_temporal`，`H' = H_lat / patch_spatial`，`W' = W_lat / patch_spatial`。
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
- 权重语法 `(word:1.2)` 由该引擎自身的 `parse_prompt_attention` 处理，在区域提示词内仍然有效。

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
│   ├─ layout.py        Divide ratio 字符串 → 归一化矩形 + 逐区域 base 权重
│   ├─ masks.py         矩形 → token 掩码张量（带缓存）
│   ├─ prompts.py       主提示词按 ADDCOMM/ADDBASE/BREAK 切段
│   ├─ conditioning.py  各段 → (N, L, 1024) cond 张量
│   ├─ attention.py     SelfCrossAttention.forward 的 patch + 分区注意力
│   └─ state.py         参数归一化、模型识别、RegionalConfig 构建与校验
├─ tests\
├─ docs\superpowers\specs\
└─ README.md
```

`layout` / `masks` / `prompts` / `state` 是纯函数，零 Forge 依赖。`attention` 只依赖 torch。
因此除 `scripts/` 外全部可以在不启动 Forge 的情况下 pytest。

## 4. 分区几何（layout.py）

**采用 RP 原版语法，不做任何"改良"。** 迁移兼容性优先于语法美观。

### 4.1 Divide ratio 语法

`;` 分隔外层带（band），`,` 分隔带内单元（cell）。
**每个带的第一个数字是该带自身的尺寸，其余数字是带内细分。**
（依据：RP `scripts/regions.py:45` 注释「First value in each row is the row's ratio,
the rest are col ratios」，及 `split_l2(..., indsingles=True)` 的实现。）

```
1,2,1,1;2,4,6      Main Splitting: Columns

┌──────┬───┬───┐
│  区0  │区1 │区2 │   带 1：尺寸 1，带内 2:1:1
├──────┴┬──┴───┤
│  区3   │  区4  │   带 2：尺寸 2，带内 4:6
└───────┴──────┘
```

特例：

- **不含 `;`** → 一维。整个字符串是单个带的 cells，带尺寸为 1。
  `1,1,1` → 3 个等分单元。
- **某带只有一个数字** → 该带尺寸为此数，带内为单个 cell 铺满。
- 尺寸与 cell 值各自按其所在层归一化（带尺寸按所有带之和，cell 按同带内之和）。

### 4.2 方向语义（`mmode`）

取值 `"Columns"` / `"Rows"`，与 RP 的 Matrix submode 一致。
**指的是带内细分的方向**，不是外层带的方向：

| `mmode` | 外层带 | 带内 cell |
|---------|--------|-----------|
| `Columns`（默认） | 行，纵向堆叠 | 列，横向排列 |
| `Rows` | 列，横向排列 | 行，纵向堆叠 |

依据：RP `regions.py:275-276` 把 `Columns`→`Horizontal`、`Rows`→`Vertical`，
再由 `indflip = ("Ver" in mode)` 在 296–299 行决定 `ocell`（外层）落在 h 轴还是 w 轴。

`"Random"` 不支持，按 §9 的降级规则处理。

UI 上提供**实时布局预览**（CSS grid 渲染 + 区域编号），使方向语义无需记忆。

### 4.3 `flipper`

RP 的 `flipper` 参数（UI 标签 `flip "," and ";"`）在解析前交换两个分隔符，
等价于 RP 的 `changecs()`（`regions.py:268`）。实现为解析入口处的一次字符串置换。

### 4.4 区域编号

带优先、带内其次（band-major），与 RP 计数器一致（`regions.py:307-313`）。
提示词段按此顺序一一对应。

### 4.5 Base ratio（`baseratios`）

与 Divide ratio **同构的字符串**，逐区域给出 base 混合权重，默认 `"0.2"`。
`fmap` 默认值为 0（RP `regions.py:373` 的 `ffloatd(0)`）。

广播规则（RP `split_l2` 的 `basestruct` 行为）：值不足时**重复最后一个**填满，多余的忽略。
故 `"0.2"` 对任意布局都展开成全区域 0.2。

> **与 RP 的已知差异**：RP 还有一条「某行值过多时溢出到下一行」的规则。
> 本实现不复刻该行为——它在 RP 里也属于意外语义。超出当前带 cell 数的值直接截断。
> 此差异写入 README。

解析结果是长度 N 的权重向量，与区域一一对应。

### 4.6 接口与错误处理

```python
@dataclass(frozen=True)
class Rect:
    x0: float; y0: float; x1: float; y1: float   # 归一化 [0, 1]

def parse_layout(ratios: str, mmode: str, flipper: bool) -> list[Rect]
def parse_base_weights(baseratios: str, layout: list[Rect], mmode: str,
                       flipper: bool) -> list[float]
```

`parse_layout` 须对以下输入抛 `ValueError`：非数字、负数、全零比例、空字符串。

**区域数与提示词段数不匹配同样抛 `ValueError`，不静默降级。** 这是跨模块检查
（`parse_layout` 的输出 vs `split_prompt` 的输出），放在 `state.build_config()` 里，
在注入 `transformer_options` 之前完成。RP 在这里会静默补齐最后一段提示词
（`regions.py:383-384`），导致「少写一个 BREAK 后画面莫名其妙」这类难以排查的问题。
错误信息须包含实际区域数、实际段数、以及解析出的带/cell 结构。

## 5. 条件注入（conditioning.py）

采用**旁路注入**：不改动 Forge 的提示词解析与批处理管道。

```
Forge 管道（原样不动）：
    正向 prompt → cond   (B, 1, L, 1024)
    负向 prompt → uncond (B, 1, L, 1024)     形状一致 → 仍能合批一次前向

本扩展的旁路：
    transformer_options["anima_rp"].region_conds  = (N, L, 1024)
    transformer_options["anima_rp"].base_cond     = (L, 1024) | None
    transformer_options["anima_rp"].base_weights  = (N,) float

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
**权重语法 `(word:1.2)` 仍然生效**（见 §2.3）。

## 6. 提示词切分（prompts.py）

```
masterpiece, best quality ADDCOMM 温暖的房间 ADDBASE 左边人物 BREAK 右边人物
└────────┬────────┘      └───┬──┘       └──┬──┘       └──┬──┘
      common               base          region 0      region 1
```

- `ADDCOMM`：最多一次，之前的内容是 common prompt，**以文本前缀拼到每个区域提示词和
  base prompt 之前**。仅当 `usecom=True` 时生效；`usecom=False` 时该关键字按普通分隔符
  处理其左侧内容为第一个区域（与 RP 一致）。
- `ADDBASE`：最多一次，之后到下一个分隔符之间是 base prompt。仅当 `usebase=True` 时生效；
  `usebase=False` 时同样按普通分隔符处理。
- `usecom` / `usebase` 为 `True` 但提示词中不存在对应关键字时：该功能静默不生效
  （无 common 前缀 / 无 base 混合），不报错。这与 RP 一致，且是 x/y 图之类场景的常见状态。
- `BREAK` / `ADDCOL` / `ADDROW`：区域分隔符，三者等价（本扩展的分区由 Divide ratio
  显式决定，RP 中三者的方向含义在此无差别）。
- 分隔符必须大写且是独立词（正则 `\b(?:ADDCOMM|ADDBASE|ADDCOL|ADDROW|BREAK)\b`），
  避免误伤含这些子串的单词。
- 每段去首尾空白与多余逗号。空的区域段抛错。

### 6.1 负面提示词

不分区。若负面提示词含 `ADDCOMM` / `BREAK` 等标记（`usencom=True` 的旧 payload 会有），
**剥掉标记并以逗号合并全部内容**，控制台警告一次。

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
q  = q_norm(rearrange(q_proj(xc)))                     # 只算一次（图像 token 多，这步最贵）

acc = torch.zeros(nc, TOK, inner_dim)                  # 未经 output_proj 的累加器
for i, ctx_i in enumerate(region_conds):               # 文本 token 少，投影很便宜
    k_i = k_norm(rearrange(k_proj(ctx_i)))
    v_i = rearrange(v_proj(ctx_i))
    a_i = torch_attention_op(q, k_i, v_i)              # (nc, TOK, inner_dim)
    acc += masks[i][None, :, None] * a_i

if base_cond is not None:
    a_b   = torch_attention_op(q, k_base, v_base)
    w_map = Σ_i masks[i] * base_weights[i]             # (TOK,) 逐 token 的 base 权重
    acc   = (1 - w_map)[None, :, None] * acc + w_map[None, :, None] * a_b

out = torch.empty(b, TOK, x_dim, ...)
out[rc] = output_proj(acc)                             # 正向只投影一次（见 §2.1 性质 3）

# --- 负向行：原路径，自带 output_proj ---
if ru:
    out[ru] = _ORIG_FORWARD(self, x[ru], context[ru], rope_emb, transformer_options)
```

`output_proj` 在两条路径上各作用一次，不重复也不遗漏。

`w_map` 由逐区域权重（§4.5）与掩码合成，因此 `baseratios` 支持逐区域不同的值，
而混合运算仍是单次逐 token 插值。

若 `cond_mark` 全为 1（纯 uncond 批），跳过分区路径，直接委托 `_ORIG_FORWARD` 处理整个 batch。

**开销估计**：cross-attn 的 attention op 重复 N（+1）次，q 投影与 output 投影各一次。
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
否则会静默禁用其他扩展。向上取整与 `pad_to_patch_size` 的行为一致。

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

缓存 key：`(ratios, mmode, flipper, grid, device, dtype)`。hires 二阶段分辨率变化时自动重建。

## 9. API 契约（RP 兼容）

`title()` 返回 `"ANIMA Regional Prompter"`。API 调用：

```json
"alwayson_scripts": {
  "ANIMA Regional Prompter": {
    "args": [true, false, "Matrix", "Columns", "Mask", "Prompt", "1,1", "0.2",
             true, false, false, "Attention", false, "0", "0", "0.4", null,
             "0", "0", false]
  }
}
```

`ui()` **必须返回恰好 20 个组件，顺序与 RP 的 `scripts/rp.py:344` 完全一致**，
否则位置参数会错位。未使用的参数以 `gr.State` 占位（RP 自身也用 `gr.State`
承载 `rp_selected_tab`，此做法与宿主 API 兼容）。

| # | 参数 | RP 默认 | 组件 | 处理 |
|---|------|---------|------|------|
| 0 | `active` | `False` | Checkbox | 启用开关 |
| 1 | `debug` | `False` | Checkbox | 详细日志 |
| 2 | `rp_selected_tab` | `"Matrix"` | State | 非 `Matrix` → 降级 |
| 3 | `mmode` | `"Columns"` | Radio | `Columns`/`Rows`；`Random` → 降级 |
| 4 | `xmode` | `"Mask"` | State | 忽略 |
| 5 | `pmode` | `"Prompt"` | State | 忽略 |
| 6 | `ratios` | `"1,1"` | Textbox | Divide ratio |
| 7 | `baseratios` | `"0.2"` | Textbox | 逐区域 base 权重 |
| 8 | `usebase` | `False` | Checkbox | 启用 `ADDBASE` |
| 9 | `usecom` | `False` | Checkbox | 启用 `ADDCOMM` |
| 10 | `usencom` | `False` | State | `True` → 降级（见 §6.1） |
| 11 | `calcmode` | `"Attention"` | State | 非 `Attention` → 降级 |
| 12 | `nchangeand` | `False` | State | 忽略 |
| 13 | `lnter` | `"0"` | State | 忽略 |
| 14 | `lnur` | `"0"` | State | 忽略 |
| 15 | `threshold` | `"0.4"` | State | 忽略 |
| 16 | `polymask` | `None` | State | 忽略 |
| 17 | `lstop` | `"0"` | State | 忽略 |
| 18 | `lstop_hr` | `"0"` | State | 忽略 |
| 19 | `flipper` | `False` | Checkbox | 交换 `,` `;` |

### 9.1 降级策略

「降级」= 控制台警告一次 + 写入 PNG info + 按 Attention/Matrix 模式继续生成。
**不抛异常**，以免旧 payload 里无关的默认值让整条批量任务失败。

PNG info 记录形如：

```
RP Unsupported: calcmode=Latent, tab=Mask
```

「忽略」= 完全不读该参数，不警告。这些参数（LoRA 比例、threshold、polymask 等）
对本扩展的输出没有任何影响。

## 10. UI（scripts/anima_regional_prompter.py）

`InputAccordion`，`scripts.AlwaysVisible`（`show()` 时还不知道当前模型）。

可见控件：`active`、`mmode`、`ratios`、`baseratios`、`usebase`、`usecom`、`flipper`、`debug`，
外加一个**实时布局预览**（gr.HTML，随 `ratios`/`mmode`/`flipper` 更新，非输入组件、不进 arg 列表）。

其余 12 个参数为 `gr.State` 占位，UI 上不可见。README 中说明它们仅为 API 兼容而存在。

### 10.1 模型检测

`p.sd_model.forge_objects.unet.model.diffusion_model.__class__.__name__ == "Anima"`
（与 `sd_forge_radial` 检测 `WanModel` 的方式一致）。非 ANIMA 时不注入
`transformer_options`，控制台警告一次。

### 10.2 元数据

写入 `p.extra_generation_params`，使 PNG info 可复现：

```
RP Active:       True
RP Divide ratio: 1,2,1,1;2,4,6
RP Main splitting: Columns
RP Base ratio:   0.2
RP Use base:     True
RP Use common:   False
RP Flip:         False
RP Unsupported:  (仅在发生降级时出现)
```

## 11. 测试

用 forge-neo 的 venv Python（3.13）跑 pytest，**不需要启动 Forge**。

| 文件 | 覆盖 |
|------|------|
| `test_layout.py` | RP 语法：一维 / 二维 / 单值带 / `flipper`；`Columns` 与 `Rows` 的矩形对调；畸形输入抛错且信息可读 |
| `test_base_weights.py` | `"0.2"` 广播到全区域；逐区域值；不足补最后一个；超出截断 |
| `test_masks.py` | `Σmask ≡ 1`；已知比例下各区 token 数符合预期；极窄区域退化；`(t h w)` 展平顺序 |
| `test_prompts.py` | `ADDCOMM`/`ADDBASE`/`BREAK` 切分；`ADDCOL`/`ADDROW` 等价；`usecom`/`usebase` 开关的影响；负面提示词标记剥离；不误伤含关键字子串的单词 |
| `test_state.py` | 20 参数归一化；各降级分支产生正确的警告文案；区域数不匹配抛错 |
| `test_attention_math.py` | **正确性核心**，见下 |

### 11.1 注意力等价性测试

用纯 torch 构造一个 `SelfCrossAttention`（随机权重），验证：

1. **单区域全覆盖**：`regional_forward` 输出 == 原 `forward(x, ctx)`（数值容差内）。
2. **两区域隔离**：区域 A 内 token 的输出 == 只喂 `ctx_A` 的原 `forward` 对应行输出。
   这验证了硬掩码下「分别 softmax」与「联合 masked softmax」的等价性。
3. **Base 混合**：`base_weights` 全 0 时等同无 base；全 1 时等同全局 base；
   逐区域不同权重时按 `w_map` 正确插值。
4. **投影外提**：验证 `Σ m_i·proj(a_i) == proj(Σ m_i·a_i)`（即 §2.1 性质 3 成立）。
5. **cond/uncond 分流**：混合 batch 中负向行的输出 == 原 `forward` 的对应行。

### 11.2 与 RP 的布局一致性验证

RP 的 `regions.py` 除 numpy/PIL/torch 外只额外 `from modules import devices`，
测试中注入一个 stub 模块即可导入。用一组 Divide ratio 样例分别跑 RP 的
`split_l2` + `ratiosdealer` 与我们的 `parse_layout`，断言矩形一致。
这把「格式兼容」从人工比对变成可执行断言。

RP 源码路径由环境变量 `RP_REFERENCE_PATH` 指定（开发机上为
`F:\magic\extensions\sd-webui-regional-prompter`）。未设置或路径不存在时该测试 skip 而非 fail。

### 11.3 端到端验证

跑一张实图：`1girl, red dress BREAK 1boy, blue suit`，`ratios="1,1"`，`mmode="Columns"`。
预期左半红裙女性、右半蓝西装男性。作为交付验收标准。

再跑一次同参数的 API 调用（§9 的 payload 形态），确认与 UI 路径产出一致。

## 12. 安装

代码住在 `F:\anima-regional-prompter`（独立 git 仓库），
在 `F:\sd-webui-forge-neo\extensions\anima-regional-prompter` 建 Windows 目录联接（junction）指过去。
junction 在 Windows 上不需要管理员权限，改代码即时生效。

## 13. 分阶段交付

| 阶段 | 内容 | 验收 |
|------|------|------|
| 1 | `layout` / `masks` / `prompts` / `state` | 前五个测试文件全绿，含 §11.2 的 RP 一致性断言 |
| 2 | `attention` patch | `test_attention_math` 全绿 |
| 3 | `scripts/` 接线 + UI + junction | Forge 能加载，UI 出现，非 ANIMA 模型不受影响 |
| 4 | 端到端验证 | §11.3 的实图与 API 调用均符合预期 |
