# ANIMA Regional Prompter

为 [sd-webui-forge-neo](https://github.com/Haoming02/sd-webui-forge-neo) 提供的分区提示词扩展，
专门适配 **ANIMA** 模型（`circlestone-labs/Anima`）。

功能对标 [sd-webui-regional-prompter](https://github.com/hako-mikan/sd-webui-regional-prompter)（下称 RP）
的 Attention 模式：把画面按比例切成若干区域，每个区域使用各自的提示词。
**提示词语法与 API 参数契约与 RP 保持一致**，现有 RP 调用只需改动 `alwayson_scripts` 的 key 一行。

> 状态：开发中。设计文档见
> [`docs/superpowers/specs/2026-07-31-anima-regional-prompter-design.md`](docs/superpowers/specs/2026-07-31-anima-regional-prompter-design.md)。

## 为什么单独做一个

ANIMA 是 Cosmos-Predict2 系的 cross-attention DiT，不是 SD1.5/SDXL 的 UNet，也不是 MMDiT。
原版 RP 的 hook 点在这里并不存在。

不过 ANIMA 的架构对分区渲染反而更友好：**文本只通过 cross-attention 进入模型**，
没有 pooled embedding、也没有文本参与的 adaLN 调制，所以遮罩 cross-attention
就能完整控制分区的文本影响——SDXL 做不到这一点，它的 pooled embedding 是全局的。

## 用法

在主提示词框里用 `BREAK` 分隔各区域提示词：

```
1girl, red hair, red dress BREAK
1boy, blue hair, blue suit
```

配合扩展面板：

```
Main Splitting: Columns
Divide Ratio:   1,1
```

得到左半边红发女性、右半边蓝发男性。面板内有实时布局预览，不必记方向。

可选的两个前置段（需勾选对应开关）：

```
masterpiece, best quality ADDCOMM 温暖的房间, 柔和光线 ADDBASE 1girl, red dress BREAK 1boy, blue suit
└──────── ADDCOMM 段 ────────┘ └──────── ADDBASE 段 ────────┘ └── 区域 0 ──┘  └── 区域 1 ──┘
```

- **Common prompt**（`ADDCOMM`，开关 `Use common prompt`）：文本前缀，拼到每个区域提示词前面。
- **Base prompt**（`ADDBASE`，开关 `Use base prompt`）：单独编码，在注意力输出层按
  `Base Ratio` 混入各区域。用于统一背景与整体构图，缓解区域边界的割裂感。

`ADDCOL` / `ADDROW` 是 `BREAK` 的等价别名（本扩展的分区由 Divide Ratio 显式决定）。

### Divide Ratio 语法

沿用 RP 原版语法。`;` 分隔外层带，`,` 分隔带内单元，
**每个带的第一个数字是该带自身的尺寸，其余数字是带内细分**：

```
Divide Ratio:   1,2,1,1;2,4,6
Main Splitting: Columns

┌──────┬───┬───┐
│  区0  │区1 │区2 │   带 1：尺寸 1，带内 2:1:1
├──────┴┬──┴───┤
│  区3   │  区4  │   带 2：尺寸 2，带内 4:6
└───────┴──────┘
```

不含 `;` 时为一维：`1,1,1` 就是三个等分单元。

`Main Splitting` 指的是**带内细分的方向**：

| 取值 | 外层带 | 带内单元 |
|------|--------|----------|
| `Columns`（默认） | 行，纵向堆叠 | 列，横向排列 |
| `Rows` | 列，横向排列 | 行，纵向堆叠 |

`Base Ratio` 与 Divide Ratio 同构，可逐区域给不同权重（`"0.2,0.4;0.3"`）；
给单个值则应用到所有区域。

## API

```json
"alwayson_scripts": {
  "ANIMA Regional Prompter": {
    "args": [true, false, "Matrix", "Columns", "Mask", "Prompt", "1,1", "0.2",
             true, false, false, "Attention", false, "0", "0", "0.4", null,
             "0", "0", false]
  }
}
```

20 个位置参数的顺序与 RP 完全一致，从 RP 迁移时只需把 key 从 `"Regional Prompter"`
改成 `"ANIMA Regional Prompter"`。

未实现的参数分两类处理：

- **降级**（控制台警告 + 写入 PNG info，继续生成）：`calcmode="Latent"`、
  `rp_selected_tab` 为 `Mask`/`Prompt`、`mmode="Random"`、`usencom=true`
- **忽略**（无影响，不警告）：`xmode`、`pmode`、`nchangeand`、`lnter`、`lnur`、
  `threshold`、`polymask`、`lstop`、`lstop_hr`

## 与 RP 的已知差异

- 不支持 Latent couple 模式、Mask 模式、Prompt 模式、`Random` 分区。
- 区域提示词内的步数调度 `[a:b:5]`、交替 `[a|b]`、AND 组合不生效。
  权重语法 `(word:1.2)` 仍然有效。
- 负面提示词不分区。
- 区域数与提示词段数不匹配时**抛错**，而不是像 RP 那样静默补齐最后一段。

## 安装

克隆到 Forge Neo 的 `extensions` 目录：

```
git clone https://github.com/sadlywish/anima-regional-prompter.git
```

## 许可

AGPL-3.0
