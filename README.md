# ANIMA Regional Prompter

为 [sd-webui-forge-neo](https://github.com/Haoming02/sd-webui-forge-neo) 提供的分区提示词扩展，
专门适配 **ANIMA** 模型（`circlestone-labs/Anima`）。

功能对标 [sd-webui-regional-prompter](https://github.com/hako-mikan/sd-webui-regional-prompter)（下称 RP）
的 Attention 模式：把画面按比例切成若干区域，每个区域使用各自的提示词。
**提示词语法与 API 参数契约与 RP 保持一致**，现有 RP 调用只需改动 `alwayson_scripts` 的 key 一行。

> 状态：可用，已在 Forge Neo + ANIMA 上完成端到端验证。设计文档见
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

### Base Ratio 语法

`Base Ratio` 与 Divide Ratio 同构，可逐区域给不同权重；给单个值（默认的 `0.2`）则应用到所有区域。

取值按**带**分配：某个带的取值不够就重复该带自己的最后一个值，取值过多则溢出到下一带，
带用完后剩余的值丢弃。这与 RP 完全一致。

```
Divide Ratio: 1,2,1,1;2,4,6      带1 = 3 区，带2 = 2 区
Base Ratio:   0.1,0.2;0.3,0.4    →  0.1  0.2  0.2 │ 0.3  0.4
                                          ↑ 带1 不够，重复它自己的 0.2
```

> **一个容易踩的坑：`Rows` 模式下 Base Ratio 的分隔符也跟着换。**
> 和 Divide Ratio 一样，`Rows` 模式里 `,` 分隔带、`;` 分隔带内单元。
> 但 Divide Ratio 有个「不含 `;` 就按一维处理」的特判，Base Ratio 没有——
> 所以一维布局下两者的写法会错位：
>
> ```
> Divide Ratio: 1,1,1      Main Splitting: Rows      →  三行
> Base Ratio:   0.1,0.2,0.3   ✗  得到 [0.1, 0.1, 0.1]（被当成三个带，后两个丢弃）
> Base Ratio:   0.1;0.2;0.3   ✓  得到 [0.1, 0.2, 0.3]
> ```
>
> 这是 RP 自身的不一致，本扩展为兼容而如实复刻。`Columns` 模式和单值写法不受影响。

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

## 验证方式

### 恒等式检验（不依赖模型行为）

最硬的一条判据：**所有区域用同一段提示词时，输出必须与不启用扩展逐像素完全一致。**

因为 `Σ mask_i ≡ 1` 且 `output_proj` 无偏置、线性，所以
`Σ mask_i · attn(ctx)` 必须精确退化成 `attn(ctx)`。这一条覆盖掩码构建、
context 索引、注意力累加、投影外提整条链路，且完全不受模型对提示词的理解影响。

实测（`anima_aestheticV11`，1024×1024）：

```
N=1  MAE=0.00      N=2  MAE=0.00
N=3  MAE=0.00      N=4  MAE=0.00      （MAE 为逐像素平均绝对差）
```

### 分区效果（依赖模型行为）

用强色彩提示词生成，再测量各区域的像素统计量（平均 R 减平均 B）。
关键是**对照组**——同一段提示词、同一个种子，在扩展关闭时再跑一次。

在 `anima_aestheticV11` 上的实测（1024×1024，Euler，16 步）：

```
                        左半 R-B   右半 R-B   分离度
对照组（扩展关闭）        +77.3     +77.5      -0.2
启用 Columns 1,1         +117.6     -20.2    +137.8
```

右半从 +77 翻到 -20，说明它确实被第二段提示词接管了；对照组的分离度接近零，
排除了「模型自己碰巧那么画」的可能。

`Rows` 模式的纵向分离强度天然弱于横向（模型对上下结构有更强的构图先验），
所以那里用的判据是**颜色对调时分离度必须翻转符号**——这个判据与模型无关，
只有分区真在驱动版面时才成立。实测 +26.2 → −24.0，且横向分离仅 5.4，
确认轴向没有对调。

### 分区数量对效果的影响

分离效果随区域变窄而减弱，这是 attention 模式分区的固有性质，不是缺陷：
文本只在 cross-attention 里被分区约束，而 self-attention 会在 10 个 Block
之间把相邻区域的内容混合。

实测下来，两个区域各占一半时分离非常干脆；三个及以上、每区只有三分之一
画面宽度时，若各区提示词互相矛盾（例如三段不同颜色的「纯色背景」），
模型往往会放弃条带结构去画一个完整主体。相邻区域共享同一主题时则依然稳固
（例如三区中前两区都是蓝、第三区是红，实测左 2/3 蓝、右 1/3 红，分毫不差）。

建议：区域数多时，给各区提示词一个能共存的整体设定（用 `ADDBASE` 尤其有效），
不要让它们在同一张图里互相打架。

## 与 RP 的已知差异

- 不支持 Latent couple 模式、Mask 模式、Prompt 模式、`Random` 分区。
- 区域提示词内的步数调度 `[a:b:5]`、交替 `[a|b]`、AND 组合不生效。
  权重语法 `(word:1.2)` 仍然有效。
- 负面提示词不分区。若负面提示词里含分隔关键字，会被剥离并以逗号合并，控制台警告一次。
- 区域数与提示词段数不匹配时**不静默补齐**（RP 会自动重复最后一段提示词凑数）。
  本扩展的做法是：本次生成不分区，控制台警告，并把错误写进 PNG info 的 `RP Error`。

  之所以不是直接中止生成，是因为 Forge 的 `modules/scripts.py:846` 会吞掉扩展
  `process()` 抛出的异常并继续跑——中止不在扩展的能力范围内。所以退而求其次，
  让错误在产物里留下痕迹：出图时不会有 `RP Active`，只有 `RP Error`，事后可查。
- `process()` 读的是 `p.prompt`。若调用方传入的是逐图不同的提示词列表（典型是
  API 路径，网页 UI 不会），整批会共用第一张图的分区配置。上游 RP 的
  `rp.py:358` 是同一行代码，此处为忠实复刻而非新引入的问题。

## 安装

克隆到 Forge Neo 的 `extensions` 目录：

```
git clone https://github.com/sadlywish/anima-regional-prompter.git
```

## 许可

AGPL-3.0
