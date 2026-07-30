# ANIMA Regional Prompter

为 [sd-webui-forge-neo](https://github.com/Haoming02/sd-webui-forge-neo) 提供的分区提示词扩展，
专门适配 **ANIMA** 模型（`circlestone-labs/Anima`）。

功能对标 [sd-webui-regional-prompter](https://github.com/hako-mikan/sd-webui-regional-prompter)
的 Attention 模式：把画面按比例切成若干区域，每个区域使用各自的提示词。

> 状态：开发中。设计文档见
> [`docs/superpowers/specs/2026-07-31-anima-regional-prompter-design.md`](docs/superpowers/specs/2026-07-31-anima-regional-prompter-design.md)。

## 为什么单独做一个

ANIMA 是 Cosmos-Predict2 系的 cross-attention DiT，不是 SD1.5/SDXL 的 UNet，也不是 MMDiT。
原版 Regional Prompter 的 hook 点在这里并不存在。

不过 ANIMA 的架构对分区渲染反而更友好：**文本只通过 cross-attention 进入模型**，
没有 pooled embedding、也没有文本参与的 adaLN 调制，所以遮罩 cross-attention
就能完整控制分区的文本影响——SDXL 做不到这一点，它的 pooled embedding 是全局的。

## 用法

在主提示词框里用 `BREAK` 分隔各区域提示词，用 `ADDBASE` 指定一段混入所有区域的全局提示词：

```
a cozy room, warm lighting ADDBASE
1girl, red hair, red dress BREAK
1boy, blue hair, blue suit
```

配合扩展面板里的 `Divide ratio` 与 `Main split`：

```
Divide ratio: 1,1
Main split:   Vertical
```

得到左半边红发女性、右半边蓝发男性。面板内有实时布局预览，不必记方向。

### Divide ratio 语法

`;` 分段。第一段是顶层比例，其后每段细分顶层的对应切片。

```
Divide ratio: 1,2;1,1,1
Main split:   Horizontal

┌───────────────────────┐
│        区域 1          │  ← 高度占 1/3
├───────┬───────┬───────┤
│  区域2 │  区域3 │  区域4 │  ← 高度占 2/3，横向 1:1:1
└───────┴───────┴───────┘
```

方向以**分割线的方向**定义：`Horizontal` 的分割线是水平的，所以顶层切成行。

## 安装

克隆到 Forge Neo 的 `extensions` 目录：

```
git clone https://github.com/sadlywish/anima-regional-prompter.git
```

## 许可

AGPL-3.0
