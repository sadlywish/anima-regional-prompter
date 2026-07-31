# 从 sd-webui-regional-prompter 迁移

写给已经在用 RP 的 API 调用方。**结论先说：改 key 一行即可跑通，但有一条行为差异会决定你的容错逻辑要不要改。**

本文最后一节针对 `F:\anima` 的 koishi 插件给出逐处对照。

---

## 一、必须改的：脚本 key

```diff
  payload.alwayson_scripts = {
-   'Regional Prompter': { args: [...] },
+   'ANIMA Regional Prompter': { args: [...] },
  }
```

20 个位置参数的**顺序、含义、默认值与 RP 完全一致**（对照 `rp.py:344` 逐项核对过），
数组本身一个字都不用动。

---

## 二、不用改的

### Divide ratio 语法

完全一致。`;` 分带、`,` 分带内单元，**每带的第一个数字是该带自身的尺寸**：

```
"2,1,1"           单带三单元，宽度 2:1:1
"1,2,1,1;2,4,6"   带1 高 1、内部 2:1:1；带2 高 2、内部 4:6
```

这套解析有 47 个断言背书，期望值直接调用 RP 真实源码的 `split_l2` / `ratiosdealer`
算出，不是凭理解复刻。

### 分隔关键字

`BREAK` / `ADDCOL` / `ADDROW` 在本扩展里**完全等价**——分区结构由 Divide ratio 决定，
关键字只负责切分。所以你现有的「行内 ADDCOL、行间 ADDROW」拼法原样可用。

`ADDCOMM`（公共前缀）与 `ADDBASE`（全局底稿）语义与 RP 一致。

### 区域顺序

band-major：先第一带内的各单元（左→右），再第二带。`mmode="Columns"` 时带即是行，
所以**与「行主序」完全一致**，你按行拼的提示词顺序不用调整。

已验证：

```
ratios "1,2,1,1;2,4,6"  +  r0 ADDCOL r1 ADDCOL r2 ADDROW r3 ADDCOL r4
→ 区域0 x[0.00,0.50] y[0.00,0.33]    区域1 x[0.50,0.75] y[0.00,0.33]
  区域2 x[0.75,1.00] y[0.00,0.33]    区域3 x[0.00,0.40] y[0.33,1.00]
  区域4 x[0.40,1.00] y[0.33,1.00]
```

---

## 三、唯一要紧的行为差异

**RP 以提示词结构为准，本扩展以 Divide ratio 为准。**

RP 在提示词含 `ADDROW`/`ADDCOL` 时，走 `matrixdealer` 的 `lbreaks` 分支——先从提示词
数出网格结构，再把 ratios 广播上去。ratios 与提示词对不上时，RP **静默适配**
（`regions.py:383-384` 甚至会自动重复最后一段提示词来凑数）。

本扩展不这么做。区域数完全由 Divide ratio 决定，提示词段数必须与之相等。**不相等时：**

- 本次生成**不分区**
- 控制台警告
- PNG info 里写入 `RP Error`，包含区域数、段数、Divide ratio 原文
- **不写** `RP Active`

这是有意的设计。RP 那种静默补齐会让你拿到一张「看起来正常但分区其实错了」的图，而排查
起来毫无线索。我们宁可让它响亮地失败。

**对你的影响**：如果你原来依赖 RP 的自动适配来兜底 ratios 与提示词不一致的情况，那部分
容错要挪到你自己这边——在发请求前保证两者一致。

判断方法（不需要出图）：

```python
from lib_anima_rp.layout import parse_layout
n_regions = len(parse_layout(ratios, mmode, flipper))
# 断言 n_regions == 你的提示词段数
```

或者直接从响应里检查：

```python
info = json.loads(resp["info"])["infotexts"][0]
if "RP Error" in info:
    ...  # 分区没生效，info 里有可读的原因
```

---

## 四、后端也要换

本扩展只适配 **ANIMA** 模型，宿主是 **sd-webui-forge-neo**，不是 Reforge。
非 ANIMA 模型下扩展完全旁路（控制台警告一次，不影响出图，但也不分区）。

所以迁移不只是改 key，还要把 API 地址指向跑着 ANIMA 的 Forge Neo 实例。

---

## 五、不支持的参数怎么处理

不会报错，走**降级 + 留痕**：控制台警告一次，并在 PNG info 写
`RP Unsupported: <原因>`，然后按 Attention/Matrix 模式继续出图。

| 参数取值 | 处理 |
|---|---|
| `calcmode="Latent"` | 降级为 Attention |
| `rp_selected_tab` 为 `Mask` / `Prompt` | 降级为 Matrix |
| `mmode="Random"` | 降级为 Columns |
| `usencom=true` | 负面提示词不分区；其中的关键字会被剥离并以逗号合并 |
| `xmode` `pmode` `nchangeand` `lnter` `lnur` `threshold` `polymask` `lstop` `lstop_hr` | 完全不读，不警告 |

最后一行意味着：这些参数你随便传什么都行（包括类型不对的值），不会有任何影响。

---

## 六、值得顺手启用的：ADDBASE

这不是迁移必需，但能直接改善多区域的效果。

本扩展的分区是遮罩 cross-attention 实现的，**区域越窄、各区提示词越互相矛盾，分离越软**——
因为文本只在 cross-attention 里被约束，而 self-attention 会跨 10 个 Block 把相邻区域混合。
实测两区各半时分离非常干脆，三区以上且提示词互斥时模型往往会放弃条带结构。

`ADDBASE` 给所有区域一个共同的底稿，按 `Base Ratio` 权重混入，能显著缓解这一点：

```
温暖的室内, 柔和光线 ADDBASE 左边人物 ADDCOL 右边人物
```

对应参数：`usebase=true`（arg 8），`baseratios`（arg 7）默认 `"0.2"`。

`baseratios` 与 ratios 同构、可逐区域给不同权重。**注意 `Rows` 模式下它的分隔符也跟着换**
（`,` 分带、`;` 分单元），这是 RP 自身的不一致，我们照搬了——详见 README。

---

## 七、针对 `F:\anima\koishi` 的逐处对照

目标文件 `src/reforge.ts` 的 `generateImageRegional`。

### 需要改的

**1. 脚本 key**（约 713 行）

```diff
- payload.alwayson_scripts['Regional Prompter'] = {
+ payload.alwayson_scripts['ANIMA Regional Prompter'] = {
```

**2. arg 12 传的是 `[]`**（约 731 行，注释写作 `options`）

RP 那一位是 `nchangeand`（bool）。本扩展完全不读它，传 `[]` 无害，但建议改成 `false`
以免将来有人照着这份代码理解参数含义。

**3. `ratios` 与 `rows` 的一致性校验要收紧**（约 618-641 行）

现有的自动修正只覆盖「值数 == 列数 → 缺行高，补 1」这一种情况。其余不一致会原样传下去，
在 RP 那边被静默适配，在本扩展这边会触发 `RP Error` 且不分区。

建议在发请求前加一道断言：**每带的单元数之和必须等于 `rows` 摊平后的区域数**。

单带（`ratios` 不含 `;`）时，单元数就是逗号分隔的值数；多带时，每带的单元数是该带值数减一
（首个是带高）。

### 不用改的

- ` ADDCOL ` / ` ADDROW ` 的拼接方式（约 670-671 行）——等价，且顺序与我们的 band-major 一致
- `ratios` 的构造格式（工具 schema 里描述的「行高,列宽1,列宽2」）——与我们完全一致
- 其余 18 个参数的取值

### 建议新增

`usebase` + `ADDBASE` 段（见第六节）。你的工具 schema 里已经有 `style` / `artist` / `quality`
这些注入每个区域的全局字段——它们目前是**文本拼接**到每段前面，等价于 `ADDCOMM`。
把「场景/光线/氛围」这类整体设定改走 `ADDBASE`（独立编码、权重混入）通常比拼接更有效，
尤其在区域多的时候。

### 迁移后的自检

改完先跑一次这个，不用出图就能确认参数是否被正确解析：

```bash
curl -s -X POST http://<forge-neo>/sdapi/v1/txt2img -H "Content-Type: application/json" \
  -d @payload.json | python -c "
import json,sys
info=json.loads(json.load(sys.stdin)['info'])['infotexts'][0]
print([p for p in info.split(', ') if p.startswith('RP ')])
"
```

预期看到 `RP Active: True` 与 `RP Divide ratio: ...`；若出现 `RP Error` 说明区域数与段数
不匹配，消息里会写清两边各是多少。
