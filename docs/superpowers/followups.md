# 后续清理项

来自合并前最终审查（2026-07-31）。四条 P0（1 Critical + 3 Important）已在 `27c1e53` 修复；
以下 Minor 项判定为不阻塞合并，留作后续一次性清理。

| # | 位置 | 问题 |
|---|------|------|
| M1 | `lib_anima_rp/layout.py:88-90` vs `176-178` | 「strip + flipper 时 swap_delimiters」这段前置处理写了两遍。两个公开函数**必须**保持一致（修复轮 4 的 Critical 就是它们不一致造成的），却仍靠人工同步。 |
| M2 | `lib_anima_rp/layout.py:108-111` vs `183-186` | `mmode == ROWS → 交换分隔符角色` 这条规则同样写了两遍。同上。 |
| M3 | `scripts/anima_regional_prompter.py:44` | `render_preview` 把异常消息原样插进 `gr.HTML`，而异常消息含用户输入（`layout.py:41` 的 `{text!r}`）。`--listen` / `--share` 下是可注入的 HTML。转义一下即可。 |
| M4 | `scripts/anima_regional_prompter.py:281-294` | `_pad_and_stack` 是纯 torch 逻辑、零 Forge 依赖，却住在唯一那个 Forge 耦合文件里。设计文档 §3 / §5 明确规划了 `lib_anima_rp/conditioning.py` 来放它——该模块根本不存在。代价是它只能靠两处 stub 加载体操才能被测。 |
| M5 | `scripts/anima_regional_prompter.py:39-60` | 同上：`render_preview` 也是纯函数（`layout → HTML`），零 Forge 依赖，却因为住在 `scripts/` 而需要 `tests/test_preview.py:26-78` 那 50 行 stub 加载器才能测。 |
| M6 | `tests/test_preview.py:26-78` 与 `tests/test_process_error_handling.py:41-95` | 同一个 `modules` stub 加载器复制了两份（各约 50 行），逐字近似。 |
| M7 | `tests/test_patch_anima_attention.py:23-95` 与 `tests/test_process_error_handling.py:103-163` | 同一个 `backend` stub 复制了两份，且**已经开始分叉**：前者的 `SelfCrossAttention.forward` 返回 `"orig-forward-result"` 并计数，后者返回 `x`。两份 stub 的 `attention_function` 签名都与真实后端不一致（progress.md 第 60 行已记录）。三套 stub（conftest 的 `modules`、backend、preview 的 `modules`）实测**不互相干扰**，但重复本身是维护负担，应收敛到 `conftest.py`。 |
| M8 | `lib_anima_rp/attention.py:160-171` | `anima_attention_op` 闭包（q/k/v 转置 + 调 `attention_function`）**零自动化覆盖**——`tests/test_patch_anima_attention.py` 的五个测试全都 monkeypatch 掉了 `regional_forward`，从不驱动它。它在每次 ANIMA 生成的热路径上，目前只有端到端跑过。 |
| M9 | `tests/conftest.py:33` | `_load_rp_regions(path, rp_path_explicit)` 的第二个参数在函数体里从未被使用（docstring 还专门描述了它）。 |
| M10 | `tests/test_base_weights.py:67-78` | 注释仍写着「本函数干脆不接受 flipper 参数，从签名上就杜绝了这个可能性」——修复轮 4 之后签名已经**有** `flipper` 了。陈旧注释，会误导后来人。 |
| M11 | `tests/test_base_weights.py:238-258` 与 `tests/test_base_weights_rp_parity.py:152-175` | 长度契约断言写了三遍：`test_flipper_weight_count_matches_layout_region_count`（7 组）、`LENGTH_CASES` 参数化（12 组，含前者 7 组中的 6 组）、以及 `test_matches_rp_weights` 内嵌的第三份。 |
| M12 | `lib_anima_rp/masks.py:51-53` | 用裸 `assert` 承载 Σmask≡1 这个正确性不变量，`python -O` 下被剥掉（progress.md 第 42 行已记录）。目前由 `scatter_` 的构造保证，风险低，但语义上应是 `if not ...: raise`。 |
| M13 | `scripts/anima_regional_prompter.py:163-169` | 非 ANIMA 模型时只有一次控制台警告，**不写任何 PNG info**。其余每一条降级路径（`RP Unsupported`、`RP Error`）都会在产物里留痕，唯独这条不留——用户拿到一张没有任何 RP 键的图，无从判断是「没启用」还是「模型不对」。 |
| M14 | 设计文档 §4.6:251-252 | 签名过时：文档写 `parse_base_weights(baseratios, layout: list[Rect], mmode, flipper)`，实现是 `(baseratios: str, ratios: str, mmode: str, flipper: bool = False)`。第二个参数的类型从一开始就是错的。 |
| M15 | 设计文档 §3:122、§5:268 | 规划了 `lib_anima_rp/conditioning.py`，该文件不存在（见 M4）。计划文档的 File Structure 表也没有它——两份文档互相矛盾。 |
| M16 | 设计文档 §5.3:298-300 | 描述了一个「按 `(tuple(segments), id(p.sd_model))` 缓存」的缓存层，实现里没有（`scripts/anima_regional_prompter.py:186-188` 的注释诚实说明了「由编码时机本身满足」）。文档该改。 |
| M17 | 设计文档 §7.3:420-424 | 依据不成立：文档说「`compute_dtype` 就是 `crossattn_emb` 的 dtype」，实际 `backend/nn/anima.py:291` 是 `compute_dtype = emb_B_T_D.dtype`（timestep embedding 的 dtype，源自 latent，与文本编码器无关）。**实现不受影响**（代码取的是 `x.dtype`，本来就更稳），错的只是论证。 |
| M18 | 设计文档 §8:460 | 「记录警告」与实现不符，见 I3。 |
| M19 | 设计文档 §11:553-560 | 测试文件表只列了 6 个文件，实际 12 个。 |
| M20 | 设计文档 §4.6:262-266 | 「区域数不匹配时抛 ValueError，不静默降级」——`build_config` 确实抛了，但 `scripts/` 在边界处捕获并降级成「不分区 + `RP Error` 留痕」。README:184-192 完整记录了这个最终行为，**设计文档一个字也没提**。文档现在描述的是一个中间态。 |
| M21 | `lib_anima_rp/layout.py:58` | `_spans` 最后一段强制贴 1.0 的那行仍然没有测试锁住（progress.md 第 9 行的遗留项）。我复核后认为它**永远不可能有影响**：token 中心最大是 `1 - 0.5/W`，浮点残差量级 1e-16，两者相差六个数量级；预览的 `:.4f` 也吃不到。建议加注释说明「这是防御性的、不可测」，而不是硬造一个测试。 |

## 建议的处理顺序

1. **M4 / M5 —— 把纯函数移出 `scripts/`**。`_pad_and_stack` 与 `render_preview` 都是零 Forge
   依赖的纯逻辑，却住在唯一那个 Forge 耦合文件里，导致测试要靠约 50 行 stub 加载体操。
   设计文档 §3/§5 本来就规划了 `lib_anima_rp/conditioning.py`。移出去之后 M6 的重复 stub
   自然消失一半。
2. **M6 / M7 —— 把三套 stub 收敛到 `conftest.py`**。目前 `modules` stub 复制了两份、
   `backend` stub 复制了两份且**已经开始分叉**（两份的 `SelfCrossAttention.forward` 返回值不同）。
   实测三套互不干扰，但分叉本身是未来 bug 的温床。
3. **M1 / M2 —— `layout.py` 里两处前置处理去重**。`parse_layout` 与 `parse_base_weights` 各写了
   一份「strip + flipper 时 swap_delimiters」和「`Rows` 交换分隔符角色」，靠人工保持同步——
   而修复轮 4 的那条 Critical 正是它们不一致造成的。
4. **M8 —— 给 `anima_attention_op` 补自动化覆盖**。它在每次 ANIMA 生成的热路径上，
   目前只被端到端跑过，单元测试全部 monkeypatch 掉了 `regional_forward`。
5. **M13 —— 非 ANIMA 模型时也在 PNG info 留痕**。其余每条降级路径都会留痕，唯独这条只有控制台警告。
6. **M3 / M12 —— `render_preview` 的 HTML 转义、`masks.py` 的裸 `assert` 改 `raise`**。
7. **M9 / M10 / M11 / M14–M21 —— 陈旧注释、重复断言、文档与实现的签名/描述同步**。
