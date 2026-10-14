# 发现：`omg` 的输出是非确定性的（同一 ONNX，每次字节不同）

记录时间：2026-10-13 21:0x
记录人：Lead
状态：**已实测确认**

## 实测数据

同一份 `models/ops/r1_mul_bcast.onnx`，**同一参数、同一目录、连续转换 4 次**：

```
run1  2832 B  093f3df284bbee3d19b3b686aea8dc966c511361ba38dbb4
run2  2832 B  2207490ce36430b1618539945d0faea0c66823341258f000
run3  2832 B  0f1cf22688cef06ed105a032bdb24af055478bf3498a8d8e
run4  2832 B  776052cf86e68d5dc9eec88ab58adadfd0e513ae6e530ae9
distinct hashes: 4
```

对 `r3_div_tensor` 与 `r4_mul_flat` 各转 3 次，同样是 **3/3 哈希各不相同**（字节数完全一致）。

## 差异在哪里

`cmp -l r1_1.om r1_2.om`：**916 个字节不同**，从 offset 371 起。
把差异区的八进制解成字符，出现的是 **`index`** 与 **`version`** 两个字符串：

```
371  13  15        (\v  vs  \r)
374 151 166       i  v
375 156 145       n  e
376 144 162       d  r
377 145 163       e  s
378 170 151       x  i
379  22 157       ... o
380   2 156       ... n
```

→ 差异落在**元数据/标识区**（疑似内部索引与版本/时间戳字段），不是计算图主体。

## 这带来三个后果

1. **字节哈希不能用来判断"两次转换是不是同一个模型"。**
   我此前若用哈希做等价性判断，结论会是错的。**必须改为按"真机兼容性判定"比对。**

2. **排查"缺平台插件"那条线索时曾被误导。**
   我装了 kirin9020/kirinx90/kirin9030 三个官方平台插件后重转，发现哈希变了，
   一度以为是插件起了作用。现在看**更可能是非确定性本身**造成的。
   （旁证：装了插件后 `libcustom_op.so` **仍然报缺失**，说明插件压根没被 `omg` 用上。）

3. **可能解释我此前探针结果的自相矛盾。**
   `r1_mul_bcast`/`r2_add_bcast` 判为兼容，而结构几乎相同的
   `r3_div_tensor`/`r4_mul_flat` 判为不兼容 —— 如果每次转换的产物在
   **元数据层**不同、而设备侧的兼容性判定又恰好读了那部分，
   那结论就会随转换批次漂移。

## 尚未验证的关键问题

**同一个 ONNX 转出的多份 `.om`，真机 `compatibility` 判定是否一致？**

- 若**一致** → 判定是"图的属性"，我此前那批探针结论**仍然可信**，r1/r2 与 r3/r4 的差异另有原因
- 若**不一致** → 判定受非确定性影响，**我此前整批探针结论作废**，必须改为"同一模型多次转换取多数票"

**实验已就绪但未执行**（原因见下）：
`om/determinism/` 下已备好 `r1_{1,2,3}.om`、`r3_{1,2,3}.om`、`r4_{1,2,3}.om`；
把它们放进 rawfile 并让 App 逐个测即可判定。

## 阻塞：写冲突

**App.uvue 目前由 codex 会话在改**（它加了一行
`Logger.info('[App] COMPAT_REVIEW ' + probeHiaiOmFromFile('/data/local/tmp/hiai-review-selection.txt'))`，
用"选择文件"控制测哪些模型，一次编译多次复用）。

我原计划往 App.uvue 里加 `probe_det_*` 的名单，几次"锚点未匹配"就是两边同时写同一文件造成的。
**我已停止写 App.uvue，把该文件完全让给 codex**，避免互相破坏。

**因此上述关键实验交由 codex 执行**，或等它写完 App.uvue 后我再补测。

## 对结论的影响（诚实说明）

- 本发现**不推翻**已确证的真机结果：官方 SqueezeNet 与 codex 的 tiny_conv 都实测
  `compatibility=0` + `Build=SUCCESS` + `RunSync` 输出非零合理 —— 那是**单次产物**上的直接观测。
- 本发现**削弱**的是：**从小探针黑箱反推"哪个算子不兼容"这类结论的可靠性**。
  在完成"多份产物取多数票"的实验之前，不应把 r/q 系列的算子归因当作定论。

---

## 【已解决】2026-10-13 21:1x 更新 —— 担心被证否，且规律已找到

codex 会话用 b/c/d 多轮重复测同一批模型，结论：

```
models with MORE THAN ONE distinct compatibility value:
  (none -- every model gave a consistent compatibility across its rounds)
```

**每个模型在所有轮次里 compatibility 完全一致**（c00_sqrt 两轮都是 0、c03_div_row 两轮都是 1、
c10_ln_div 两轮都是 0、old_r1_mul_bcast 两轮都是 0、old_r4_mul_flat 两轮都是 1）。

**→ `omg` 在元数据层的非确定性，不影响设备侧的 compatibility 判定。判定是稳定的图属性。**
我上一节担心的"整批探针结论作废"**不成立**。

## 真正的原因（codex 探针系列给出）

逐元素算子的**第二操作数形态**决定兼容性：

| 探针 | 第二操作数 | compatibility |
|---|---|---|
| c05_div_scalar | 标量 | **0** |
| c06_mul_scalar | 标量 | **0** |
| c01_mul_vec | 向量（可广播） | **0** |
| c02_add_vec | 向量（可广播） | **0** |
| c07_mul_row | 行向量 | **1** |
| c03_div_row | 行向量 | **1** |
| c04_mul_full | 完整 shape | **1** |
| c08_add_full | 完整 shape | **1** |
| c09_div_full | 完整 shape | **1** |

**→ 逐元素算子（Mul/Add/Div）：第二操作数为标量或可广播 = 兼容；
为行向量或与主操作数同形 = 不兼容。**

这解释了我此前 `r1_mul_bcast`(g 形状 (8,)) 判 0、而 `r4_mul_flat`(g2 形状 (4,8)) 判 1 的矛盾。

## 对改写 Qwen 的意义

Qwen 图里凡是"与主操作数同形的逐元素 Mul/Add/Div"都是候选改写点 ——
若能把它们改写成广播形式（或标量形式），整图有机会从 compatibility=1 变成 0。
**这是当前最有希望的路线。**

## 方法论教训（记下来）

**codex 的验证比我严格**：它除了 compatibility，还记 `oracle_pass` 与 `max_abs_error`，
即拿 NPU 输出与参考值对账（d21_mul err=0.00041、d30_mul err=0.0、d32_div err=0.0059）。
**"兼容 + Build 成功"仍不等于"算对了"，必须有数值对账。** 这一条我此前只在日志里看了眼"非零合理"，
不够。后续一律照 codex 的口径来。
