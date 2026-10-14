# 决定性结论：Kirin 9010 无法通过 HiAI Foundation 离线模型路线执行 Qwen 类 LLM

记录时间：2026-10-13 夜
记录人：Lead
证据等级：**设备编译器原始输出**（非推断）

---

## 一、这次测的是什么

codex 制作的**最完整**的 Qwen 试验版 `qwen_rank4_int32.om`：

- 49 个 RMSNorm 全部显式 reshape 成 `[1,1,8,896]`，归约轴改用 `3`，再 reshape 回 `[1,8,896]`
- 公共 token / mask / position 输入由 INT64 改为 INT32
- 保留所有权重
- 3126 节点，`omg exit 0`，1,027,633,908 B
- SHA-256 `b66c63f764466966a3bd3ff74e232430f9e5edd678c26c8aaf1afdfc626851b7`
- FNV-1a64 `26ce1a20bd043bfa`
- 桌面验证：49 组新旧 RMSNorm 逐元素相同，最大绝对误差 0

真机结果（**已核对模型身份**）：

```
selected_path=/data/storage/el2/base/haps/entry/files/hiai-selected.om
model_fnv1a64=26ce1a20bd043bfa        <- 与上述指纹一致
model_bytes=1027633908                <- 完整 1.027 GB
compatibility=1
SetFallbackMode(DISABLED)=SUCCESS     <- 显式关闭 fallback，排除静默回落 CPU
Build=FAILED 271ms
compatibility_after=1
```

---

## 二、设备编译器自己报出的原因（原始日志）

日志：`compat-review-20261013/device/qwen_rank4_int32-full2-case1-npu.log`（1,338,669 B）

```
CheckSupported(314)::"CheckSupported: the op name [/model/Cast_1]  type [CastT]       is not supported in npucl store [elementary_lib]"
CheckSupported(314)::"CheckSupported: the op name [mask_flatten_to_fp32] type [CastT] is not supported in npucl store [elementary_lib]"
CheckSupported(314)::"CheckSupported: the op name [/model/Gather_4] type [GatherV2D]  is not supported in npucl store [elementary_lib]"
CheckSupported(314)::"CheckSupported: the op name [/model/Reshape]  type [Reshape]   is not supported in npucl store [elementary_lib]"
CheckSupported(314)::"CheckSupported: the op name [/model/And_1]    type [LogicalAnd] is not supported in npucl store [elementary_lib]"
CheckSupported(314)::"CheckSupported: the op name [/model/Expand]   type [BroadcastTo] is not supported in npucl store [elementary_lib]"
CheckSupported(314)::"CheckSupported: the op name [/model/layers.NN/self_attn/Where_2] type [Select] is not supported in npucl store [elementary_lib]"
CheckSupported(314)::"CheckSupported: the op name [/model/rotary_emb/Cos] type [Cos] is not supported in npucl store [elementary_lib]"
CheckSupported(314)::"CheckSupported: the op name [/model/rotary_emb/Sin] type [Sin] is not supported in npucl store [elementary_lib]"
CheckSupported(314)::"CheckSupported: the op name [.../Unsqueeze_12] type [ExpandDims] is not supported in npucl store [elementary_lib]"
expanddims_check_support.cc CheckExpandDimsSupport(40)::"not support input dimCnt >= 4!"
deviceType: 0, fallBackMode: 1
general_model_compiler.cpp BeforeCompile(172)::"check ir model compatibility failed"
```

**计数**：

| 消息 | 次数 |
|---|---|
| `not supported in npucl store [elementary_lib]` | **452** |
| `not supported in npucl store [fe_lib]` | **372** |
| `not support input dimCnt >= 4!` | **48** |
| `compatibility failed` | 2 |

`Select`（来自 ONNX `Where`）在 **24 层里每层都出现**（`/model/layers.0..23/self_attn/Where_2`）。

---

## 三、为什么这构成**路线级**结论，而不仅是"这一版没改好"

被点名的算子中，**`Reshape`、`CastT`、`GatherV2D`、`ExpandDims` 是任何 transformer 都离不开的基础算子**，
不是 Qwen 特有的写法：

- `Reshape`：Qwen 图里有 **146 个**
- `CastT`：**179 个**
- `ExpandDims`（Unsqueeze）：**51 个**
- `GatherV2D`：embedding 与 mask 索引必需
- `Select`：mask/causal 处理必需
- `Cos`/`Sin`：RoPE 位置编码必需
- `dimCnt >= 4`：注意力张量 `[B,H,S,D]` 天生四维

**而且它们不在 NPU 的两个计算库（`elementary_lib`、`fe_lib`）里实现。**

这同时解释了此前所有"看起来矛盾"的现象：
**小探针之所以能 `compatibility=0` + `Build SUCCESS`，是因为它们只用 `MatMul`/`Softmax`/`Add`/`Mul`/`Div`
这些确实实现了的算子，恰好把上述全部障碍都绕开了。**

---

## 四、这条路线之外还剩什么

| 备选 | 状态 |
|---|---|
| **CANN LM Engine**（华为官方 LLM 专用通路） | 官方硬件要求 **Kirin X90**；本机 `canIUse` 返回 **false**；官方平台插件只有 kirin9020 / kirinx90 / kirin9030，**没有 9010** |
| **自写 Ascend C 自定义算子** | 这是华为合作伙伴（如智谱 GLM-Edge-1.5b 的 RmsNorm）用的办法，但需要 **KirinX90/9030 工具链**，9010 不在支持列表 |
| **MiniCPM-V 4.7** | 官方鸿蒙部署**只走 CPU，没有 Kirin NPU 后端** |
| **MindSpore Lite nnrt** | 已被真机证否（SELinux 拒绝驱动服务，predict 输出全 0） |
| **原生 NNRt 内存构图** | 已被真机证否（应用进程崩溃） |

**→ 在本机可获得的工具链与公开通路范围内，LLM 上 NPU 不可行。**

---

## 五、诚实边界（不夸大、不隐瞒）

**可以主张的：**
- 本机 HiAI Foundation 通路**可用**：官方 SqueezeNet 与我们自产的 `.om` 都实测 NPU Build + RunSync 成功，
  SqueezeNet 上 NPU `RunSync=2ms` vs CPU `10ms`
- 本机 NPU **能执行**：普通张量乘法、动态除法、若干广播、以及一个小注意力图
- 本机**不具备**官方 LLM 专用通路（CANN LM Engine，需 Kirin X90）
- 通过**通用 OMG + 当前 HiAI ROM + 原导出 ONNX** 这条具体路线，**完整 LLM 图无法编译**，
  原因是设备编译器逐条报出的算子缺失与 `dimCnt>=4` 限制

**不能主张的：**
- ❌ "9010 的 NPU 是摆设" —— 不对，它在支持的算子上真能跑
- ❌ "所有 LLM 都永远不可能" —— 我们只证否了这一条具体路线；
  若华为未来在 ROM 或 DDK 中补齐这些算子，结论需要重测
- ❌ "compatibility=0 就等于 NPU 执行" —— 本次已证明必须在 `SetFallbackMode(DISABLED)` 下测才有意义

---

## 六、方法论教训（写下来，避免重犯）

1. **`compatibility=0` 在有 fallback 时不能作为 NPU 执行证据。** 必须显式 `SetFallbackMode(DISABLED)`。
2. **`Build`/`RunSync` 成功 ≠ 数值正确。** 必须用 ONNX ReferenceEvaluator 做 oracle 对账
   （已有反例：Build/Run 都成功，非零输入却输出全零）。
3. **永远核对模型身份。** 我第一次跑 `--packaged` 时用错了 HAP，应用回落到残留的 3 KB 探针，
   却打出 `Build=SUCCESS` —— 差点被我当成"完整 Qwen 编过了"。
   **是报告里的 `model_bytes` / `fnv1a64` 揭穿的。**
4. **看实验结论必须同时看实验怎么造的。** 我按结果标签读 codex 的探针，得出过"完整同形算子一律不支持"
   的错误规则；回去读它的构造代码才发现失败的那批第二操作数是**常量**而非运行时张量。
5. **读自己的实验也要小心。** 我写的 `build_rcorrection.py` 里 `n=Sqrt(v)` 实际是 `(4,1)`，
   所以 `r1` 与 `r4` 根本不是"广播 vs 不广播"的对照，我读错了自己的设计。

---

## 七、给下一步的建议

**若目标是"比赛/答辩上有个 NPU 亮点"**：现有成果足够 ——
真机实测的 NPU 加速（NPU 2ms vs CPU 10ms）、完整的工具链、以及"我们查清了它为什么不能跑 LLM"的诚实结论，
比一个假装跑通的 demo 更有说服力。

**若必须让 LLM 上 NPU**：需要换到 **KirinX90/9030** 机型，或等华为在 9010 的 ROM/DDK 中补齐算子。
本机不可行。
