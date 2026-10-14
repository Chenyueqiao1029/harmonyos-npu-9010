# 灵枢 · 鸿蒙端侧 NPU 算子实战

**做比赛项目的时候心血来潮，想把大模型塞进手机 NPU 里看看 —— 结果真弄出来了。**

顺手把这一路的东西全记下来：能用的通路、真实的算子支持边界、
一个官方文档里没有的性能发现，**以及我们被自己的实验推翻的每一个错误结论。**

测试机：**华为 Pura 70 Pro+（麒麟 9010）**，HarmonyOS NEXT，普通三方签名、无 root。

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
![Platform](https://img.shields.io/badge/platform-HarmonyOS%20NEXT-green)
![Chip](https://img.shields.io/badge/chip-Kirin%209010-orange)

---

## 一句话结论

**麒麟 9010 的 NPU 对第三方应用**不是"没开放"，而是"开放了一条窄路"**：
HiAI Foundation 的 `.om` 离线模型通路可用，但只支持一小撮算子，
且**算子写法本身会造成约 500 倍的性能差距**。

---

## 三个值得一看的发现

### 发现 1：算子表达形式决定性能量级（约 500 倍）

同一份真实 Qwen2.5-0.5B 权重、同一个输入、同一个序列长度（S=64），
**只把 `MatMul` 改写成数学等价的 `1×1 Conv`**：

| 表达形式 | RunSync 耗时 | 有效吞吐 |
|---|---|---|
| `MatMul(X, W)` | **549 ms** | 0.19 GFLOP/s |
| `Conv1x1(X, W)` | **1 ms** | **≈103 GFLOP/s** |

**两种表达的输出都与 CPU 参考逐元素吻合。**

这解释了为什么官方 SqueezeNet 在 NPU 上很快（它是全卷积网络），
而一整块裸 MatMul 极慢——**这块 NPU 对卷积有原生快路径，对裸 MatMul 走慢路径。**

> 📄 复现：`qproj/build_qproj_real.py` → `convert_qproj.sh` → `verify/verify_qproj_numeric.py`

### 发现 2：真实 Qwen 权重确实在 NPU 上跑对了

取 Qwen2.5-0.5B 第 0 层 `q_proj` 的**真实权重**，加上从完整模型 CPU 前向
**捕获的真实激活**（不是随机数），在本机执行：

```
权重     model.layers.0.self_attn.q_proj.weight  [896,896] float16
输入     捕获自 /model/layers.0/input_layernorm/Mul_1_output_0  [1,8,896] float16
模型     qp_matmul.om  1,606,786 B   FNV-1a64 = 80c9c03ebeb23820

SetModelDeviceOrder(NPU) = SUCCESS
SetFallbackMode(DISABLED) = SUCCESS        ← 这一步是判据的关键
Build   = SUCCESS   32 ms
RunSync = SUCCESS   69 ms

逐元素对账（对比 CPU 参考）：
  最大绝对误差 0.0143189
  平均绝对误差 0.0010786
  2e-2 容差内   7168 / 7168 = 100.00%
  非有限值      0
```

> 📄 详见 [`docs/03-真实-Qwen-层在-NPU-上正确执行.md`](docs/03-真实-Qwen-层在-NPU-上正确执行.md)

### 发现 3：`SetFallbackMode(DISABLED)` 必须显式设，否则你在自欺欺人

**这是最容易踩、也最致命的坑。**

`HIAI_FALLBACK_ENABLED` 是**默认值**。不关掉它，
**不支持的算子会被静默路由到 CPU**，你得到一个"NPU 跑通了"的假象。

```cpp
// 必须显式禁用回落，再 Build —— 否则 Build 成功不代表算子跑在 NPU 上
HiAI_ExecuteDevice order[] = { HIAI_EXECUTE_DEVICE_NPU };
HMS_HiAIOptions_SetModelDeviceOrder(c, order, 1);
HMS_HiAIOptions_SetFallbackMode(c, HIAI_FALLBACK_DISABLED);   // ← 缺这行，结论全废
OH_NNCompilation_Build(c);
```

**我们差点因此把一次 CPU 冒充的结果当成突破。**
在这个仓里，所有"N 跑通了"的结论都要求在 `DISABLED` 下取得。

---

## 算子支持边界（11 个最小探针实测）

用「**每个探针只含一个算子、单一 dtype、显式 axis、不掺 Cast**」的方式重测，
结果与我们早期的判断有重大出入：

| 算子 | 结果 | 算子 | 结果 |
|---|---|---|---|
| `Reshape`（常量 shape） | ✅ 可用 | `ExpandDims` 4D→5D | ❌ 不可用 |
| `Cast`（fp16↔fp32） | ✅ 可用 | `Select`（同形状） | ❌ 不可用 |
| `Gather`（axis=0） | ✅ 可用 | `Cos` | ❌ 不可用 |
| `ExpandDims` 3D→4D | ✅ 可用 | `Sin` | ❌ 不可用 |
| `Conv 1×1` | ✅ 可用 | `BroadcastTo` | ❌ 不可用 |
| | | `LogicalAnd` | ❌ 不可用 |

**完整 Qwen 整图 `.om` 目前仍不能通过设备兼容性检查**（`compatibility=1` / `Build=FAILED`），
设备日志逐条点名上述不可用算子，另有 48 处 `dimCnt>=4` 检查失败。

> 📄 详见 [`docs/02-算子支持边界-11-个最小探针.md`](docs/02-算子支持边界-11-个最小探针.md)

---

## 我们被自己的实验推翻的四处结论

写出来是因为：**能证伪自己的假设，比声称一次成功更接近可信。**

| 错误结论 | 被什么推翻 |
|---|---|
| ① `dimCnt>=4` 是全局硬限制 | 真机四维张量 `d34_mul` 成功运行。**真实含义是「不能从 4 维再扩到 5 维」** |
| ② "11/11 探针独立验证了算子缺失" | `u6_where` 实为形状不匹配（`SelectVerify: the dimnum of x1 and x2 must be the same, but now is 2 1`）；`u4/u5` 混入 Cast 与多形状输入，无法归因 |
| ③ RunSync 中混有首次 kernel 编译开销 | 连续 12 轮实测耗时完全不变（S=8 恒为 67~69 ms，S=64 恒为 548~553 ms） |
| ④ NPU 算错了（`max_abs_err=79.79`） | 我们的参考公式用错：图里是 `MatMul(X, W)`，我们却拿 `X @ W.T` 作参考 |

---

## 不宣称的部分（重要）

- ❌ **不宣称端侧大模型已整体跑在 NPU 上。** 宣称的是：真实 Qwen 的**单个投影层**已在 NPU 上正确执行，通路与验证方法已打通。
- ❌ **不宣称 NPU 带来端到端 LLM 加速。** 端侧逐 token 生成属内存带宽瓶颈（算术强度约 1 FLOP/字节），
  而现行 CPU 方案是 4-bit 量化，每 token 读取字节远少于 fp16。**这是可量化的工程取舍，不是"做不到"。**
- ❌ **不宣称算子边界是普适的。** 上表只适用于本次设备、dtype、shape 与工具链版本。

---

## 环境

| 项 | 值 |
|---|---|
| 设备 | 华为 Pura 70 Pro+（HBN-AL10） |
| 芯片 | 麒麟 9010 |
| 系统 | HarmonyOS NEXT |
| HiAI 版本 | `106.636.120.010` |
| `canIUse` | `HiAIFoundation=true`、`MindSporeLite=true`、**`CANN.LLMEngine=false`** |
| SDK | `compatibleSdkVersion 5.0.1(13)`、`targetSdkVersion 6.1.1(24)` |
| 权限 | 普通三方 debug 签名（**无 root**，uid 2000 shell） |

### 另一条路为什么不通

MindSpore Lite 的 `nnrt` 后端**在本机被 SELinux 拒绝**：

```
avc: denied { get } for service=nnrt_device_service scontext=u:r:debug_hap:s0
```

`RegisterBackend` 失败后 `predict` 静默返回全 0；
继续调原生 `Add` 探针会直接段错误杀进程（实测 `HandleAppDied`，距调用 1.4 s）。

三条候选通路的状态：

| 通路 | 状态 |
|---|---|
| CANN LLM Engine | `canIUse=false`；官方材料写明仅支持 Kirin X90（笔记本芯片） |
| MindSpore Lite + nnrt | SELinux 拒绝，不可用 |
| **HiAI Foundation + `.om`** | ✅ **可用，本仓内容基于此** |

---

## 仓库结构

```
docs/        研究记录（含被推翻的结论、平台插件缺失、omg 不确定性等）
probes/      11 个最小算子探针的构建脚本
qproj/       真实权重子图构建 + 1×1 Conv 改写 + 离线转换脚本
verify/      逐元素数值对账、设备侧结果读取
runner/      HiAI Foundation 执行器（C++ / N-API）+ NNRt 枚举探针
results/     真机实测原始数据（JSON）
```

## 怎么跑

```bash
# 1) 构建算子探针（需要 onnx + numpy）
python probes/build_clean_probes.py

# 2) 用 HiAI DDK 的 omg 转成 .om（DDK 需自行从华为开发者站点获取）
bash qproj/convert_clean_probes.sh

# 3) 推到设备并执行（runner/ 里的 N-API 模块嵌入应用，或参考其调用序列）
hdc file send <xxx.om> /data/local/tmp/
# 应用侧调 probeHiaiOmFromFile()

# 4) 数值对账
python verify/verify_qproj_numeric.py
```

> ⚠️ **本仓不含**：华为 HiAI DDK 工具（专有，请自行从官方获取）、
> 任何模型权重、`libnpu_llama.so`。见 [`NOTICE`](NOTICE)。

---

## 给华为的两条建议

基于这次实测，我们认为有两点值得改进：

1. **发布分机型的支持算子清单。** 现状是开发者只能靠真机逐条探针反推，
   我们为此写了 11 个最小探针才定清边界，期间还因为探针设计不干净得出过错误结论。
2. **把「禁用回落」写进默认最佳实践。** `HIAI_FALLBACK_ENABLED` 是默认值，
   开发者极易得到"NPU 跑通了"的假象。

另可参考 [`docs/09-关于麒麟9020-KirinX90-的调研.md`](docs/09-关于麒麟9020-KirinX90-的调研.md)：
按官方文档，**麒麟 9020（工具链代号 `KirinX90`）与 Ascend 910B/910C 同代**，
有完整的 CANN 平台插件与 AscendC 支持——**与 9010 的处境完全不同**。

---

## 许可

[Apache-2.0](LICENSE)。第三方来源与致谢见 [`NOTICE`](NOTICE)。

本项目是「灵枢 LingShu」民生 AI 助手（光谷杯鸿蒙创新赛参赛作品）的 NPU 研究副产品。
