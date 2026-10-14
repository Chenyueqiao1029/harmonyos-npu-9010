# 真实 Qwen 计算在 Kirin 9010 NPU 上执行 —— 实测记录与边界

> 日期：2026-10-13
> 设备：HUAWEI Pura 70 Pro+（HBN-AL10），Kirin 9010，HiAI `106.636.120.010`
> 起因：`gpt-6-astra` 独立复审指出我此前"9010 不可能跑任何 transformer LLM"的结论**过度外推**，
> 并给出了一个可执行的最小实验。本文是该实验的记录。

---

## 一、先认错：我之前的三个过度外推（已核实，全部成立）

| # | 我此前的说法 | 反证 | 判定 |
|---|---|---|---|
| 1 | `dimCnt >= 4` 是全局硬限制，任何 LLM 都不可能 | `d34_mul` 真机：**四维张量 `1,8,4,1`** `Build=SUCCESS 4ms`、`RunSync=SUCCESS 1ms`、`OUTPUT n=32 sum=22.25` 非零 | **我错** |
| 2 | "11/11 独立验证算子缺失" | `u4_less`/`u5_and` 混入 Cast 与多形状；`u6_where` 的真实原因是 `SelectVerify::"the dimnum of x1 and x2 must be the same, but now is 2 1"` —— **形状不匹配，不是算子不支持** | **我错** |
| 3 | 用 `atc --framework` 无 Paddle 来反驳 Paddle-Lite | `atc --framework` 是**转换器接受的输入格式**，不是"能调底层运行时的框架"；Paddle-Lite 有自己的 `NNAdapter → HiAI IR` 路径 | **我的反驳本身错** |

**结论修正**：应说「**当前完整 Qwen `.om` 在当前环境下不能执行**」，
**不能**说「HiAI Foundation 在 Kirin 9010 上不可能执行任何 transformer LLM」。

---

## 二、按复审建议做的实验：真实权重投影子图

**设计要点**（复审明确要求，避免"随机矩阵证明一切"）：
- `Wq`、`bq` 取自**真实 Qwen2.5-0.5B 权重**（从我们已有的 fp16 ONNX initializer 提取）
- `X` 取自**完整模型 CPU 前向的真实激活**，捕获点
  `/model/layers.0/input_layernorm/Mul_1_output_0`（正是喂给 `q_proj` 的 RMSNorm 输出）

**产物**：

| 项 | 值 |
|---|---|
| 权重 `W` | `[896, 896]` float16 —— 真实 `model.layers.0.self_attn.q_proj` |
| 偏置 `b` | `[896]` float16 |
| 真实激活 `X` | `[1, 8, 896]` float16，14336 B |
| CPU 参考 | `n=7168`，`sum=-1099.108154`，`max=47.832813`，`min=-79.241570` |
| `.om` | `qp_matmul.om` = **1,606,786 B**；`qproj_subgraph.om`（含 bias）= 1,608,903 B |
| `.om` FNV-1a64 | `80c9c03ebeb23820` |

**CPU 参考前 8 个值**：`-0.226348, 0.293741, -0.036812, 0.215967, -14.820379, 0.408097, 0.449141, 0.038552`

---

## 三、真机结果：**NPU 成功编译并执行了这一层**

```
model_bytes    = 1606786              <- 真实 q_proj 投影
model_fnv1a64  = 80c9c03ebeb23820     <- 身份核对一致
compatibility  = 0
SetModelDeviceOrder(NPU) = SUCCESS
SetFallbackMode(DISABLED) = SUCCESS   <- 关键：显式禁用回落，排除 CPU 冒充
Build          = SUCCESS 23ms
dtype          = 10 (FLOAT16)  shape = 1,1,8,896
RunSync        = SUCCESS 70ms
OUTPUT=0 n=7168 nonfinite=0 sum=-57.083572
VALUES=0.595703125,1.26171875,0.687011719,-0.92578125,...
compatibility_after=0
```

**`SetFallbackMode(DISABLED)` 下 `Build` 与 `RunSync` 均成功** —— 这是本项目第一次做到
**真实 Qwen 权重 + 真实激活形状在 Kirin 9010 NPU 上被执行**。

**顺带再次印证纠正 1**：设备报的形状是 **`1,1,8,896` 四维**，**四维确实能跑**。

---

## 四、逐元素数值对账：**通过**

### 4.1 中途踩的三个坑（都已定位并绕过）

**坑 1：`hdc file send -b` 写出的文件 SELinux 标签未分类**

```
hiai-input.bin   u:object_r:debug_hap_data_file:s0                 <- 无类别集，应用读不了
应用能读的必须是  o:object_r:debug_hap_data_file:s0:x71,x335,x512,x868,x1024
```

`u:` = unclassified。**`hdc -b` 的标签行为不稳定**（同一命令有时标签又对），
所以整条通路**不能依赖 hdc 推文件**。

**绕过**：模型与输入改为**跟随 HAP 一起分发（rawfile）**，由**应用自己**读出并写进 `filesDir`
—— 应用写的文件标签天然正确。落地日志：

```
[LingshuNpu] 落地 rawfile：model=1606786B input=14336B
[LingshuNpu] 选择文件已写：.../hiai-review-selection.txt -> .../hiai-selected.om
```

**坑 2：原生侧的"触发字面量"**

`hiai_om_napi.cpp` 用
`if (selected == "/data/local/tmp/hiai-review-selection.txt")` 作为**触发开关**：
只有匹配它才会去读沙盒里的选择文件。传沙盒路径不匹配，`selected` 不被替换，
**会把 63 字节的选择文件当模型加载**（实测 `model_bytes=63`、`Build=FAILED`）。

**坑 3：我自己的参考公式错了**（见 4.3）

### 4.2 最终真机结果（喂入真实 `X`）

```
selected_path = .../hiai-selected.om
model_bytes   = 1606786
compatibility = 0
SetFallbackMode(DISABLED) = SUCCESS
Build         = SUCCESS 32ms
dtype         = 10 (FLOAT16)   shape = 1,1,8,896
INPUT_FROM_FILE   = 14336bytes          <- 真实激活喂进去了
RunSync           = SUCCESS 69ms
OUTPUT=0 n=7168 nonfinite=0 sum=45.934439
OUTPUT_WRITTEN_TO_FILE = 7168elements
```

### 4.3 对账：**PASS**

**我最初用 `X @ Wqᵀ` 作参考，结果 `max_abs_err=79.79`、`within2e-2=1.41%`。**
但图里构造的是 `MatMul(X, Wq)`（与原始 Qwen ONNX 图一致，**不转置**）——
**两个不同公式，当然对不上。错在我，不在 NPU。**

改用与图一致的参考后：

| 参考表达式 | sum | max_abs_err | 2e-2 内占比 |
|---|---|---|---|
| **`X @ Wq`（图的真实语义）** | **45.8913** | **0.0143189** | **1.0000** ← MATCH |
| `X @ Wqᵀ`（我原来的错误参考） | -9.6137 | 9.52487 | 0.0363 |

**NPU 输出 `sum = 45.934441`，与 `X @ Wq` 仅差 0.043（fp16 累积误差）。**

**逐元素（first8）**：
```
NPU   : [ 0.009537,  0.102051,  0.004669, -0.549805, -0.813477,  0.262207, -0.015015,  0.392578]
X @ Wq: [ 0.009620,  0.102032,  0.004645, -0.550131, -0.816367,  0.262253, -0.014601,  0.392841]
```

**7168 个元素 100% 落在 2e-2 容差内，最大绝对误差 0.0143，无 NaN/Inf。**

> **结论：Kirin 9010 的 NPU 用真实 Qwen2.5-0.5B 权重，正确计算了 `layers.0.self_attn.q_proj`
> 投影，且在显式禁用回落的条件下完成。**

**注**：`qp_matmul.onnx` 不含偏置（`bq` 未加入该变体），故 `+b` 与不加 `b` 结果相同；
含偏置的变体 `qproj_subgraph.om`（1,608,903 B）已转好，尚未单独测。

---

## 五、当前该怎么说（三条都能站住）

| 说法 | 状态 |
|---|---|
| 端侧 HiAI Foundation 通路可用，NPU 计算真实 | ✅ 官方 SqueezeNet：NPU `RunSync 2ms` vs CPU `10ms` |
| 完整 Qwen `.om` 在当前环境下不能整图执行 | ✅ `compatibility=1`、`Build=FAILED`（禁用回落） |
| **真实 Qwen 的单个投影层可在 NPU 上编译并执行** | ✅ **`Build=SUCCESS 23ms`、`RunSync=SUCCESS 70ms`** |
| 该层在 NPU 上**算得对** | ⏳ **待逐元素对账**（被沙盒回归阻塞） |
| 全模型在 NPU 上推理 | ❌ 未实现，也无方案 |
| NPU 带来的 token 吞吐提升 | ❌ **无数据**。端侧 CPU 实测 1.5B ≈18–20 tok/s、0.5B ≈22 tok/s |

**不能说**："NPU 让 LLM 提速 X 倍"。**没有这个数据。**

---

## 六、下一步（解锁即做）

1. **定位沙盒读取回归**：重装 HAP 后原生侧无法打开
   `/data/storage/el2/base/haps/entry/files/hiai-review-selection.txt`。
   可行绕法：把选择文件内容改为指向 `/data/local/tmp` 之外的**应用 filesDir 绝对路径**，
   或改用 HAP rawfile 通道（即 `test_device.py --packaged`，但需把 `.om` 打进 HAP）。
2. **跑完逐元素对账**：输入、输出文件入口已写好并编入 `.so`，通道一通即可得结论。
3. **若对账通过**：按复审设计接回完整 CPU 推理，替换该层结果，跑一次真实 next-token 生成。
   **届时能说的是**「真实 Qwen 的实际生成过程使用了本机 NPU 执行模型计算」，
   **仍不能说**全模型在 NPU，也不能说加速。

---

## 七、材料位置

| 内容 | 路径 |
|---|---|
| 真实子图构建脚本 | `<workdir>\build_qproj_real.py` |
| 转换脚本 | `<workdir>\convert_qproj.sh` |
| 真机验证脚本 | `<workdir>\verify_qproj_numeric.py` |
| 子图与权重 | `<workdir>\qproj-real\` |
| 真机日志 | `<workdir>\qproj-real\logs\` |
| 第三方复审全文（astra） | `<workdir>\codex-astra-review-answer.md` |
| 复审 brief | `<workdir>\codex-astra-review-brief.md` |
