"""Is graph SCALE the reason Qwen is rejected? Test it with synthetic graphs.

Ruled out so far:
  - opset 18                     (converted fine)
  - symbolic dims                (converted fine)
  - Identity nodes               (fails, but Qwen has none)
  - empty-shape ConstantOfShape  (was the parse blocker; codex fixed it)
  - element-wise ops             (all 582 in Qwen are runtime x runtime, a category
                                  the device accepts with oracle_pass=True)
  - LLM-specific ops             (attention block p6_tblock is compatible)

Remaining candidates include pure SCALE: ~3028 nodes, 290 initializers, 980 MB.
The other session is probing shape ranks and input signs, not scale, so test it here.

Method: build synthetic graphs that repeat a benign, known-compatible pattern
(MatMul -> Add -> Relu) N times, for several N, and let the device judge each.
Because the pattern itself is compatible at N=1, any flip to incompatible names
scale as the cause and gives the threshold.

  s001 -> 1 repetition
  s050 -> 50
  s200 -> 200
  s600 -> 600
  s1200 -> 1200
  s2400 -> 2400
"""
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT = "/mnt/e/npu_work/hiai-llm-plan/models/scale"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(23)

D = 32          # hidden dim
S = 8           # sequence


def build(reps, path):
    nodes = []
    inits = []
    prev = "x"
    for i in range(reps):
        w = "W%d" % i
        inits.append(numpy_helper.from_array(
            rng.standard_normal((D, D)).astype(np.float32) * 0.1, w))
        mm = "mm%d" % i
        ar = "ar%d" % i
        rl = "rl%d" % i
        nodes.append(helper.make_node("MatMul", [prev, w], [mm]))
        nodes.append(helper.make_node("Add", [mm, "bias"], [ar]))
        nodes.append(helper.make_node("Relu", [ar], [rl]))
        prev = rl
    inits.append(numpy_helper.from_array(
        rng.standard_normal((D,)).astype(np.float32) * 0.01, "bias"))

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [S, D])
    y = helper.make_tensor_value_info(prev, TensorProto.FLOAT, [S, D])
    g = helper.make_graph(nodes, "scale", [x], [y], initializer=inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    onnx.save(m, path)
    print("  %-14s reps=%-5d nodes=%-5d inits=%-5d" % (
        os.path.basename(path), reps, len(nodes), len(inits)))


for reps in (1, 50, 200, 600, 1200, 2400):
    build(reps, "%s/s%04d.onnx" % (OUT, reps))

print("SCALE PROBE BUILD DONE")
