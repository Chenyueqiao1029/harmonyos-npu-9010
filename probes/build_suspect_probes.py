"""Probe the Qwen operators nobody has tested yet.

Qwen's audited op inventory (fixed graph, 3028 nodes) contains operators that are
unusual for a CNN-oriented NPU and that the other session's probes (c/d/axis
series) never covered:

    Cos 1, Sin 1        -> RoPE (rotary position embedding)
    And 1               -> boolean logic
    Less 24             -> comparison
    Neg 48              -> negation
    Where 48            -> select
    Expand 121          -> dynamic broadcast
    Constant 1119       -> shape arithmetic
    Sigmoid 24, Softmax 24

Cos/Sin/And are prime suspects: a high-performance NPU rarely implements
transcendental or boolean ops, which would be exactly why an LLM graph is
rejected while CNN/attention probes pass.

Each probe is kept minimal -- a MatMul baseline plus ONE suspect op -- so a
failure names that op. Build here; conversion is done by the companion shell
script, and device testing waits until the phone is free.
"""
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT = "/mnt/e/npu_work/hiai-llm-plan/models/suspect"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(31)
S, H = 4, 8
W = rng.standard_normal((H, H)).astype(np.float32)


def save(nodes, path, note, extra=None, out_dtype=TensorProto.FLOAT, out_shape=None):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [S, H])
    y = helper.make_tensor_value_info("y", out_dtype, out_shape or [S, H])
    inits = [numpy_helper.from_array(W, "W1")]
    if extra:
        inits += extra
    g = helper.make_graph(nodes, "probe", [x], [y], initializer=inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    onnx.save(m, path)
    print("  %-24s nodes=%-3d %s" % (os.path.basename(path), len(nodes), note))


def base():
    return [helper.make_node("MatMul", ["x", "W1"], ["mm"])]


# u1 Cos  (RoPE)
save(base() + [helper.make_node("Cos", ["mm"], ["y"])],
     OUT + "/u1_cos.onnx", "Cos (RoPE)")

# u2 Sin  (RoPE)
save(base() + [helper.make_node("Sin", ["mm"], ["y"])],
     OUT + "/u2_sin.onnx", "Sin (RoPE)")

# u3 Neg
save(base() + [helper.make_node("Neg", ["mm"], ["y"])],
     OUT + "/u3_neg.onnx", "Neg")

# u4 Less (comparison -> bool), then Cast back to float so the graph output stays float
save(base() + [helper.make_node("Less", ["mm", "zero"], ["lt"]),
               helper.make_node("Cast", ["lt"], ["y"], to=TensorProto.FLOAT)],
     OUT + "/u4_less.onnx", "Less -> Cast(bool->float)",
     [numpy_helper.from_array(np.zeros((H,), dtype=np.float32), "zero")])

# u5 And (boolean) on Less results
save(base() + [helper.make_node("Less", ["mm", "zero"], ["a"]),
               helper.make_node("Less", ["zero", "mm"], ["b"]),
               helper.make_node("And", ["a", "b"], ["and_out"]),
               helper.make_node("Cast", ["and_out"], ["y"], to=TensorProto.FLOAT)],
     OUT + "/u5_and.onnx", "And on two bool tensors",
     [numpy_helper.from_array(np.zeros((H,), dtype=np.float32), "zero")])

# u6 Where (select)
save(base() + [helper.make_node("Less", ["mm", "zero"], ["cond"]),
               helper.make_node("Where", ["cond", "mm", "zero"], ["y"])],
     OUT + "/u6_where.onnx", "Where (select)",
     [numpy_helper.from_array(np.zeros((H,), dtype=np.float32), "zero")])

# u7 Expand
save(base() + [helper.make_node("Expand", ["mm", "shp"], ["y"])],
     OUT + "/u7_expand.onnx", "Expand to a static shape",
     [numpy_helper.from_array(np.array([S, H], dtype=np.int64), "shp")])

# u8 Flatten
save(base() + [helper.make_node("Flatten", ["mm"], ["y"], axis=0)],
     OUT + "/u8_flatten.onnx", "Flatten", out_shape=[S * H])

# u9 Sigmoid
save(base() + [helper.make_node("Sigmoid", ["mm"], ["y"])],
     OUT + "/u9_sigmoid.onnx", "Sigmoid")

# u10 Softmax
save(base() + [helper.make_node("Softmax", ["mm"], ["y"], axis=-1)],
     OUT + "/u10_softmax.onnx", "Softmax")

# u11 Slice
save(base() + [helper.make_node("Slice", ["mm", "st", "en", "ax"], ["y"])],
     OUT + "/u11_slice.onnx", "Slice",
     [numpy_helper.from_array(np.array([0], dtype=np.int64), "st"),
      numpy_helper.from_array(np.array([4], dtype=np.int64), "en"),
      numpy_helper.from_array(np.array([1], dtype=np.int64), "ax")])

print("SUSPECT PROBE BUILD DONE")
