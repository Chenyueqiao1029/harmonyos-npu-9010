"""Build a set of small probe ONNX models that map HiAI op support on Kirin 9010.

Question: why is the Qwen .om HIAI_COMPATIBILITY_INCOMPATIBLE on this device while
SqueezeNet and tiny MatMul models are compatible?

Two competing explanations:
  (a) OP SET  -- the transformer-oriented ops are unsupported
  (b) SCALE   -- the graph/weights are simply too large

These probes separate them: each is small (< 2 MB) but adds one op group at a
time, so a failure names the op group rather than the size.

  p1 matmul          MatMul                     (known good, control)
  p2 softmax         MatMul + Softmax
  p3 layernorm       MatMul + ReduceMean/Sub/Pow/Sqrt/Div/Mul/Add
  p4 where_expand    MatMul + Expand + Where
  p5 gather          MatMul + Gather + Cast
  p6 tblock          a minimal transformer block (attention shape math)
"""
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT = "/mnt/e/npu_work/hiai-llm-plan/models/ops"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(7)

S = 4          # sequence
H = 8          # hidden
V = 16         # vocab


def vi(name, dt, shape):
    return helper.make_tensor_value_info(name, dt, shape)


def save(nodes, inputs, outputs, inits, path, note):
    g = helper.make_graph(nodes, "probe", inputs, outputs, initializer=inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    onnx.save(m, path)
    print("  %-16s nodes=%-3d %s" % (os.path.basename(path), len(nodes), note))


def f32(shape):
    return rng.standard_normal(shape).astype(np.float32)


# p1: matmul only
w1 = f32((H, H))
save([helper.make_node("MatMul", ["x", "W1"], ["y"])],
     [vi("x", TensorProto.FLOAT, [S, H])], [vi("y", TensorProto.FLOAT, [S, H])],
     [numpy_helper.from_array(w1, "W1")], OUT + "/p1_matmul.onnx", "control")

# p2: + softmax
save([helper.make_node("MatMul", ["x", "W1"], ["mm"]),
      helper.make_node("Softmax", ["mm"], ["y"], axis=-1)],
     [vi("x", TensorProto.FLOAT, [S, H])], [vi("y", TensorProto.FLOAT, [S, H])],
     [numpy_helper.from_array(w1, "W1")], OUT + "/p2_softmax.onnx", "adds Softmax")

# p3: + layer-norm style reduction chain
c = np.ones((H,), dtype=np.float32)
save([helper.make_node("MatMul", ["x", "W1"], ["mm"]),
      helper.make_node("ReduceMean", ["mm"], ["mu"], axes=[-1], keepdims=1),
      helper.make_node("Sub", ["mm", "mu"], ["d"]),
      helper.make_node("Pow", ["d", "two"], ["d2"]),
      helper.make_node("ReduceMean", ["d2"], ["v"], axes=[-1], keepdims=1),
      helper.make_node("Sqrt", ["v"], ["s"]),
      helper.make_node("Div", ["d", "s"], ["n"]),
      helper.make_node("Mul", ["n", "g"], ["ng"]),
      helper.make_node("Add", ["ng", "b"], ["y"])],
     [vi("x", TensorProto.FLOAT, [S, H])], [vi("y", TensorProto.FLOAT, [S, H])],
     [numpy_helper.from_array(w1, "W1"),
      numpy_helper.from_array(np.array(2.0, dtype=np.float32), "two"),
      numpy_helper.from_array(c, "g"), numpy_helper.from_array(c * 0, "b")],
     OUT + "/p3_layernorm.onnx", "adds ReduceMean/Pow/Sqrt/Div")

# p4: + Expand and Where
mask = np.array([[1, 1, 0, 0]], dtype=np.bool_)
save([helper.make_node("MatMul", ["x", "W1"], ["mm"]),
      helper.make_node("Expand", ["mask", "shp"], ["m2"]),
      helper.make_node("Where", ["m2", "mm", "zero"], ["y"])],
     [vi("x", TensorProto.FLOAT, [S, H])], [vi("y", TensorProto.FLOAT, [S, H])],
     [numpy_helper.from_array(w1, "W1"),
      numpy_helper.from_array(mask, "mask"),
      numpy_helper.from_array(np.array([S, H], dtype=np.int64), "shp"),
      numpy_helper.from_array(np.zeros((S, H), dtype=np.float32), "zero")],
     OUT + "/p4_where_expand.onnx", "adds Expand/Where")

# p5: + Gather and Cast
emb = f32((V, H))
idx = np.array([[1, 2, 3, 4]], dtype=np.int64)
save([helper.make_node("Gather", ["emb", "idx"], ["g"], axis=0),
      helper.make_node("Cast", ["g"], ["y"], to=TensorProto.FLOAT)],
     [vi("x", TensorProto.FLOAT, [1, S])], [vi("y", TensorProto.FLOAT, [1, S, H])],
     [numpy_helper.from_array(emb, "emb"), numpy_helper.from_array(idx, "idx")],
     OUT + "/p5_gather.onnx", "adds Gather/Cast")

# p6: minimal transformer block shape math (attention without KV cache)
wq, wk, wv, wo = (f32((H, H)) for _ in range(4))
save([helper.make_node("MatMul", ["x", "Wq"], ["q"]),
      helper.make_node("MatMul", ["x", "Wk"], ["k"]),
      helper.make_node("MatMul", ["x", "Wv"], ["v"]),
      helper.make_node("Transpose", ["k"], ["kt"], perm=[1, 0]),
      helper.make_node("MatMul", ["q", "kt"], ["s"]),
      helper.make_node("Div", ["s", "scale"], ["ss"]),
      helper.make_node("Softmax", ["ss"], ["a"], axis=-1),
      helper.make_node("MatMul", ["a", "v"], ["o"]),
      helper.make_node("MatMul", ["o", "Wo"], ["y"])],
     [vi("x", TensorProto.FLOAT, [S, H])], [vi("y", TensorProto.FLOAT, [S, H])],
     [numpy_helper.from_array(wq, "Wq"), numpy_helper.from_array(wk, "Wk"),
      numpy_helper.from_array(wv, "Wv"), numpy_helper.from_array(wo, "Wo"),
      numpy_helper.from_array(np.array(2.0, dtype=np.float32), "scale")],
     OUT + "/p6_tblock.onnx", "minimal attention block")

print("OPS PROBE BUILD DONE")
