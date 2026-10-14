"""Variant: q_proj matmul with the weight as a RUNTIME INPUT instead of a graph constant.

Motivation. Same device and same API:
  official SqueezeNet  ~0.7 GFLOP in   2 ms  -> ~350 GFLOP/s   (normal NPU behaviour)
  our q_proj (S=256)  0.411 GFLOP in 1018 ms -> ~0.4 GFLOP/s   (875x worse)

So the NPU is not the problem; our specific .om is. The leading suspect is the 896x896
weight baked in as an initializer: if omg lays it out in a way the NPU cannot stream
efficiently, every pass re-walks 1.6 MB. Moving W to a runtime input tests exactly that.

We also emit an int8-quantised runtime-weight variant, since the official fast sample is
very likely int8 rather than fp16, and dtype alone can account for a large factor.
"""
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

DIR = "/mnt/e/npu_work/hiai-llm-plan/qproj-real"
SRC = os.path.join(DIR, "qp_matmul.onnx")
HID = 896
S = 64


def build_runtime_weight(W, name, elem="f2"):
    """MatMul(X, Wrt) with both X and W supplied at run time."""
    xinfo = helper.make_tensor_value_info("X", TensorProto.FLOAT16, [1, S, HID])
    winfo = helper.make_tensor_value_info("Wrt", TensorProto.FLOAT16, [HID, HID])
    yinfo = helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [1, S, HID])
    g = helper.make_graph([helper.make_node("MatMul", ["X", "Wrt"], ["Y"])], name,
                          [xinfo, winfo], [yinfo])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    p = os.path.join(DIR, name + ".onnx")
    onnx.save(m, p)
    return p


def main():
    m = onnx.load(SRC, load_external_data=False)
    W = numpy_helper.to_array([i for i in m.graph.initializer if i.name == "Wq"][0])
    print("  W", W.shape, W.dtype)

    p = build_runtime_weight(W, "qp_rtw")
    print("  built", p)

    # runtime weight bytes (raw fp16 little endian)
    wb = os.path.join(DIR, "qp_rtw.W.bin")
    open(wb, "wb").write(W.astype("<f2").tobytes())
    print("  W bin %d B -> %s" % (os.path.getsize(wb), wb))

    # same X as the S=64 scalable variant, so only the weight's location changes
    xb = os.path.join(DIR, "qp_s64.input.bin")
    print("  X bin %d B (reused from qp_s64)" % os.path.getsize(xb))

    # also keep a constant-weight S=64 twin for a clean A/B in the same session
    g = helper.make_graph(
        [helper.make_node("MatMul", ["X", "Wq"], ["Y"])], "qp_c64",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT16, [1, S, HID])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [1, S, HID])],
        initializer=[numpy_helper.from_array(W, "Wq")])
    mm = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    mm.ir_version = 8
    onnx.save(mm, os.path.join(DIR, "qp_c64.onnx"))
    print("  built qp_c64.onnx (constant weight twin, S=64)")


if __name__ == "__main__":
    main()
