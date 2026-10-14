"""Build seq_len-scaled copies of the SAME real q_proj matmul, for a timing curve.

The question this answers: is the NPU's 69 ms dominated by per-call overhead or by compute?
If RunSync stays flat as seq_len grows, it is overhead. If it scales linearly, it is compute.

Weights are identical to qp_matmul (the genuine Qwen2.5-0.5B layer-0 q_proj); only the
sequence length changes. Input values do not affect timing, so a deterministic fp16
pattern is enough here -- the earlier numeric check already used the real activation.
"""
import json
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

DIR = "/mnt/e/npu_work/hiai-llm-plan/qproj-real"
SRC = os.path.join(DIR, "qp_matmul.onnx")
HID = 896


def main():
    m = onnx.load(SRC, load_external_data=False)
    W = numpy_helper.to_array([i for i in m.graph.initializer if i.name == "Wq"][0])
    print("  W", W.shape, W.dtype)

    meta = {}
    for S in (8, 64, 256):
        name = "qp_s%d" % S
        g = helper.make_graph(
            [helper.make_node("MatMul", ["X", "Wq"], ["Y"])], name,
            [helper.make_tensor_value_info("X", TensorProto.FLOAT16, [1, S, HID])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [1, S, HID])],
            initializer=[numpy_helper.from_array(W, "Wq")],
        )
        mm = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
        mm.ir_version = 8
        onnx.checker.check_model(mm)
        p = os.path.join(DIR, name + ".onnx")
        onnx.save(mm, p)

        # deterministic fp16 input of the right size
        x = (((np.arange(S * HID) * 7) % 23) - 11).astype(np.float32) / 8.0
        x16 = x.astype(np.float16)
        bp = os.path.join(DIR, name + ".input.bin")
        x16.astype("<f2").tobytes() and open(bp, "wb").write(x16.astype("<f2").tobytes())

        flops = 2 * S * HID * HID
        meta[name] = {"seq_len": S, "input_bytes": os.path.getsize(bp), "flops": flops}
        print("  %-8s S=%-4d input=%-8d B  flops=%.3f MFLOP" %
              (name, S, os.path.getsize(bp), flops / 1e6))

    json.dump(meta, open(os.path.join(DIR, "scale_meta.json"), "w"), indent=2)
    print("  meta ->", os.path.join(DIR, "scale_meta.json"))


if __name__ == "__main__":
    main()
