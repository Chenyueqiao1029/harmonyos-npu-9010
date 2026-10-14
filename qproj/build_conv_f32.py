"""fp32 twin of the 1x1-conv projection, so NPU and CPU can run the SAME model.

The fp16 version would not build under SetModelDeviceOrder(CPU) -- most likely the HiAI
CPU backend has no fp16 path. This fp32 twin lets both device orders compile the identical
graph, which is the only way to get an honest same-model NPU-vs-CPU factor.

Weights are still the genuine Qwen2.5-0.5B layer-0 q_proj; only dtype changes.
"""
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

DIR = "/mnt/e/npu_work/hiai-llm-plan/qproj-real"
SRC = os.path.join(DIR, "qp_matmul.onnx")
HID = 896


def main():
    m = onnx.load(SRC, load_external_data=False)
    W = numpy_helper.to_array([i for i in m.graph.initializer if i.name == "Wq"][0]).astype(np.float32)
    Wc = W.T.reshape(HID, HID, 1, 1)  # [out, in, 1, 1] fp32

    for S in (64,):
        name = "qp_conv%d_f32" % S
        nodes = [
            helper.make_node("Transpose", ["X"], ["Xt"], perm=[0, 2, 1]),
            helper.make_node("Unsqueeze", ["Xt", "ax"], ["Xn"]),
            helper.make_node("Conv", ["Xn", "Wc"], ["Yn"], kernel_shape=[1, 1]),
            helper.make_node("Squeeze", ["Yn", "ax"], ["Yt"]),
            helper.make_node("Transpose", ["Yt"], ["Y"], perm=[0, 2, 1]),
        ]
        g = helper.make_graph(
            nodes, name,
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, S, HID])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, S, HID])],
            initializer=[numpy_helper.from_array(Wc, "Wc"),
                         numpy_helper.from_array(np.array([3], dtype=np.int64), "ax")],
        )
        mm = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
        mm.ir_version = 8
        p = os.path.join(DIR, name + ".onnx")
        onnx.save(mm, p)

        # fp32 input, same values as the fp16 run
        x = (((np.arange(S * HID) * 7) % 23) - 11).astype(np.float32) / 8.0
        x = x.reshape(1, S, HID).astype(np.float32)
        open(os.path.join(DIR, name + ".input.bin"), "wb").write(x.astype("<f4").tobytes())

        # CPU sanity: conv form must equal X @ W
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        sess = ort.InferenceSession(p, so, providers=["CPUExecutionProvider"])
        y = sess.run(None, {"X": x})[0].reshape(-1)
        ref = (x @ W).reshape(-1)
        print("  %-16s S=%d  input=%d B  conv-vs-matmul max_abs_err=%.6g"
              % (name, S, os.path.getsize(os.path.join(DIR, name + ".input.bin")),
                 np.abs(y - ref).max()))
    print("  done")


if __name__ == "__main__":
    main()
