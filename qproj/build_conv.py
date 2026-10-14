"""Express the SAME q_proj matmul as a 1x1 convolution, to test whether the NPU's fast
path is convolution-specific.

Evidence pointing here. On this one device and this one API:
  official SqueezeNet   0.7 GFLOP in   2 ms  -> ~350 GFLOP/s   (convolutions)
  our q_proj (S=8)     12.8 MFLOP in  68 ms  -> ~0.19 GFLOP/s  (one big MatMul)
SqueezeNet does 55x the arithmetic in 34x less time. The largest structural difference is
operator form, and 1x1 convolution is mathematically identical to a MatMul:
    Y[b, :, s] = sum_c X[b, c, s] * W[o, c]   for each output channel o

So this builds the same projection as Conv with kernel 1x1, stride 1, and compares the
latency of the two forms under otherwise identical conditions.
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
    # ONNX MatMul stores W as [in, out]; a Conv weight for 1x1 is [out, in, 1, 1]
    Wc = W.T.reshape(HID, HID, 1, 1).astype(np.float16)

    for S in (8, 64):
        name = "qp_conv%d" % S
        # X is [1, S, 896] in the matmul form -> transpose to NCHW [1, 896, S, 1] for conv
        nodes = [
            helper.make_node("Transpose", ["X"], ["Xt"], perm=[0, 2, 1]),
            helper.make_node("Unsqueeze", ["Xt", "ax"], ["Xn"]),
            helper.make_node("Conv", ["Xn", "Wc"], ["Yn"], kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0]),
            helper.make_node("Squeeze", ["Yn", "ax"], ["Yt"]),
            helper.make_node("Transpose", ["Yt"], ["Y"], perm=[0, 2, 1]),
        ]
        g = helper.make_graph(
            nodes, name,
            [helper.make_tensor_value_info("X", TensorProto.FLOAT16, [1, S, HID])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [1, S, HID])],
            initializer=[
                numpy_helper.from_array(Wc, "Wc"),
                numpy_helper.from_array(np.array([3], dtype=np.int64), "ax"),
            ],
        )
        mm = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
        mm.ir_version = 8
        try:
            onnx.checker.check_model(mm)
        except Exception as e:
            print("  checker warned:", str(e)[:120])
        p = os.path.join(DIR, name + ".onnx")
        onnx.save(mm, p)

        # numeric sanity on CPU: conv form must reproduce X @ W
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        sess = ort.InferenceSession(p, so, providers=["CPUExecutionProvider"])
        x = (((np.arange(S * HID) * 7) % 23) - 11).astype(np.float32) / 8.0
        x = x.reshape(1, S, HID).astype(np.float16)
        y = sess.run(None, {"X": x})[0].astype(np.float32).reshape(-1)
        ref = (x.astype(np.float32).reshape(1, S, HID) @ W).reshape(-1)
        print("  %-10s S=%-3d  conv-vs-matmul max_abs_err = %.6g" % (name, S, np.abs(y - ref).max()))

        # matching input bytes for the device
        open(os.path.join(DIR, name + ".input.bin"), "wb").write(x.astype("<f2").tobytes())
    print("  done")


if __name__ == "__main__":
    main()
