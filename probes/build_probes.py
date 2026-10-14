"""Build probe ONNX files and lower the Qwen graph's opset.

Kept as a real file rather than a shell heredoc: a heredoc in this environment
dropped the final character of two import lines ('numpy_helper',
'version_converter'), which broke the run. A standalone file avoids that path
entirely.
"""
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

MODELS = "/mnt/e/npu_work/hiai-llm-plan/models"
QWEN = "/mnt/e/npu_work/out/qwen2.5-0.5b-fp16-patched.onnx"


def build(path, opset, symbolic):
    shape = ["N", 16] if symbolic else [1, 16]
    out_shape = ["N", 4] if symbolic else [1, 4]
    w = np.random.default_rng(1).standard_normal((16, 4)).astype(np.float32)
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, out_shape)
    node = helper.make_node("MatMul", ["input", "W"], ["output"])
    graph = helper.make_graph(
        [node], "probe", [inp], [out],
        initializer=[numpy_helper.from_array(w, "W")],
    )
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    onnx.save(m, path)
    print("  wrote %s (opset %d, %s)" % (path, opset, "symbolic" if symbolic else "static"))


def lower_qwen(src, dst, target_opset=13):
    from onnx import version_converter
    print("  loading qwen graph (about 1 GB, weights included) ...")
    m = onnx.load(src)
    print("  loaded; opset imports:", [(o.domain or "ai.onnx", o.version) for o in m.opset_import])
    print("  converting opset 18 -> %d ..." % target_opset)
    m2 = version_converter.convert_version(m, target_opset)
    m2.ir_version = 8
    onnx.save(m2, dst)
    print("  saved", dst)
    chk = onnx.load(dst, load_external_data=False)
    print("  verify: nodes=%d opset=%s" % (
        len(chk.graph.node), [(o.domain or "ai.onnx", o.version) for o in chk.opset_import]))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "probes"):
        build(MODELS + "/t1_opset18.onnx", 18, False)
        build(MODELS + "/t2_symbolic.onnx", 13, True)
    if what in ("all", "lower"):
        try:
            lower_qwen(QWEN, MODELS + "/qwen05_opset13.onnx")
        except Exception as e:  # noqa: BLE001 - report, never raise past this point
            print("  OPSET LOWER FAILED: %s: %s" % (type(e).__name__, e))
    print("PROBE BUILD DONE")
