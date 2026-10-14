"""Build probe ONNX files that isolate the remaining Qwen suspects.

Already excluded by earlier probes (both converted fine):
  - opset 18
  - symbolic dimensions

Remaining suspects, one probe each:
  t3  INT64 graph inputs        -> omg --input_type has no INT64 option
  t4  wide logits output        -> Qwen emits [.., 151936]
  t5  many nodes (6000)         -> parser scale limit
  t6  fp16 output              -> Qwen's logits are FP16
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

MODELS = "/mnt/e/npu_work/hiai-llm-plan/models"


def save(m, path, note):
    m.ir_version = 8
    onnx.checker.check_model(m)
    onnx.save(m, path)
    n = len(m.graph.node)
    print("  wrote %-42s nodes=%-5d %s" % (path.split("/")[-1], n, note))


def matmul_probe(path, in_dtype, out_dtype, note, wide=False, extra_nodes=0):
    N = 151936 if wide else 4
    w = np.random.default_rng(2).standard_normal((16, N)).astype(np.float32)
    shape = ["N", 16]
    out_shape = ["N", N] if wide else ["N", 4]
    inp = helper.make_tensor_value_info("input", in_dtype, shape)
    out = helper.make_tensor_value_info("output", out_dtype, out_shape)

    nodes = []
    src = "input"
    if in_dtype == TensorProto.INT64:
        # keep the graph import-free: cast int64 -> float so the matmul is valid
        nodes.append(helper.make_node("Cast", ["input"], ["input_f"], to=TensorProto.FLOAT))
        src = "input_f"

    nodes.append(helper.make_node("MatMul", [src, "W"], ["mm"]))
    tail = "mm"
    # pad the graph with cheap identity-ish nodes to reach a target node count
    for i in range(extra_nodes):
        nodes.append(helper.make_node("Identity", [tail], ["id_%d" % i]))
        tail = "id_%d" % i
    if out_dtype != TensorProto.FLOAT:
        nodes.append(helper.make_node("Cast", [tail], ["output"], to=out_dtype))
    else:
        nodes.append(helper.make_node("Identity", [tail], ["output"]))

    g = helper.make_graph(nodes, "probe", [inp], [out],
                          initializer=[numpy_helper.from_array(w, "W")])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    save(m, path, note)


if __name__ == "__main__":
    # t3: int64 input (Qwen's actual input dtype)
    matmul_probe(MODELS + "/t3_int64_in.onnx", TensorProto.INT64, TensorProto.FLOAT,
                 "int64 graph input + Cast (Qwen-like)")
    # t4: wide output tensor (Qwen logits dim 151936)
    matmul_probe(MODELS + "/t4_wide_out.onnx", TensorProto.FLOAT, TensorProto.FLOAT,
                 "wide output dim 151936 (Qwen-like)", wide=True)
    # t5: node-count scale
    matmul_probe(MODELS + "/t5_many_nodes.onnx", TensorProto.FLOAT, TensorProto.FLOAT,
                 "6000 nodes (Qwen-like scale)", extra_nodes=6000)
    # t6: fp16 output
    matmul_probe(MODELS + "/t6_fp16_out.onnx", TensorProto.FLOAT, TensorProto.FLOAT16,
                 "fp16 graph output (Qwen-like)")
    print("PROBE BUILD DONE")
