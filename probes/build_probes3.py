"""Minimal probes: is `Identity` as the graph's output node what breaks omg?

Evidence that motivated this: among the second-wave probes, the three that FAILED
(t3 int64-in, t4 wide-out, t5 many-nodes) all ended with `Identity -> output`,
while the one that PASSED (t6 fp16-out) ended with `Cast -> output`. Node count
and output width therefore did not explain the split.

t7  MatMul -> Identity -> output     (suspect)
t8  MatMul -> output                 (control, known good shape)
t9  MatMul -> Relu -> Identity -> output
t10 MatMul -> output, but request --output_type output:FP16 (dtype override)
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

MODELS = "/mnt/e/npu_work/hiai-llm-plan/models"
W = np.random.default_rng(3).standard_normal((16, 4)).astype(np.float32)


def build(path, tail_ops, note):
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 16])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])
    nodes = [helper.make_node("MatMul", ["input", "W"], ["mm"])]
    prev = "mm"
    for i, op in enumerate(tail_ops):
        dst = "output" if i == len(tail_ops) - 1 else ("t%d" % i)
        nodes.append(helper.make_node(op, [prev], [dst]))
        prev = dst
    g = helper.make_graph(nodes, "probe", [inp], [out],
                          initializer=[numpy_helper.from_array(W, "W")])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    onnx.save(m, path)
    print("  %-26s ops=%-28s %s" % (path.split("/")[-1], ",".join(["MatMul"] + tail_ops), note))


if __name__ == "__main__":
    build(MODELS + "/t7_id_tail.onnx", ["Identity"], "Identity as output node (SUSPECT)")
    build(MODELS + "/t8_direct.onnx", [], "MatMul output directly (control)")
    build(MODELS + "/t9_relu_id.onnx", ["Relu", "Identity"], "Relu then Identity")
    build(MODELS + "/t10_relu.onnx", ["Relu"], "Relu only (control)")
    print("PROBE BUILD DONE")
