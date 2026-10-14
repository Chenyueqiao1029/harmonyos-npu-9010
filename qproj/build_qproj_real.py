"""Build a REAL Qwen2.5-0.5B layer-0 q_proj subgraph: Y = X @ Wq^T + bq.

Follows the reviewer's design, whose point was that random matrices prove nothing:
  - Wq, bq are taken from the genuine Qwen2.5-0.5B checkpoint (as embedded in the
    fp16 ONNX we already have)
  - X is a REAL activation: captured from a CPU forward pass of the full model,
    at /model/layers.0/input_layernorm/Mul_1_output_0 (the RMSNorm output that
    feeds q_proj) -- not synthetic noise

Outputs, for the device test:
  qproj_subgraph.onnx   the subgraph, fp16
  qproj_input_x.npy     the captured activation X
  qproj_cpu_ref.npy     the CPU reference Y (onnxruntime), for element-wise diff
  qproj_meta.json       shapes/dtypes/SHA-256
"""
import hashlib
import json
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

SRC = "/mnt/e/npu_work/out/qwen2.5-0.5b-fp16-patched.onnx"
OUTDIR = "/mnt/e/npu_work/hiai-llm-plan/qproj-real"
os.makedirs(OUTDIR, exist_ok=True)

CAPTURE_TENSOR = "/model/layers.0/input_layernorm/Mul_1_output_0"
QPROJ_MATMUL = "/model/layers.0/self_attn/q_proj/MatMul"
BIAS_NAME = "model.layers.0.self_attn.q_proj.bias"


def main():
    print("loading graph (weights not loaded into python yet)")
    m = onnx.load(SRC, load_external_data=False)

    # ---- 1) locate the q_proj weight initializer ----
    qnode = next(n for n in m.graph.node if n.name == QPROJ_MATMUL)
    wname = qnode.input[1]
    print("  q_proj MatMul:", QPROJ_MATMUL)
    print("  weight initializer:", wname)

    inits = {i.name: i for i in m.graph.initializer}
    if wname not in inits or BIAS_NAME not in inits:
        raise SystemExit("missing weight/bias initializer")

    W = numpy_helper.to_array(inits[wname])
    b = numpy_helper.to_array(inits[BIAS_NAME])
    print("  W shape=%s dtype=%s" % (W.shape, W.dtype))
    print("  b shape=%s dtype=%s" % (b.shape, b.dtype))

    # ---- 2) capture the REAL activation X with onnxruntime ----
    print("building a capture copy with the RMSNorm output exposed as a graph output")
    m2 = onnx.load(SRC, load_external_data=False)
    want = helper.make_tensor_value_info(CAPTURE_TENSOR, TensorProto.FLOAT16, None)
    existing = {o.name for o in m2.graph.output}
    if CAPTURE_TENSOR not in existing:
        m2.graph.output.append(want)
    cap_path = os.path.join(OUTDIR, "_capture.onnx")
    onnx.save(m2, cap_path)
    print("  capture copy saved:", cap_path)

    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    print("creating session (this loads ~1 GB of fp16 weights) ...")
    sess = ort.InferenceSession(cap_path, so, providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    print("  session inputs:", names)

    S = 8  # sequence length, matching the shape the .om was built with
    feeds = {}
    for n in names:
        if n == "input_ids":
            # real token ids would come from the tokenizer; ids in-vocab are what matter
            # for capturing a genuine activation through the genuine weights.
            feeds[n] = np.array([[151643, 9707, 1879, 112, 2593, 151643, 13, 4]], dtype=np.int64)[:, :S]
        elif n == "attention_mask":
            feeds[n] = np.ones((1, S), dtype=np.int64)
        else:  # position_ids
            feeds[n] = np.arange(S, dtype=np.int64).reshape(1, S)

    outs = sess.run(None, feeds)
    onames = [o.name for o in sess.get_outputs()]
    print("  session outputs:", onames)
    X = outs[onames.index(CAPTURE_TENSOR)]
    print("  captured X: shape=%s dtype=%s" % (X.shape, X.dtype))

    # ---- 3) CPU reference for the projection ----
    Xf = X.astype(np.float32)
    Y = (Xf @ W.astype(np.float32).T) + b.astype(np.float32)
    print("  CPU reference Y: shape=%s" % (Y.shape,))

    # ---- 4) build the subgraph ONNX: MatMul(X, W^T) + b ----
    # W in the original graph is stored so that MatMul(X, W) is correct, so reuse it as-is.
    Wt = numpy_helper.from_array(W, "Wq")
    bt = numpy_helper.from_array(b, "bq")
    node_mm = helper.make_node("MatMul", ["X", "Wq"], ["mm"])
    node_add = helper.make_node("Add", ["mm", "bq"], ["Y"])
    graph = helper.make_graph(
        [node_mm, node_add], "qproj",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT16, [1, S, 896])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [1, S, 896])],
        initializer=[Wt, bt],
    )
    sub = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    sub.ir_version = 8
    onnx.checker.check_model(sub)
    sub_path = os.path.join(OUTDIR, "qproj_subgraph.onnx")
    onnx.save(sub, sub_path)

    X16 = X.astype(np.float16)
    np.save(os.path.join(OUTDIR, "qproj_input_x.npy"), X16)
    np.save(os.path.join(OUTDIR, "qproj_cpu_ref.npy"), Y.astype(np.float32))

    meta = {
        "capture_tensor": CAPTURE_TENSOR,
        "weight_initializer": wname,
        "bias_initializer": BIAS_NAME,
        "W_shape": list(W.shape), "W_dtype": str(W.dtype),
        "b_shape": list(b.shape), "b_dtype": str(b.dtype),
        "X_shape": list(X.shape), "X_dtype": str(X.dtype),
        "seq_len": S,
        "subgraph_sha256": hashlib.sha256(open(sub_path, "rb").read()).hexdigest(),
        "cpu_ref_sum": float(Y.sum()),
        "cpu_ref_max": float(Y.max()),
        "cpu_ref_min": float(Y.min()),
        "cpu_ref_first8": [float(v) for v in Y.reshape(-1)[:8]],
    }
    json.dump(meta, open(os.path.join(OUTDIR, "qproj_meta.json"), "w"),
              indent=2, ensure_ascii=False)
    print(json.dumps(meta, indent=2, ensure_ascii=False)[:1200])
    print("QPROJ BUILD DONE")


if __name__ == "__main__":
    main()
