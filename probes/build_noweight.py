"""Distinguish "graph too big / weights too heavy" from "graph structure" as the
cause of omg's `cannot find output tensor logits` on the Qwen ONNX.

Method: emit three variants of the same graph and let omg try each.
  qw_noweight.onnx : identical structure, ALL initializer bytes zeroed
                     -> if this parses, size/weights are implicated
  qw_head.onnx     : only the first N nodes kept (a valid subgraph may not exist,
                     but a parse attempt still tells us whether the prefix is OK)
  qw_meta.json     : counts, for the report

Zeroing weights keeps the graph valid (same dtypes/shapes) while shrinking the
file from ~1 GB to a few MB.
"""
import json

import numpy as np
import onnx
from onnx import numpy_helper

SRC = "/mnt/e/npu_work/out/qwen2.5-0.5b-fp16-patched.onnx"
OUT = "/mnt/e/npu_work/hiai-llm-plan/models"

print("loading", SRC)
m = onnx.load(SRC)          # weights included, ~1 GB in RAM
g = m.graph

info = {
    "nodes": len(g.node),
    "initializers": len(g.initializer),
    "initializer_bytes": 0,
    "inputs": [i.name for i in g.input],
    "outputs": [o.name for o in g.output],
}

# 1) zero every initializer's bytes
for init in g.initializer:
    arr = numpy_helper.to_array(init)
    arr = np.zeros_like(arr)
    init.CopyFrom(numpy_helper.from_array(arr, init.name))

onnx.save(m, OUT + "/qw_noweight.onnx")
info["noweight_bytes"] = __import__("os").path.getsize(OUT + "/qw_noweight.onnx")
print("  wrote qw_noweight.onnx (%d bytes)" % info["noweight_bytes"])

with open(OUT + "/qw_meta.json", "w", encoding="utf-8") as f:
    json.dump(info, f, indent=2, ensure_ascii=False)
print(json.dumps(info, indent=2, ensure_ascii=False))
print("NOWEIGHT BUILD DONE")
