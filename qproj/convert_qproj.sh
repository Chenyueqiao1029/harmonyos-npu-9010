#!/usr/bin/env bash
# Convert the real q_proj subgraph to .om, in two variants:
#   qp_matmul  Y = X @ Wq
#   qp_add     Y = X @ Wq + bq
# so that if the Add triggers a restriction we can tell it apart from MatMul.
set -u
TASK=/mnt/e/npu_work/hiai-llm-plan
DDK="$TASK/ddk/DDK-tools-next-6.1.1.0"
OMG_DIR="$DDK/tools/tools_omg"
DIR=$TASK/qproj-real
OUT=$DIR/om
LOG=$TASK/logs/qproj_conv.log
mkdir -p "$OUT" "$TASK/logs"
export TMPDIR="$TASK/tmp"; export XDG_CACHE_HOME="$TASK/cache"
export LD_LIBRARY_PATH="$OMG_DIR/master/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CCE_LIB_DIR="$OMG_DIR/master/lib64"
OMG=(/lib64/ld-linux-x86-64.so.2 "$OMG_DIR/master/omg")
PY=/mnt/e/ls/refs/mslite-work/venv/bin/python

{
  echo "########## build the matmul-only variant ##########"
  "$PY" - <<'PYEOF'
import numpy as np, onnx
from onnx import TensorProto, helper, numpy_helper
DIR = "/mnt/e/npu_work/hiai-llm-plan/qproj-real"
m = onnx.load(DIR + "/qproj_subgraph.onnx", load_external_data=False)
W = numpy_helper.to_array([i for i in m.graph.initializer if i.name == "Wq"][0])
g = helper.make_graph(
    [helper.make_node("MatMul", ["X", "Wq"], ["Y"])], "qp_matmul",
    [helper.make_tensor_value_info("X", TensorProto.FLOAT16, [1, 8, 896])],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [1, 8, 896])],
    initializer=[numpy_helper.from_array(W, "Wq")])
mm = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
mm.ir_version = 8
onnx.checker.check_model(mm)
onnx.save(mm, DIR + "/qp_matmul.onnx")
print("  wrote qp_matmul.onnx")
PYEOF

  echo
  echo "########## convert both ##########"
  for k in qp_matmul qproj_subgraph; do
    tag=$k
    printf -- "---- %s\n" "$tag"
    timeout --signal=TERM --kill-after=10s 10m "${OMG[@]}" \
      --model "$DIR/$k.onnx" --framework 5 \
      --input_shape 'X:1,8,896' \
      --output "$OUT/$tag" > "$TASK/logs/conv_$tag.log" 2>&1
    if [ -s "$OUT/$tag.om" ]; then
      printf "     CONVERTED %9s B\n" "$(stat -c%s "$OUT/$tag.om")"
    else
      printf "     CONVERT FAILED\n"
      grep -E 'E/AI|E/OMG' "$TASK/logs/conv_$tag.log" | tail -4
    fi
  done
} > "$LOG" 2>&1

echo "DONE -> $LOG"
