#!/usr/bin/env bash
# Convert the op-support probes to .om, so they can be checked for device
# compatibility one by one on the phone.
set -u
TASK=/mnt/e/npu_work/hiai-llm-plan
OMG_DIR="$TASK/linux-ddk/DDK-tools-next-6.1.1.0/tools/tools_omg"
OUT=$TASK/om/ops
export TMPDIR="$TASK/tmp"; export XDG_CACHE_HOME="$TASK/cache"
export LD_LIBRARY_PATH="$OMG_DIR/master/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CCE_LIB_DIR="$OMG_DIR/master/lib64"
OMG=(/lib64/ld-linux-x86-64.so.2 "$OMG_DIR/master/omg")
mkdir -p "$OUT" "$TASK/models/ops"

# probe : input_shape
declare -A SHAPES=(
  [p1_matmul]='x:4,8'
  [p2_softmax]='x:4,8'
  [p3_layernorm]='x:4,8'
  [p4_where_expand]='x:4,8'
  [p5_gather]='x:1,4'
  [p6_tblock]='x:4,8'
)

for k in p1_matmul p2_softmax p3_layernorm p4_where_expand p5_gather p6_tblock; do
  echo "================ $k ================"
  timeout --signal=TERM --kill-after=10s 5m "${OMG[@]}" \
    --model "$TASK/models/ops/$k.onnx" --framework 5 \
    --input_shape "${SHAPES[$k]}" \
    --output "$OUT/$k" 2>&1 \
    | grep -E 'E/AI|OMG generate|OMG precheck|OMG run' | tail -4
  if [ -s "$OUT/$k.om" ]; then
    echo "  >>> CONVERTED $(stat -c%s "$OUT/$k.om") B"
  else
    echo "  >>> CONVERT FAILED"
  fi
  echo
done

echo "================ summary ================"
ls -l "$OUT" 2>&1
echo "OPS CONVERT DONE"
