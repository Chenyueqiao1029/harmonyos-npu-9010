#!/usr/bin/env bash
# Convert the runtime-weight variant and its constant-weight twin.
set -u
TASK=/mnt/e/npu_work/hiai-llm-plan
DDK="$TASK/ddk/DDK-tools-next-6.1.1.0"
OMG_DIR="$DDK/tools/tools_omg"
DIR=$TASK/qproj-real
OUT=$DIR/om
mkdir -p "$OUT" "$TASK/logs"
export TMPDIR="$TASK/tmp"
export XDG_CACHE_HOME="$TASK/cache"
export LD_LIBRARY_PATH="$OMG_DIR/master/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CCE_LIB_DIR="$OMG_DIR/master/lib64"
OMG=(/lib64/ld-linux-x86-64.so.2 "$OMG_DIR/master/omg")

/mnt/e/ls/refs/mslite-work/venv/bin/python "$TASK/build_rtw.py"

echo "---- qp_rtw  (W as runtime input)"
timeout --signal=TERM --kill-after=10s 10m "${OMG[@]}" \
  --model "$DIR/qp_rtw.onnx" --framework 5 \
  --input_shape 'X:1,64,896;Wrt:896,896' \
  --output "$OUT/qp_rtw" > "$TASK/logs/conv_qp_rtw.log" 2>&1
if [ -s "$OUT/qp_rtw.om" ]; then
  printf "     CONVERTED %10s B\n" "$(stat -c%s "$OUT/qp_rtw.om")"
else
  printf "     CONVERT FAILED\n"
  grep -E 'E/AI|E/OMG' "$TASK/logs/conv_qp_rtw.log" | tail -5
fi

echo "---- qp_c64  (constant weight twin, S=64)"
timeout --signal=TERM --kill-after=10s 10m "${OMG[@]}" \
  --model "$DIR/qp_c64.onnx" --framework 5 \
  --input_shape 'X:1,64,896' \
  --output "$OUT/qp_c64" > "$TASK/logs/conv_qp_c64.log" 2>&1
if [ -s "$OUT/qp_c64.om" ]; then
  printf "     CONVERTED %10s B\n" "$(stat -c%s "$OUT/qp_c64.om")"
else
  printf "     CONVERT FAILED\n"
  grep -E 'E/AI|E/OMG' "$TASK/logs/conv_qp_c64.log" | tail -5
fi
echo ALLDONE
