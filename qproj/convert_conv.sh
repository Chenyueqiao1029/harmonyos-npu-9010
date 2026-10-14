set -u
TASK=/mnt/e/npu_work/hiai-llm-plan
DDK="$TASK/ddk/DDK-tools-next-6.1.1.0"
OMG_DIR="$DDK/tools/tools_omg"
DIR=$TASK/qproj-real
OUT=$DIR/om
mkdir -p "$OUT" "$TASK/logs"
export TMPDIR="$TASK/tmp"; export XDG_CACHE_HOME="$TASK/cache"
export LD_LIBRARY_PATH="$OMG_DIR/master/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CCE_LIB_DIR="$OMG_DIR/master/lib64"
OMG=(/lib64/ld-linux-x86-64.so.2 "$OMG_DIR/master/omg")
/mnt/e/ls/refs/mslite-work/venv/bin/python "$TASK/build_conv.py"
for S in 8 64; do
  tag="qp_conv$S"
  printf -- "---- %s\n" "$tag"
  timeout --signal=TERM --kill-after=10s 10m "${OMG[@]}" --model "$DIR/$tag.onnx" --framework 5 \
    --input_shape "X:1,$S,896" --output "$OUT/$tag" > "$TASK/logs/conv_$tag.log" 2>&1
  if [ -s "$OUT/$tag.om" ]; then
    printf "     CONVERTED %10s B\n" "$(stat -c%s "$OUT/$tag.om")"
  else
    printf "     CONVERT FAILED\n"; grep -E 'E/AI|E/OMG' "$TASK/logs/conv_$tag.log" | tail -4
  fi
done
echo ALLDONE
