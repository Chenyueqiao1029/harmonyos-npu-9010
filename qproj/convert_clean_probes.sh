#!/usr/bin/env bash
# Convert the clean single-op probes and report which ones omg can even produce a .om for.
set -u
TASK=/mnt/e/npu_work/hiai-llm-plan
DDK="$TASK/ddk/DDK-tools-next-6.1.1.0"
OMG_DIR="$DDK/tools/tools_omg"
DIR=$TASK/clean-probes
OUT=$DIR/om
mkdir -p "$OUT" "$TASK/logs"
export TMPDIR="$TASK/tmp"
export XDG_CACHE_HOME="$TASK/cache"
export LD_LIBRARY_PATH="$OMG_DIR/master/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CCE_LIB_DIR="$OMG_DIR/master/lib64"
OMG=(/lib64/ld-linux-x86-64.so.2 "$OMG_DIR/master/omg")

/mnt/e/ls/refs/mslite-work/venv/bin/python "$TASK/build_clean_probes.py" || exit 1

printf "%-26s %-8s %s\n" "probe" "conv" "note"
for f in "$DIR"/p*.onnx; do
  tag=$(basename "$f" .onnx)
  timeout --signal=TERM --kill-after=10s 8m "${OMG[@]}" \
    --model "$f" --framework 5 --output "$OUT/$tag" > "$TASK/logs/conv_$tag.log" 2>&1
  if [ -s "$OUT/$tag.om" ]; then
    printf "%-26s %-8s %s B\n" "$tag" "OK" "$(stat -c%s "$OUT/$tag.om")"
  else
    reason=$(grep -oE 'E/AI_FMK.*|E/OMG.*' "$TASK/logs/conv_$tag.log" | head -1 | cut -c1-90)
    printf "%-26s %-8s %s\n" "$tag" "FAIL" "${reason:-unknown}"
  fi
done
echo ALLDONE
