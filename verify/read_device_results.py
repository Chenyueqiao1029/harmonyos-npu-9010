"""Read codex's device-results.jsonl in full and summarise verdicts per model per round."""
import json
from collections import defaultdict

P = r"<workdir>\compat-review-20261013\device-results.jsonl"

rows = []
for line in open(P, encoding="utf-8", errors="replace"):
    line = line.strip()
    if not line:
        continue
    try:
        rows.append(json.loads(line))
    except Exception as e:  # noqa: BLE001
        print("  unparsed:", line[:120], e)

print("total records:", len(rows))
if not rows:
    raise SystemExit

print("keys seen:", sorted({k for r in rows for k in r.keys()}))
print()

# group by model + round
g = defaultdict(dict)
for r in rows:
    m = r.get("model")
    rd = r.get("round", "?")
    cpu = r.get("cpu")
    key = "%s/%s" % (rd, "cpu" if cpu else "npu")
    g[m][key] = r

print("%-24s %s" % ("model", "round/order -> verdict fields"))
print("-" * 110)
for m in sorted(g):
    for key in sorted(g[m]):
        r = g[m][key]
        # find the interesting fields
        interesting = {k: v for k, v in r.items()
                       if k not in ("model", "round", "cpu", "local_sha256", "remote_sha256")}
        print("%-24s %-12s %s" % (m, key, json.dumps(interesting, ensure_ascii=False)[:250]))
    print()
