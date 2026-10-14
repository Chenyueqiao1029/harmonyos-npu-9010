"""Focused view of codex's device results: compatibility verdict per model per round."""
import json
from collections import defaultdict

P = r"<workdir>\compat-review-20261013\device-results.jsonl"

rows = []
for line in open(P, encoding="utf-8", errors="replace"):
    line = line.strip()
    if line:
        try:
            rows.append(json.loads(line))
        except Exception:
            pass

g = defaultdict(list)
for r in rows:
    g[r.get("model")].append(r)

FIELDS = ("round", "compatibility", "oracle_pass", "identity_ok", "max_abs_error", "build_only", "cpu")

print("%-22s %-28s %s" % ("model", "rounds -> compatibility", "oracle/identity/err"))
print("-" * 112)
for m in sorted(g):
    recs = sorted(g[m], key=lambda r: str(r.get("round")))
    comp = " ".join("%s=%s" % (r.get("round"), r.get("compatibility")) for r in recs)
    extra = []
    for r in recs:
        extra.append("%s:op=%s id=%s err=%s" % (
            r.get("round"), r.get("oracle_pass"), r.get("identity_ok"), r.get("max_abs_error")))
    print("%-22s %-28s %s" % (m, comp, " | ".join(extra)[:120]))

print()
print("models with MORE THAN ONE distinct compatibility value:")
any_var = False
for m in sorted(g):
    vals = {r.get("compatibility") for r in g[m]}
    if len(vals) > 1:
        any_var = True
        print("  %-22s -> %s" % (m, vals))
if not any_var:
    print("  (none -- every model gave a consistent compatibility across its rounds)")
