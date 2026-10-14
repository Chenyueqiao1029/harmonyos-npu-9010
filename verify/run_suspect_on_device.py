"""Run the suspect-op probes on the device, reusing the other session's harness.

Why the device is not free right now: the compat-review session is actively
driving the phone (its selection.txt was written seconds ago). Running a second
test cycle concurrently would corrupt both sets of results, because the app reads
a single selection file and only one app instance exists. So this driver refuses
to start unless the device looks idle, and it refuses to collide with the other
session's files.

It reuses:
  - test_device.py   (hdc file send -b into the app sandbox, SHA-256 read-back,
                      aa start with --ps/--pi, hilog capture, oracle comparison)
  - its models.json   (only for oracle values; our probes set out of range simply
                      skip the numeric check)

Usage:
    python run_suspect_on_device.py --check          # only report whether it is safe
    python run_suspect_on_device.py --run            # actually run the probes
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

OTHER = Path(r"<workdir>\compat-review-20261013")
MINE = Path(r"<workdir>")
HARNESS = OTHER / "test_device.py"
PROBES = [
    "u1_cos", "u2_sin", "u3_neg", "u4_less", "u5_and",
    "u7_expand", "u8_flatten", "u9_sigmoid", "u10_softmax", "u11_slice",
]

IDLE_SECONDS = 150  # require the other session to be quiet this long


def other_session_age():
    newest = None
    for p in [OTHER / "device-results.jsonl", OTHER / "selection.txt"]:
        if p.exists():
            t = p.stat().st_mtime
            newest = t if newest is None else max(newest, t)
    d = OTHER / "device"
    if d.exists():
        for f in d.glob("*"):
            t = f.stat().st_mtime
            newest = t if newest is None else max(newest, t)
    return None if newest is None else time.time() - newest


def build_models_json():
    """test_device.py reads models.json for oracle values; give ours an empty oracle
    so it records compatibility/build/run but skips the numeric check."""
    recs = []
    for name in PROBES:
        p = MINE / "om" / "suspect" / (name + ".om")
        if p.exists():
            recs.append({"name": name, "inputs": {"x": [4, 8]},
                         "shape": [4, 8], "nodes": [], "oracle": {}})
    out = MINE / "suspect-models.json"
    out.write_text(json.dumps(recs, indent=2), encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--force", action="store_true", help="ignore the idle gate")
    args = ap.parse_args()

    age = other_session_age()
    print("other session last device write: %s" % (
        "never" if age is None else "%.0f s ago" % age))
    print("our probes present: %d / %d" % (
        sum(1 for n in PROBES if (MINE / "om" / "suspect" / (n + ".om")).exists()),
        len(PROBES)))

    safe = age is None or age >= IDLE_SECONDS
    if not safe:
        print("NOT SAFE: the other session is still driving the device "
              "(needs %.0f s of quiet)." % IDLE_SECONDS)
        if not args.force:
            print("re-run later, or pass --force if you know it has stopped.")
            return 2
        print("--force given: proceeding anyway")

    if args.check:
        return 0

    build_models_json()
    print("running probes via the shared harness ...")
    done, failed = [], []
    for name in PROBES:
        om = MINE / "om" / "suspect" / (name + ".om")
        if not om.exists():
            failed.append((name, "no .om"))
            continue
        cmd = [sys.executable, str(HARNESS), str(om), "--case", "1",
               "--round", "s", "--timeout", "40"]
        print("---- %s" % name, flush=True)
        r = subprocess.run(cmd, cwd=str(OTHER), capture_output=True,
                           encoding="utf-8", errors="replace", timeout=300)
        tail = (r.stdout or "").strip().splitlines()
        print("   " + ("\n   ".join(tail[-4:]) if tail else "(no output)"))
        if r.returncode == 0:
            done.append(name)
        else:
            failed.append((name, "exit %d" % r.returncode))
        time.sleep(3)

    print()
    print("completed: %d  %s" % (len(done), done))
    print("failed   : %d  %s" % (len(failed), failed))
    print("SUGGEST SENDING"
          if False else "")
    print("SUSPECT DEVICE RUN DONE")


if __name__ == "__main__":
    raise SystemExit(main())
