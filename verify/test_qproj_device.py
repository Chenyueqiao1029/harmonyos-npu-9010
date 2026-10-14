"""Device test for the REAL q_proj subgraph, bypassing the HAP rawfile route.

codex's test_device.py --packaged reads the model from the HAP's rawfile; rebuilding a
1 GB HAP just to swap a 1.6 MB .om is wasteful. The installed app already exposes
probeHiaiOmFromFile(), and its COMPAT_REVIEW block reads whatever sits at
    /data/storage/el2/base/haps/entry/files/hiai-selected.om
so we push the .om straight into the sandbox under that name and read the verdict
from hilog.

The probe disables fallback (SetFallbackMode(DISABLED) -> HIAI_FALLBACK_DISABLED) and
forces NPU device order, so a Build success here means the NPU really tried it.
"""
import argparse
import os
import re
import subprocess
import sys
import time

HDC = r"<sdk>\DevEco Studio\sdk\default\openharmony\toolchains\hdc.exe"
T = "<device-id>"
BUNDLE = "com.lingshu.smart"
OMDIR = r"<workdir>\compat-review-20261013\om"
SANDBOX = "/data/storage/el2/base/haps/entry/files/hiai-selected.om"
LOGDIR = r"<workdir>\qproj-real\logs"


def sh(*a, timeout=180):
    p = subprocess.run([HDC, "-t", T, *a], capture_output=True, timeout=timeout)
    return (p.stdout + p.stderr).decode("utf-8", "replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--wait", type=int, default=45)
    args = ap.parse_args()

    os.makedirs(LOGDIR, exist_ok=True)
    local = os.path.join(OMDIR, args.label + ".om")
    if not os.path.exists(local):
        sys.exit("no such .om: " + local)
    print("  .om: %s (%d B)" % (args.label, os.path.getsize(local)))

    print("  channel:", sh("shell", "echo PONG").strip()[:40])

    # push into the sandbox under the name the app's COMPAT_REVIEW block reads
    print("  pushing ->", SANDBOX)
    out = sh("file", "send", "-b", BUNDLE, local, SANDBOX, timeout=600)
    print("   ", out.strip()[:120])

    # verify by size
    rb = os.path.join(LOGDIR, args.label + ".readback.om")
    if os.path.exists(rb):
        os.remove(rb)
    sh("file", "recv", "-b", BUNDLE, SANDBOX, rb, timeout=600)
    if os.path.exists(rb):
        ok = os.path.getsize(rb) == os.path.getsize(local)
        print("  readback %d B  match=%s" % (os.path.getsize(rb), ok))

    log = os.path.join(LOGDIR, args.label + ".log")
    if os.path.exists(log):
        os.remove(log)
    proc = subprocess.Popen([HDC, "-t", T, "hilog"],
                            stdout=open(log, "w", encoding="utf-8", errors="replace"),
                            stderr=subprocess.DEVNULL)
    time.sleep(3)
    sh("shell", "aa force-stop " + BUNDLE)
    time.sleep(2)
    sh("shell", "aa start -a EntryAbility -b " + BUNDLE)
    time.sleep(args.wait)
    proc.terminate()
    time.sleep(1)

    txt = open(log, encoding="utf-8", errors="replace").read()
    lines = [re.sub(r"^.*?JSAPP:\s*", "", l).strip()
             for l in txt.splitlines() if "JSAPP" in l]
    keys = ("model_bytes", "model_fnv1a64", "compatibility", "Build=", "RunSync=",
            "OUTPUT=", "VALUES=", "in=", "out=", "dtype=", "SetFallbackMode",
            "SetModelDeviceOrder", "device#", "startup time", "not supported")
    seen = []
    started = False
    for l in lines:
        if "=== HiAI Foundation .om model probe ===" in l:
            started = True
        if started:
            if any(k in l for k in keys):
                seen.append(l)
            if "=== done ===" in l:
                seen.append(l)
                break
    print("  ---- probe output ----")
    for l in seen:
        print("   ", l[:150])
    if not seen:
        print("    (no probe output captured)")
    print("  raw log:", log)
    print("  DEVICE TEST DONE")


if __name__ == "__main__":
    main()
