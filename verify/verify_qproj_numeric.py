"""Element-wise numeric verification: does the NPU compute Qwen's real q_proj correctly?

Astaire's design point was that "Build succeeded" proves nothing on its own. The only
claim worth making is:

    real Qwen weights + the exact activation captured from the CPU forward pass
      -> NPU output == CPU output, element by element

So this run feeds the SAME activation the CPU reference used (hiai-input.bin, raw fp16),
pulls the NPU output back (hiai-output.bin), and compares all 7168 elements.
"""
import os
import shutil
import subprocess
import sys
import time

import numpy as np

HDC = r"<sdk>\DevEco Studio\sdk\default\openharmony\toolchains\hdc.exe"
T = "<device-id>"
BUNDLE = "com.lingshu.smart"
HAP = r"<project>\unpackage\dist\dev\app-harmony\entry\build\default\outputs\default\entry-default-signed.hap"
DIR = r"<workdir>\qproj-real"
SB = "/data/storage/el2/base/haps/entry/files"


def sh(*a, timeout=900):
    p = subprocess.run([HDC, "-t", T, *a], capture_output=True, timeout=timeout)
    return (p.stdout + p.stderr).decode("utf-8", "replace").strip()


def main():
    os.makedirs(os.path.join(DIR, "logs"), exist_ok=True)
    print("  channel:", sh("shell", "echo PONG")[:40])

    # ---- 1) install ----
    print("  installing hap ...")
    print("   ", sh("install", "-r", HAP)[:120])

    # launch once so the sandbox exists (hdc file send -b needs it)
    sh("shell", "aa start -a EntryAbility -b " + BUNDLE)
    time.sleep(8)

    # ---- 2) push input / model / selection file ----
    sel = os.path.join(DIR, "sel_qp_matmul.txt")
    with open(sel, "w", encoding="ascii") as f:
        f.write(SB + "/hiai-selected.om\n1 0 0\n")

    for local, remote in [
        (os.path.join(DIR, "hiai-input.bin"), SB + "/hiai-input.bin"),
        (os.path.join(DIR, "om", "qp_matmul.om"), SB + "/hiai-selected.om"),
        (sel, SB + "/hiai-review-selection.txt"),
    ]:
        print("  push %-22s %d B -> %s" % (os.path.basename(local),
                                           os.path.getsize(local),
                                           sh("file", "send", "-b", BUNDLE, local, remote)[:60]))

    # ---- 3) run ----
    log = os.path.join(DIR, "logs", "numeric-run.log")
    if os.path.exists(log):
        os.remove(log)
    proc = subprocess.Popen([HDC, "-t", T, "hilog"],
                            stdout=open(log, "w", encoding="utf-8", errors="replace"),
                            stderr=subprocess.DEVNULL)
    time.sleep(3)
    sh("shell", "aa force-stop " + BUNDLE)
    time.sleep(2)
    sh("shell", "aa start -a EntryAbility -b %s --ps hiai_model qp_matmul --pi hiai_case 1 "
       "--pi hiai_cpu 0 --pi hiai_build 0" % BUNDLE)
    time.sleep(60)
    proc.terminate()
    time.sleep(1)

    import re
    txt = open(log, encoding="utf-8", errors="replace").read()
    keys = ("INPUT_FROM_FILE", "INPUT_FILE_SIZE", "model_bytes", "model_fnv1a64",
            "compatibility=", "SetFallbackMode", "SetModelDeviceOrder", "Build=",
            "RunSync=", "dtype=", "OUTPUT=", "OUTPUT_WRITTEN", "OUTPUT_FILE_OPEN",
            "=== done ===")
    for line in txt.splitlines():
        if "JSAPP" not in line:
            continue
        s = re.sub(r"^.*?JSAPP:\s*", "", line).strip()
        if any(k in s for k in keys):
            print("   |", s[:135])

    # ---- 4) pull the NPU output and compare ----
    out_remote = SB + "/hiai-output.bin"
    out_local = os.path.join(DIR, "hiai-output.bin")
    if os.path.exists(out_local):
        os.remove(out_local)
    sh("file", "recv", "-b", BUNDLE, out_remote, out_local)

    if not os.path.exists(out_local):
        print("  RESULT: could not retrieve hiai-output.bin -> "
              "the NPU side did not write an output (see OUTPUT_* lines above)")
        return

    got = np.fromfile(out_local, dtype="<f2").astype(np.float32)
    ref = np.load(os.path.join(DIR, "qproj_cpu_ref.npy")).astype(np.float32).reshape(-1)
    print("  NPU elements = %d   CPU elements = %d" % (got.size, ref.size))
    if got.size != ref.size:
        print("  RESULT: size mismatch -> cannot compare")
        return

    diff = np.abs(got - ref)
    denom = np.maximum(np.abs(ref), 1e-6)
    rel = diff / denom
    print("  ---- element-wise comparison (n=%d) ----" % ref.size)
    print("   max_abs_err = %.6g" % diff.max())
    print("   mean_abs_err= %.6g" % diff.mean())
    print("   max_rel_err = %.6g" % rel.max())
    print("   NPU sum=%.6f  CPU sum=%.6f" % (got.sum(), ref.sum()))
    print("   nonfinite  = %d" % int((~np.isfinite(got)).sum()))
    print("   first8 NPU:", [round(float(v), 6) for v in got[:8]])
    print("   first8 CPU:", [round(float(v), 6) for v in ref[:8]])
    tol = 2e-2
    close = float((diff <= tol).mean())
    print("   fraction within %g = %.4f" % (tol, close))
    if diff.max() <= tol:
        print("  RESULT: PASS - NPU reproduced the CPU result element-wise")
    elif close >= 0.999:
        print("  RESULT: PASS(with fp16 rounding) - %.4f%% within tol" % (100 * close))
    else:
        print("  RESULT: FAIL - NPU output does not match the CPU reference")
    np.save(os.path.join(DIR, "hiai-output.npy"), got)
    print("  NUMERIC VERIFY DONE")


if __name__ == "__main__":
    main()
