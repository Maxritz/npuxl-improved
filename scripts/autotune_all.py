#!/usr/bin/env python3
"""Run autotune across model-like shapes and emit one recommendation JSON.

Usage:
  python3 scripts/autotune_all.py 512x4096x11008 1x4096x11008
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLI = os.path.join(ROOT, "bin", "vnpu-cli")

DEFAULT_SHAPES = [
    "1x4096x4096",       # decode: M=1
    "32x4096x4096",      # small prefill
    "512x4096x11008",    # FFN gate/up
    "512x11008x4096",    # FFN down
    "512x4096x12288",    # QKV
]


def tune(shape: str, dtype: str, iterations: int = 3) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        path = tmp.name
    try:
        cmd = [
            CLI, "autotune", "--shape", shape, "--dtype", dtype,
            "--iterations", str(iterations), "--json", path,
        ]
        result = subprocess.run(cmd, cwd=ROOT, capture_output=True,
                                text=True, check=False)
        if result.returncode != 0:
            return {"shape": shape, "dtype": dtype,
                    "error": result.stderr.strip() or "autotune failed"}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["shape"] = shape
        data["dtype"] = dtype
        return data
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def main() -> int:
    shapes = sys.argv[1:] or DEFAULT_SHAPES
    out = {
        "version": 1,
        "recommendations": [],
        "global": {"f32": None, "i8": None},
    }
    best = {"f32": None, "i8": None}
    for shape in shapes:
        for dtype in ("f32", "i8"):
            rec = tune(shape, dtype)
            out["recommendations"].append(rec)
            if "error" in rec:
                continue
            score = rec.get("gflops", 0.0)
            if best[dtype] is None or score > best[dtype].get("gflops", 0.0):
                best[dtype] = rec
    out["global"] = best
    os.makedirs(os.path.join(ROOT, "benchmarks"), exist_ok=True)
    path = os.path.join(ROOT, "benchmarks", "autotune-recommendations.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
