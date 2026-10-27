#!/usr/bin/env python3
"""Summarize a Sherlock Chrome trace JSON without external dependencies."""
from __future__ import annotations

import json
import sys
from collections import defaultdict

def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: trace_report.py trace.json [top_n]", file=sys.stderr)
        return 2
    path = sys.argv[1]
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    events = [e for e in data.get("traceEvents", []) if e.get("ph") == "X"]
    total = sum(float(e.get("dur", 0.0)) for e in events)
    by_cat = defaultdict(float)
    by_name = defaultdict(float)
    by_op = defaultdict(float)
    by_tid = defaultdict(float)
    flops = 0
    bytes_rw = 0
    for e in events:
        dur = float(e.get("dur", 0.0))
        by_cat[e.get("cat", "unknown")] += dur
        by_name[e.get("name", "unknown")] += dur
        args = e.get("args", {})
        by_op[args.get("op", "unknown")] += dur
        by_tid[str(e.get("tid", "?"))] += dur
        flops += int(args.get("flops", 0))
        bytes_rw += int(args.get("bytes_read", 0)) + int(args.get(
            "bytes_written", 0))
    print(f"Trace: {path}")
    print(f"Events: {len(events)}  total: {total/1000.0:.3f} ms")
    print(f"FLOPs: {flops}  bytes read+written: {bytes_rw}")
    print("\nBy category:")
    for k, v in sorted(by_cat.items(), key=lambda x: -x[1]):
        print(f"  {k:<12} {v/1000.0:10.3f} ms {100*v/max(total,1):6.1f}%")
    print("\nBy opcode:")
    for k, v in sorted(by_op.items(), key=lambda x: -x[1])[:top_n]:
        print(f"  {k:<22} {v/1000.0:10.3f} ms {100*v/max(total,1):6.1f}%")
    print("\nBy worker:")
    for k, v in sorted(by_tid.items(), key=lambda x: -x[1]):
        print(f"  tid {k:<8} {v/1000.0:10.3f} ms {100*v/max(total,1):6.1f}%")
    slowest = sorted(events, key=lambda e: -float(e.get("dur", 0.0)))[:top_n]
    print("\nSlowest events:")
    for e in slowest:
        args = e.get("args", {})
        print(f"  {float(e.get('dur',0.0))/1000.0:9.3f} ms  "
              f"{e.get('name',''):<24} op={args.get('op','?'):<14} "
              f"unit={args.get('unit','?'):<8} tid={e.get('tid','?')}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
