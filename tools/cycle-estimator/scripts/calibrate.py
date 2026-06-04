#!/usr/bin/env python3
"""Calibrate the estimator against CGRA-Cocotb-Sim ground-truth cycles.

The simulator (CGRA-Cocotb-Sim/server/test_runif.py) clocks at CLOCKPERIOD=2 ns
(:1469) and logs per-kernel execution time as:

    EXE.F task, stream <id>, resp ..., tile ..., time <T> ns.   (:1430)

where T = exef_time - exes_time. Ground-truth cycles = T / CLOCKPERIOD.

This harness parses such log lines, runs the estimator on the matching dot, and
reports estimate vs. ground truth + error. Run it once a sim log is available:

    calibrate.py --log sim.log --dot-dir <dir> --mlir kernel.mlir
"""

from __future__ import annotations

import argparse
import glob
import os
import re

from core.dot_parser import parse_dot
from core.cycle_model import estimate_cycles
from extract.loop_info import kernel_for_dot


CLOCKPERIOD_NS = 2  # test_runif.py:1469

# "EXE.F task, stream 3, resp ..., tile ..., time 1950 ns."
_EXE_TIME = re.compile(r'stream\s+(\d+).*?time\s+(\d+(?:\.\d+)?)\s*ns', re.IGNORECASE)


def parse_sim_log(log_path: str) -> dict[int, float]:
    """stream_id -> ground-truth cycles."""
    out: dict[int, float] = {}
    with open(log_path) as f:
        for line in f:
            m = _EXE_TIME.search(line)
            if m:
                out[int(m.group(1))] = float(m.group(2)) / CLOCKPERIOD_NS
    return out


def calibrate(log_path: str, dot_dir: str, mlir_path: str | None,
              num_alus: int, route_lat: int):
    gt = parse_sim_log(log_path)
    if not gt:
        print("no 'time ... ns' lines found in log")
        return

    dots = sorted(glob.glob(os.path.join(dot_dir, "*_CDFG.dot")))
    print(f"{'kernel':16} {'estimate':>10} {'ground_truth':>14} {'err%':>8}")
    print("-" * 52)
    rows = []
    for dot in dots:
        name = os.path.basename(dot).replace("_CDFG.dot", "")
        loops = kernel_for_dot(mlir_path, name) if mlir_path else None
        if loops is None:
            continue
        cdfg = parse_dot(dot)
        est = estimate_cycles(cdfg, loops, num_alus=num_alus, route_lat=route_lat)
        rows.append((name, est.total))

    # Ground truth is keyed by stream id; without a dot<->stream map we report
    # the estimate list and the GT list for manual / future automatic matching.
    print("Estimates:")
    for name, total in rows:
        print(f"  {name:16} {total:>10}")
    print("\nGround-truth cycles (by stream id):")
    for sid, cyc in sorted(gt.items()):
        print(f"  stream {sid:<4} {cyc:>10.0f}")
    print("\nNOTE: provide a stream<->kernel mapping to auto-compute err%.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="cocotb sim log with 'time ... ns' lines")
    ap.add_argument("--dot-dir", required=True)
    ap.add_argument("--mlir")
    ap.add_argument("--num-alus", type=int, default=16)
    ap.add_argument("--route-lat", type=int, default=0)
    a = ap.parse_args()
    calibrate(a.log, a.dot_dir, a.mlir, a.num_alus, a.route_lat)
