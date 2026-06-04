#!/usr/bin/env python3
"""ADORA MLIR cycle-count estimator -- entry point.

Pipeline (dot path):
  1. From MLIR: run cgra-opt --adora-kernel-dfg-gen to produce <kernel>_CDFG.dot,
     then estimate. Loop trip-counts are read strong-typed from the MLIR via the
     upstream MLIR Python bindings (extract.loop_info), not regex.
  2. From an existing dot (+ MLIR for trip-counts): --dot / --mlir.

Usage:
  run.py --mlir kernel.mlir [--adg adg.json] [--num-alus 16]
  run.py --dot kernel_CDFG.dot [--mlir kernel.mlir] [--cfg-num N]
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile

from core.dot_parser import parse_dot
from core.cycle_model import estimate_cycles
from extract.loop_info import extract_kernel_loops, KernelLoops
from arch.adg import load_adg


DEFAULT_CGRA_OPT = os.path.expanduser("~/adora/adora-compiler/build/bin/cgra-opt")
# cgra-opt looks up lib/DFG/Documents/GeneralOpName.txt relative to its cwd,
# so the pass pipeline must run from the compiler root.
DEFAULT_COMPILER_ROOT = os.path.expanduser("~/adora/adora-compiler")

# Polygeist emits a module attribute dict (dlti.dl_spec) that this cgra-opt
# build cannot parse; strip it before feeding the pipeline.
_MODULE_ATTRS = re.compile(r'module\s+attributes\s*\{.*?\}\s*\{', re.DOTALL)


def _strip_module_attrs(mlir_text: str) -> str:
    return _MODULE_ATTRS.sub('module {', mlir_text, count=1)


def gen_dots_from_mlir(mlir_path: str, cgra_opt: str, out_dir: str,
                       compiler_root: str = DEFAULT_COMPILER_ROOT) -> list[str]:
    """Raw MLIR -> ADORA.KernelOp -> CDFG dot."""
    with open(mlir_path) as _f:
        stripped = _strip_module_attrs(_f.read())
    tmp_mlir = os.path.join(out_dir, "stripped.mlir")
    with open(tmp_mlir, "w") as f:
        f.write(stripped)

    cmd = [cgra_opt, tmp_mlir,
           "--adora-extract-affine-for-to-kernel", "--adora-kernel-dfg-gen"]
    # The pass aborts during teardown AFTER writing dots; ignore non-zero exit
    # and instead check whether dots were produced.
    subprocess.run(cmd, cwd=compiler_root,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    produced = glob.glob(os.path.join(compiler_root, "*_CDFG.dot"))
    moved = []
    for src in produced:
        dst = os.path.join(out_dir, os.path.basename(src))
        shutil.move(src, dst)
        moved.append(dst)
    return sorted(moved)


def _loops_by_name(mlir_path: str | None) -> dict[str, KernelLoops]:
    if not mlir_path:
        return {}
    return {k.name: k for k in extract_kernel_loops(mlir_path)}


def _match_loops(loops_map: dict[str, KernelLoops], dot_name: str) -> KernelLoops | None:
    if dot_name in loops_map:
        return loops_map[dot_name]
    # dot names like "gemm_0" vs kernel "kernel_gemm"; try loose contains-match.
    for name, loops in loops_map.items():
        if dot_name in name or name in dot_name:
            return loops
    return None


def estimate_one(dot_path: str, loops: KernelLoops | None, *,
                 cfg_num=None, load_bytes=None, store_bytes=None,
                 num_alus=16, route_lat=0, overlap=True,
                 adg=None, num_tiles=1):
    cdfg = parse_dot(dot_path)
    if loops is None:
        loops = KernelLoops(name=os.path.basename(dot_path),
                            trip_counts=[1], elt_bytes=4)
    return estimate_cycles(cdfg, loops, cfg_num=cfg_num,
                           load_bytes=load_bytes, store_bytes=store_bytes,
                           num_alus=num_alus, route_lat=route_lat,
                           overlap=overlap, adg=adg, num_tiles=num_tiles)


def main():
    ap = argparse.ArgumentParser(description="ADORA MLIR cycle estimator")
    ap.add_argument("--mlir", help="kernel MLIR (trip-counts; or source for dot gen)")
    ap.add_argument("--dot", help="existing _CDFG.dot (skip cgra-opt)")
    ap.add_argument("--cgra-opt", default=DEFAULT_CGRA_OPT)
    ap.add_argument("--adg", help="ADG json: PE count / SPAD / cfg width")
    ap.add_argument("--num-alus", type=int, default=16,
                    help="PE/ALU count (ignored if --adg given)")
    ap.add_argument("--num-tiles", type=int, default=1,
                    help="kernel tile count (for tile-based cfgNum)")
    ap.add_argument("--route-lat", type=int, default=0,
                    help="routing latency per edge (0 = optimistic lower bound)")
    ap.add_argument("--no-overlap", action="store_true",
                    help="disable mem/compute overlap (serial sum)")
    ap.add_argument("--cfg-num", type=int, default=None,
                    help="override config words")
    ap.add_argument("--load-bytes", type=int, default=None)
    ap.add_argument("--store-bytes", type=int, default=None)
    args = ap.parse_args()

    if not args.dot and not args.mlir:
        ap.error("need --dot or --mlir")

    adg = load_adg(args.adg) if args.adg else None
    overlap = not args.no_overlap
    loops_map = _loops_by_name(args.mlir)

    common = dict(cfg_num=args.cfg_num, load_bytes=args.load_bytes,
                  store_bytes=args.store_bytes, num_alus=args.num_alus,
                  route_lat=args.route_lat, overlap=overlap,
                  adg=adg, num_tiles=args.num_tiles)

    if args.dot:
        dot_name = os.path.basename(args.dot).replace("_CDFG.dot", "")
        loops = _match_loops(loops_map, dot_name)
        print(estimate_one(args.dot, loops, **common).report())
        return

    # --mlir only: generate dots, estimate each kernel.
    with tempfile.TemporaryDirectory() as wd:
        dots = gen_dots_from_mlir(args.mlir, args.cgra_opt, wd)
        if not dots:
            print("no _CDFG.dot produced", file=sys.stderr)
            sys.exit(1)
        for dot in dots:
            dot_name = os.path.basename(dot).replace("_CDFG.dot", "")
            loops = _match_loops(loops_map, dot_name)
            print(estimate_one(dot, loops, **common).report())
            print()


if __name__ == "__main__":
    main()
