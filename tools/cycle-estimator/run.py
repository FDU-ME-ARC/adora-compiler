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
    ap.add_argument("--viz", default=None,
                    help="render a PE-array occupancy Gantt (PDF/PNG) of the "
                         "task schedule; requires a SCHEDULED MLIR (with "
                         "adora.tile_set) and matplotlib")
    ap.add_argument("--viz-sram", default=None,
                    help="render a SPAD/SRAM occupancy timeline (PDF/PNG); "
                         "same inputs as --viz")
    # ---- cycle-accurate event-level simulator (docs/EVENT_SIM_ARCH.md) ----
    ap.add_argument("--event-sim", action="store_true",
                    help="run the cycle-accurate event-level simulator on a "
                         "SCHEDULED MLIR (adora.scheduled); prints makespan + "
                         "overlap and (with --viz/--viz-sram) emits event-driven "
                         "Gantt / SPAD figures")
    ap.add_argument("--spec", default=None,
                    help="vitra_spec.json (DMA bandwidth / SPAD banks / PE); "
                         "if omitted, built-in defaults are used")
    ap.add_argument("--dma-bpc", type=int, default=None,
                    help="override DMA bytes/cycle (default from --spec)")
    ap.add_argument("--dma-setup", type=int, default=0,
                    help="DMA startup latency (calibration residual, default 0)")
    ap.add_argument("--max-cycles", type=int, default=None,
                    help="zoom event figures to the first N cycles")
    args = ap.parse_args()

    if args.event_sim:
        _run_event_sim(args)
        return

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
        estimates = {}                       # name -> CycleEstimate (source order)
        for dot in dots:
            dot_name = os.path.basename(dot).replace("_CDFG.dot", "")
            loops = _match_loops(loops_map, dot_name)
            est = estimate_one(dot, loops, **common)
            estimates[est.kernel] = est
            print(est.report())
            print()

        if args.viz or args.viz_sram:
            _render_viz(args.mlir, estimates,
                        viz=args.viz, viz_sram=args.viz_sram,
                        num_alus=(adg.num_alus if adg else args.num_alus),
                        adg=adg)


def _align(mlir_path, estimates):
    """Align estimate keys (CDFG dot names) with schedule keys (KernelNames) by
    source order; the dot `_<i>_CDFG` suffix is generated in kernel source order.
    Returns (aligned_est, aligned_sched) keyed by KernelName."""
    from extract.sched_info import extract_kernel_sched
    sched_list = extract_kernel_sched(mlir_path)
    est_items = list(estimates.items())
    aligned_est, aligned_sched = {}, {}
    if len(est_items) == len(sched_list):
        for (ename, est), ks in zip(est_items, sched_list):
            aligned_est[ks.name] = est
            aligned_sched[ks.name] = ks
    else:
        print(f"[viz] warning: {len(est_items)} estimates vs {len(sched_list)} "
              "scheduled kernels; tile mapping may be approximate",
              file=sys.stderr)
        sched_by = {s.name: s for s in sched_list}
        for ename, est in est_items:
            aligned_est[ename] = est
            aligned_sched[ename] = sched_by.get(ename)
    if not any(s and s.tiles != [0] for s in aligned_sched.values()):
        print("[viz] warning: no kernel has a non-[0] adora.tile_set; is this "
              "MLIR scheduled? (run cgra-opt --llm-pipeline-schedule first)",
              file=sys.stderr)
    return aligned_est, aligned_sched


def _spad_capacity(adg):
    """Total per-tile SPAD capacity in bytes, approximated from the ADG."""
    if adg is None:
        return None
    # iob_spad_bank_size (words) * num_input lanes * (data_width/8) bytes/word
    elt_b = max(1, (adg.data_width or 32) // 8)
    cap = (adg.iob_spad_bank_size or 0) * max(1, adg.num_input) * elt_b
    return cap or None


def _render_viz(mlir_path, estimates, viz=None, viz_sram=None,
                num_alus=None, adg=None):
    """Render figure A (PE-array Gantt) and/or figure B (SPAD occupancy)."""
    try:
        from viz.timeline import render, render_sram
    except Exception as e:                   # pragma: no cover
        print(f"[viz] skipped (import failed: {e})", file=sys.stderr)
        return
    aligned_est, aligned_sched = _align(mlir_path, estimates)

    if viz:
        out = render(aligned_est, aligned_sched, viz, num_alus=num_alus)
        print(f"[viz] wrote {out}"
              + (f" (+ {out[:-4]}.png)" if out.lower().endswith('.pdf') else ""),
              file=sys.stderr)
    if viz_sram:
        cap = _spad_capacity(adg)
        out = render_sram(aligned_est, aligned_sched, viz_sram,
                          spad_capacity=cap)
        print(f"[viz] wrote {out}"
              + (f" (+ {out[:-4]}.png)" if out.lower().endswith('.pdf') else ""),
              file=sys.stderr)


def _run_event_sim(args):
    """Cycle-accurate event-level simulation entry point.

    Requires a SCHEDULED MLIR (adora.scheduled).  Builds the Event graph
    (extract.event_build), runs the discrete-event simulator (core.event_sim),
    prints the makespan/overlap summary, and — if --viz/--viz-sram given —
    renders the event-driven Gantt (figure A) and SPAD occupancy (figure B).
    """
    if not args.mlir:
        print("--event-sim needs --mlir <scheduled.mlir>", file=sys.stderr)
        sys.exit(1)

    from extract.event_build import build_event_graph, CostModel
    from core.event_sim import simulate, summarize

    spec = None
    if args.spec:
        from arch.spec import load_spec
        spec = load_spec(args.spec)

    dma_bpc = args.dma_bpc or (spec.dma_bpc if spec else 16)
    cost = CostModel(dma_bpc=dma_bpc, dma_setup=args.dma_setup)

    g = build_event_graph(args.mlir, cost=cost, spec=spec)
    tl = simulate(g)
    print(summarize(g, tl))

    if args.viz:
        from viz.timeline import render_event_gantt
        out = render_event_gantt(tl, args.viz, max_cycles=args.max_cycles,
                                 title=f"{os.path.basename(args.mlir)} — event timeline")
        print(f"[viz] wrote {out}", file=sys.stderr)
    if args.viz_sram:
        from viz.timeline import render_event_sram
        bank = spec.spad_bank_size if spec else None
        out = render_event_sram(tl, args.viz_sram, bank_size=bank,
                                max_cycles=args.max_cycles,
                                title=f"{os.path.basename(args.mlir)} — SPAD occupancy")
        print(f"[viz] wrote {out}", file=sys.stderr)
    # When BOTH panels are requested, also emit a combined shared-x figure so
    # the hardware Gantt and the SPAD address map line up vertically in time.
    if args.viz and args.viz_sram:
        from viz.timeline import render_event_combined
        bank = spec.spad_bank_size if spec else None
        base = args.viz
        combo = (base[:-4] + "_combined" + base[-4:]) if base.lower().endswith(
            (".png", ".pdf")) else base + "_combined.png"
        out = render_event_combined(
            tl, combo, bank_size=bank, max_cycles=args.max_cycles,
            title=f"{os.path.basename(args.mlir)} — HW timeline + SPAD")
        print(f"[viz] wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
