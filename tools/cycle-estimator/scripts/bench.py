"""Batch-run the cycle estimator over a tree of benchmarks and tabulate the
results, so kernels can be compared side by side.

For each benchmark dir it pairs the final CDFG dots (under `2_dfgs/` or
`1_kernels_opt/`) with a kernel MLIR (under `kernels/`, `1_kernels_opt/` or
`0_kernels/`) for strong-typed trip-counts, runs estimate_cycles, and prints a
table of (benchmark, kernel, II, inner, outer, drain, load, store, total).

Usage:
  ADORA_MLIR_CORE=<...>/mlir_core python3 scripts/bench.py \
      --root experiment/Cbenchmarks [--adg adg.json] [--no-overlap] [--csv out.csv]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.dot_parser import parse_dot
from core.cycle_model import estimate_cycles
from extract.loop_info import extract_kernel_loops, KernelLoops
from arch.adg import load_adg


def _final_dots(bench_dir: str) -> list[str]:
    """Prefer 2_dfgs/ final CDFGs; fall back to 1_kernels_opt/."""
    for sub in ("2_dfgs", "1_kernels_opt", "0_kernels_opt_beforefinal"):
        d = os.path.join(bench_dir, sub)
        hits = sorted(glob.glob(os.path.join(d, "*_CDFG.dot")))
        # skip before_map / intermediate variants, keep the plain final ones
        hits = [h for h in hits if "before_map" not in h]
        if hits:
            return hits
    return []


def _kernel_mlirs(bench_dir: str) -> list[str]:
    out = []
    for sub in ("kernels", "1_kernels_opt", "0_kernels"):
        d = os.path.join(bench_dir, sub)
        out += sorted(glob.glob(os.path.join(d, "*.mlir")))
    return out


def _loops_for(bench_dir: str) -> dict[str, KernelLoops]:
    """Extract all kernels' loops from every candidate MLIR in the bench."""
    loops: dict[str, KernelLoops] = {}
    for mlir in _kernel_mlirs(bench_dir):
        try:
            for k in extract_kernel_loops(mlir):
                loops.setdefault(k.name, k)
        except Exception:
            continue
    return loops


def _match(loops: dict[str, KernelLoops], dot_name: str) -> KernelLoops | None:
    if dot_name in loops:
        return loops[dot_name]
    for name, lp in loops.items():
        if dot_name in name or name in dot_name:
            return lp
    return None


def find_benchmarks(root: str) -> list[str]:
    """A benchmark dir is one that contains a dir with *_CDFG.dot under it."""
    benches = set()
    for dot in glob.glob(os.path.join(root, "**", "*_CDFG.dot"), recursive=True):
        # bench dir = parent of the 2_dfgs/1_kernels_opt/... subdir
        sub = os.path.dirname(dot)
        bench = os.path.dirname(sub)
        if "before_map" in dot or "DesignSpace" in dot:
            continue
        benches.add(bench)
    return sorted(benches)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="benchmark tree root")
    ap.add_argument("--adg", help="ADG json")
    ap.add_argument("--no-overlap", action="store_true")
    ap.add_argument("--num-alus", type=int, default=16)
    ap.add_argument("--csv", help="also write rows to this csv")
    ap.add_argument("--limit", type=int, default=0, help="cap benchmarks (0=all)")
    args = ap.parse_args()

    adg = load_adg(args.adg) if args.adg else None
    overlap = not args.no_overlap

    benches = find_benchmarks(args.root)
    if args.limit:
        benches = benches[:args.limit]

    rows = []
    hdr = ("benchmark", "kernel", "II", "inner", "outer", "drain",
           "load", "store", "total")
    print(f"{'benchmark':28} {'kernel':22} {'II':>3} {'inner':>6} "
          f"{'outer':>6} {'drain':>6} {'load':>7} {'store':>6} {'total':>8}")
    print("-" * 100)

    for bench in benches:
        short = os.path.relpath(bench, args.root)
        dots = _final_dots(bench)
        if not dots:
            continue
        loops = _loops_for(bench)
        for dot in dots:
            dot_name = os.path.basename(dot).replace("_CDFG.dot", "")
            lp = _match(loops, dot_name)
            try:
                cdfg = parse_dot(dot)
                if lp is None:
                    lp = KernelLoops(name=dot_name, trip_counts=[1], elt_bytes=4)
                est = estimate_cycles(cdfg, lp, num_alus=args.num_alus,
                                      overlap=overlap, adg=adg)
            except Exception as e:
                print(f"{short[:28]:28} {dot_name[:22]:22}  ERROR: {e}")
                continue
            trip_note = "" if _match(loops, dot_name) else " (trip?)"
            print(f"{short[:28]:28} {dot_name[:22]:22} {est.II:>3} "
                  f"{est.inner_trip:>6} {est.outer_trip:>6} {est.drain:>6} "
                  f"{est.load:>7} {est.store:>6} {est.total:>8}{trip_note}")
            rows.append((short, dot_name, est.II, est.inner_trip,
                         est.outer_trip, est.drain, est.load, est.store,
                         est.total))

    print("-" * 100)
    print(f"{len(rows)} kernels across {len(benches)} benchmarks "
          f"(overlap={'on' if overlap else 'off'}"
          f"{', adg' if adg else ''})")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(hdr)
            w.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
