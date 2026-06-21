"""Regression tests for the cycle-accurate event-level simulator.

Covers the three structural cases:
  * gesummv : outer affine.for ENCLOSES the ADORA ops (event expansion ×64),
              cross-kernel RAR via async token.
  * tri     : ADORA ops run ONCE; affine.for nests live INSIDE kernels (cost).
  * attn    : outer affine.for with iter_args threading 4 loop-carried tokens
              (cross-iteration store->store dependency).

Run:  python3 -m pytest tests/test_event_sim.py -q
  or:  python3 tests/test_event_sim.py
"""
from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extract.event_build import build_event_graph, CostModel
from core.event_sim import simulate
from core.event import Opcode

SPEC = "/data00/home/loujiahang/adora/adora-compiler/test/spec/cgra_bf16/vitra_spec.json"
GESUMMV = "/tmp/g_full_cc/adora-cc-ir/3_task-schedule/gesummv.final.mlir"
TRI = "/tmp/tri_test/adora-cc-ir/3_task-schedule/tri.final.mlir"
ATTN = "/tmp/adoracc_attn/adora-cc-ir/3_task-schedule/attn.final.mlir"


def _spec():
    from arch.spec import load_spec
    return load_spec(SPEC)


def _sim(path):
    spec = _spec()
    g = build_event_graph(path, cost=CostModel(dma_bpc=spec.dma_bpc), spec=spec)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tl = simulate(g, conflict_mode="warn")
        conflicts = len(w)
    return g, tl, conflicts


# ---------------------------------------------------------------------------
def test_gesummv_expansion_and_cross_kernel():
    g, tl, conflicts = _sim(GESUMMV)
    # 64 outer iters x (3 load + alloc + kernel + store) x 2 kernels = 768
    assert g.outer_trip == 64
    assert len(g.events) == 768
    loads = [e for e in g.events if e.opcode == Opcode.LOAD]
    assert len(loads) == 384, "inter-iteration loads must be expanded (B-cyc-1)"
    # cross-kernel RAR: gesummv_1.Id2 (E8) depends on gesummv_0.Id2 (E2)
    e8 = g.eid_index[8]
    assert 2 in e8.parents, "cross-kernel async-token RAR edge missing"
    # makespan must exceed the old compute-only formula (4480) and show overlap
    assert tl.makespan > 4480
    dma = sum(e.cost for e in g.events if e.opcode in (Opcode.LOAD, Opcode.STORE))
    pe = sum(e.cost for e in g.events if e.opcode == Opcode.KERNEL)
    assert (dma + pe) / tl.makespan > 1.0, "no DMA||compute overlap"
    assert conflicts == 0, "SPAD bank conflicts (slot recycling broken)"


def test_tri_ops_run_once_loops_are_cost():
    g, tl, conflicts = _sim(TRI)
    # ADORA ops at func top level -> run once; affine.for nests are INSIDE kernels
    assert g.outer_trip == 1
    assert len(g.events) == 7   # 1 load + 2 alloc + 2 kernel + 2 store
    kernels = [e for e in g.events if e.opcode == Opcode.KERNEL]
    assert len(kernels) == 2
    # k_0 has a 16x16 compute nest -> cost must reflect ~256 inner iters
    k0 = next(e for e in kernels if e.kernel == "k_0")
    assert k0.cost > 200, "kernel internal affine.for nest not folded into cost"
    assert conflicts == 0


def test_attn_loop_carried_tokens():
    g, tl, conflicts = _sim(ATTN)
    assert g.outer_trip == 8
    # iter-1's first attention_0 store (E25) must depend on iter-0 stores
    # (E9 / E13) via the affine.yield carried tokens -> loop-carried dep.
    e25 = g.eid_index[25]
    assert any(p in (9, 12, 13) for p in e25.parents), \
        "loop-carried token threading (iter_args/yield) not captured"
    # event.create initial tokens are zero-cost markers
    markers = [e for e in g.events if e.label == "event.create"]
    assert len(markers) == 4 and all(e.cost == 0 for e in markers)
    assert conflicts == 0


def test_all_examples_run_without_hang():
    for path in (GESUMMV, TRI, ATTN):
        g, tl, conflicts = _sim(path)
        assert tl.makespan > 0
        assert all(e.start >= 0 and e.finish >= e.start for e in g.events)


if __name__ == "__main__":
    fns = [test_gesummv_expansion_and_cross_kernel,
           test_tri_ops_run_once_loops_are_cost,
           test_attn_loop_carried_tokens,
           test_all_examples_run_without_hang]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print("\nall event-sim regression tests passed")
