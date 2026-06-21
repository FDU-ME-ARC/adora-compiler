"""Cycle-accurate event-level simulator.

Replays the Event graph from extract/event_build under three resource units:

  * DMA engine (serial, n_dma=1): LOAD/STORE queue here.
  * PE array per tile: KERNEL runs here.
  * SPAD bank slots (core/sram_track.BankAllocator): a LOAD waits for its
    rotating slot to free; buffering depth (double/triple/serial) emerges from
    bank availability — never a knob.

Because the dependency DAG is already topologically ordered by
(iteration, program-position) — every edge points forward within an iteration
and cross-iteration ordering is carried by resource/slot free-times, not graph
edges — we can schedule events in their emission order and compute, for each:

    start  = max(parents' finish, resource free-time, slot free-time)
    finish = start + cost

Inter-iteration overlap (the fix for B-cyc-1) emerges naturally: iteration i's
KERNEL runs on the PE while iteration i+1's LOADs stream on the DMA into a free
bank slot.  A scalar accumulator's tiny re-load is expanded per iteration too,
so nothing is counted once.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .event import Event, Opcode, ResKind, Timeline
from .sram_track import SRAMAccess, SRAMTracker


def simulate(graph, conflict_mode: str = "off") -> Timeline:
    events: List[Event] = graph.events
    idx: Dict[int, Event] = graph.eid_index
    alloc = graph.allocator

    dma_free = 0                       # serial DMA engine (n_dma = 1)
    pe_free: Dict[int, int] = {}       # per-tile PE array
    tracker = SRAMTracker(mode=conflict_mode)

    # Proper list-scheduling: at each step, among READY events (all parents
    # finished), pick the one with the smallest feasible start time so an idle
    # resource is filled by whatever is ready — letting iteration i+1's loads
    # stream on the DMA while iteration i's KERNEL occupies the PE.  Tie-break by
    # eid keeps in-order issue stable.  This is what produces DMA||compute
    # overlap (and hence the inter-iteration pipeline) instead of serializing.
    ready = sorted(e.eid for e in events if e.ready_counter == 0)
    ready_set = set(ready)
    scheduled = 0
    n = len(events)
    no_progress = 0

    def feasible_start(e: Event) -> int:
        dep_ready = 0
        for pid in e.parents:
            dep_ready = max(dep_ready, idx[pid].finish)
        slot_ready = 0
        if e.res_kind == ResKind.DMA:
            res_ready = dma_free
            if e.opcode == Opcode.LOAD and e.buf:
                slot_ready = alloc.slot_free_at(e.buf, e.it)
        elif e.res_kind == ResKind.PE:
            res_ready = pe_free.get(e.tile, 0)
        else:
            res_ready = 0
        return max(dep_ready, res_ready, slot_ready)

    while scheduled < n:
        if not ready_set:
            stuck = [idx[i] for i in range(n) if idx[i].start < 0]
            raise RuntimeError(
                f"simulation hang at {scheduled}/{n} scheduled; "
                f"{len(stuck)} events deadlocked (dependency cycle?). "
                f"first stuck: {stuck[0] if stuck else None}")

        # choose the ready event with the smallest feasible start (tie: eid)
        best_eid = None
        best_start = None
        for eid in ready_set:
            st = feasible_start(idx[eid])
            if best_start is None or st < best_start or (st == best_start and eid < best_eid):
                best_start, best_eid = st, eid

        e = idx[best_eid]
        e.start = best_start
        e.finish = e.start + e.cost
        ready_set.discard(best_eid)
        scheduled += 1

        if e.res_kind == ResKind.DMA:
            dma_free = e.finish
        elif e.res_kind == ResKind.PE:
            pe_free[e.tile] = e.finish

        if e.opcode in (Opcode.LOAD, Opcode.ALLOC) and e.buf:
            _, bank, lo, hi = alloc.slot_window(e.buf, e.it)
            e.bank = bank

        for (buf, bit) in graph.consumes.get(e.eid, []):
            alloc.occupy(buf, bit, e.finish)

        # unlock children
        for cid in sorted(e.children):
            c = idx[cid]
            c.ready_counter -= 1
            if c.ready_counter == 0 and c.start < 0:
                ready_set.add(cid)

    # ----- post-pass: SRAM occupancy windows for Figure B -----
    # A buffer is held from when it is first written until its last reader
    # finishes.  For a LOAD the write starts at LOAD.start; for an ALLOC output
    # buffer the meaningful window starts when the KERNEL begins producing it.
    for e in events:
        if e.opcode not in (Opcode.LOAD, Opcode.ALLOC) or not e.buf:
            continue
        win_start = e.start
        release = e.finish
        for cid in e.children:
            release = max(release, idx[cid].finish)
            if e.opcode == Opcode.ALLOC:
                win_start = idx[cid].start  # alloc bank used once kernel writes
        _, bank, lo, hi = alloc.slot_window(e.buf, e.it)
        kind = "load" if e.opcode == Opcode.LOAD else "alloc"
        tracker.register(SRAMAccess(bank=bank, addr_lo=lo, addr_hi=hi,
                                    start=win_start, finish=release,
                                    kind=kind, buf=f"{e.buf}#{e.it}"))

    makespan = max((e.finish for e in events), default=0)
    return Timeline(events=list(events), makespan=makespan,
                    sram=tracker.accesses)


# ----------------------------------------------------------------------------
def summarize(graph, tl: Timeline) -> str:
    """Human-readable summary + sanity stats (inter-iteration load accounting)."""
    loads = [e for e in tl.events if e.opcode == Opcode.LOAD]
    kernels = [e for e in tl.events if e.opcode == Opcode.KERNEL]
    stores = [e for e in tl.events if e.opcode == Opcode.STORE]
    dma_busy = sum(e.cost for e in tl.events if e.res_kind == ResKind.DMA)
    pe_busy = sum(e.cost for e in kernels)
    total_load_cycles = sum(e.cost for e in loads)

    lines = [
        f"makespan            = {tl.makespan} cycles",
        f"outer_trip          = {graph.outer_trip}",
        f"events              = {len(tl.events)} "
        f"(loads={len(loads)} kernels={len(kernels)} stores={len(stores)})",
        f"DMA busy (serial)   = {dma_busy} cycles  "
        f"(= sum of all LOAD+STORE; this is the inter-iteration load that the "
        f"old formula counted ONCE)",
        f"  of which LOADs    = {total_load_cycles} cycles across "
        f"{len(loads)} load events ({len(loads)//max(1,graph.outer_trip)} per iter)",
        f"PE busy (compute)   = {pe_busy} cycles",
        f"overlap factor      = {(dma_busy + pe_busy) / max(1, tl.makespan):.2f}x "
        f"(>1 means DMA||compute overlap is happening)",
        f"bank buffering depth:",
    ]
    for s in graph.allocator.summary():
        lines.append("  " + s)
    return "\n".join(lines)
