"""Resource-occupancy timeline (Gantt) for an ADORA task schedule.

Turns per-kernel `CycleEstimate` (real cycles) + per-kernel `KernelSched`
(tile_set / hw_dep_type, from the llm-pipeline-schedule pass) into a Gantt
chart of PE-array occupancy over time, one row per CGRA tile.

Story: independent kernels (dep=LD_DEP_NONE) on DIFFERENT tiles start together
and their compute phases OVERLAP in time; dependent kernels (dep!=NONE) serialise
after the task they wait on. Same-tile kernels always serialise (one PE array).

This is a *schedule-decision view* built on the cycle model, NOT a cycle-accurate
RTL simulation (SPAD-bank conflicts / DMA contention / token sync are abstracted).

Public API:
    schedule(estimates, scheds) -> list[Bar]
    render(estimates, scheds, out_path, title=...) -> out_path
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Phase colours (paper-friendly, colour-blind-safe-ish).
_PHASE_COLORS = {
    "config":  "#9aa0a6",   # gray
    "load":    "#4c78a8",   # blue
    "compute": "#59a14f",   # green
    "store":   "#e8893a",   # orange
}


@dataclass
class Phase:
    kind: str       # config | load | compute | store
    start: int
    dur: int

    @property
    def end(self) -> int:
        return self.start + self.dur


@dataclass
class Bar:
    kernel: str
    tile: int
    start: int
    end: int
    phases: list[Phase] = field(default_factory=list)


def _kernel_phases(est, base: int) -> list[Phase]:
    """Lay out a single kernel's phases starting at absolute time `base`,
    mirroring core.cycle_model.estimate_cycles:
       config precedes the body; in overlap mode load/compute/store run
       concurrently inside the body window (body = max(load+store, compute));
       in serial mode they are laid end-to-end.
    """
    phases: list[Phase] = []
    t = base
    if est.config > 0:
        phases.append(Phase("config", t, est.config))
        t += est.config

    compute = est.outer_trip * (est.II * est.inner_trip + est.drain)
    if getattr(est, "overlap", True):
        # Heterogeneous-token HW overlaps data movement with compute, but the
        # data-flow order still holds: LOAD feeds the pipeline at the head,
        # COMPUTE runs concurrently, and STORE writes results back only AFTER
        # compute has produced them -> store sits at the COMPUTE TAIL, not right
        # after load. (Previously store was drawn at t+load, which looked like
        # "load then immediately store".)
        body = max(est.load + est.store, compute)  # body window length
        if est.load > 0:
            phases.append(Phase("load", t, est.load))
        if compute > 0:
            phases.append(Phase("compute", t, compute))
        if est.store > 0:
            # tail-align store to the end of the body window
            phases.append(Phase("store", t + body - est.store, est.store))
    else:
        if est.load > 0:
            phases.append(Phase("load", t, est.load)); t += est.load
        if compute > 0:
            phases.append(Phase("compute", t, compute)); t += compute
        if est.store > 0:
            phases.append(Phase("store", t, est.store)); t += est.store
    return phases


def schedule(estimates: dict, scheds: dict) -> list[Bar]:
    """List-scheduling: assign each kernel an absolute [start,end) window.

    estimates : {kernel_name -> CycleEstimate}
    scheds    : {kernel_name -> KernelSched}  (tiles, dep_type, dep_on)

    Rules:
      - same tile  -> serialise (a tile's PE array runs one kernel at a time)
      - dep_on set -> start no earlier than max(end of deps)
      - else (independent, different tile) -> start at 0 -> overlap
    """
    # source order = order of estimates dict (py3.7+ preserves insertion)
    order = list(estimates.keys())
    end_of: dict[str, int] = {}
    tile_free: dict[int, int] = {}     # earliest free time per tile
    bars: list[Bar] = []

    for name in order:
        est = estimates[name]
        ks = scheds.get(name)
        tiles = ks.tiles if ks else [0]
        tile = tiles[0]                # one kernel -> its first assigned tile
        total = est.total

        # earliest start from dependencies
        dep_ready = 0
        if ks and ks.dep_on:
            dep_ready = max((end_of.get(d, 0) for d in ks.dep_on), default=0)
        # earliest start from tile resource (same-tile serialisation)
        res_ready = tile_free.get(tile, 0)

        start = max(dep_ready, res_ready)
        end = start + total
        end_of[name] = end
        tile_free[tile] = end

        bars.append(Bar(kernel=name, tile=tile, start=start, end=end,
                         phases=_kernel_phases(est, start)))
    return bars


def render(estimates: dict, scheds: dict, out_path: str,
           title: str = "ADORA task schedule — PE-array occupancy",
           num_alus: int | None = None) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    bars = schedule(estimates, scheds)
    if not bars:
        raise ValueError("no kernels to plot")

    tiles = sorted({b.tile for b in bars})
    row_of = {t: i for i, t in enumerate(tiles)}
    makespan = max(b.end for b in bars)

    fig_h = 1.2 + 0.9 * len(tiles)
    fig_w = max(7.0, makespan / 120.0 + 3.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    row_h = 0.7
    for b in bars:
        y = row_of[b.tile]
        # faint full-kernel envelope
        ax.broken_barh([(b.start, b.end - b.start)], (y - row_h/2, row_h),
                       facecolors="none", edgecolors="#444", linewidth=0.8,
                       zorder=2)
        # phase sub-bars (compute drawn slightly thinner so concurrent
        # load/store remain visible at the band edges in overlap mode)
        for ph in b.phases:
            if ph.dur <= 0:
                continue
            if ph.kind == "compute":
                yy, hh = y - row_h*0.30, row_h*0.60
            elif ph.kind == "load":
                yy, hh = y + row_h*0.10, row_h*0.30
            elif ph.kind == "store":
                yy, hh = y - row_h*0.40, row_h*0.30
            else:  # config
                yy, hh = y - row_h/2, row_h
            ax.broken_barh([(ph.start, ph.dur)], (yy, hh),
                           facecolors=_PHASE_COLORS[ph.kind], alpha=0.92,
                           zorder=3)
        # kernel label
        ax.text(b.start + (b.end - b.start) / 2, y, b.kernel,
                ha="center", va="center", fontsize=9, fontweight="bold",
                color="white", zorder=4,
                bbox=dict(boxstyle="round,pad=0.15", fc="#33333388", ec="none"))

    ax.set_yticks(range(len(tiles)))
    ax.set_yticklabels([f"tile {t}" + (f"\n({num_alus} PE)" if num_alus else "")
                        for t in tiles])
    ax.set_ylim(-0.6, len(tiles) - 0.4)
    ax.set_xlim(0, makespan * 1.02)
    ax.set_xlabel("cycle (estimated)")
    ax.set_title(title)
    ax.grid(axis="x", linestyle=":", alpha=0.4, zorder=0)
    ax.invert_yaxis()

    legend = [Patch(facecolor=_PHASE_COLORS[k], label=k)
              for k in ("config", "load", "compute", "store")]
    ax.legend(handles=legend, loc="upper right", ncol=4, fontsize=8,
              framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    # also a PNG sibling if a PDF was requested
    if out_path.lower().endswith(".pdf"):
        png = out_path[:-4] + ".png"
        fig.savefig(png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _occupancy_steps(intervals):
    """Given [(start, end, bytes)], return (xs, ys) of a step function = total
    occupied bytes over time (stacked, since concurrent buffers coexist).

    All deltas at the SAME time point are merged before emitting a value, so a
    buffer ending exactly when another begins does NOT produce a spurious dip to
    zero (the previous version processed -b then +b separately, drawing a fake
    sawtooth down-and-up at shared boundaries)."""
    from collections import defaultdict
    delta = defaultdict(int)
    for s, e, b in intervals:
        if b <= 0 or e <= s:
            continue
        delta[s] += b
        delta[e] -= b
    if not delta:
        return [0], [0]
    xs, ys, cur = [], [], 0
    times = sorted(delta)
    # start baseline at first event time
    xs.append(times[0]); ys.append(0)
    for x in times:
        # value just before x (left edge of the step)
        xs.append(x); ys.append(cur)
        cur += delta[x]          # apply ALL deltas at this instant at once
        xs.append(x); ys.append(cur)
    return xs, ys


def render_sram(estimates: dict, scheds: dict, out_path: str,
                title: str = "ADORA task schedule — SPAD/SRAM occupancy",
                spad_capacity: int | None = None) -> str:
    """Figure B: on-chip SPAD/SRAM occupancy over time.

    Each kernel holds (load_bytes + store_bytes) of SPAD for its active window
    [start, end). Concurrent (overlapping) kernels stack, so the curve peaks
    when independent kernels run together -> shows on-chip memory pressure.
    A dashed line marks the SPAD capacity if known.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bars = schedule(estimates, scheds)
    if not bars:
        raise ValueError("no kernels to plot")

    # per-kernel buffer footprint
    foot = {}
    for name, est in estimates.items():
        foot[name] = (getattr(est, "load_bytes", 0) or 0) + \
                     (getattr(est, "store_bytes", 0) or 0)

    intervals = [(b.start, b.end, foot.get(b.kernel, 0)) for b in bars]
    xs, ys = _occupancy_steps(intervals)
    makespan = max(b.end for b in bars)
    peak = max(ys) if ys else 0

    fig_w = max(7.0, makespan / 120.0 + 3.0)
    fig, ax = plt.subplots(figsize=(fig_w, 3.6))

    # total occupancy curve
    ax.fill_between(xs, ys, step="post", alpha=0.25, color="#4c78a8",
                    zorder=2, label="total SPAD occupied")
    ax.step(xs, ys, where="post", color="#4c78a8", linewidth=1.6, zorder=3)

    # per-kernel contribution bands (stacked) for readability
    cmap = plt.get_cmap("tab10")
    base = [0.0]  # running baseline as a step fn is complex; annotate instead
    for i, b in enumerate(bars):
        fb = foot.get(b.kernel, 0)
        if fb <= 0:
            continue
        ax.broken_barh([(b.start, b.end - b.start)], (0, fb),
                       facecolors=cmap(i % 10), alpha=0.0, zorder=1)
        ax.text(b.start + (b.end - b.start) / 2, fb * 0.5,
                f"{b.kernel}\n{fb}B", ha="center", va="center",
                fontsize=8, color="#222", zorder=5)

    if spad_capacity:
        ax.axhline(spad_capacity, ls="--", color="#d62728", linewidth=1.4,
                   zorder=4, label=f"SPAD capacity ({spad_capacity}B)")
        ax.set_ylim(0, max(peak, spad_capacity) * 1.15)
    else:
        ax.set_ylim(0, peak * 1.25 if peak else 1)

    ax.set_xlim(0, makespan * 1.02)
    ax.set_xlabel("cycle (estimated)")
    ax.set_ylabel("SPAD bytes occupied")
    ax.set_title(title)
    ax.grid(axis="both", linestyle=":", alpha=0.4, zorder=0)
    ax.annotate(f"peak = {peak} B", xy=(0.99, 0.96), xycoords="axes fraction",
                ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="#ffffcc", ec="#aaa"))
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    if out_path.lower().endswith(".pdf"):
        fig.savefig(out_path[:-4] + ".png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path
