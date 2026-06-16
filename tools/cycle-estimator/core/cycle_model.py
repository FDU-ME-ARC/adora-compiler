"""Cycle-count model for an ADORA kernel.

Cycles ~= config_overhead(cfgNum)
        + block_load(bytes)
        + II * inner_trip            (pipelined innermost loop)
        + pipeline_drain             (DFG critical-path latency)
        + block_store(bytes)
all repeated `outer_trip` times for the loop body.

Sources:
  - config_overhead: CGRA-Cocotb-Sim/server/test_runif.py:1209-1211
        enable_config delay = config_ptr.size // 3  (config words / 3)
  - II: ii_model (mirrors mapping.cpp:1685)
  - pipeline_drain: longest op-latency path in the DFG (~ mapper latencyBound,
        mapping.cpp:1214)
  - block_load/store: bytes / dma_bytes_per_cycle  (bandwidth TBD; default 4 B/cyc)
"""

from __future__ import annotations

from dataclasses import dataclass

from .dot_parser import CDFG
from .ii_model import estimate_ii
from .latency_table import op_latency
from extract.loop_info import KernelLoops
from arch.adg import AdgParams


DEFAULT_DMA_BYTES_PER_CYCLE = 4   # TODO: calibrate against test_runif.py / hardware


def critical_path_latency(cdfg: CDFG) -> int:
    """Longest op-latency path over forward edges (pipeline fill/drain)."""
    fwd = cdfg.forward_edges()
    succ: dict[str, list[str]] = {}
    indeg: dict[str, int] = {n: 0 for n in cdfg.nodes}
    for e in fwd:
        succ.setdefault(e.src, []).append(e.dst)
        indeg[e.dst] = indeg.get(e.dst, 0) + 1

    def lat(name: str) -> int:
        return op_latency(cdfg.nodes[name].opcode) if name in cdfg.nodes else 0

    # Longest path via topological relaxation (DFG forward graph is acyclic).
    dist = {n: lat(n) for n in cdfg.nodes}
    order = [n for n in cdfg.nodes if indeg.get(n, 0) == 0]
    seen = list(order)
    i = 0
    while i < len(seen):
        cur = seen[i]; i += 1
        for nxt in succ.get(cur, []):
            if dist[cur] + lat(nxt) > dist.get(nxt, 0):
                dist[nxt] = dist[cur] + lat(nxt)
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                seen.append(nxt)
    return max(dist.values()) if dist else 0


def io_bytes_from_dot(cdfg: CDFG) -> tuple[int, int]:
    """Sum data-movement bytes from Input (load) and Output (store) nodes.

    Input/Output nodes carry a `size` attribute = total bytes streamed for that
    operand across the whole loop nest (see dot `pattern` = base,count,...).
    """
    load = sum(n.size or 0 for n in cdfg.nodes.values() if n.opcode == "Input")
    store = sum(n.size or 0 for n in cdfg.nodes.values() if n.opcode == "Output")
    return load, store


def cfg_num_from_tiles(num_tiles: int, adg: AdgParams) -> int:
    """Estimate total config words from the kernel's tile count and the ADG's
    per-tile config capacity. Each tile loads its own configuration."""
    return max(num_tiles, 1) * adg.max_cfg_data_num


def config_overhead(cfg_num: int | None) -> int:
    if not cfg_num:
        return 0
    return cfg_num // 3


def dma_cycles(byte_count: int, bw: int = DEFAULT_DMA_BYTES_PER_CYCLE) -> int:
    if not byte_count:
        return 0
    return -(-byte_count // bw)  # ceil


@dataclass
class CycleEstimate:
    kernel: str
    II: int
    rec_mii: int
    res_mii: int
    inner_trip: int
    outer_trip: int
    drain: int
    config: int
    load: int
    store: int
    total: int
    overlap: bool = True
    load_bytes: int = 0     # raw data-movement bytes (for SRAM-occupancy viz)
    store_bytes: int = 0

    def report(self) -> str:
        return (
            f"kernel        : {self.kernel}\n"
            f"II            : {self.II}  (RecMII={self.rec_mii}, ResMII={self.res_mii})\n"
            f"inner_trip    : {self.inner_trip}\n"
            f"outer_trip    : {self.outer_trip}\n"
            f"pipeline_drain: {self.drain}\n"
            f"config        : {self.config}\n"
            f"block_load    : {self.load}\n"
            f"block_store   : {self.store}\n"
            f"mem/compute   : {'overlap (max)' if self.overlap else 'serial (sum)'}\n"
            f"---------------------------------\n"
            f"TOTAL cycles  : {self.total}  (optimistic; routeLat=0)"
        )


def estimate_cycles(cdfg: CDFG, loops: KernelLoops, *,
                    cfg_num: int | None = None,
                    load_bytes: int | None = None, store_bytes: int | None = None,
                    num_alus: int = 16, route_lat: int = 0,
                    overlap: bool = True,
                    adg: AdgParams | None = None,
                    num_tiles: int = 1) -> CycleEstimate:
    if adg is not None:
        num_alus = adg.num_alus
        if cfg_num is None:
            cfg_num = cfg_num_from_tiles(num_tiles, adg)
    ii_info = estimate_ii(cdfg, num_alus=num_alus, route_lat=route_lat)
    II = ii_info["II"]
    drain = critical_path_latency(cdfg)
    inner = loops.inner_trip
    outer = loops.outer_trip

    # Auto-extract whole-nest data-movement bytes from the dot if not supplied.
    if load_bytes is None or store_bytes is None:
        auto_load, auto_store = io_bytes_from_dot(cdfg)
        load_bytes = auto_load if load_bytes is None else load_bytes
        store_bytes = auto_store if store_bytes is None else store_bytes

    cfg_c = config_overhead(cfg_num)
    load_c = dma_cycles(load_bytes)
    store_c = dma_cycles(store_bytes)

    # Compute pipeline overlaps the whole loop nest; load/store cover the full
    # nest (dot `size` is per-nest), so they are charged once, not per outer iter.
    compute = outer * (II * inner + drain)
    if overlap:
        # Heterogeneous-token hardware overlaps data movement with compute:
        # the two run concurrently, so the kernel body costs max(memory, compute)
        # rather than their sum. Config load still precedes the body.
        total = cfg_c + max(load_c + store_c, compute)
    else:
        total = cfg_c + load_c + compute + store_c

    return CycleEstimate(
        kernel=loops.name, II=II,
        rec_mii=ii_info["RecMII"], res_mii=ii_info["ResMII"],
        inner_trip=inner, outer_trip=outer, drain=drain,
        config=cfg_c, load=load_c, store=store_c, total=total, overlap=overlap,
        load_bytes=load_bytes, store_bytes=store_bytes,
    )
