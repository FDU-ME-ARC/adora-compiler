"""Read CGRA hardware parameters from a ``vitra_spec.json`` file.

This is the AUTHORITATIVE source for structural + bandwidth constants that the
event-level simulator needs.  Note the distinction from ``adg.py``:

* ``vitra_cgra_adg.json`` is a *connection / placement graph* (GIB instances,
  module ids, x/y coordinates).  It carries NO width / bandwidth fields, so the
  old ``adg.py`` reader silently fell back to zeros for everything.  (That is a
  pre-existing bug; ``adg.py`` was pointed at the wrong file.)
* ``vitra_spec.json`` is the *architecture spec* — it has the real bus width,
  SPAD bank count / size, DMA burst parameters and PE-array dimensions.

Key fields (cgra_bf16 spec, cross-checked 2026-06):

    system_bus_beat_bits : 128   -> DMA bandwidth = 128/8 = 16 bytes/cycle
    spad_data_width      : 128   -> on-chip SPAD path 16 bytes/cycle (matches)
    dma_num_req_in_flight: 8     -> DMA outstanding requests (latency hiding)
    dma_lg_max_burst_size: 6     -> max burst = 2**6 beats
    tile_spad_num_banks  : 4     -> SPAD banks per tile (double/triple buffering)
    spad_bank_lg_size    : 14    -> bank capacity = 2**14 = 16384 bytes
    tile_num_row x col   : 8 x 2 -> 16 PEs per tile (ResMII)
    cgra_tile_num        : 8     -> tiles
    cgra_cfg_data_width  : 32    -> config word bits
    cgra_data_width      : 16    -> datapath element bits (bf16)
"""
from __future__ import annotations
import json
import math
from dataclasses import dataclass


@dataclass
class SpecParams:
    # --- PE array ---
    tile_num_row: int          # PE rows per tile
    tile_num_column: int       # PE columns per tile
    num_alus: int              # PEs per tile = row * column
    cgra_tile_num: int         # number of tiles
    # --- config ---
    cfg_data_width: int        # bits per config word
    cfg_blk_offset: int        # config block offset (cfg_num // cfg_blk_offset)
    # --- datapath ---
    data_width: int            # datapath element bit width
    # --- DMA / bus ---
    system_bus_beat_bits: int  # system bus width in bits
    dma_bpc: int               # DMA bytes per cycle = system_bus_beat_bits // 8
    dma_num_req_in_flight: int
    dma_max_burst: int         # 2 ** dma_lg_max_burst_size (beats)
    # --- SPAD ---
    spad_num_banks: int        # SPAD banks per tile
    spad_bank_size: int        # bytes per bank = 2 ** spad_bank_lg_size
    spad_data_width: int       # SPAD data path bits


def load_spec(path: str) -> SpecParams:
    with open(path) as f:
        d = json.load(f)

    def g(key, default=0):
        v = d.get(key, default)
        return v if isinstance(v, (int, float)) else default

    row = int(g("tile_num_row", 1))
    col = int(g("tile_num_column", 1))
    bus_bits = int(g("system_bus_beat_bits", 128))

    return SpecParams(
        tile_num_row=row,
        tile_num_column=col,
        num_alus=row * col,
        cgra_tile_num=int(g("cgra_tile_num", 1)),
        cfg_data_width=int(g("cgra_cfg_data_width", 32)),
        cfg_blk_offset=int(g("cgra_cfg_blk_offset", 3)),
        data_width=int(g("cgra_data_width", 16)),
        system_bus_beat_bits=bus_bits,
        dma_bpc=max(1, bus_bits // 8),
        dma_num_req_in_flight=int(g("dma_num_req_in_flight", 1)),
        dma_max_burst=1 << int(g("dma_lg_max_burst_size", 0)),
        spad_num_banks=int(g("tile_spad_num_banks", 1)),
        spad_bank_size=1 << int(g("spad_bank_lg_size", 14)),
        spad_data_width=int(g("spad_data_width", bus_bits)),
    )


if __name__ == "__main__":
    import sys
    p = load_spec(sys.argv[1])
    print(p)
    print(f"\nDMA bandwidth = {p.dma_bpc} bytes/cycle  "
          f"(system_bus_beat_bits={p.system_bus_beat_bits})")
    print(f"SPAD = {p.spad_num_banks} banks x {p.spad_bank_size} bytes/tile")
    print(f"PE  = {p.num_alus} ALUs/tile x {p.cgra_tile_num} tiles")
