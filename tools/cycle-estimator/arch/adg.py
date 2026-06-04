"""Read CGRA architecture parameters from an ADG (Architecture Description
Graph) JSON file.

The ADG describes the PE-array structure, configuration storage and on-chip
SPAD capacity. It does NOT carry per-op latencies or DMA bandwidth (those were
deliberately left out of the ADG schema); the estimator's latency_table covers
that. This reader only extracts the structural / resource parameters the
cycle model needs.
"""
import json
from dataclasses import dataclass


@dataclass
class AdgParams:
    num_alus: int          # PE-array size = num_row * num_colum
    num_row: int
    num_colum: int
    cfg_data_width: int    # bits per config word
    cfg_spad_size: int     # config SPAD depth (words)
    max_cfg_data_num: int  # max config words per tile
    sum_blk_cfg_bits: int  # total config bits across all blocks
    data_width: int        # datapath / element bit width
    num_input: int         # IOB input lanes (memory parallelism)
    num_output: int
    iob_spad_bank_size: int


def load_adg(path: str) -> AdgParams:
    with open(path) as f:
        d = json.load(f)

    def g(key, default=0):
        v = d.get(key, default)
        return v if isinstance(v, int) else default

    num_row = g("num_row")
    num_colum = g("num_colum")
    return AdgParams(
        num_alus=num_row * num_colum,
        num_row=num_row,
        num_colum=num_colum,
        cfg_data_width=g("cfg_data_width"),
        cfg_spad_size=g("cfg_spad_size"),
        max_cfg_data_num=g("max_cfg_data_num"),
        sum_blk_cfg_bits=g("sum_blk_cfg_bits"),
        data_width=g("data_width"),
        num_input=g("num_input"),
        num_output=g("num_output"),
        iob_spad_bank_size=g("iob_spad_bank_size"),
    )


if __name__ == "__main__":
    import sys
    p = load_adg(sys.argv[1])
    print(p)
