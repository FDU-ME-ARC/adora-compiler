"""Initiation-interval (II) estimation.

Mirrors mapper/src/mapper/mapping.cpp:1685 evaluateII():
    for each backedge:
        newII = ceil((srclat + routeLat - dstInportLat) / iterDist)
    II = max(all newII, ResMII)

Limitation: the structural dot has no routing latency (routeLat) and no final
schedule, so we approximate:
  - routeLat = 0           (optimistic; real routing adds delay -> we UNDER-estimate II)
  - dstInportLat = 0       (consumer input port at schedule slot 0 of its op)
  - srclat = sum of op latencies along the recurrence cycle (ASAP within the cycle)

ResMII = ceil(#compute_nodes / num_alus) -- resource lower bound.
"""

from __future__ import annotations

import math

from .dot_parser import CDFG
from .latency_table import op_latency


# Default CGRA array resources. Override via estimate_ii(..., num_alus=...).
DEFAULT_NUM_ALUS = 16     # e.g. 4x4 PE array; one ALU per PE
DEFAULT_ROUTE_LAT = 0     # dot carries no routing info; 0 = optimistic lower bound


def _recurrence_latency(cdfg: CDFG, backedge, route_lat: int) -> int:
    """Latency around the cycle closed by a backedge src->dst.

    The recurrence travels dst -> ... -> src along forward edges, then the
    backedge src -> dst closes it. We sum op latencies on the forward path.
    """
    src, dst = backedge.src, backedge.dst

    # BFS/DFS forward from dst back to src, accumulating max op-latency path.
    best = {dst: op_latency(cdfg.nodes[dst].opcode) if dst in cdfg.nodes else 0}
    stack = [dst]
    while stack:
        cur = stack.pop()
        for e in cdfg.forward_edges():
            if e.src != cur:
                continue
            nxt = e.dst
            nxt_lat = op_latency(cdfg.nodes[nxt].opcode) if nxt in cdfg.nodes else 0
            cand = best[cur] + route_lat + nxt_lat
            if cand > best.get(nxt, -1):
                best[nxt] = cand
                stack.append(nxt)
    # srclat = accumulated latency reaching the backedge source.
    return best.get(src, op_latency(cdfg.nodes[src].opcode) if src in cdfg.nodes else 0)


def rec_mii(cdfg: CDFG, route_lat: int = DEFAULT_ROUTE_LAT) -> int:
    ii = 1
    for be in cdfg.backedges():
        iter_dist = max(be.iterdist or 1, 1)
        srclat = _recurrence_latency(cdfg, be, route_lat)
        dst_inport_lat = 0
        new_ii = math.ceil((srclat + route_lat - dst_inport_lat) / iter_dist)
        ii = max(ii, new_ii)
    return ii


def res_mii(cdfg: CDFG, num_alus: int = DEFAULT_NUM_ALUS) -> int:
    n_compute = len(cdfg.compute_nodes())
    return max(1, math.ceil(n_compute / max(1, num_alus)))


def estimate_ii(cdfg: CDFG, num_alus: int = DEFAULT_NUM_ALUS,
                route_lat: int = DEFAULT_ROUTE_LAT) -> dict:
    rec = rec_mii(cdfg, route_lat)
    res = res_mii(cdfg, num_alus)
    return {"II": max(rec, res), "RecMII": rec, "ResMII": res}


if __name__ == "__main__":
    import sys
    from dot_parser import parse_dot
    g = parse_dot(sys.argv[1])
    print(estimate_ii(g))
