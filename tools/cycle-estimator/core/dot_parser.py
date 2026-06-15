"""Parser for ADORA CDFG .dot files produced by the `adora-kernel-dfg-gen` pass.

The dot is purely structural: nodes carry an opcode (+ memref size/pattern for
load/store, value for const); edges carry an operand index, an optional
loop-carried iteration distance (`iterdist`), and a dependency type.
Latency and II are NOT in the dot -- they come from latency_table / ii_model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# `Add5`, `FMUL321`, `for6` -> opcode prefix + trailing numeric id.
_NODE_LINE = re.compile(r'^\s*([A-Za-z_]\w*?)(\d+)\s*\[(.*)\]\s*;?\s*$')
_EDGE_LINE = re.compile(r'^\s*(\w+)\s*->\s*(\w+)\s*\[(.*)\]\s*;?\s*$')
_ATTR = re.compile(r'(\w+)\s*=\s*"([^"]*)"|(\w+)\s*=\s*([^,\]]+)')


def _parse_attrs(body: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for m in _ATTR.finditer(body):
        if m.group(1) is not None:
            attrs[m.group(1)] = m.group(2)
        else:
            attrs[m.group(3)] = m.group(4).strip()
    return attrs


@dataclass
class Node:
    name: str          # full dot name, e.g. "FMUL321"
    opcode: str        # e.g. "FMUL32"
    nid: int           # trailing numeric id
    color: str = "black"
    # load/store-only:
    ref_name: str | None = None
    size: int | None = None       # memref size in bytes
    pattern: str | None = None
    value: str | None = None      # const hex value


@dataclass
class Edge:
    src: str
    dst: str
    operand: int = 0
    iterdist: int | None = None   # loop-carried distance; None for forward edges
    deptype: str | None = None
    style: str = "bold"

    @property
    def is_backedge(self) -> bool:
        # dashed edges carrying an iterdist are loop-carried (recurrence) edges.
        return self.iterdist is not None


@dataclass
class CDFG:
    name: str = "kernel"
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)

    def succ(self, name: str) -> list[Edge]:
        return [e for e in self.edges if e.src == name]

    def pred(self, name: str) -> list[Edge]:
        return [e for e in self.edges if e.dst == name]

    def backedges(self) -> list[Edge]:
        return [e for e in self.edges if e.is_backedge]

    def forward_edges(self) -> list[Edge]:
        return [e for e in self.edges if not e.is_backedge]

    def compute_nodes(self) -> list[Node]:
        """Nodes that consume an ALU slot (exclude structural for/yield/const/io)."""
        structural = {"for", "yield", "CONST", "Input", "Output",
                      "LocalAlloc", "BlockLoad", "BlockStore"}
        return [n for n in self.nodes.values() if n.opcode not in structural]


def parse_dot(path: str) -> CDFG:
    cdfg = CDFG()
    with open(path) as f:
        for raw in f:
            line = raw.rstrip("\n")
            stripped = line.strip()
            if not stripped or stripped.startswith(("Digraph", "digraph", "}", "{")):
                continue

            em = _EDGE_LINE.match(line)
            if em:
                src, dst, body = em.group(1), em.group(2), em.group(3)
                attrs = _parse_attrs(body)
                cdfg.edges.append(Edge(
                    src=src, dst=dst,
                    operand=int(attrs.get("operand", 0)),
                    iterdist=int(attrs["iterdist"]) if "iterdist" in attrs else None,
                    deptype=attrs.get("deptype"),
                    style=attrs.get("style", "bold"),
                ))
                continue

            nm = _NODE_LINE.match(line)
            if nm:
                prefix, num, body = nm.group(1), nm.group(2), nm.group(3)
                attrs = _parse_attrs(body)
                name = f"{prefix}{num}"
                cdfg.nodes[name] = Node(
                    name=name,
                    opcode=attrs.get("opcode", prefix),
                    nid=int(num),
                    color=attrs.get("color", "black"),
                    ref_name=attrs.get("ref_name"),
                    size=int(attrs["size"]) if "size" in attrs else None,
                    pattern=attrs.get("pattern"),
                    value=attrs.get("value"),
                )
                continue
    return cdfg


if __name__ == "__main__":
    import sys
    g = parse_dot(sys.argv[1])
    print(f"nodes={len(g.nodes)} edges={len(g.edges)} "
          f"backedges={len(g.backedges())} compute={len(g.compute_nodes())}")
    for n in g.nodes.values():
        extra = f" size={n.size}" if n.size else ""
        print(f"  {n.name:12} opcode={n.opcode:10} color={n.color}{extra}")
    for e in g.edges:
        bk = f" iterdist={e.iterdist}" if e.is_backedge else ""
        dt = f" deptype={e.deptype}" if e.deptype else ""
        print(f"  {e.src} -> {e.dst} operand={e.operand}{bk}{dt}")
