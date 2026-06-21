"""Event model for the cycle-accurate event-level simulator.

Borrowed paradigm (not copied) from AdaTileSim's ``Instruction``: a
dependency-driven event whose ``ready_counter`` tracks how many predecessors
are still pending; when it hits zero the event may execute, and on ``finish``
it notifies its children.

Each ADORA op (BlockLoad / LocalMemAlloc / kernel / BlockStore), replicated
once per outer-loop iteration, becomes one Event.  Dependencies come from the
scheduled MLIR's explicit async tokens + memref SSA (see extract/event_build.py),
so we never *guess* the schedule — we replay it.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set


class Opcode(Enum):
    LOAD = auto()    # ADORA.BlockLoad
    STORE = auto()   # ADORA.BlockStore
    ALLOC = auto()   # ADORA.LocalMemAlloc (zero cost, reserves a bank)
    KERNEL = auto()  # ADORA.kernel
    CONFIG = auto()  # per-tile configuration load


class ResKind(Enum):
    """The three resource-unit classes (== Gantt row groups)."""
    DMA = auto()    # LOAD / STORE occupy the (serial) DMA engine
    PE = auto()     # KERNEL occupies a tile's PE array
    NONE = auto()   # ALLOC / CONFIG: no timed resource (ALLOC only reserves bank)


@dataclass
class Event:
    eid: int
    opcode: Opcode
    kernel: str                 # owning KernelName, e.g. "gesummv_0"
    it: int                     # outer-loop iteration index (0 .. outer_trip-1)
    cost: int                   # duration in cycles (see core/cost.py / event_build)
    res_kind: ResKind
    tile: int = 0               # which PE tile this runs on (PE events)

    # --- buffer / SRAM ---
    buf: str = ""               # logical buffer name, e.g. "gesummv_0.Id1" (None for kernel)
    nbytes: int = 0             # bytes moved (LOAD/STORE) or buffer size (ALLOC)

    # --- dependency-driven scheduling (AdaTileSim paradigm) ---
    ready_counter: int = 0      # # of predecessors not yet finished
    parents: Set[int] = field(default_factory=set)
    children: Set[int] = field(default_factory=set)

    # --- simulation outputs (filled by event_sim) ---
    start: int = -1
    finish: int = -1
    bank: int = -1              # SPAD bank assigned (LOAD/ALLOC), -1 = n/a
    label: str = ""             # human label for the Gantt / SRAM figures

    # ------------------------------------------------------------------
    def is_ready(self) -> bool:
        return self.ready_counter == 0 and self.start < 0

    def do_finish(self, cyc: int, events: Dict[int, "Event"]) -> List[int]:
        """Mark finished at ``cyc`` and decrement children's ready_counter.

        Returns the list of children that became ready (ready_counter hit 0).
        """
        self.finish = cyc
        newly_ready: List[int] = []
        for cid in sorted(self.children):
            c = events[cid]
            c.ready_counter -= 1
            if c.ready_counter == 0 and c.start < 0:
                newly_ready.append(cid)
        return newly_ready

    def __repr__(self) -> str:
        return (f"E{self.eid}({self.opcode.name} {self.kernel} it={self.it} "
                f"cost={self.cost} buf={self.buf!r} "
                f"[{self.start},{self.finish}) bank={self.bank})")


def link(parent: Event, child: Event) -> None:
    """Wire a dependency edge parent -> child (idempotent)."""
    if child.eid in parent.children:
        return
    parent.children.add(child.eid)
    child.parents.add(parent.eid)
    child.ready_counter += 1


@dataclass
class Timeline:
    """Result of a simulation run: scheduled events + makespan."""
    events: List[Event]
    makespan: int
    # SRAM occupancy windows (one per LOAD/STORE/kernel access), filled by sim
    sram: list = field(default_factory=list)

    def by_resource(self):
        """Group events into Gantt rows: ('DMA',), ('PE', tile), ('SPAD', bank)."""
        rows: Dict[tuple, List[Event]] = {}
        for e in self.events:
            if e.res_kind == ResKind.DMA:
                key = ("DMA",)
            elif e.res_kind == ResKind.PE:
                key = ("PE", e.tile)
            else:
                continue
            rows.setdefault(key, []).append(e)
        return rows
