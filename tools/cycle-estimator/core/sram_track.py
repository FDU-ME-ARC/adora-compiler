"""SPAD occupancy tracking + bank allocator (replicates io_scheduler).

Two responsibilities:

1. ``SRAMAccess`` — one occupancy window (bank, address range, time window, the
   owning op).  Borrowed from AdaTileSim's ``sram_tracker``; feeds Figure B
   (time x address x who x which-instruction).

2. ``BankAllocator`` — replicates the mapper ``io_scheduler.cpp`` cur/old/older
   model so that buffering DEPTH is an *outcome* of bank availability, not a
   knob.  Each logical buffer stream (e.g. "gesummv_0.Id1") rotates through up
   to ``depth`` physical slots; iteration ``it`` uses ``slot = it % depth``.  A
   slot stays busy from its LOAD.start until the buffer's last reader (kernel)
   finishes.  With depth>=2 the next iteration's load can prefetch into a free
   slot while the current iteration computes -> inter-iteration overlap.

Capacity is honoured by bump-pointer allocation over ``num_banks * bank_size``
bytes: if too many streams x depth would overflow SPAD, depth is reduced (down
to 1 = serial) for the streams that no longer fit.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class SRAMAccess:
    bank: int
    addr_lo: int          # byte offset start (inclusive)
    addr_hi: int          # byte offset end (exclusive)
    start: int
    finish: int
    kind: str             # "load" | "store" | "kernel_read" | "kernel_write"
    buf: str              # logical buffer name (which MLIR op)


@dataclass
class _Stream:
    name: str
    buf_bytes: int
    depth: int
    base: int             # byte base address in the SPAD arena
    # per-slot free time (slot index -> cycle the slot becomes available)
    free: Dict[int, int] = field(default_factory=dict)


class BankAllocator:
    """cur/old/older style rotating bank allocator over a shared SPAD arena."""

    MAX_DEPTH = 3  # cur / old / older

    def __init__(self, num_banks: int, bank_size: int, default_depth: int = 3):
        self.num_banks = max(1, num_banks)
        self.bank_size = bank_size
        self.capacity = self.num_banks * bank_size
        self.default_depth = min(default_depth, self.MAX_DEPTH)
        self._streams: Dict[str, _Stream] = {}
        self._bump = 0  # next free byte in the arena

    # ------------------------------------------------------------------
    def register(self, name: str, buf_bytes: int) -> _Stream:
        """Register a logical buffer stream, reserving depth*buf_bytes bytes.

        Depth degrades from default_depth down to 1 if SPAD capacity is tight.
        """
        if name in self._streams:
            return self._streams[name]
        buf_bytes = max(1, buf_bytes)
        depth = self.default_depth
        while depth > 1 and self._bump + depth * buf_bytes > self.capacity:
            depth -= 1
        if self._bump + depth * buf_bytes > self.capacity:
            # Even depth=1 doesn't fit: clamp (over-subscribe, flag via depth=1).
            depth = 1
        s = _Stream(name=name, buf_bytes=buf_bytes, depth=depth, base=self._bump)
        self._bump += depth * buf_bytes
        self._streams[name] = s
        return s

    def slot_window(self, name: str, it: int) -> Tuple[int, int, int, int]:
        """Return (slot, bank, addr_lo, addr_hi) for stream ``name`` at iter ``it``."""
        s = self._streams[name]
        slot = it % s.depth
        addr_lo = s.base + slot * s.buf_bytes
        addr_hi = addr_lo + s.buf_bytes
        bank = addr_lo // self.bank_size
        return slot, bank, addr_lo, addr_hi

    def slot_free_at(self, name: str, it: int) -> int:
        """Earliest cycle the slot for (name, it) is free (0 if never used)."""
        s = self._streams[name]
        slot = it % s.depth
        return s.free.get(slot, 0)

    def occupy(self, name: str, it: int, until: int) -> None:
        """Mark the slot for (name, it) busy until ``until`` (last reader finish)."""
        s = self._streams[name]
        slot = it % s.depth
        s.free[slot] = max(s.free.get(slot, 0), until)

    def depth_of(self, name: str) -> int:
        return self._streams[name].depth

    def summary(self) -> List[str]:
        out = []
        for s in self._streams.values():
            out.append(f"{s.name}: depth={s.depth} buf={s.buf_bytes}B "
                       f"base={s.base} banks={s.base // self.bank_size}.."
                       f"{(s.base + s.depth * s.buf_bytes - 1) // self.bank_size}")
        return out


class SRAMTracker:
    """Collects SRAMAccess windows and (optionally) checks bank conflicts."""

    def __init__(self, mode: str = "warn"):
        self.accesses: List[SRAMAccess] = []
        self.mode = mode

    def register(self, acc: SRAMAccess) -> None:
        if self.mode != "off":
            self._check(acc)
        self.accesses.append(acc)

    def _check(self, new: SRAMAccess) -> None:
        for ex in self.accesses:
            if ex.bank != new.bank:
                continue
            if new.addr_lo >= ex.addr_hi or new.addr_hi <= ex.addr_lo:
                continue  # no address overlap
            if new.start >= ex.finish or new.finish <= ex.start:
                continue  # no time overlap (sequential reuse is fine)
            msg = (f"SRAM conflict on bank {new.bank}: "
                   f"{new.kind} '{new.buf}' [{new.addr_lo},{new.addr_hi}) "
                   f"@[{new.start},{new.finish}) vs "
                   f"{ex.kind} '{ex.buf}' [{ex.addr_lo},{ex.addr_hi}) "
                   f"@[{ex.start},{ex.finish})")
            if self.mode == "error":
                raise RuntimeError(msg)
            else:
                import warnings
                warnings.warn(msg, stacklevel=2)
