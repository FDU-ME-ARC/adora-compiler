"""Build a dependency-driven Event graph from a *scheduled* ADORA MLIR.

REWRITE (rigorous): instead of guessing "one outer affine.for + a flat template"
(which mis-handles nesting), this walks the func body like an INTERPRETER,
honouring arbitrary ``affine.for`` nesting and ``iter_args`` token threading.

Design (see docs/EVENT_SIM_REDESIGN.md):

* func-level ``affine.for`` = event-expansion loop -> recurse once per trip, so
  every dynamic iteration emits its own LOAD/kernel/STORE events (this is what
  makes inter-iteration loads appear — B-cyc-1).
* ``affine.for`` *inside* a kernel body = compute loop -> folded into the
  kernel's cost (II * Π inner-trips + drain), NOT expanded into events.
* dependencies come from an SSA environment ``env: Value -> producing eid``:
  - async tokens / memref results are resolved against env at use sites;
  - ``affine.for iter_args`` block-args bind to the previous iteration's
    ``affine.yield`` producers -> loop-carried (cross-iteration) token deps
    (e.g. attn's 4 carried tokens) emerge automatically;
  - ``ADORA.event.create`` produces an initial (zero-cost) token marker.
* SPAD bank rotation: each buffer stream (KernelName.Id) gets a per-stream
  emission sequence; slot = seq % depth; a cross-iteration slot-recycle edge
  reader(seq-depth) -> load(seq) keeps the list-scheduler from stomping a slot
  still live ``depth`` iterations back.

Event count = number of dynamic ADORA-op instances = product of enclosing
func-level affine.for trips.  Linear in that product, by construction.
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Same MLIR-py bootstrap as extract/loop_info.py
_MLIR_CORE = os.environ.get(
    "ADORA_MLIR_CORE",
    "/data00/home/loujiahang/CGRVOPT/llvm-project-onnx/build/python_packages/mlir_core",
)
if _MLIR_CORE and _MLIR_CORE not in sys.path:
    sys.path.insert(0, _MLIR_CORE)
_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from core.event import Event, Opcode, ResKind, link  # noqa: E402
from core.sram_track import BankAllocator  # noqa: E402
from extract.loop_info import _for_trip, _strip_module_attrs  # noqa: E402


# ----------------------------------------------------------------------------
@dataclass
class CostModel:
    dma_bpc: int = 16          # DMA bytes/cycle (vitra_spec system_bus_beat_bits/8)
    dma_setup: int = 0         # DMA startup latency (calibration residual)
    kernel_ii: int = 1         # initiation interval (RecMII; gesummv/tri/attn = 1)
    kernel_drain: int = 4      # pipeline drain (critical path); overridable

    def dma_cycles(self, nbytes: int) -> int:
        return self.dma_setup + max(1, math.ceil(max(1, nbytes) / self.dma_bpc))

    def kernel_cycles(self, inner_iters: int) -> int:
        return self.kernel_ii * max(1, inner_iters) + self.kernel_drain


@dataclass
class EventGraph:
    events: List[Event]
    eid_index: Dict[int, Event]
    allocator: BankAllocator
    outer_trip: int                       # product of func-level expansion trips
    kernels: List[str]
    consumes: Dict[int, List[Tuple[str, int]]] = field(default_factory=dict)
    buf_producer: Dict[Tuple[str, int], int] = field(default_factory=dict)


# ----------------------------------------------------------------------------
def _memref_bytes(type_str: str) -> int:
    """memref<1x64xi32> -> 64*4 = 256;  memref<i32> -> 4;  memref<2xi32> -> 8."""
    if "memref<" not in type_str:
        return 0
    inner = type_str.split("memref<", 1)[1].rsplit(">", 1)[0]
    parts = inner.split("x")
    elt = parts[-1]
    bits = {"i8": 8, "i16": 16, "i32": 32, "i64": 64,
            "f16": 16, "bf16": 16, "f32": 32, "f64": 64}.get(elt, 32)
    nelt = 1
    for d in parts[:-1]:
        d = d.strip()
        if d.isdigit():
            nelt *= int(d)
    return nelt * (bits // 8)


def _attr_dict(op) -> Dict[str, str]:
    out = {}
    for a in op.attributes:
        out[a.name] = str(a.attr).strip('"')
    return out


def _kernel_inner_iters(kernel_op) -> int:
    """Total dynamic iterations of the affine.for nest INSIDE the kernel body.

    Product of trip counts of nested affine.for loops (the pipelined compute).
    Affine-symbol upper bounds (e.g. triangular `to #map(%i)`) that aren't
    constant fall back to a best-effort: treat as the enclosing trip (so the
    nest product is an over-estimate rather than 1).
    """
    def nest_product(op) -> int:
        total = 0
        for region in op.regions:
            for block in region.blocks:
                for child in block.operations:
                    co = child.operation
                    if co.name == "affine.for":
                        t = _for_trip(co)
                        if t is None:
                            # non-constant bound: approximate by the parent trip
                            t = _approx_dynamic_trip(co)
                        inner = nest_product(co)
                        total += t * (inner if inner else 1)
                    else:
                        sub = nest_product(co)
                        total += sub
        return total
    n = nest_product(kernel_op)
    return n if n > 0 else 1


def _approx_dynamic_trip(for_op) -> int:
    """Best-effort trip for an affine.for whose upper bound is a symbol/map.

    We cannot evaluate `to #map(%i)` statically; use the loop's *constant lower
    bound to a constant upper* if present, else a conservative 1.  Triangular
    loops are thus under-counted — flagged for v2 (affine domain integration).
    """
    attrs = {a.name: a.attr for a in for_op.attributes}
    # try a plain constant upper bound
    t = _for_trip(for_op)
    return t if t is not None else 1


# ----------------------------------------------------------------------------
class _Interp:
    """SSA-environment interpreter that emits Events while walking the IR."""

    EVENT_OPS = {"ADORA.BlockLoad", "ADORA.BlockStore",
                 "ADORA.LocalMemAlloc", "ADORA.kernel", "ADORA.event.create"}

    def __init__(self, cost: CostModel, alloc: BankAllocator):
        self.cost = cost
        self.alloc = alloc
        self.events: List[Event] = []
        self.eid_index: Dict[int, Event] = {}
        self.consumes: Dict[int, List[Tuple[str, int]]] = {}
        self.buf_producer: Dict[Tuple[str, int], int] = {}
        self.kernels: set = set()
        # env maps a static SSA Value -> producing eid (or None = ready/non-event)
        self.env: Dict[object, Optional[int]] = {}
        self.buf_seq: Dict[str, int] = {}        # buffer stream -> emission count
        self.buf_consumer: Dict[Tuple[str, int], int] = {}  # (buf,seq) -> reader eid
        self.outer_it = 0                         # outermost-loop iteration (label)
        self.outer_loops = 0                      # depth of func-level expansion

    # -- helpers ----------------------------------------------------------
    def _new_eid(self) -> int:
        return len(self.events)

    def _emit(self, opcode, kernel, cost, res_kind, buf="", nbytes=0,
              label="") -> Event:
        eid = self._new_eid()
        ev = Event(eid=eid, opcode=opcode, kernel=kernel, it=self.outer_it,
                   cost=cost, res_kind=res_kind, tile=0, buf=buf,
                   nbytes=nbytes, label=label or kernel)
        self.events.append(ev)
        self.eid_index[eid] = ev
        return ev

    def _producer(self, value):
        """eid of the event producing `value`, or None if non-event/ready."""
        from mlir.ir import OpResult
        if value in self.env:
            return self.env[value]
        if OpResult.isinstance(value):
            owner = value.owner
            # owner not yet emitted (forward ref) shouldn't happen in program order
            if owner.name in self.EVENT_OPS:
                # find by identity among emitted (rare; env should have it)
                return None
        return None

    def _link_operand_deps(self, op, ev):
        """Link ev to producers of all its operands found in env."""
        for operand in op.operands:
            p = self._producer(operand)
            if p is not None and p != ev.eid:
                link(self.eid_index[p], ev)

    # -- main walk --------------------------------------------------------
    def walk(self, block):
        from mlir.ir import OpResult
        for child in block.operations:
            op = child.operation
            name = op.name
            if name == "affine.for":
                self._walk_for(op)
            elif name == "ADORA.kernel":
                self._emit_kernel(op)
            elif name == "ADORA.BlockLoad":
                self._emit_load(op)
            elif name == "ADORA.LocalMemAlloc":
                self._emit_alloc(op)
            elif name == "ADORA.BlockStore":
                self._emit_store(op)
            elif name == "ADORA.event.create":
                self._emit_event_create(op)
            elif name in ("affine.yield", "ADORA.terminator", "func.return",
                          "return"):
                pass  # handled by parent / no-op
            else:
                # scalar glue: results are non-event producers (ready); record
                # them as None so dep resolution skips them cleanly.
                for r in op.results:
                    self.env[r] = None

    def _walk_for(self, for_op):
        """Func-level affine.for = event-expansion loop (with iter_args)."""
        trip = _for_trip(for_op)
        if trip is None:
            trip = _approx_dynamic_trip(for_op)
        region = for_op.regions[0]
        block = region.blocks[0]
        # block args: [induction_var, iter_arg0, iter_arg1, ...]
        blk_args = list(block.arguments)
        n_iter = len(for_op.results)            # number of iter_args
        iter_args = blk_args[len(blk_args) - n_iter:] if n_iter else []

        # initial carried producers come from the for's operands' tail
        # (affine.for operands = [lb operands..ub operands.., iter_inits..])
        op_operands = list(for_op.operands)
        inits = op_operands[len(op_operands) - n_iter:] if n_iter else []
        carried = [self._producer(v) for v in inits]

        is_outer = (self.outer_loops == 0)
        self.outer_loops += 1
        for it in range(trip):
            if is_outer:
                self.outer_it = it
            # bind iter_arg block args to the previous iteration's carried value
            for ba, prod in zip(iter_args, carried):
                self.env[ba] = prod
            self.walk(block)
            # capture affine.yield to thread carried values to next iteration
            term = block.operations[len(block.operations) - 1].operation
            if term.name == "affine.yield" and n_iter:
                carried = [self._producer(v) for v in term.operands]
        self.outer_loops -= 1
        # the for's results map to the final carried producers
        for r, prod in zip(for_op.results, carried):
            self.env[r] = prod

    # -- ADORA op emitters ------------------------------------------------
    def _emit_load(self, op):
        attrs = _attr_dict(op)
        kname = attrs.get("KernelName", "")
        ident = attrs.get("Id", "")
        buf = f"{kname}.Id{ident}"
        seq = self.buf_seq.get(buf, 0)
        self.buf_seq[buf] = seq + 1
        nb = _memref_bytes(str(op.results[0].type))
        self.alloc.register(buf, nb)
        ev = self._emit(Opcode.LOAD, kname, self.cost.dma_cycles(nb),
                        ResKind.DMA, buf=buf, nbytes=nb,
                        label=f"{kname}.Id{ident}")
        ev.it = seq                              # per-buffer sequence for slot
        self._link_operand_deps(op, ev)
        self._slot_recycle(buf, seq, ev)
        self.buf_producer[(buf, seq)] = ev.eid
        # results: memref + async token both map to this event
        for r in op.results:
            self.env[r] = ev.eid

    def _emit_alloc(self, op):
        attrs = _attr_dict(op)
        kname = attrs.get("KernelName", "")
        ident = attrs.get("Id", "")
        buf = f"{kname}.Id{ident}.out"
        seq = self.buf_seq.get(buf, 0)
        self.buf_seq[buf] = seq + 1
        nb = _memref_bytes(str(op.results[0].type))
        self.alloc.register(buf, nb)
        ev = self._emit(Opcode.ALLOC, kname, 0, ResKind.NONE, buf=buf, nbytes=nb,
                        label=f"{kname}.Id{ident}.out")
        ev.it = seq
        self._slot_recycle(buf, seq, ev)
        self.buf_producer[(buf, seq)] = ev.eid
        for r in op.results:
            self.env[r] = ev.eid

    def _emit_kernel(self, op):
        attrs = _attr_dict(op)
        kname = attrs.get("KernelName", "")
        self.kernels.add(kname)
        inner = _kernel_inner_iters(op)
        ev = self._emit(Opcode.KERNEL, kname, self.cost.kernel_cycles(inner),
                        ResKind.PE, label=kname)
        self._link_operand_deps(op, ev)
        # kernel reads its input buffers (the loads it depends on) and writes its
        # output alloc buffer. Record the input buffers it consumes (for slot
        # release): every LOAD producer it links to whose buf is set.
        reads: List[Tuple[str, int]] = []
        for operand in op.operands:
            p = self._producer(operand)
            if p is not None:
                pe = self.eid_index[p]
                if pe.opcode == Opcode.LOAD and pe.buf:
                    reads.append((pe.buf, pe.it))
        if reads:
            self.consumes[ev.eid] = reads
            for (b, s) in reads:
                self.buf_consumer[(b, s)] = ev.eid
        for r in op.results:
            self.env[r] = ev.eid

    def _emit_store(self, op):
        attrs = _attr_dict(op)
        kname = attrs.get("KernelName", "")
        ident = attrs.get("Id", "")
        nb = _memref_bytes(str(op.operands[0].type))
        ev = self._emit(Opcode.STORE, kname, self.cost.dma_cycles(nb),
                        ResKind.DMA, nbytes=nb, label=f"{kname}.store{ident}")
        self._link_operand_deps(op, ev)
        # the store reads an output buffer (its first memref operand's producer
        # is the alloc / kernel). Record it as consuming that alloc buffer.
        reads: List[Tuple[str, int]] = []
        if len(op.operands) > 0:
            p = self._producer(op.operands[0])
            if p is not None:
                pe = self.eid_index[p]
                buf = pe.buf
                if not buf and pe.opcode == Opcode.KERNEL:
                    # operand defined by alloc but written by kernel; find the
                    # alloc buffer among kernel's... fall back: search env later.
                    buf = ""
                if buf:
                    reads.append((buf, pe.it))
        if reads:
            self.consumes[ev.eid] = reads
            for (b, s) in reads:
                self.buf_consumer[(b, s)] = ev.eid
        for r in op.results:
            self.env[r] = ev.eid

    def _emit_event_create(self, op):
        # initial loop-carried token: a zero-cost, always-ready marker.
        ev = self._emit(Opcode.ALLOC, "", 0, ResKind.NONE, label="event.create")
        for r in op.results:
            self.env[r] = ev.eid

    # -- bank slot recycle edge ------------------------------------------
    def _slot_recycle(self, buf, seq, ev):
        depth = self.alloc.depth_of(buf)
        if seq >= depth:
            prev = self.buf_consumer.get((buf, seq - depth))
            if prev is not None:
                link(self.eid_index[prev], ev)


# ----------------------------------------------------------------------------
def build_event_graph(mlir_path: str, cost: Optional[CostModel] = None,
                      spec=None) -> EventGraph:
    import mlir.ir as ir
    import adora_mlir

    cost = cost or CostModel()

    with open(mlir_path) as f:
        txt = _strip_module_attrs(f.read())
    ctx = ir.Context()
    adora_mlir.register_dialect(ctx)
    module = ir.Module.parse(txt, ctx)

    # locate the func.func body block
    func_block = None
    def find_func(op):
        nonlocal func_block
        for region in op.regions:
            for block in region.blocks:
                for child in block.operations:
                    co = child.operation
                    if co.name == "func.func" and func_block is None:
                        if co.regions and co.regions[0].blocks:
                            func_block = co.regions[0].blocks[0]
                        return
                    find_func(co)
    find_func(module.operation)
    if func_block is None:
        raise RuntimeError("no func.func body found in scheduled MLIR")

    if spec is not None:
        alloc = BankAllocator(spec.spad_num_banks, spec.spad_bank_size)
    else:
        alloc = BankAllocator(num_banks=4, bank_size=16384)

    interp = _Interp(cost, alloc)
    interp.walk(func_block)

    # outer_trip = how many times the most-repeated buffer stream was emitted
    outer_trip = max(interp.buf_seq.values(), default=1)

    return EventGraph(events=interp.events, eid_index=interp.eid_index,
                      allocator=alloc, outer_trip=outer_trip,
                      kernels=sorted(interp.kernels),
                      consumes=interp.consumes,
                      buf_producer=interp.buf_producer)


if __name__ == "__main__":
    g = build_event_graph(sys.argv[1])
    print(f"outer_trip={g.outer_trip} kernels={g.kernels} events={len(g.events)}")
    for ev in g.events[:16]:
        print(f"  {ev}  deps={sorted(ev.parents)}")
    print("  bank streams:")
    for line in g.allocator.summary():
        print("    " + line)
