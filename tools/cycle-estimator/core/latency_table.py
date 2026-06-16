"""Per-operation latency table.

Single source of truth: VITRA-CGRA/cgra-mg/src/main/scala/op/Operations.scala:71+
(comment: "latency including the register outside ALU").
DIV_LATENCY = 6 (Common.scala:37).

Do NOT use adora-compiler/lib/DFG/Documents/operations20241118.json -- it is a
stale copy whose values diverge from the hardware.
"""

DIV_LATENCY = 6

# OpName -> latency (cycles). Names match Operations.scala exactly (upper-case).
LATENCY: dict[str, int] = {
    # --- BasicOpInfoMap (integer ALU) ---
    "PASS": 1, "ADD": 1, "SUB": 1, "MUL": 1,
    "UDIV": DIV_LATENCY, "SDIV": DIV_LATENCY, "UREM": DIV_LATENCY, "SREM": DIV_LATENCY,
    "MOD": 1, "MIN": 1, "MAX": 1, "AND": 1, "OR": 1, "NOT": 1, "XOR": 1,
    "SHL": 1, "LSHR": 1, "ASHR": 1, "CSHL": 1, "CSHR": 1,
    "EQ": 1, "NE": 1, "ULT": 1, "ULE": 1, "SLT": 1, "SLE": 1, "SEL": 1,
    "MulAdd": 2,
    "INTLV4BASE": 1, "INTLV3BASE": 1, "INTLV2BASE": 1,
    "DEINTLV4BASE": 4, "DEINTLV3BASE": 3, "DEINTLV2BASE": 2,
    # --- Interleaver ---
    "INTLV4": 1, "INTLV3": 1, "INTLV2": 1,
    "DEINTLV4": 4, "DEINTLV3": 3, "DEINTLV2": 2,
    # --- FP32 ---
    "FMUL32": 3, "FADD32": 1, "FSUB32": 1, "FDIV32": 7, "FSQRT": 17,
    "FEQ32": 1, "FOLT32": 1, "FOLE32": 1, "FUNO32": 1, "FMA32": 4,
    # --- BF16 ---
    "BFMUL16": 3, "BFADD16": 1, "BFSUB16": 1, "BFDIV16": 4, "BFSQRT": 17,
    "BFEQ16": 1, "BFOLT16": 1, "BFOLE16": 1, "BFUNO16": 1, "BFMA16": 4,
    # --- Accumulative ---
    "ACC": 1, "ASUB": 1, "AMUL": 1, "ADIV": 1, "AMOD": 1,
    "AAND": 1, "AOR": 1, "AXOR": 1, "ASHL": 1, "ALSHR": 1, "AASHR": 1,
    "FACC32": 1, "FASUB32": 1, "BFACC16": 1, "BFASUB16": 1,
    # --- MAC ---
    "MAC": 2, "FMAC32": 4, "BFMAC32": 4,
    # --- Conditional accumulative ---
    "CACC": 1, "CASUB": 1, "CAMUL": 1, "CADIV": 1, "CAMOD": 1,
    "CAAND": 1, "CAOR": 1, "CAXOR": 1, "CASHL": 1, "CALSHR": 1, "CAASHR": 1,
    "CIACC": 1, "CIASUB": 1, "CIAMUL": 1, "CIADIV": 1, "CIAMOD": 1,
    "CIAAND": 1, "CIAOR": 1, "CIAXOR": 1, "CIASHL": 1, "CIALSHR": 1, "CIAASHR": 1,
    "ISEL": 1, "IACC": 1,
    # --- Load/Store ---
    "INPUT": 2, "OUTPUT": 1, "LOAD": 2, "STORE": 1,
    "CINPUT": 2, "COUTPUT": 1, "CLOAD": 2, "CSTORE": 1,
}

# Structural / control nodes carry no compute latency for the recurrence path,
# but `for` and `yield` route the loop-carried value (1 reg delay each).
_STRUCTURAL = {"for": 0, "yield": 0, "CONST": 0,
               "Input": 2, "Output": 1, "LocalAlloc": 0,
               "BlockLoad": 0, "BlockStore": 0}

# MLIR arith/math dialect op names (as emitted into the CDFG dot) -> hardware
# op names in LATENCY above. The DFG dot sometimes carries raw arith.* mnemonics
# (e.g. "divsi", "muli") instead of the hardware names (SDIV, MUL); map them so
# the estimator stays robust to front-end op-name choices.
_ARITH_ALIAS = {
    "addi": "ADD", "subi": "SUB", "muli": "MUL",
    "divsi": "SDIV", "divui": "UDIV", "remsi": "SREM", "remui": "UREM",
    "andi": "AND", "ori": "OR", "xori": "XOR",
    "shli": "SHL", "shrsi": "ASHR", "shrui": "LSHR",
    "addf": "FADD32", "subf": "FSUB32", "mulf": "FMUL32", "divf": "FDIV32",
    "negf": "FSUB32", "cmpi": "EQ", "cmpf": "FEQ32", "select": "SEL",
    "maxsi": "MAX", "minsi": "MIN", "maxf": "MAX", "minf": "MIN",
    "extf": "PASS", "truncf": "PASS", "extsi": "PASS", "extui": "PASS",
    "trunci": "PASS", "sitofp": "PASS", "fptosi": "PASS", "bitcast": "PASS",
    "sqrt": "FSQRT", "exp": "FSQRT", "log": "FSQRT",  # transcendental ~ heavy
}


def op_latency(opcode: str) -> int:
    """Return forward latency (cycles) for an op. Case-insensitive fallback;
    MLIR arith/math mnemonics are mapped to hardware op names."""
    if opcode in LATENCY:
        return LATENCY[opcode]
    if opcode in _STRUCTURAL:
        return _STRUCTURAL[opcode]
    # strip a leading dialect prefix like "arith." / "math." if present
    bare = opcode.split(".")[-1]
    if bare in _ARITH_ALIAS:
        return LATENCY[_ARITH_ALIAS[bare]]
    up = opcode.upper()
    if up in LATENCY:
        return LATENCY[up]
    # Unknown op: assume single-cycle ALU op, surface a warning to the caller.
    raise KeyError(f"unknown opcode {opcode!r}; add it to latency_table.LATENCY")


if __name__ == "__main__":
    for op in ("FMUL32", "FADD32", "ADD", "SDIV", "FDIV32", "MAC", "LOAD", "for"):
        print(f"{op:8} -> {op_latency(op)}")
