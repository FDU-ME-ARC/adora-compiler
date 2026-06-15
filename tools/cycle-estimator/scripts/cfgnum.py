"""Extract per-kernel cfgNum from a generated CGRA pytest kernel.

The emitter (mapper EmitPytest) writes a docstring per kernel:

    \"\"\" kernel: jacobi_1d_kernel_0,  cfgNum: 30\"\"\"

cfgNum = number of config words. The simulator's config-load overhead is
cfgNum // 3 cycles (test_runif.py:1209-1211).
"""

from __future__ import annotations

import re


_CFGNUM = re.compile(r'kernel:\s*(\w+)\s*,\s*cfgNum:\s*(\d+)')


def parse_cfgnums(py_kernel_path: str) -> dict[str, int]:
    """kernel_name -> cfgNum, from a generated *.py kernel file."""
    out: dict[str, int] = {}
    with open(py_kernel_path) as f:
        for line in f:
            m = _CFGNUM.search(line)
            if m:
                out[m.group(1)] = int(m.group(2))
    return out


def cfgnum_for(py_kernel_path: str, kernel_name: str) -> int | None:
    table = parse_cfgnums(py_kernel_path)
    if kernel_name in table:
        return table[kernel_name]
    for name, n in table.items():
        if kernel_name.startswith(name) or name.startswith(kernel_name):
            return n
    return None


if __name__ == "__main__":
    import sys
    print(parse_cfgnums(sys.argv[1]))
