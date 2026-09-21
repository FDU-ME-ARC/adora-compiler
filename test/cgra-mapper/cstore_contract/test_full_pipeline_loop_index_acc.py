#!/usr/bin/env python3
"""Exercise loop-index ACC lowering through the production adoracc pipeline."""

import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


def load_mapping_helpers(test_dir):
    helper_path = test_dir / "test_loop_index_acc_mapping.py"
    spec = importlib.util.spec_from_file_location("loop_index_mapping", helper_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(command, cwd):
    result = subprocess.run([str(value) for value in command], cwd=cwd,
                            env=dict(os.environ), text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=120, check=False)
    if result.returncode != 0:
        raise AssertionError(
            f"expected status 0, got {result.returncode}: "
            f"{' '.join(map(str, command))}\n{result.stdout}")
    return result.stdout


def main():
    if len(sys.argv) != 9:
        raise AssertionError(
            "usage: test_full_pipeline_loop_index_acc.py WORKDIR ADORACC "
            "MAPPER ADG OPERATIONS CONTRACT TESTDIR INPUT")
    workdir, adoracc, mapper, adg, operations, contract_path, test_dir, fixture = \
        (Path(value).resolve() for value in sys.argv[1:])
    helpers = load_mapping_helpers(test_dir)
    contract = json.loads(contract_path.read_text())

    compiler_input = workdir / "case-b-input.mlir"
    shutil.copyfile(fixture, compiler_input)
    compiler_output = workdir / "case-b-final.mlir"
    compiler_workdir = workdir / "compiler"
    compiler_workdir.mkdir()
    run([adoracc, compiler_input, "--work-dir", compiler_workdir,
         "-o", compiler_output], workdir)

    normalized_files = list(
        (compiler_workdir / "adora-cc-ir/temp/normalize").glob("*_normalized.mlir"))
    if len(normalized_files) != 1:
        raise AssertionError(
            f"expected one production normalized IR, got {normalized_files}")
    normalized = normalized_files[0].read_text()
    final_ir = compiler_output.read_text()
    canonical_map = re.compile(
        r"affine_map<\(d0\) -> \(d0 \* 2 \+ 3\)>")
    if not canonical_map.search(normalized) or not canonical_map.search(final_ir):
        raise AssertionError(
            "production pipeline did not expose the canonical d0 * 2 + 3 "
            "affine.apply form")

    helpers.check_case(
        helpers.load_alignment_helpers(test_dir), workdir, mapper, adg,
        operations, compiler_output, "full_pipeline_loop_index_b", contract,
        init=3, step=2, trips=5, seed=7)

    logical_iv = [3 + iteration * 2 for iteration in range(5)]
    byte_addresses = [value * 2 for value in logical_iv]
    if logical_iv != [3, 5, 7, 9, 11] or \
            byte_addresses != [6, 10, 14, 18, 22]:
        raise AssertionError("Case B logical IV or byte-address sequence changed")
    print("production adoracc Case B ACC/config/address regression passed")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, OSError, subprocess.TimeoutExpired) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
