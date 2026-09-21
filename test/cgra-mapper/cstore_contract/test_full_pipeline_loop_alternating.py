#!/usr/bin/env python3
"""Prove a production loop routes a loaded predicate stream to CSTORE."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def load_mapping_helpers(test_dir):
    path = test_dir / "test_loop_index_acc_mapping.py"
    spec = importlib.util.spec_from_file_location("loop_index_mapping", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(command, cwd):
    result = subprocess.run(
        [str(value) for value in command], cwd=cwd, env=dict(os.environ),
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=120, check=False)
    if result.returncode != 0:
        raise AssertionError(
            f"expected status 0, got {result.returncode}: "
            f"{' '.join(map(str, command))}\n{result.stdout}")
    return result.stdout


def main():
    if len(sys.argv) != 9:
        raise AssertionError(
            "usage: test_full_pipeline_loop_alternating.py WORKDIR ADORACC "
            "MAPPER ADG OPERATIONS CONTRACT TESTDIR INPUT")
    workdir, adoracc, mapper, adg, operations, contract_path, test_dir, fixture = \
        (Path(value).resolve() for value in sys.argv[1:])
    helpers = load_mapping_helpers(test_dir)
    alignment = helpers.load_alignment_helpers(test_dir)
    contract = json.loads(contract_path.read_text())

    compiler_input = workdir / "alternating-input.mlir"
    shutil.copyfile(fixture, compiler_input)
    compiler_output = workdir / "alternating-final.mlir"
    compiler_workdir = workdir / "compiler"
    compiler_workdir.mkdir()
    run([adoracc, compiler_input, "--disable-schedule-tasks",
         "--work-dir", compiler_workdir, "-o", compiler_output], workdir)

    run_dir = workdir / "mapping"
    run_dir.mkdir()
    mapped_py = run_dir / "mapped.py"
    run(helpers.mapper_command(
        mapper, 7, adg, operations, compiler_output, mapped_py), run_dir)
    result_dir = run_dir / "full_pipeline_loop_alternating_map_result"
    nodes, edges = helpers.parse_mapped_dfg(result_dir / "mapped_dfg.dot")
    helpers.assert_no_physical_for(nodes, result_dir / "mapped_adg.dot")

    with adg.open() as stream:
        adg_data = json.load(stream)
    opcodes = helpers.read_operation_opcodes(operations)
    packet_map, acc = helpers.assert_loop_configuration(
        alignment, adg_data, contract,
        result_dir, mapped_py, "full_pipeline_loop_alternating",
        0, 1, 4, nodes, opcodes)
    byte_mul = helpers.assert_byte_scaling_mul(
        alignment, adg_data, packet_map,
        result_dir / "mapped_adg.dot", nodes, edges, result_dir, acc,
        opcodes, element_bytes=2)

    cstore = helpers.one_node(nodes, "CSTORE")
    incoming = {operand: source for source, destination, _, operand in edges
                if destination == cstore and operand >= 0}
    if set(incoming) != {0, 1, 2} or incoming[1] != byte_mul or \
            helpers.node_operation(incoming[2]) != "SLT":
        raise AssertionError(
            f"expected data/address/loaded-SLT predicate at CSTORE, got {incoming}")
    predicate = incoming[2]
    predicate_inputs = [source for source, destination, _, operand in edges
                        if destination == predicate and operand >= 0]
    if not any(helpers.node_operation(source) == "Input"
               for source in predicate_inputs):
        raise AssertionError(
            f"SLT predicate is not driven by a mapped LOAD/Input: {predicate_inputs}")

    rows = [row for row in helpers.read_rows(result_dir / "mapped_routes.tsv")
            if row["dst_operation"] == "CSTORE"]
    alignment.validate_emitted_configuration(
        adg, result_dir / "mapped_routes.tsv", mapped_py,
        "full_pipeline_loop_alternating", "CSTORE", rows)
    target_ids = {int(row["dst_adg"]) for row in rows}
    if len(target_ids) != 1:
        raise AssertionError(f"expected one CSTORE IOB, got {target_ids}")
    instance, attributes = alignment.find_iob(adg_data, target_ids.pop())
    cycles = alignment.decode_controller_field(
        packet_map, instance, attributes, adg_data, "Cycles0")
    if cycles != 4:
        raise AssertionError(f"alternating CSTORE Cycles0 expected 4, got {cycles}")
    print("production alternating loaded-predicate loop regression passed")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, OSError, subprocess.TimeoutExpired) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
