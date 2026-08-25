#!/usr/bin/env python3
"""Lock the original loop-bearing if_store through production configuration."""

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
            "usage: test_full_pipeline_real_if_store.py WORKDIR ADORACC "
            "MAPPER ADG OPERATIONS CONTRACT TESTDIR INPUT")
    workdir, adoracc, mapper, adg, operations, contract_path, test_dir, fixture = \
        (Path(value).resolve() for value in sys.argv[1:])
    helpers = load_mapping_helpers(test_dir)
    alignment = helpers.load_alignment_helpers(test_dir)
    contract = json.loads(contract_path.read_text())

    compiler_input = workdir / "if-store-input.mlir"
    shutil.copyfile(fixture, compiler_input)
    compiler_output = workdir / "if-store-final.mlir"
    compiler_workdir = workdir / "compiler"
    compiler_workdir.mkdir()
    compiler_log = run(
        [adoracc, compiler_input, "--disable-schedule-tasks",
         "--work-dir", compiler_workdir, "-o", compiler_output], workdir)
    if "FOR is not supported!" in compiler_log:
        raise AssertionError("production compiler still reports unsupported FOR")

    run_dir = workdir / "mapping"
    run_dir.mkdir()
    mapped_py = run_dir / "mapped.py"
    mapper_log = run(helpers.mapper_command(
        mapper, 7, adg, operations, compiler_output, mapped_py), run_dir)
    if "FOR is not supported!" in mapper_log:
        raise AssertionError("physical mapper still reports unsupported FOR")
    result_dir = run_dir / "if_store_map_result"
    nodes, edges = helpers.parse_mapped_dfg(result_dir / "mapped_dfg.dot")
    helpers.assert_no_physical_for(nodes, result_dir / "mapped_adg.dot")

    with adg.open() as stream:
        adg_data = json.load(stream)
    opcodes = helpers.read_operation_opcodes(operations)
    packet_map, acc = helpers.assert_loop_configuration(
        alignment, adg_data, contract, result_dir, mapped_py, "if_store",
        0, 1, 16, nodes, opcodes)
    byte_mul = helpers.assert_byte_scaling_mul(
        alignment, adg_data, packet_map, result_dir / "mapped_adg.dot",
        nodes, edges, result_dir, acc, opcodes, element_bytes=4)

    cstore = helpers.one_node(nodes, "CSTORE")
    if "immIdx=" in nodes[cstore]:
        raise AssertionError(
            f"CSTORE IOB illegally retained an immediate: {nodes[cstore]}")
    incoming = {operand: source for source, destination, _, operand in edges
                if destination == cstore and operand >= 0}
    if set(incoming) != {0, 1, 2} or incoming[1] != byte_mul:
        raise AssertionError(
            f"if_store CSTORE data/address/predicate routes changed: {incoming}")
    data_source = incoming[0]
    if helpers.node_operation(data_source) != "PASS" or \
            "imm=7\\nimmIdx=0" not in nodes[data_source]:
        raise AssertionError(
            "constant 7 was not materialized by a routed immediate PASS at "
            f"CSTORE operand 0: {data_source} {nodes.get(data_source)}")

    rows = [row for row in helpers.read_rows(result_dir / "mapped_routes.tsv")
            if row["dst_operation"] == "CSTORE"]
    alignment.validate_emitted_configuration(
        adg, result_dir / "mapped_routes.tsv", mapped_py,
        "if_store", "CSTORE", rows)
    target_ids = {int(row["dst_adg"]) for row in rows}
    if len(target_ids) != 1:
        raise AssertionError(f"expected one CSTORE IOB, got {target_ids}")
    instance, attributes = alignment.find_iob(adg_data, target_ids.pop())
    cycles = alignment.decode_controller_field(
        packet_map, instance, attributes, adg_data, "Cycles0")
    if cycles != 16:
        raise AssertionError(f"real if_store CSTORE Cycles0 expected 16, got {cycles}")
    print("production real loop-bearing if_store configuration passed")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, OSError, subprocess.TimeoutExpired) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
