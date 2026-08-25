#!/usr/bin/env python3
"""Decode the emitted VITRA config for loop-index ACC CSTORE mappings."""

import csv
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


SEEDS = (7, 19)
HEADER = (
    "edge_id", "src_dfg", "dst_dfg", "dst_operation", "logical_operand",
    "dst_adg", "dst_physical_input", "src_latency", "route_latency",
    "rdu_delay", "arrival_latency", "target_latency",
)
NODE_LINE = re.compile(r'^"([^"]+)"\[label = "(.*)"\];$')
EDGE_LINE = re.compile(
    r'^"([^"]+)"->"([^"]+)"\[label = "lat=(\d+)\\nop=(\d+)')


def load_alignment_helpers(test_dir):
    helper_path = test_dir / "test_cstore_alignment.py"
    spec = importlib.util.spec_from_file_location("cstore_alignment", helper_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(command, cwd, environment):
    result = subprocess.run([str(value) for value in command], cwd=cwd,
                            env=environment, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=90, check=False)
    if result.returncode != 0:
        raise AssertionError(
            f"expected status 0, got {result.returncode}: "
            f"{' '.join(map(str, command))}\n{result.stdout}")
    return result.stdout


def mapper_command(mapper, seed, adg, operations, source, output):
    op_name_file = operations.parents[4] / "lib/DFG/Documents/GeneralOpName.txt"
    return (mapper, f"--seed={seed}", f"--adg={adg}",
            f"--op-file={operations}", "--output-type=pytest",
            f"--op-name-file={op_name_file}",
            "--obj-opt=false", "--max-iters=30", "--timeout=30000",
            source, f"--output={output}")


def read_rows(path):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if tuple(reader.fieldnames or ()) != HEADER:
            raise AssertionError(f"{path}: unexpected manifest header")
        return list(reader)


def parse_mapped_dfg(path):
    nodes = {}
    edges = []
    for line in path.read_text().splitlines():
        node = NODE_LINE.match(line)
        if node:
            name, label = node.groups()
            if name in nodes:
                raise AssertionError(f"{path}: duplicate mapped DFG node {name}")
            nodes[name] = label
            continue
        edge = EDGE_LINE.match(line)
        if edge:
            source, destination, latency, operand = edge.groups()
            edges.append((source, destination, int(latency), int(operand)))
    if not nodes:
        raise AssertionError(f"{path}: no mapped DFG nodes")
    return nodes, edges


def node_operation(name):
    match = re.fullmatch(r'([A-Za-z_][A-Za-z_]*)(\d+)', name)
    if not match:
        raise AssertionError(f"unexpected mapped DFG node name {name!r}")
    return match.group(1)


def node_id(name):
    match = re.fullmatch(r'[A-Za-z_][A-Za-z_]*(\d+)', name)
    if not match:
        raise AssertionError(f"unexpected mapped DFG node id {name!r}")
    return int(match.group(1))


def one_node(nodes, operation, label_fragment=None):
    matches = [name for name, label in nodes.items()
               if node_operation(name) == operation and
               (label_fragment is None or label_fragment in label)]
    if len(matches) != 1:
        raise AssertionError(f"expected one {operation} node, got {matches}")
    return matches[0]


def mapped_gpe_for_node(mapped_adg, node_name):
    match = re.search(r'GPE(\d+)\[label = "GPE\d+\\nDFG:' +
                      re.escape(node_name) + r'", color = red\];',
                      mapped_adg.read_text())
    if not match:
        raise AssertionError(f"mapped {node_name} has no GPE placement")
    return int(match.group(1))


def mapped_gpe_for_acc(mapped_adg, nodes, require_immediate=True):
    accs = [name for name, label in nodes.items()
            if node_operation(name) == "ACC" and
            (not require_immediate or "immIdx=0" in label)]
    if len(accs) != 1:
        raise AssertionError(f"expected one mapped loop-index ACC, got {accs}")
    return mapped_gpe_for_node(mapped_adg, accs[0]), accs[0]


def read_operation_opcodes(path):
    operations = json.loads(path.read_text()).get("Operations", [])
    opcodes = {entry.get("name"): entry.get("OPC") for entry in operations}
    for operation in ("ACC", "MUL"):
        if not isinstance(opcodes.get(operation), int):
            raise AssertionError(f"{path}: missing {operation} opcode")
    return opcodes


def find_gpe(adg, node_id):
    instances = [item for item in adg["instances"]
                 if item.get("id") == node_id and item.get("type") == "GPE"]
    if len(instances) != 1:
        raise AssertionError(f"expected GPE instance {node_id}")
    modules = [item for item in adg["sub_modules"]
               if item.get("id") == instances[0].get("module_id") and
               item.get("type") == "GPE"]
    if len(modules) != 1:
        raise AssertionError(f"expected GPE module for instance {node_id}")
    return instances[0], modules[0]["attributes"]


def decode_gpe_field(helpers, packet_map, instance, attributes, adg, field):
    config_ids = attributes["affine_ctrl_reg_cfg_id"]
    config_id = config_ids[field]
    low, high = helpers.config_range(attributes, config_id, field)
    return helpers.decode_range(packet_map, instance, adg, low, high)


def decode_gpe_opcode(helpers, packet_map, instance, attributes, adg):
    connections = list(attributes["connections"].values())
    alu_ids = {int(edge[3]) for edge in connections if edge[4] == "ALU"}
    if len(alu_ids) != 1:
        raise AssertionError("expected one GPE ALU configuration")
    low, high = helpers.config_range(attributes, alu_ids.pop(), "ALU")
    return helpers.decode_range(packet_map, instance, adg, low, high)


def constant_path_to_alu_operand(attributes, operand):
    connections = [tuple(edge) for edge in attributes["connections"].values()]
    paths = []
    for const_to_mux in connections:
        if const_to_mux[1] != "Const" or const_to_mux[4] != "Muxn":
            continue
        for mux_to_delay in connections:
            if (mux_to_delay[0], mux_to_delay[1], mux_to_delay[4]) != \
                    (const_to_mux[3], "Muxn", "DelayPipe"):
                continue
            for delay_to_alu in connections:
                if (delay_to_alu[0], delay_to_alu[1], delay_to_alu[2],
                    delay_to_alu[4], delay_to_alu[5]) != \
                        (mux_to_delay[3], "DelayPipe", mux_to_delay[5],
                         "ALU", operand):
                    continue
                paths.append((const_to_mux, mux_to_delay, delay_to_alu))
    if len(paths) != 1:
        raise AssertionError(
            f"expected one Const-to-ALU operand {operand} path, got {paths}")
    return paths[0]


def decode_constant_path(helpers, adg, packet_map, instance, attributes,
                         operand):
    const_to_mux, mux_to_delay, delay_to_alu = constant_path_to_alu_operand(
        attributes, operand)
    const_id, mux_id, delay_id = (int(const_to_mux[0]), int(const_to_mux[3]),
                                  int(mux_to_delay[3]))
    low, high = helpers.config_range(attributes, const_id, "Const")
    actual_constant = helpers.decode_range(packet_map, instance, adg, low, high)
    low, high = helpers.config_range(attributes, mux_id, "Muxn")
    actual_mux = helpers.decode_range(packet_map, instance, adg, low, high)
    expected_mux = int(const_to_mux[5])
    low, high = helpers.config_range(attributes, delay_id, "DelayPipe")
    packed_delay = helpers.decode_range(packet_map, instance, adg, low, high)
    operands = int(attributes["num_operands"])
    width = high - low + 1
    if width % operands:
        raise AssertionError(
            f"DelayPipe{delay_id}: width {width} does not divide {operands} lanes")
    lane_width = width // operands
    lane = int(mux_to_delay[5])
    if lane != int(delay_to_alu[2]):
        raise AssertionError("Const route DelayPipe lane does not feed the selected ALU operand")
    actual_lane_delay = (packed_delay >> (lane * lane_width)) & ((1 << lane_width) - 1)
    return {"operand": operand, "const": actual_constant, "mux": actual_mux,
            "expected_mux": expected_mux, "delay_pipe": packed_delay,
            "lane": lane, "lane_delay": actual_lane_delay}


def assert_constant_operand(helpers, adg, packet_map, instance, attributes,
                            operand, expected_value):
    decoded = decode_constant_path(helpers, adg, packet_map, instance, attributes,
                                   operand)
    if decoded["const"] != expected_value:
        raise AssertionError(
            f"ALU operand {operand}: expected Const {expected_value}, "
            f"got {decoded['const']}")
    if decoded["mux"] != decoded["expected_mux"]:
        raise AssertionError(
            f"ALU operand {operand}: expected Muxn select {decoded['expected_mux']}, "
            f"got {decoded['mux']}")
    if decoded["lane_delay"] != 0:
        raise AssertionError(
            f"ALU operand {operand}: local Const DelayPipe lane {decoded['lane']} "
            f"expected 0, got {decoded['lane_delay']} from packed "
            f"0x{decoded['delay_pipe']:x}")
    return decoded


def selected_constant_operand(helpers, adg, packet_map, instance, attributes,
                              expected_value, excluded_operand, active_operands):
    candidates = []
    for operand in range(active_operands):
        if operand == excluded_operand:
            continue
        decoded = decode_constant_path(helpers, adg, packet_map, instance,
                                       attributes, operand)
        if decoded["const"] == expected_value and \
                decoded["mux"] == decoded["expected_mux"]:
            candidates.append(decoded)
    if len(candidates) != 1:
        raise AssertionError(
            f"expected one selected Const={expected_value} path outside ALU operand "
            f"{excluded_operand}, got {candidates}")
    decoded = candidates[0]
    if decoded["lane_delay"] != 0:
        raise AssertionError(
            f"ALU operand {decoded['operand']}: local Const DelayPipe lane "
            f"{decoded['lane']} expected 0, got {decoded['lane_delay']} from packed "
            f"0x{decoded['delay_pipe']:x}")
    return decoded


def external_path_to_alu_operand(attributes, physical_input):
    connections = [tuple(edge) for edge in attributes["connections"].values()]
    paths = []
    for this_to_mux in connections:
        if (this_to_mux[1], this_to_mux[2], this_to_mux[4]) != \
                ("This", physical_input, "Muxn"):
            continue
        for mux_to_delay in connections:
            if (mux_to_delay[0], mux_to_delay[1], mux_to_delay[4]) != \
                    (this_to_mux[3], "Muxn", "DelayPipe"):
                continue
            for delay_to_alu in connections:
                if (delay_to_alu[0], delay_to_alu[1], delay_to_alu[2],
                    delay_to_alu[4]) != \
                        (mux_to_delay[3], "DelayPipe", mux_to_delay[5], "ALU"):
                    continue
                paths.append((this_to_mux, mux_to_delay, delay_to_alu))
    if len(paths) != 1:
        raise AssertionError(
            f"expected one physical input {physical_input} path to ALU, got {paths}")
    return paths[0]


def assert_no_physical_for(nodes, mapped_adg):
    physical_fors = [name for name in nodes if node_operation(name).upper() == "FOR"]
    if physical_fors:
        raise AssertionError(f"mapped DFG contains physical FOR nodes: {physical_fors}")
    placements = re.findall(r'DFG:([^"\\]+)', mapped_adg.read_text())
    placed_fors = [name for name in placements
                   if node_operation(name).upper() == "FOR"]
    if placed_fors:
        raise AssertionError(f"mapped ADG contains physical FOR placements: {placed_fors}")


def assert_acc_route(helpers, adg, packet_map, mapped_adg, nodes, step,
                     expected_opcode):
    gpe, acc = mapped_gpe_for_acc(mapped_adg, nodes)
    if f"imm={step}\\nimmIdx=0" not in nodes[acc]:
        raise AssertionError("positive loop step is not folded at ACC operand 0")
    instance, attributes = find_gpe(adg, gpe)
    actual_opcode = decode_gpe_opcode(helpers, packet_map, instance, attributes, adg)
    if actual_opcode != expected_opcode:
        raise AssertionError(f"ACC opcode expected {expected_opcode}, got {actual_opcode}")
    assert_constant_operand(helpers, adg, packet_map, instance, attributes, 0, step)
    return acc


def assert_cstore_routes(helpers, adg, packet_map, output, result_dir, kernel,
                         nodes, edges, exact_mul, expected_cycles):
    rows = [row for row in read_rows(result_dir / "mapped_routes.tsv")
            if row["dst_operation"] == "CSTORE"]
    if len(rows) != 3:
        raise AssertionError(f"{kernel}: expected data/address/predicate CSTORE routes")
    by_port = {int(row["logical_operand"]): row for row in rows}
    if set(by_port) != {0, 1, 2}:
        raise AssertionError(f"{kernel}: CSTORE logical ports changed")
    if any(int(row["arrival_latency"]) != int(row["target_latency"])
           for row in rows):
        raise AssertionError(f"{kernel}: CSTORE route alignment changed")

    cstore = one_node(nodes, "CSTORE")
    incoming_edges = [(source, operand) for source, destination, _, operand in edges
                      if destination == cstore and operand >= 0]
    incoming = {operand: source for source, operand in incoming_edges}
    if len(incoming_edges) != 3 or len(incoming) != len(incoming_edges) or \
            set(incoming) != {0, 1, 2}:
        raise AssertionError(f"{kernel}: mapped DFG CSTORE inputs changed: {incoming_edges}")
    for port, row in by_port.items():
        if int(row["dst_dfg"]) != node_id(cstore) or \
                int(row["src_dfg"]) != node_id(incoming[port]):
            raise AssertionError(
                f"{kernel}: route port {port} is not the mapped DFG edge "
                f"{incoming[port]} -> {cstore}")
    if node_operation(incoming[0]) != "Input" or \
            node_operation(incoming[2]) != "Input" or incoming[0] == incoming[2]:
        raise AssertionError(
            f"{kernel}: expected distinct data/predicate INPUT sources, got "
            f"port0={incoming[0]}, port2={incoming[2]}")
    if incoming[1] != exact_mul:
        raise AssertionError(
            f"{kernel}: CSTORE address port is {incoming[1]}, not byte MUL {exact_mul}")
    helpers.validate_emitted_configuration(
        adg, result_dir / "mapped_routes.tsv", output, kernel, "CSTORE", rows)
    with adg.open() as stream:
        adg_data = json.load(stream)
    target_ids = {int(row["dst_adg"]) for row in rows}
    if len(target_ids) != 1:
        raise AssertionError(
            f"{kernel}: expected one mapped CSTORE IOB, got {target_ids}")
    instance, attributes = helpers.find_iob(adg_data, target_ids.pop())
    actual_cycles = helpers.decode_controller_field(
        packet_map, instance, attributes, adg_data, "Cycles0")
    if actual_cycles != expected_cycles:
        raise AssertionError(
            f"{kernel}: CSTORE Cycles0 expected {expected_cycles}, "
            f"got {actual_cycles}")
    return rows


def assert_loop_configuration(helpers, adg_data, contract, result_dir, output,
                              kernel, init, step, trips, nodes, opcodes):
    packets = helpers.read_config_packets(result_dir / "config.bit",
                                          int(adg_data["cfg_data_width"]))
    if packets != helpers.read_pytest_packets(output, kernel, adg_data):
        raise AssertionError(f"{kernel}: emitted pytest config differs from config.bit")
    packet_map = dict(packets)
    gpe, _ = mapped_gpe_for_acc(result_dir / "mapped_adg.dot", nodes)
    instance, attributes = find_gpe(adg_data, gpe)
    if decode_gpe_opcode(helpers, packet_map, instance, attributes, adg_data) != \
            opcodes["ACC"]:
        raise AssertionError(f"{kernel}: ACC opcode did not decode from the operation catalog")
    if contract["operations"]["ACC"]["opcode"] != opcodes["ACC"]:
        raise AssertionError(f"{kernel}: contract and operations catalog disagree on ACC opcode")
    expected = {"InitVal": init, "WI": 1, "Latency": 0,
                "Cycles": trips, "Repeats": 1, "SkipFirst": 1}
    actual = {field: decode_gpe_field(helpers, packet_map, instance, attributes,
                                      adg_data, field)
              for field in expected}
    if actual != expected:
        raise AssertionError(f"{kernel}: loop-index ACC fields expected {expected}, got {actual}")
    if "loop_index_acc=1" not in (result_dir / "mapped_dfg.dot").read_text():
        raise AssertionError(f"{kernel}: mapper lost the loop-index ACC marker")
    acc = assert_acc_route(helpers, adg_data, packet_map,
                           result_dir / "mapped_adg.dot", nodes, step,
                           opcodes["ACC"])
    return packet_map, acc


def assert_byte_scaling_mul(helpers, adg_data, packet_map, mapped_adg, nodes,
                            edges, result_dir, acc, opcodes, element_bytes):
    muls = [name for name, label in nodes.items()
            if node_operation(name) == "MUL" and
            f"imm={element_bytes}\\nimmIdx=1" in label and
            any(source == acc and destination == name and operand == 0
                for source, destination, _, operand in edges)]
    if len(muls) != 1:
        raise AssertionError(f"expected one ACC-fed byte-scaling MUL, got {muls}")
    mul = muls[0]
    gpe = mapped_gpe_for_node(mapped_adg, mul)
    instance, attributes = find_gpe(adg_data, gpe)
    actual_opcode = decode_gpe_opcode(helpers, packet_map, instance, attributes,
                                      adg_data)
    if actual_opcode != opcodes["MUL"]:
        raise AssertionError(f"MUL opcode expected {opcodes['MUL']}, got {actual_opcode}")
    mul_rows = [row for row in read_rows(result_dir / "mapped_routes.tsv")
                if row["dst_operation"] == "MUL" and
                int(row["src_dfg"]) == node_id(acc) and
                int(row["dst_dfg"]) == node_id(mul)]
    if len(mul_rows) != 1 or int(mul_rows[0]["logical_operand"]) != 0:
        raise AssertionError(
            f"expected one ACC -> {mul} logical operand-0 route, got {mul_rows}")
    physical_input = int(mul_rows[0]["dst_physical_input"])
    _, _, delay_to_alu = external_path_to_alu_operand(attributes, physical_input)
    acc_physical_operand = int(delay_to_alu[5])
    selected_constant_operand(
        helpers, adg_data, packet_map, instance, attributes, element_bytes,
        acc_physical_operand, active_operands=2)
    return mul


def check_case(helpers, workdir, mapper, adg, operations, source, kernel,
               contract, init, step, trips, seed):
    run_dir = workdir / f"{kernel}-seed-{seed}"
    run_dir.mkdir()
    output = run_dir / "mapped.py"
    environment = dict(os.environ)
    log = run(mapper_command(mapper, seed, adg, operations, source, output),
              run_dir, environment)
    if log.count(f"Random seed: {seed}\n") != 1:
        raise AssertionError(f"{kernel}: mapper did not log deterministic seed {seed}")
    result_dir = run_dir / f"{kernel}_map_result"
    with adg.open() as stream:
        adg_data = json.load(stream)
    opcodes = read_operation_opcodes(operations)
    nodes, edges = parse_mapped_dfg(result_dir / "mapped_dfg.dot")
    assert_no_physical_for(nodes, result_dir / "mapped_adg.dot")
    packet_map, acc = assert_loop_configuration(
        helpers, adg_data, contract, result_dir, output, kernel, init, step, trips,
        nodes, opcodes)
    exact_mul = assert_byte_scaling_mul(
        helpers, adg_data, packet_map, result_dir / "mapped_adg.dot", nodes,
        edges, result_dir, acc, opcodes, element_bytes=2)
    rows = assert_cstore_routes(
        helpers, adg, packet_map, output, result_dir, kernel, nodes, edges,
        exact_mul, expected_cycles=trips)
    return (result_dir / "mapped_routes.tsv").read_bytes(), \
        (result_dir / "config.bit").read_bytes(), rows


def check_normal_store(helpers, workdir, mapper, adg, operations, test_dir):
    source = workdir / "normal-store.mlir"
    shutil.copyfile(test_dir / "normal_store.mlir.in", source)
    run_dir = workdir / "normal-store"
    run_dir.mkdir()
    output = run_dir / "mapped.py"
    run(mapper_command(mapper, 7, adg, operations, source, output), run_dir,
        dict(os.environ))
    result_dir = run_dir / "normal_store_route_map_result"
    with adg.open() as stream:
        adg_data = json.load(stream)
    rows = [row for row in read_rows(result_dir / "mapped_routes.tsv")
            if row["dst_operation"] == "STORE"]
    helpers.validate_emitted_configuration(
        adg, result_dir / "mapped_routes.tsv", output, "normal_store_route",
        "STORE", rows)


def check_generic_acc(helpers, workdir, mapper, adg, operations, test_dir):
    source = test_dir.parents[1] / "cgra-opt/cdfggen/mmul_relu/mmul_relu_opt.mlir"
    run_dir = workdir / "generic-acc"
    run_dir.mkdir()
    output = run_dir / "mapped.py"
    run(mapper_command(mapper, 7, adg, operations, source, output), run_dir,
        dict(os.environ))
    result_dir = run_dir / "mmul_relu_map_result"
    with adg.open() as stream:
        adg_data = json.load(stream)
    packets = helpers.read_config_packets(result_dir / "config.bit",
                                          int(adg_data["cfg_data_width"]))
    nodes, _ = parse_mapped_dfg(result_dir / "mapped_dfg.dot")
    gpe, _ = mapped_gpe_for_acc(result_dir / "mapped_adg.dot",
                                 nodes,
                                 require_immediate=False)
    instance, attributes = find_gpe(adg_data, gpe)
    packet_map = dict(packets)
    expected = {"InitVal": 0, "WI": 1, "Latency": 11,
                "Cycles": 25, "Repeats": 625, "SkipFirst": 0}
    actual = {field: decode_gpe_field(helpers, packet_map, instance, attributes,
                                      adg_data, field)
              for field in expected}
    if actual != expected:
        raise AssertionError(f"generic ACC config changed: expected {expected}, got {actual}")
    if "loop_index_acc=1" in (result_dir / "mapped_dfg.dot").read_text():
        raise AssertionError("ordinary ACC was marked as a loop-index ACC")


def main():
    if len(sys.argv) != 8:
        raise AssertionError("usage: test_loop_index_acc_mapping.py WORKDIR CGRA_OPT "
                             "MAPPER ADG OPERATIONS CONTRACT TESTDIR")
    workdir, cgra_opt, mapper, adg, operations, contract_path, test_dir = \
        (Path(value).resolve() for value in sys.argv[1:])
    del cgra_opt
    helpers = load_alignment_helpers(test_dir)
    contract = json.loads(contract_path.read_text())
    if contract["configuration_formula"]["WI"] != 1 or \
            contract["configuration_formula"]["Latency"] != 0:
        raise AssertionError("pinned loop-index contract has unexpected ACC formula")
    chunks = (test_dir / "loop_index_acc_mapping.mlir").read_text().split("// -----\n")
    if len(chunks) != 2:
        raise AssertionError("expected Case A and Case B fixture chunks")
    sources = []
    for name, chunk in zip(("loop_index_a", "loop_index_b"), chunks):
        source = workdir / f"{name}.mlir"
        source.write_text(chunk)
        sources.append(source)

    for seed in SEEDS:
        check_case(
            helpers, workdir, mapper, adg, operations, sources[0], "loop_index_a",
            contract, 0, 1, 4, seed)
        check_case(
            helpers, workdir, mapper, adg, operations, sources[1], "loop_index_b",
            contract, 3, 2, 5, seed)
    check_normal_store(helpers, workdir, mapper, adg, operations, test_dir)
    check_generic_acc(helpers, workdir, mapper, adg, operations, test_dir)
    print("loop-index ACC config, byte-scaled CSTORE routing, and STORE regression passed")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, OSError, subprocess.TimeoutExpired) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
