#!/usr/bin/env python3
"""Real VITRA placement, routing, and input-alignment checks."""

import ast
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys


HEADER = (
    "edge_id", "src_dfg", "dst_dfg", "dst_operation", "logical_operand",
    "dst_adg", "dst_physical_input", "src_latency", "route_latency",
    "rdu_delay", "arrival_latency", "target_latency",
)
PHYSICAL_INPUTS = {0: {0, 1}, 1: {2, 3}, 2: {4, 5}}
SEEDS = (7, 19, 101)


def run(command, cwd, environment, expected=0):
    result = subprocess.run(
        [str(value) for value in command], cwd=cwd, env=environment,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=60, check=False)
    if result.returncode != expected:
        raise AssertionError(
            f"expected status {expected}, got {result.returncode}: "
            f"{' '.join(map(str, command))}\n{result.stdout}")
    return result.stdout


def assert_pre_mapping_cdfg(cgra_opt, kernel, workdir, environment):
    preflight = workdir / "cdfg-preflight"
    preflight.mkdir()
    run((cgra_opt, "--adora-kernel-dfg-gen", kernel), preflight, environment)
    dot_path = preflight / "cstore_route_CDFG.dot"
    if not dot_path.is_file():
        raise AssertionError("CDFG preflight produced no cstore_route_CDFG.dot")

    node_pattern = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*)\[opcode = "([^"]+)"')
    edge_pattern = re.compile(
        r'^([A-Za-z_][A-Za-z0-9_]*) -> ([A-Za-z_][A-Za-z0-9_]*)'
        r'\[.*operand = (-?[0-9]+)')
    opcodes = {}
    incoming = {}
    for line in dot_path.read_text().splitlines():
        node = node_pattern.match(line)
        if node:
            opcodes[node.group(1)] = node.group(2)
        edge = edge_pattern.match(line)
        if edge:
            incoming.setdefault(edge.group(2), []).append(
                (edge.group(1), int(edge.group(3))))

    cstores = [node for node, opcode in opcodes.items() if opcode == "CSTORE"]
    if len(cstores) != 1:
        raise AssertionError(f"expected exactly one CSTORE in CDFG, got {cstores}")
    cstore = cstores[0]
    logical_edges = [(src, port) for src, port in incoming.get(cstore, [])
                     if port >= 0]
    if sorted(port for _, port in logical_edges) != [0, 1, 2]:
        raise AssertionError(f"CSTORE logical inputs are not exactly 0/1/2: {logical_edges}")

    def producer_depth(node, active):
        if node in active:
            raise AssertionError(f"cycle in CDFG producer chain at {node}")
        opcode = opcodes.get(node)
        if opcode in ("Input", "CONST"):
            return 0
        predecessors = [src for src, port in incoming.get(node, []) if port >= 0]
        if not predecessors:
            raise AssertionError(f"dynamic producer {node} has no predecessor")
        return 1 + max(producer_depth(src, active | {node}) for src in predecessors)

    depths = {port: producer_depth(src, set()) for src, port in logical_edges}
    if depths != {0: 1, 1: 2, 2: 3}:
        raise AssertionError(f"unexpected CSTORE producer depths: {depths}")


def mapper_command(mapper, seed, adg, operations, kernel, output,
                   parallel_cores=None):
    command = [
        mapper, f"--seed={seed}", f"--adg={adg}",
        f"--op-file={operations}", "--output-type=pytest",
        "--obj-opt=false", "--max-iters=30", "--timeout=30000",
        kernel, f"--output={output}",
    ]
    if parallel_cores is not None:
        command.append(f"--parallel-cores={parallel_cores}")
    return command


def run_mapping(label, seed, workdir, mapper, adg, operations, kernel,
                kernel_name, environment, expected=0, parallel_cores=None):
    run_dir = workdir / label
    run_dir.mkdir()
    output = run_dir / "mapped.py"
    log = run(mapper_command(mapper, seed, adg, operations, kernel, output,
                             parallel_cores),
              run_dir, environment, expected=expected)
    if log.count(f"Random seed: {seed}\n") != 1:
        raise AssertionError(f"{label}: mapper did not log seed {seed} exactly once")
    if expected != 0:
        return log, None, None
    if not output.is_file():
        raise AssertionError(f"{label}: mapper produced no output")
    manifest = run_dir / f"{kernel_name}_map_result" / "mapped_routes.tsv"
    if not manifest.is_file():
        raise AssertionError(f"{label}: mapper produced no route manifest")
    return log, manifest, output


def read_manifest(path):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if tuple(reader.fieldnames or ()) != HEADER:
            raise AssertionError(f"{path}: unexpected manifest header {reader.fieldnames}")
        return list(reader)


def validate_cstore_manifest(path):
    rows = [row for row in read_manifest(path)
            if row["dst_operation"] == "CSTORE"]
    if len(rows) != 3:
        raise AssertionError(f"{path}: expected three CSTORE rows, got {len(rows)}")
    by_port = {int(row["logical_operand"]): row for row in rows}
    if set(by_port) != {0, 1, 2} or len(by_port) != len(rows):
        raise AssertionError(f"{path}: CSTORE logical operands are not exactly 0/1/2")
    physical = []
    for logical, row in by_port.items():
        physical_input = int(row["dst_physical_input"])
        if physical_input not in PHYSICAL_INPUTS[logical]:
            raise AssertionError(
                f"{path}: logical {logical} used physical input {physical_input}")
        physical.append(physical_input)
        if int(row["arrival_latency"]) != int(row["target_latency"]):
            raise AssertionError(f"{path}: CSTORE logical {logical} is misaligned")
    if len(set(physical)) != 3:
        raise AssertionError(f"{path}: CSTORE physical inputs alias: {physical}")
    return rows


def validate_store_manifest(path, expected_routed_ports):
    rows = read_manifest(path)
    store_rows = [row for row in rows if row["dst_operation"] == "STORE"]
    if not store_rows:
        raise AssertionError(f"{path}: normal STORE has no route rows")
    routed_ports = [int(row["logical_operand"]) for row in store_rows]
    if sorted(routed_ports) != sorted(expected_routed_ports):
        raise AssertionError(
            f"{path}: expected STORE routed ports {sorted(expected_routed_ports)}, "
            f"got {sorted(routed_ports)}")
    if any(int(row["logical_operand"]) == 2 for row in rows):
        raise AssertionError(f"{path}: normal STORE unexpectedly uses logical operand 2")
    return store_rows


def read_config_packets(path, cfg_data_width):
    if cfg_data_width != 32:
        raise AssertionError(
            f"{path}: test decoder requires the pinned 32-bit configuration width")
    packets = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) != 2:
            raise AssertionError(f"{path}: malformed configuration line {line!r}")
        packets.append((int(fields[0], 16), int(fields[1], 16)))
    if not packets:
        raise AssertionError(f"{path}: empty configuration packet stream")
    if len({address for address, _ in packets}) != len(packets):
        raise AssertionError(f"{path}: duplicate configuration packet address")
    return packets


def read_pytest_packets(path, kernel_name, adg):
    array_name = f"cfgbit_{kernel_name}"
    source = path.read_text()
    match = re.search(rf"^{re.escape(array_name)}\s*=\s*(\[)", source,
                      flags=re.MULTILINE)
    if match is None:
        raise AssertionError(f"{path}: missing {array_name}")
    start = match.start(1)
    depth = 0
    end = None
    for offset, character in enumerate(source[start:], start=start):
        if character == "[":
            depth += 1
        elif character == "]":
            depth -= 1
            if depth == 0:
                end = offset + 1
                break
    if end is None:
        raise AssertionError(f"{path}: unterminated {array_name}")
    values = ast.literal_eval(source[start:end])

    cfg_addr_width = int(adg["cfg_addr_width"])
    cfg_data_width = int(adg["cfg_data_width"])
    align_width = 32 if cfg_addr_width > 16 else 16
    data_chunks = cfg_data_width // align_width
    packet_width = data_chunks + 1
    if cfg_data_width % align_width or len(values) % packet_width:
        raise AssertionError(f"{path}: malformed emitted packet alignment")

    packets = []
    for offset in range(0, len(values), packet_width):
        packet = values[offset:offset + packet_width]
        data = sum(int(packet[index]) << (align_width * index)
                   for index in range(data_chunks))
        packets.append((int(packet[-1]), data))
    return packets


def find_iob(adg, node_id):
    instances = [instance for instance in adg["instances"]
                 if instance.get("id") == node_id and instance.get("type") == "IOB"]
    if len(instances) != 1:
        raise AssertionError(f"expected one IOB instance {node_id}, got {instances}")
    instance = instances[0]
    modules = [module for module in adg["sub_modules"]
               if module.get("id") == instance.get("module_id") and
               module.get("type") == "IOB"]
    if len(modules) != 1:
        raise AssertionError(
            f"expected one IOB module {instance.get('module_id')}, got {modules}")
    return instance, modules[0]["attributes"]


def config_range(attributes, config_id, expected_name=None):
    entry = attributes["configuration"].get(str(config_id))
    if not isinstance(entry, list) or len(entry) != 3:
        raise AssertionError(f"missing configuration range for id {config_id}")
    name, high, low = entry
    if expected_name is not None and name != expected_name:
        raise AssertionError(
            f"configuration id {config_id}: expected {expected_name}, got {name}")
    if not isinstance(high, int) or not isinstance(low, int) or low < 0 or high < low:
        raise AssertionError(f"configuration id {config_id}: invalid range {entry}")
    return low, high


def decode_range(packet_map, instance, adg, low, high):
    cfg_data_width = int(adg["cfg_data_width"])
    base_address = int(instance["cfg_blk_index"]) << int(adg["cfg_blk_offset"])
    value = 0
    for source_bit in range(low, high + 1):
        address = base_address | (source_bit // cfg_data_width)
        if address not in packet_map:
            raise AssertionError(
                f"missing packet address 0x{address:x} for aggregate bit {source_bit}")
        packet_bit = source_bit % cfg_data_width
        value |= ((packet_map[address] >> packet_bit) & 1) << (source_bit - low)
    return value


def decode_controller_field(packet_map, instance, attributes, adg, field):
    controller = attributes["io_controller_cfg_id"]
    config_id = controller[field]
    low, high = config_range(attributes, config_id, field)
    return decode_range(packet_map, instance, adg, low, high)


def mapped_iob_ids(path, operation):
    prefix = f"{operation}_"
    ids = []
    for line in path.read_text().splitlines():
        fields = [field.strip() for field in line.split(",")]
        if fields and fields[0].startswith(prefix) and len(fields) >= 3:
            ids.append(int(fields[2]))
    return ids


def validate_iob_semantics(packet_map, instance, attributes, adg, expected):
    actual = {field: decode_controller_field(
        packet_map, instance, attributes, adg, field) for field in expected}
    if actual != expected:
        raise AssertionError(
            f"IOB {instance['id']}: expected controller fields {expected}, got {actual}")


def route_configuration(attributes, rows):
    connections = list(attributes["connections"].values())
    expected_mux = {}
    expected_delays = {}
    delay_id = None
    for row in rows:
        physical_input = int(row["dst_physical_input"])
        input_connections = [edge for edge in connections
                             if edge[0] == 0 and edge[1] == "This" and
                             edge[2] == physical_input and edge[4] == "Muxn"]
        if len(input_connections) != 1:
            raise AssertionError(
                f"physical input {physical_input}: expected one mux connection")
        input_connection = input_connections[0]
        mux_id = int(input_connection[3])
        expected_mux[mux_id] = int(input_connection[5])

        delay_connections = [edge for edge in connections
                             if edge[0] == mux_id and edge[1] == "Muxn" and
                             edge[4] in ("DelayPipe", "RDU")]
        if len(delay_connections) != 1:
            raise AssertionError(f"mux {mux_id}: expected one delay connection")
        delay_connection = delay_connections[0]
        current_delay_id = int(delay_connection[3])
        if delay_id is None:
            delay_id = current_delay_id
        elif delay_id != current_delay_id:
            raise AssertionError("IOB routes do not share one delay pipe")
        expected_delays[int(delay_connection[5])] = int(row["rdu_delay"])
    return expected_mux, delay_id, expected_delays


def validate_route_configuration(packet_map, instance, attributes, adg, rows):
    expected_mux, delay_id, expected_delays = route_configuration(attributes, rows)
    for mux_id, expected_select in expected_mux.items():
        low, high = config_range(attributes, mux_id, "Muxn")
        actual_select = decode_range(packet_map, instance, adg, low, high)
        if actual_select != expected_select:
            raise AssertionError(
                f"IOB {instance['id']} mux {mux_id}: expected select "
                f"{expected_select}, got {actual_select}")

    if delay_id is None:
        return
    low, high = config_range(attributes, delay_id, "DelayPipe")
    width = high - low + 1
    operands = int(attributes["num_operands"])
    if width % operands:
        raise AssertionError(
            f"IOB {instance['id']}: delay width {width} is not divisible by {operands}")
    lane_width = width // operands
    expected = sum(delay << (lane * lane_width)
                   for lane, delay in expected_delays.items())
    actual = decode_range(packet_map, instance, adg, low, high)
    if actual != expected:
        raise AssertionError(
            f"IOB {instance['id']} DelayPipe: expected packed delays "
            f"0x{expected:x}, got 0x{actual:x}")


def validate_emitted_configuration(adg_path, manifest_path, output_path,
                                   kernel_name, operation, rows):
    with adg_path.open() as stream:
        adg = json.load(stream)
    result_dir = manifest_path.parent
    config_path = result_dir / "config.bit"
    config_packets = read_config_packets(config_path, int(adg["cfg_data_width"]))
    emitted_packets = read_pytest_packets(output_path, kernel_name, adg)
    if emitted_packets != config_packets:
        raise AssertionError(
            f"{kernel_name}: emitted pytest packets differ from config.bit")
    packet_map = dict(config_packets)

    target_ids = {int(row["dst_adg"]) for row in rows}
    if len(target_ids) != 1:
        raise AssertionError(
            f"{kernel_name}: expected one mapped {operation} IOB, got {target_ids}")
    instance, attributes = find_iob(adg, target_ids.pop())
    expected = {"IsStore": 1, "UseAddr": 1,
                "UseEn": 1 if operation == "CSTORE" else 0}
    validate_iob_semantics(packet_map, instance, attributes, adg, expected)
    validate_route_configuration(packet_map, instance, attributes, adg, rows)

    mapped_dfgio = result_dir / "mapped_dfgio.txt"
    for input_id in mapped_iob_ids(mapped_dfgio, "INPUT"):
        input_instance, input_attributes = find_iob(adg, input_id)
        validate_iob_semantics(
            packet_map, input_instance, input_attributes, adg,
            {"IsStore": 0, "UseAddr": 0, "UseEn": 0})


def validate_parallel_seed(mapper, adg, operations, kernel, workdir,
                           environment):
    reference = None
    kernel_names = ("parallel_cstore_a", "parallel_cstore_b")
    for repetition in range(4):
        run_dir = workdir / f"parallel-repeat-{repetition}"
        run_dir.mkdir()
        output = run_dir / "mapped.py"
        log = run(mapper_command(mapper, 7, adg, operations, kernel, output,
                                 parallel_cores=2),
                  run_dir, environment)
        if log.count("Random seed: 7\n") != 1:
            raise AssertionError(
                f"parallel repeat {repetition}: seed was not logged exactly once")
        manifests = []
        for kernel_name in kernel_names:
            manifest = run_dir / f"{kernel_name}_map_result" / "mapped_routes.tsv"
            if not manifest.is_file():
                raise AssertionError(
                    f"parallel repeat {repetition}: missing {kernel_name} manifest")
            validate_cstore_manifest(manifest)
            manifests.append(manifest.read_bytes())
        if reference is None:
            reference = manifests
        elif manifests != reference:
            raise AssertionError(
                "parallel seed 7 manifests differ across repeated two-kernel runs")


def make_no_cstore_adg(source, destination):
    with source.open() as stream:
        adg = json.load(stream)
    removed = 0
    for module in adg.get("sub_modules", []):
        operations = module.get("attributes", {}).get("operations", [])
        while "CSTORE" in operations:
            operations.remove("CSTORE")
            removed += 1
    if removed == 0:
        raise AssertionError("pinned ADG has no CSTORE capability to remove")
    with destination.open("w") as stream:
        json.dump(adg, stream, indent=2)
        stream.write("\n")


def main():
    if len(sys.argv) != 9:
        raise AssertionError(
            "usage: test_cstore_alignment.py WORKDIR CGRA_OPT MAPPER ADG "
            "OPERATIONS CSTORE_KERNEL STORE_KERNEL PARALLEL_KERNEL")
    (workdir, cgra_opt, mapper, adg, operations, cstore_kernel, store_kernel,
     parallel_kernel) = (
        Path(value).resolve() for value in sys.argv[1:])
    workdir.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)

    assert_pre_mapping_cdfg(cgra_opt, cstore_kernel, workdir, environment)

    manifests = []
    for seed in SEEDS:
        _, manifest, output = run_mapping(
            f"cstore-seed-{seed}", seed, workdir, mapper, adg, operations,
            cstore_kernel, "cstore_route", environment)
        manifests.append((manifest, output))
    _, repeated, repeated_output = run_mapping(
        "cstore-seed-7-repeat", 7, workdir, mapper, adg, operations,
        cstore_kernel, "cstore_route", environment)

    cstore_rows = []
    for manifest, output in manifests + [(repeated, repeated_output)]:
        rows = validate_cstore_manifest(manifest)
        cstore_rows.extend(rows)
        validate_emitted_configuration(
            adg, manifest, output, "cstore_route", "CSTORE", rows)
    if not any(int(row["rdu_delay"]) > 0 for row in cstore_rows):
        raise AssertionError("all deterministic CSTORE inputs have zero RDU delay")
    if manifests[0][0].read_bytes() != repeated.read_bytes():
        raise AssertionError("seed 7 route manifests are not byte-identical")
    if manifests[0][0].parent.joinpath("config.bit").read_bytes() != \
            repeated.parent.joinpath("config.bit").read_bytes():
        raise AssertionError("seed 7 configuration packets are not byte-identical")

    _, store_manifest, store_output = run_mapping(
        "normal-store", 7, workdir, mapper, adg, operations, store_kernel,
        "normal_store_route", environment)
    store_rows = validate_store_manifest(store_manifest, {0})
    validate_emitted_configuration(
        adg, store_manifest, store_output, "normal_store_route", "STORE",
        store_rows)
    normal_store_dir = workdir / "normal-store"
    for kernel_name in ("store_block_arg", "store_index_cast"):
        manifest = normal_store_dir / f"{kernel_name}_map_result" / \
            "mapped_routes.tsv"
        rows = validate_store_manifest(manifest, {0, 1})
        validate_emitted_configuration(
            adg, manifest, store_output, kernel_name, "STORE", rows)

    constant_dir = normal_store_dir / "store_constant_data_map_result"
    constant_manifest = constant_dir / "mapped_routes.tsv"
    constant_rows = validate_store_manifest(constant_manifest, {0, 1})
    validate_emitted_configuration(
        adg, constant_manifest, store_output, "store_constant_data", "STORE",
        constant_rows)
    constant_dfg = (constant_dir / "mapped_dfg.dot").read_text()
    pass_match = re.search(
        r'"(PASS\d+)"\[label = ".*\\nimm=7\\nimmIdx=0"\];',
        constant_dfg)
    store_match = re.search(r'"(store\d+)"\[label = "([^"]*)"\];',
                            constant_dfg)
    if not pass_match or not store_match or "immIdx=" in store_match.group(2):
        raise AssertionError(
            "constant STORE data was not materialized outside the IOB")
    routed_data = re.search(
        rf'"{pass_match.group(1)}"->"{store_match.group(1)}"'
        r'\[label = "lat=\d+\\nop=0', constant_dfg)
    if not routed_data:
        raise AssertionError(
            "materialized constant STORE data is not routed to operand 0")

    no_cstore_adg = workdir / "no-cstore-adg.json"
    make_no_cstore_adg(adg, no_cstore_adg)
    rejection, _, _ = run_mapping(
        "no-cstore-reject", 7, workdir, mapper, no_cstore_adg, operations,
        cstore_kernel, "cstore_route", environment, expected=1)
    for message in ("No legal ADG node supports CSTORE", "required=1", "available=0"):
        if message not in rejection:
            raise AssertionError(f"no-CSTORE rejection missing {message!r}")
    _, no_cstore_store, _ = run_mapping(
        "no-cstore-normal-store", 7, workdir, mapper, no_cstore_adg,
        operations, store_kernel, "normal_store_route", environment)
    validate_store_manifest(no_cstore_store, {0})

    validate_parallel_seed(mapper, adg, operations, parallel_kernel, workdir,
                           environment)

    print("CSTORE alignment and parallel determinism passed; STORE addresses passed")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, OSError, subprocess.TimeoutExpired) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
