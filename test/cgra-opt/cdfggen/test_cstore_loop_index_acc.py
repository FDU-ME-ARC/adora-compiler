#!/usr/bin/env python3
"""Focused physical-CDFG contract checks for affine CSTORE loop indices."""

import pathlib
import json
import re
import subprocess
import sys
import tempfile


def run(command, cwd, expected=0):
    result = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    if result.returncode != expected:
        raise AssertionError(
            f"expected status {expected}, got {result.returncode}: "
            f"{' '.join(map(str, command))}\n{result.stdout}")
    return result.stdout


def nodes_and_edges(dot):
    nodes = {}
    edges = []
    for line in dot.read_text().splitlines():
        node = re.match(r'^(\w+)\[opcode = "([^"]+)"(.*)\];$', line)
        if node:
            nodes[node.group(1)] = (node.group(2), node.group(3))
            continue
        edge = re.match(r'^(\w+) -> (\w+)\[.*operand = (-?\d+)', line)
        if edge:
            edges.append((edge.group(1), edge.group(2), int(edge.group(3))))
    return nodes, edges


def check_supported(dot, init, step, trips):
    nodes, edges = nodes_and_edges(dot)
    if any(op == "for" for op, _ in nodes.values()):
        raise AssertionError(f"{dot}: supported CSTORE loop still has physical FOR")
    accs = [(name, attrs) for name, (op, attrs) in nodes.items() if op == "ACC"]
    if len(accs) != 1:
        raise AssertionError(f"{dot}: expected exactly one loop-index ACC, got {accs}")
    acc, attrs = accs[0]
    expected = f'acc_params="{init},{trips},1,1"'
    if expected not in attrs or 'loop_index_acc="1"' not in attrs or \
       'skip_first="1"' not in attrs:
        raise AssertionError(f"{dot}: missing loop-index ACC semantics: {attrs}")
    step_nodes = []
    for name, (op, attrs) in nodes.items():
        value = re.search(r'value="0x([0-9A-Fa-f]+)"', attrs)
        if op == "CONST" and value and int(value.group(1), 16) == step:
            step_nodes.append(name)
    if not step_nodes:
        raise AssertionError(f"{dot}: expected a routed step CONST {step}")
    if not any((step_node, acc, 0) in edges for step_node in step_nodes):
        raise AssertionError(f"{dot}: positive step is not ACC operand 0")
    cstores = [attrs for _, (op, attrs) in nodes.items() if op == "CSTORE"]
    if len(cstores) != 1 or f'pattern="0,{trips}"' not in cstores[0]:
        raise AssertionError(
            f"{dot}: CSTORE execution count does not match trip count {trips}")


def check_case_b_structure(dot):
    nodes, edges = nodes_and_edges(dot)
    acc = next(name for name, (op, _) in nodes.items() if op == "ACC")
    cstore = next(name for name, (op, _) in nodes.items() if op == "CSTORE")
    muls = [name for name, (op, _) in nodes.items() if op == "MUL"]
    if len(muls) != 1:
        raise AssertionError(f"{dot}: expected exactly one byte-scaling MUL")
    mul = muls[0]
    if (acc, mul, 0) not in edges or (mul, cstore, 1) not in edges:
        raise AssertionError(f"{dot}: expected ACC -> MUL -> CSTORE address path")
    if {port for _, dst, port in edges if dst == cstore} != {0, 1, 2}:
        raise AssertionError(f"{dot}: CSTORE data/address/predicate ports changed")
    if not any(op == "CONST" and (name, mul, 1) in edges and
               'value="0x00000002"' in attrs
               for name, (op, attrs) in nodes.items()):
        raise AssertionError(f"{dot}: byte-scaling MUL no longer has constant 2 on operand 1")


def check_mapped_step_route(mapped_dfg, mapped_adg, adg):
    mapped_text = mapped_dfg.read_text()
    acc_matches = [line.split('"')[1] for line in mapped_text.splitlines()
                   if line.startswith('"ACC') and '\\nimm=2\\nimmIdx=0' in line]
    if len(acc_matches) != 1:
        raise AssertionError(
            "Case B step was not folded into exactly one mapped ACC immediate on operand 0")
    acc = acc_matches[0]
    mapped_adg_text = mapped_adg.read_text()
    match = re.search(r'GPE(\d+)\[label = "GPE\d+\\nDFG:' + re.escape(acc) +
                      r'", color = red\];', mapped_adg_text)
    if not match:
        raise AssertionError("mapped ACC is not placed on a physical GPE")
    physical_gpe = int(match.group(1))
    design = json.loads(adg.read_text())
    instance = next(item for item in design["instances"] if item["id"] == physical_gpe)
    module = next(item for item in design["sub_modules"]
                  if item["id"] == instance["module_id"] and item["type"] == "GPE")
    connections = {tuple(connection) for connection in module["attributes"]["connections"].values()}
    route = {
        (1, "Const", 0, 5, "Muxn", 0),
        (5, "Muxn", 0, 4, "DelayPipe", 0),
        (4, "DelayPipe", 0, 2, "ALU", 0),
    }
    if not route.issubset(connections):
        raise AssertionError("mapped ACC GPE lacks Const -> Muxn -> DelayPipe operand-0 -> ALU route")


def check_composed_affine_apply(dot):
    nodes, edges = nodes_and_edges(dot)
    if any(op == "for" for op, _ in nodes.values()):
        raise AssertionError(f"{dot}: supported composed affine address retained physical FOR")
    acc = next((name for name, (op, _) in nodes.items() if op == "ACC"), None)
    cstore = next((name for name, (op, _) in nodes.items() if op == "CSTORE"), None)
    if not acc or not cstore:
        raise AssertionError(f"{dot}: missing ACC or CSTORE for composed affine address")
    successors = {}
    for src, dst, port in edges:
        successors.setdefault(src, []).append((dst, port))
    expected_ops = ["ACC", "MUL", "ADD", "MUL", "CSTORE"]
    expected_ports = [0, 0, 0, 1]
    pending = [(acc, [acc], [])]
    while pending:
        source, path, ports = pending.pop()
        if len(path) > len(expected_ops):
            continue
        if source == cstore:
            if ([nodes[name][0] for name in path] == expected_ops and
                    ports == expected_ports and
                    {port for _, dst, port in edges if dst == cstore} == {0, 1, 2}):
                return
            continue
        for target, port in successors.get(source, []):
            if target not in path:
                pending.append((target, path + [target], ports + [port]))
    raise AssertionError(
        f"{dot}: no single ACC -> MUL -> ADD -> MUL -> CSTORE-address path "
        "with the expected operand ports")


def main():
    cgra_opt = pathlib.Path(sys.argv[1]).resolve()
    mapper = pathlib.Path(sys.argv[2]).resolve()
    adg = pathlib.Path(sys.argv[3]).resolve()
    operations = pathlib.Path(sys.argv[4]).resolve()
    test_dir = pathlib.Path(sys.argv[5]).resolve()
    fixture = test_dir / "cstore_loop_index_acc.mlir"
    chunks = fixture.read_text().split("// -----\n")
    if len(chunks) != 14:
        raise AssertionError("expected fourteen loop-index fixture chunks")

    with tempfile.TemporaryDirectory(prefix="adora-loop-index-acc-") as temp:
        workdir = pathlib.Path(temp)
        (workdir / "lib").symlink_to(cgra_opt.parents[2] / "lib", target_is_directory=True)
        for index, chunk in enumerate(chunks[:2]):
            source = workdir / f"supported-{index}.mlir"
            source.write_text(chunk)
            run([cgra_opt, "--adora-kernel-dfg-gen", source], workdir)

        check_supported(workdir / "loop_index_a_CDFG.dot", 0, 1, 4)
        check_supported(workdir / "loop_index_b_CDFG.dot", 3, 2, 5)
        check_case_b_structure(workdir / "loop_index_b_CDFG.dot")

        static_apply = workdir / "supported-static-apply.mlir"
        static_apply.write_text(chunks[8])
        run([cgra_opt, "--adora-kernel-dfg-gen", static_apply], workdir)
        check_supported(
            workdir / "loop_index_static_apply_CDFG.dot", 1, 2, 4)
        check_case_b_structure(
            workdir / "loop_index_static_apply_CDFG.dot")

        run([mapper, "--seed=7", f"--adg={adg}", f"--op-file={operations}",
             "--output-type=pytest", "--obj-opt=false", "--max-iters=30",
             "--timeout=30000", workdir / "supported-1.mlir",
             "--output=mapped.py"], workdir)
        mapped_dfg = (workdir / "loop_index_b_map_result" / "mapped_dfg.dot")
        mapped_text = mapped_dfg.read_text()
        if '"ACC' not in mapped_text or '"FOR' in mapped_text:
            raise AssertionError("mapper did not preserve the loop-index ACC")
        check_mapped_step_route(mapped_dfg,
                                workdir / "loop_index_b_map_result" / "mapped_adg.dot", adg)

        negatives = [
            ("empty", chunks[2], "logical iteration space must be non-empty", True),
            ("iter_arg", chunks[3], "loop-carried values are unsupported", True),
            ("indirect", chunks[4], "indirect or non-address induction-value use is unsupported", True),
            ("wrong_port", chunks[5], "indirect or non-address induction-value use is unsupported", True),
            ("extreme", chunks[6], "trip count must be in the supported range", True),
            ("negative_step", chunks[7], "positive signed integer", False),
            ("nested", chunks[9], "unsupported loop-index CSTORE", True),
            ("direct_extra", chunks[10], "unsupported loop-index CSTORE", True),
            ("transformed_extra", chunks[11], "unsupported loop-index CSTORE", True),
            ("division", chunks[12], "unsupported loop-index CSTORE", True),
            ("modulo", chunks[13], "unsupported loop-index CSTORE", True),
        ]
        negative_errors = []
        for name, chunk, expected_diagnostic, needs_loop_index_diagnostic in negatives:
            unsupported = workdir / f"unsupported-{name}.mlir"
            unsupported.write_text(chunk)
            result = subprocess.run([cgra_opt, "--adora-kernel-dfg-gen", unsupported],
                                    cwd=workdir, text=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, check=False)
            output = result.stdout
            if result.returncode != 1 or expected_diagnostic not in output or \
               (needs_loop_index_diagnostic and "unsupported loop-index CSTORE" not in output):
                negative_errors.append(
                    f"missing fail-closed diagnostic for {name} "
                    f"(rc={result.returncode}):\n{output}")
        if negative_errors:
            raise AssertionError("\n".join(negative_errors))

        run([cgra_opt, "--adora-kernel-dfg-gen",
             test_dir / "mmul_relu/mmul_relu_opt.mlir"], workdir)
        generic_dot = workdir / "mmul_relu_CDFG.dot"
        generic_text = generic_dot.read_text()
        if 'opcode = "ACC"' not in generic_text or \
           'loop_index_acc="1"' in generic_text or 'acc_first=1' not in generic_text:
            raise AssertionError("ordinary ACC behavior was changed")

        run([cgra_opt, "--adora-kernel-dfg-gen",
             test_dir / "control_flow_paths/conditional_store.mlir"], workdir)
        if 'opcode = "CSTORE"' not in (workdir / "cf_direct_i8_CDFG.dot").read_text():
            raise AssertionError("existing loop-free CSTORE no longer lowers")

        run([cgra_opt, "--adora-kernel-dfg-gen",
             test_dir / "control_flow_paths/conditional_store_memory_ordering.mlir"], workdir)
        check_composed_affine_apply(workdir / "cf_affine_apply_CDFG.dot")

        run([cgra_opt, "--adora-kernel-dfg-gen",
             test_dir.parent.parent / "cgra-mapper/cstore_contract/normal_store.mlir.in"],
            workdir)


if __name__ == "__main__":
    main()
