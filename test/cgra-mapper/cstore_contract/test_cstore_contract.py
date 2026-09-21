#!/usr/bin/env python3
"""Black-box ADG and DFG contract checks for CSTORE."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


CASES = (
    "no-cstore",
    "missing-controller-map",
    "non-object-controller-map",
    "missing-use-en",
    "non-integer-use-en",
    "two-operands",
    "missing-num-operands",
    "malformed-num-operands",
    "missing-enable-path",
    "overlapping-input-paths",
    "missing-intra-endpoint",
    "cyclic-intra-connect",
    "missing-module-iob-index",
    "non-integer-module-iob-index",
    "missing-instance-iob-index",
    "non-integer-instance-iob-index",
    "non-object-connections",
    "truncated-connection",
    "non-array-connection",
    "non-integer-connection-endpoint",
    "non-integer-connection-port",
    "invalid-cstore-operands",
    "invalid-cstore-results",
)

HARDWARE_CASES = frozenset(
    case for case in CASES
    if case not in ("no-cstore", "invalid-cstore-operands",
                    "invalid-cstore-results"))


def load(path):
    with path.open() as stream:
        return json.load(stream)


def dump(path, value):
    with path.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def cstore_iob_module(adg):
    for module in adg["sub_modules"]:
        attrs = module.get("attributes", {})
        if module.get("type") == "IOB" and "CSTORE" in attrs.get("operations", []):
            return module
    raise AssertionError("pinned ADG has no CSTORE IOB")


def cstore_iob(adg):
    return cstore_iob_module(adg)["attributes"]


def cstore_iob_instance(adg):
    module_id = cstore_iob_module(adg)["id"]
    for instance in adg["instances"]:
        if instance.get("type") == "IOB" and instance.get("module_id") == module_id:
            return instance
    raise AssertionError("pinned ADG has no CSTORE IOB instance")


def mutate(case, adg, operations):
    attrs = cstore_iob(adg)
    if case == "no-cstore":
        attrs["operations"].remove("CSTORE")
        use_en_id = attrs["io_controller_cfg_id"].pop("UseEn")
        del attrs["configuration"][str(use_en_id)]
        return "No legal ADG node supports CSTORE", "required=1", "available=0"
    if case == "missing-controller-map":
        del attrs["io_controller_cfg_id"]
        return "missing io_controller_cfg_id",
    if case == "non-object-controller-map":
        attrs["io_controller_cfg_id"] = []
        return "expected io_controller_cfg_id object",
    if case == "missing-use-en":
        del attrs["io_controller_cfg_id"]["UseEn"]
        return "missing UseEn",
    if case == "non-integer-use-en":
        attrs["io_controller_cfg_id"]["UseEn"] = "enabled"
        return "UseEn must be an integer",
    if case == "two-operands":
        attrs["num_operands"] = 2
        return "expected num_operands=3",
    if case == "missing-num-operands":
        del attrs["num_operands"]
        return "expected num_operands=3",
    if case == "malformed-num-operands":
        attrs["num_operands"] = "3"
        return "expected num_operands=3",
    if case == "missing-enable-path":
        del attrs["connections"]["12"]
        return "operand 2 has no physical input path",
    if case == "overlapping-input-paths":
        for connection in attrs["connections"].values():
            if connection[0] == 0 and connection[2] == 4:
                connection[3] = 3
                connection[5] = 0
                attrs["connections"][str(max(map(int, attrs["connections"])) + 1)] = \
                    [0, "This", 4, 5, "Muxn", 0]
                return "physical input sets overlap",
        raise AssertionError("pinned CSTORE IOB has no physical input 4")
    if case == "missing-intra-endpoint":
        attrs["connections"]["12"][3] = 99
        return "malformed endpoint", "iob_index 0", "edge 12"
    if case == "cyclic-intra-connect":
        attrs["connections"]["11"][3] = 5
        attrs["connections"]["11"][4] = "Muxn"
        attrs["connections"]["11"][5] = 1
        return "operand 2 has no physical input path",
    if case == "missing-module-iob-index":
        del attrs["iob_index"]
        return "missing iob_index",
    if case == "non-integer-module-iob-index":
        attrs["iob_index"] = "zero"
        return "iob_index must be an integer",
    if case == "missing-instance-iob-index":
        del cstore_iob_instance(adg)["iob_index"]
        return "missing iob_index",
    if case == "non-integer-instance-iob-index":
        cstore_iob_instance(adg)["iob_index"] = "zero"
        return "iob_index must be an integer",
    if case == "non-object-connections":
        attrs["connections"] = []
        return "connections must be an object",
    if case == "truncated-connection":
        attrs["connections"]["12"] = [2, "DelayPipe"]
        return "connection 12 must contain 6 fields",
    if case == "non-array-connection":
        attrs["connections"]["12"] = {}
        return "connection 12 must be an array",
    if case == "non-integer-connection-endpoint":
        attrs["connections"]["12"][0] = "two"
        return "connection 12 endpoint/port fields must be integers",
    if case == "non-integer-connection-port":
        attrs["connections"]["12"][2] = "two"
        return "connection 12 endpoint/port fields must be integers",
    for operation in operations["Operations"]:
        if operation["name"] != "CSTORE":
            continue
        if case == "invalid-cstore-operands":
            operation["numOperands"] = 2
        elif case == "invalid-cstore-results":
            operation["numRes"] = 1
        else:
            raise AssertionError(f"unknown case: {case}")
        return "Invalid CSTORE DFG node",
    raise AssertionError("pinned operations has no CSTORE entry")


def store_kernel():
    return """module {
  func.func @store_only(%value: i32, %output: memref<8xi32>) {
    %index = arith.constant 0 : index
    ADORA.kernel {
      memref.store %value, %output[%index] : memref<8xi32>
      ADORA.terminator
    } {KernelName = \"store_only\"}
    return
  }
}
"""


def cstore_two_immediate_kernel():
    return """module {
  func.func @cstore_two_immediate(%index: index, %output: memref<8xi32>) {
    ADORA.kernel {
      %value = arith.constant 7 : i32
      %condition = arith.constant true
      ADORA.cond_store %value, %output[%index] if %condition : memref<8xi32>
      ADORA.terminator
    } {KernelName = \"cstore_two_immediate\"}
    return
  }
}
"""


def mixed_cstore_store_kernel():
    return """module {
  func.func @mixed_cstore_store(%input: memref<8xi32>,
                                %output: memref<8xi32>, %index: index,
                                %value: i32, %condition: i1) {
    ADORA.kernel {
      ADORA.cond_store %value, %output[%index] if %condition : memref<8xi32>
      %loaded = memref.load %input[%index] : memref<8xi32>
      memref.store %loaded, %output[%index] : memref<8xi32>
      ADORA.terminator
    } {KernelName = "mixed_cstore_store"}
    return
  }
}
"""


def unsupported_store_producer_kernel():
    return """module {
  func.func @unsupported_store_producer(%value: i32,
                                        %output: memref<?xi32>) {
    %c0 = arith.constant 0 : index
    ADORA.kernel {
      %index = memref.dim %output, %c0 : memref<?xi32>
      memref.store %value, %output[%index] : memref<?xi32>
      ADORA.terminator
    } {KernelName = "unsupported_store_producer"}
    return
  }
}
"""


def run(mapper, adg_path, operations_path, kernel, output, environment, cwd):
    command = [
        str(mapper), "--seed=7", f"--adg={adg_path}",
        f"--op-file={operations_path}", "--output-type=pytest",
        "--obj-opt=false", "--max-iters=30", "--timeout=30000",
        str(kernel), f"--output={output}",
    ]
    return subprocess.run(command, cwd=cwd, text=True, env=environment,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          timeout=60, check=False)


def run_case(case, workdir, mapper, adg_source, operations_source, kernel):
    case_dir = workdir / case
    case_dir.mkdir(parents=True, exist_ok=True)
    adg_path = case_dir / "adg.json"
    operations_path = case_dir / "operations.json"
    shutil.copyfile(adg_source, adg_path)
    shutil.copyfile(operations_source, operations_path)
    adg = load(adg_path)
    operations = load(operations_path)
    expected = mutate(case, adg, operations)
    dump(adg_path, adg)
    dump(operations_path, operations)

    environment = dict(os.environ)
    environment["GeneralOpNameFile"] = str(
        kernel.parents[3] / "lib/DFG/Documents/GeneralOpName.txt")
    result = run(mapper, adg_path, operations_path, kernel,
                 case_dir / "mapped.py", environment, case_dir)
    if result.returncode != 1:
        raise AssertionError(
            f"{case}: expected exit status 1, got {result.returncode}\n{result.stdout}")
    for message in expected:
        if message not in result.stdout:
            raise AssertionError(
                f"{case}: missing diagnostic {message!r}\n{result.stdout}")
    if case in HARDWARE_CASES and "Invalid CSTORE IOB" not in result.stdout:
        raise AssertionError(f"{case}: missing CSTORE IOB identity\n{result.stdout}")
    if case in HARDWARE_CASES:
        if (case_dir / "mapped.py").exists() or any(
                case_dir.glob("*_map_result/mapped_routes.tsv")):
            raise AssertionError(f"{case}: emitted mapped output after rejection")
    if case == "no-cstore":
        normal_store = case_dir / "normal-store.mlir"
        normal_store.write_text(store_kernel())
        normal_result = run(mapper, adg_path, operations_path, normal_store,
                            case_dir / "normal-store.py", environment, case_dir)
        if normal_result.returncode != 0:
            raise AssertionError(
                f"no-cstore normal STORE failed with {normal_result.returncode}\n"
                f"{normal_result.stdout}")
        if not (case_dir / "normal-store.py").is_file():
            raise AssertionError("no-cstore normal STORE produced no output")
        cdfg = case_dir / "store_only_map_result" / "before_map_store_only_CDFG.dot"
        if not cdfg.is_file():
            raise AssertionError("no-cstore normal STORE produced no before-map CDFG")
        if 'opcode = "store"' not in cdfg.read_text():
            raise AssertionError("no-cstore normal workload CDFG was not memref.store")
        mapped_dfgio = case_dir / "store_only_map_result" / "mapped_dfgio.txt"
        if not mapped_dfgio.is_file() or "STORE_" not in mapped_dfgio.read_text():
            raise AssertionError("no-cstore normal workload was not mapped as STORE")


def run_dfg_contract_tests(mapper, operations):
    unit = mapper.with_name("cstore-dfg-contract-test")
    expected = {
        "zero-immediate": (0, ()),
        "one-immediate": (0, ()),
        "two-immediate": (1, ("Invalid CSTORE DFG node", "immediate inputs")),
        "missing": (1, ("Invalid CSTORE DFG node", "logical operand 2")),
        "duplicate": (1, ("Invalid CSTORE DFG node", "logical operand 0")),
        "out-of-range": (1, ("Invalid CSTORE DFG node", "out of range")),
        "memory-ignored": (0, ()),
        "store-zero-immediate": (0, ()),
        "store-one-immediate": (0, ()),
        "store-two-immediate": (1, ("Invalid STORE DFG node", "immediate inputs")),
        "store-missing": (1, ("Invalid STORE DFG node", "logical operand 1")),
        "store-duplicate": (1, ("Invalid STORE DFG node", "logical operand 0")),
        "store-out-of-range": (1, ("Invalid STORE DFG node", "out of range")),
        "store-memory-ignored": (0, ()),
    }
    for case, (status, messages) in expected.items():
        result = subprocess.run([str(unit), case, str(operations)], text=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=60, check=False)
        if result.returncode != status:
            raise AssertionError(
                f"dfg-{case}: expected exit status {status}, got {result.returncode}\n"
                f"{result.stdout}")
        for message in messages:
            if message not in result.stdout:
                raise AssertionError(
                    f"dfg-{case}: missing diagnostic {message!r}\n{result.stdout}")


def run_cstore_immediate_mapper_test(workdir, mapper, adg, operations,
                                     environment):
    case_dir = workdir / "cstore-two-immediate-mapper"
    case_dir.mkdir(parents=True, exist_ok=True)
    kernel = case_dir / "kernel.mlir"
    kernel.write_text(cstore_two_immediate_kernel())
    output = case_dir / "mapped.py"
    result = run(mapper, adg, operations, kernel, output, environment, case_dir)
    if result.returncode != 1:
        raise AssertionError(
            "cstore-two-immediate-mapper: expected exit status 1, got "
            f"{result.returncode}\n{result.stdout}")
    for message in ("Invalid CSTORE DFG node", "immediate inputs"):
        if message not in result.stdout:
            raise AssertionError(
                f"cstore-two-immediate-mapper: missing {message!r}\n{result.stdout}")
    if output.exists() or (case_dir / "cstore_two_immediate_map_result" /
                           "mapped_routes.tsv").exists():
        raise AssertionError(
            "cstore-two-immediate-mapper emitted mapped output after rejection")


def run_unsupported_store_producer_test(workdir, mapper, adg, operations,
                                        environment):
    case_dir = workdir / "unsupported-store-producer"
    case_dir.mkdir(parents=True, exist_ok=True)
    kernel = case_dir / "kernel.mlir"
    kernel.write_text(unsupported_store_producer_kernel())
    output = case_dir / "mapped.py"
    result = run(mapper, adg, operations, kernel, output, environment, case_dir)
    if result.returncode != 1:
        raise AssertionError(
            "unsupported-store-producer: expected exit status 1, got "
            f"{result.returncode}\n{result.stdout}")
    for message in ("Invalid STORE CDFG node", "unsupported direct operand producer"):
        if message not in result.stdout:
            raise AssertionError(
                f"unsupported-store-producer: missing {message!r}\n{result.stdout}")
    if output.exists() or (case_dir / "unsupported_store_producer_map_result" /
                           "mapped_routes.tsv").exists():
        raise AssertionError(
            "unsupported-store-producer emitted mapped output after rejection")


def run_mixed_cstore_store_test(workdir, mapper, adg, operations, environment):
    case_dir = workdir / "mixed-cstore-store"
    case_dir.mkdir(parents=True, exist_ok=True)
    kernel = case_dir / "kernel.mlir"
    kernel.write_text(mixed_cstore_store_kernel())
    output = case_dir / "mapped.py"
    result = run(mapper, adg, operations, kernel, output, environment, case_dir)
    if result.returncode != 0:
        raise AssertionError(
            "mixed-cstore-store: expected success, got "
            f"{result.returncode}\n{result.stdout}")
    cdfg = case_dir / "mixed_cstore_store_map_result" / \
        "before_map_mixed_cstore_store_CDFG.dot"
    manifest = case_dir / "mixed_cstore_store_map_result" / "mapped_routes.tsv"
    if not output.is_file() or not manifest.is_file() or not cdfg.is_file():
        raise AssertionError("mixed-cstore-store emitted incomplete mapped output")
    text = cdfg.read_text()
    if 'opcode = "CSTORE"' not in text:
        raise AssertionError("mixed-cstore-store CDFG omitted CSTORE")
    for opcode in ('opcode = "load"', 'opcode = "store"'):
        if opcode in text:
            raise AssertionError(
                f"mixed-cstore-store CDFG leaked ordinary memory node {opcode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workdir", type=Path)
    parser.add_argument("mapper", type=Path)
    parser.add_argument("adg", type=Path)
    parser.add_argument("operations", type=Path)
    parser.add_argument("kernel", type=Path)
    parser.add_argument("--case", choices=("all",) + CASES, default="all")
    parser.add_argument("--skip-dfg-contract", action="store_true")
    args = parser.parse_args()
    args.mapper = args.mapper.resolve()
    args.adg = args.adg.resolve()
    args.operations = args.operations.resolve()
    args.kernel = args.kernel.resolve()
    args.workdir = args.workdir.resolve()
    if not args.skip_dfg_contract:
        run_dfg_contract_tests(args.mapper, args.operations)
        environment = dict(os.environ)
        environment["GeneralOpNameFile"] = str(
            args.kernel.parents[3] / "lib/DFG/Documents/GeneralOpName.txt")
        run_cstore_immediate_mapper_test(
            args.workdir, args.mapper, args.adg, args.operations, environment)
        run_unsupported_store_producer_test(
            args.workdir, args.mapper, args.adg, args.operations, environment)
        run_mixed_cstore_store_test(
            args.workdir, args.mapper, args.adg, args.operations, environment)
    cases = CASES if args.case == "all" else (args.case,)
    for case in cases:
        run_case(case, args.workdir, args.mapper, args.adg, args.operations,
                 args.kernel)
    print("CSTORE contract variants passed:", ", ".join(cases))


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, subprocess.TimeoutExpired) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
