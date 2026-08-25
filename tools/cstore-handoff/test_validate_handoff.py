#!/usr/bin/env python3
"""Black-box tests for the ADORA CSTORE handoff validator."""
import copy
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("validate_handoff.py")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def reseal(package):
    """Refresh manifest artifact hashes and SHA256SUMS after a semantic edit."""
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for key, rel in (("operations", "artifacts/operations.json"),
                     ("adg", "artifacts/adg.json")):
        manifest["artifacts"][key]["sha256"] = digest(package / rel)
    manifest["generation"]["resolved_input"]["sha256"] = digest(
        package / "artifacts/generator-input/vitra_spec.json")
    dump(manifest_path, manifest)
    payload = sorted(p for p in package.rglob("*") if p.is_file() and
                     p != package / "SHA256SUMS")
    (package / "SHA256SUMS").write_text("".join(
        f"{digest(p)}  {p.relative_to(package).as_posix()}\n" for p in payload),
        encoding="utf-8")


def valid_adg():
    config = {
        "0": ["This", 142, 0], "17": ["IsStore", 125, 125],
        "18": ["UseAddr", 126, 126], "19": ["UseEn", 127, 127],
    }
    edges = {}
    for n in range(3):
        edges[str(4 * n)] = [0, "This", 2 * n, 3 + n, "Muxn", 0]
        edges[str(4 * n + 1)] = [0, "This", 2 * n + 1, 3 + n, "Muxn", 1]
        edges[str(4 * n + 2)] = [3 + n, "Muxn", 0, 2, "DelayPipe", n]
        edges[str(4 * n + 3)] = [2, "DelayPipe", n, 1, "IOController", n]
    attrs = {
        "num_input": 6, "num_operands": 3, "iob_mode": 2,
        "operations": ["INPUT", "OUTPUT", "LOAD", "STORE", "CSTORE"],
        "io_controller_cfg_id": {"IsStore": 17, "UseAddr": 18, "UseEn": 19},
        "configuration": config,
        "instances": [{"id": 0, "type": "This"},
                      {"id": 1, "type": "IOController"},
                      {"id": 2, "type": "DelayPipe"},
                      {"id": 3, "type": "Muxn"}, {"id": 4, "type": "Muxn"},
                      {"id": 5, "type": "Muxn"}],
        "sub_modules": [{"id": 1, "type": "IOController"},
                        {"id": 2, "type": "DelayPipe"}, {"id": 3, "type": "Muxn"}],
        "connections": edges,
    }
    return {"connection_format": ["src_id", "src_type", "src_out_idx", "dst_id", "dst_type", "dst_in_idx"],
            "sub_modules": [{"id": 1, "type": "IOB", "attributes": attrs}],
            "instances": [{"id": 42, "type": "IOB", "module_id": 1}],
            "connections": {str(n): [100 + n, "GIB", 0, 42, "IOB", n] for n in range(6)}}


def valid_manifest():
    return {
        "schema_version": 1, "contract": "adora-cstore-v1", "status": "ready",
        "source": {"repository": "https://example.invalid/vitra", "branch": "cstore",
                   "commit": "a" * 40, "remote_ref": "refs/heads/cstore",
                   "remote_commit": "a" * 40, "source_snapshot": "snapshot",
                   "working_tree_clean": True},
        "generation": {"entrypoint": "gen.Main", "command": "generate",
                       "generation_id": "build-1", "timestamp_utc": "2026-08-20T12:00:00Z",
                       "toolchain_versions": {"python": "3"}, "exit_code": 0,
                       "elapsed_seconds": 1, "artifact_audit": "pass",
                       "source_files": [{"role": "generator", "path": "src/generator.scala", "blob": "c" * 40},
                                        {"role": "test", "path": "src/generator_test.scala", "blob": "d" * 40}],
                       "rtl_sha256": "e" * 64,
                       "resolved_input": {"path": "artifacts/generator-input/vitra_spec.json",
                                          "sha256": "", "generated_source_path": "spec/vitra_spec.json"}},
        "delivery_target": {"repository": "https://example.invalid/adora", "branch": "main",
                            "commit_before_delivery": "b" * 40},
        "artifacts": {"operations": {"path": "artifacts/operations.json", "sha256": "", "generated_source_path": "spec/operations.json"},
                      "adg": {"path": "artifacts/adg.json", "sha256": "", "generated_source_path": "spec/adg.json"}},
        "cstore_contract": {"operation": "CSTORE", "opcode": 4, "latency": 1,
            "operand_ports": {"data": 0, "address": 1, "enable": 2}, "address_unit": "byte",
            "use_en": {"kind": "static mode bit", "config_id": 19, "aggregate_bit": 127,
                       "zero": "normal STORE; operand 2 ignored", "one": "write permitted only when operand 2 bit 0 is one"},
            "predicate_encoding": "uniform-width operand 2, word bit 0; 0 and 2 are false, 1 is true",
            "production_add_reg_sram": 2, "cload": "out_of_scope_and_not_advertised"},
        "validation": {"true_write": {"write_count": 1}, "false_zero_write": {"write_count": 0},
                       "alternating": {"predicates": [1, 0, 1, 0], "writes": 2, "suppressions": 2},
                       "normal_store": {"passed": True},
                       "full_regression": {"tests": 4, "suites": 1, "succeeded": 4, "failed": 0}},
    }


def create_package(root):
    package = root / "vitra-cstore-hardware"
    (package / "artifacts/generator-input").mkdir(parents=True)
    (package / "evidence").mkdir()
    (package / "HANDOFF.md").write_text("handoff\n")
    (package / "evidence/hardware-validation.md").write_text("evidence\n")
    (package / "artifacts/generator-input/vitra_spec.json").write_text("{}\n")
    dump(package / "artifacts/operations.json", {"Operations": [
        {"name": "STORE", "numOperands": 2, "numRes": 0, "OPC": 3, "latency": 1},
        {"name": "CSTORE", "numOperands": 3, "numRes": 0, "OPC": 4, "latency": 1}]})
    dump(package / "artifacts/adg.json", valid_adg())
    dump(package / "manifest.json", valid_manifest())
    reseal(package)
    return package


class ValidatorCLITest(unittest.TestCase):
    def run_cli(self, package):
        return subprocess.run([sys.executable, str(SCRIPT), "--package", str(package)],
                              text=True, capture_output=True, check=False)

    def run_args(self, *args):
        return subprocess.run([sys.executable, str(SCRIPT), *args],
                              text=True, capture_output=True, check=False)

    def assert_invalid(self, package, code, category):
        result = self.run_cli(package)
        self.assertEqual(result.returncode, code, result.stderr)
        self.assertIn(f"HANDOFF PACKAGE INVALID category={category}", result.stderr)

    def test_valid_package_prints_all_identifiers(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_cli(create_package(Path(tmp)))
        self.assertEqual(result.returncode, 0, result.stderr)
        for key in ("HANDOFF PACKAGE VALID", "vitra_commit=", "manifest_sha256=",
                    "operations_sha256=", "adg_sha256="):
            self.assertIn(key, result.stdout)

    def test_missing_required_file_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = create_package(Path(tmp)); (package / "HANDOFF.md").unlink()
            self.assert_invalid(package, 10, "missing")

    def test_default_layout_rejects_extra_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"; target = repo / "tools/cstore-handoff"
            target.mkdir(parents=True); shutil.copy2(SCRIPT, target / SCRIPT.name)
            incoming = repo / ".cstore-handoff/incoming"; create_package(incoming)
            (incoming / "other-package").mkdir()
            result = subprocess.run([sys.executable, str(target / SCRIPT.name)], text=True,
                                    capture_output=True, check=False)
        self.assertEqual(result.returncode, 10, result.stderr)
        self.assertIn("HANDOFF PACKAGE INVALID category=layout", result.stderr)

    def test_tamper_is_hash_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = create_package(Path(tmp)); path = package / "artifacts/adg.json"
            path.write_bytes(path.read_bytes() + b" ")
            self.assert_invalid(package, 11, "hash")

    def test_unlisted_nested_sha256sums_is_hash_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = create_package(Path(tmp))
            (package / "artifacts/SHA256SUMS").write_text("unlisted payload\n")
            self.assert_invalid(package, 11, "hash")

    def test_invalid_utf8_root_sha256sums_is_one_line_hash_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = create_package(Path(tmp))
            (package / "SHA256SUMS").write_bytes(b"\xff")
            result = self.run_cli(package)
            self.assertEqual(result.returncode, 11, result.stderr)
            self.assertEqual(result.stderr.count("\n"), 1, result.stderr)
            self.assertTrue(result.stderr.startswith(
                "HANDOFF PACKAGE INVALID category=hash detail="), result.stderr)

    def test_listed_symlink_loop_is_one_line_hash_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = create_package(Path(tmp))
            (package / "artifacts/loop").symlink_to("loop")
            with (package / "SHA256SUMS").open("a", encoding="utf-8") as stream:
                stream.write(f"{'0' * 64}  artifacts/loop\n")
            result = self.run_cli(package)
            self.assertEqual(result.returncode, 11, result.stderr)
            self.assertEqual(result.stderr.count("\n"), 1, result.stderr)
            self.assertTrue(result.stderr.startswith(
                "HANDOFF PACKAGE INVALID category=hash detail="), result.stderr)

    def test_resealed_invalid_utf8_json_is_one_line_schema_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = create_package(Path(tmp))
            (package / "artifacts/operations.json").write_bytes(b"\xff")
            reseal(package)
            result = self.run_cli(package)
            self.assertEqual(result.returncode, 12, result.stderr)
            self.assertEqual(result.stderr.count("\n"), 1, result.stderr)
            self.assertTrue(result.stderr.startswith(
                "HANDOFF PACKAGE INVALID category=schema detail="), result.stderr)

    def test_manifest_json_schema_placeholder_and_unsafe_path_are_schema_failures(self):
        mutations = [
            lambda p: (p / "manifest.json").write_text("{"),
            lambda p: mutate_manifest(p, lambda m: m.__setitem__("contract", "wrong")),
            lambda p: mutate_manifest(p, lambda m: m["source"].__setitem__("branch", "TODO")),
            lambda p: mutate_manifest(p, lambda m: m["artifacts"]["adg"].__setitem__("path", "../adg.json")),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                package = create_package(Path(tmp)); mutation(package)
                if (package / "manifest.json").read_text() == "{":
                    payload = sorted(p for p in package.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
                    (package / "SHA256SUMS").write_text("".join(f"{digest(p)}  {p.relative_to(package).as_posix()}\n" for p in payload))
                else:
                    reseal(package)
                self.assert_invalid(package, 12, "schema")

    def test_missing_or_malformed_provenance_is_schema_failure(self):
        mutations = [
            lambda p: mutate_manifest(p, lambda m: m["generation"].pop("source_files")),
            lambda p: mutate_manifest(p, lambda m: m["generation"].__setitem__("source_files", [])),
            lambda p: mutate_manifest(p, lambda m: m["generation"]["source_files"].__setitem__(0, {"role": "generator", "path": "../bad", "blob": "c" * 40})),
            lambda p: mutate_manifest(p, lambda m: m["generation"].__setitem__("rtl_sha256", "bad")),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                package = create_package(Path(tmp)); mutation(package); reseal(package)
                self.assert_invalid(package, 12, "schema")

    def test_resealed_cstore_contract_corruptions_are_contract_failures(self):
        mutations = [remove_cstore, add_cload, remove_use_en, two_operands,
                     disconnect_operand_two, overlap_configuration, omit_top_input_five,
                     contradict_write_counts]
        for mutation in mutations:
            with self.subTest(mutation=mutation.__name__), tempfile.TemporaryDirectory() as tmp:
                package = create_package(Path(tmp)); mutation(package); reseal(package)
                self.assert_invalid(package, 13, "contract")

    def test_boolean_evidence_counts_are_contract_failures(self):
        mutations = [true_write_count_boolean, false_write_count_boolean,
                     alternating_writes_boolean, alternating_suppressions_boolean,
                     regression_tests_boolean, regression_succeeded_boolean,
                     regression_suites_boolean, regression_failed_boolean]
        for mutation in mutations:
            with self.subTest(mutation=mutation.__name__), tempfile.TemporaryDirectory() as tmp:
                package = create_package(Path(tmp)); mutation(package); reseal(package)
                self.assert_invalid(package, 13, "contract")

    def test_boolean_elapsed_seconds_is_schema_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = create_package(Path(tmp))
            mutate_manifest(package, lambda m: m["generation"].__setitem__("elapsed_seconds", True))
            reseal(package)
            self.assert_invalid(package, 12, "schema")

    def test_duplicate_iob_operand_ports_are_contract_failures(self):
        mutations = [duplicate_this_source_port, duplicate_mux_destination_port]
        for mutation in mutations:
            with self.subTest(mutation=mutation.__name__), tempfile.TemporaryDirectory() as tmp:
                package = create_package(Path(tmp)); mutation(package); reseal(package)
                self.assert_invalid(package, 13, "contract")

    def test_ghost_this_endpoint_is_contract_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = create_package(Path(tmp)); ghost_this_endpoint(package); reseal(package)
            self.assert_invalid(package, 13, "contract")

    def test_ghost_iocontroller_endpoint_is_contract_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = create_package(Path(tmp)); ghost_iocontroller_endpoint(package); reseal(package)
            self.assert_invalid(package, 13, "contract")

    def test_spoofed_top_level_iob_type_is_contract_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = create_package(Path(tmp)); spoofed_top_level_iob_type(package); reseal(package)
            self.assert_invalid(package, 13, "contract")

    def test_extra_spoofed_top_level_iob_instance_is_contract_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = create_package(Path(tmp)); extra_spoofed_top_level_iob_instance(package); reseal(package)
            self.assert_invalid(package, 13, "contract")

    def test_disconnected_delaypipe_path_is_contract_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = create_package(Path(tmp)); disconnect_delaypipe_path(package); reseal(package)
            self.assert_invalid(package, 13, "contract")

    def test_configuration_below_aggregate_low_is_contract_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = create_package(Path(tmp)); configuration_below_aggregate_low(package); reseal(package)
            self.assert_invalid(package, 13, "contract")

    def test_malformed_operation_and_adg_shapes_are_contract_failures(self):
        mutations = [operations_not_array, iob_operations_not_array,
                     top_level_port_not_hashable]
        for mutation in mutations:
            with self.subTest(mutation=mutation.__name__), tempfile.TemporaryDirectory() as tmp:
                package = create_package(Path(tmp)); mutation(package); reseal(package)
                result = self.run_cli(package)
                self.assertEqual(result.returncode, 13, result.stderr)
                self.assertEqual(result.stderr.count("\n"), 1, result.stderr)
                self.assertTrue(result.stderr.startswith(
                    "HANDOFF PACKAGE INVALID category=contract detail="), result.stderr)

    def test_unknown_cli_argument_is_one_line_usage_failure(self):
        result = self.run_args("--unknown")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(result.stderr.count("\n"), 1, result.stderr)
        self.assertTrue(result.stderr.startswith(
            "HANDOFF PACKAGE INVALID category=usage detail="), result.stderr)


def mutate_manifest(package, change):
    path = package / "manifest.json"; manifest = json.loads(path.read_text()); change(manifest); dump(path, manifest)


def json_file(package, rel):
    path = package / rel; value = json.loads(path.read_text()); return path, value


def remove_cstore(package):
    path, value = json_file(package, "artifacts/operations.json")
    value["Operations"] = value["Operations"][:-1]; dump(path, value)


def add_cload(package):
    path, value = json_file(package, "artifacts/operations.json")
    value["Operations"].append({"name": "CLOAD", "numOperands": 1, "numRes": 1, "OPC": 5, "latency": 1}); dump(path, value)


def attrs(package):
    path, value = json_file(package, "artifacts/adg.json")
    return path, value, value["sub_modules"][0]["attributes"]


def remove_use_en(package):
    path, value, a = attrs(package); del a["io_controller_cfg_id"]["UseEn"]; del a["configuration"]["19"]; dump(path, value)


def two_operands(package):
    path, value = json_file(package, "artifacts/operations.json"); value["Operations"][-1]["numOperands"] = 2; dump(path, value)


def disconnect_operand_two(package):
    path, value, a = attrs(package); del a["connections"]["11"]; dump(path, value)


def overlap_configuration(package):
    path, value, a = attrs(package); a["configuration"]["18"] = ["UseAddr", 127, 126]; dump(path, value)


def omit_top_input_five(package):
    path, value = json_file(package, "artifacts/adg.json"); del value["connections"]["5"]; dump(path, value)


def contradict_write_counts(package):
    mutate_manifest(package, lambda m: m["validation"]["false_zero_write"].__setitem__("write_count", 1))


def true_write_count_boolean(package):
    mutate_manifest(package, lambda m: m["validation"]["true_write"].__setitem__("write_count", True))


def false_write_count_boolean(package):
    mutate_manifest(package, lambda m: m["validation"]["false_zero_write"].__setitem__("write_count", False))


def alternating_writes_boolean(package):
    mutate_manifest(package, lambda m: m["validation"]["alternating"].__setitem__("writes", True))


def alternating_suppressions_boolean(package):
    mutate_manifest(package, lambda m: m["validation"]["alternating"].__setitem__("suppressions", True))


def regression_tests_boolean(package):
    def change(manifest):
        regression = manifest["validation"]["full_regression"]
        regression["tests"] = True
        regression["succeeded"] = 1
    mutate_manifest(package, change)


def regression_succeeded_boolean(package):
    def change(manifest):
        regression = manifest["validation"]["full_regression"]
        regression["tests"] = 1
        regression["succeeded"] = True
    mutate_manifest(package, change)


def regression_suites_boolean(package):
    mutate_manifest(package, lambda m: m["validation"]["full_regression"].__setitem__("suites", True))


def regression_failed_boolean(package):
    mutate_manifest(package, lambda m: m["validation"]["full_regression"].__setitem__("failed", False))


def disconnect_delaypipe_path(package):
    path, value, a = attrs(package)
    a["instances"].append({"id": 6, "type": "DelayPipe"})
    a["connections"]["11"][0] = 6
    dump(path, value)


def configuration_below_aggregate_low(package):
    path, value, a = attrs(package)
    a["configuration"]["0"] = ["This", 142, 100]
    a["configuration"]["17"] = ["IsStore", 99, 99]
    dump(path, value)


def operations_not_array(package):
    path, value = json_file(package, "artifacts/operations.json")
    value["Operations"] = 1
    dump(path, value)


def iob_operations_not_array(package):
    path, value, a = attrs(package)
    a["operations"] = 1
    dump(path, value)


def top_level_port_not_hashable(package):
    path, value = json_file(package, "artifacts/adg.json")
    value["connections"]["5"][-1] = []
    dump(path, value)


def duplicate_this_source_port(package):
    path, value, a = attrs(package)
    a["connections"]["9"][2] = 4
    dump(path, value)


def duplicate_mux_destination_port(package):
    path, value, a = attrs(package)
    a["connections"]["9"][5] = 0
    dump(path, value)


def ghost_this_endpoint(package):
    path, value, a = attrs(package)
    a["connections"]["8"][0] = 99
    dump(path, value)


def ghost_iocontroller_endpoint(package):
    path, value, a = attrs(package)
    a["connections"]["11"][3] = 99
    dump(path, value)


def spoofed_top_level_iob_type(package):
    path, value = json_file(package, "artifacts/adg.json")
    value["instances"][0]["type"] = "GIB"
    dump(path, value)


def extra_spoofed_top_level_iob_instance(package):
    path, value = json_file(package, "artifacts/adg.json")
    value["instances"].append({"id": 43, "type": "GIB", "module_id": 1})
    dump(path, value)


if __name__ == "__main__":
    unittest.main(verbosity=2)
