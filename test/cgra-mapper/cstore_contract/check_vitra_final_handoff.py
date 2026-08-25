#!/usr/bin/env python3
import hashlib
import json
import sys
from pathlib import Path


EXPECTED_VITRA_COMMIT = "da03f4ab0cf696466147ac9210518e7ead6c9589"
EXPECTED_OPERATIONS_SHA = "0eee215afdbed65fed6bd7773f40a783189d67a66accb6e7bc86aaac582bae8d"
EXPECTED_ADG_SHA = "e34fcef5b15f47718762812513d3cd7db35f06e60a17097e50429dfc41e26cf7"
EXPECTED_LOOP_SHA = "c3d358639ef9470f5abb5c3a8146e75c6920c6055e28dfba780868073312489e"
EXPECTED_CFG_FIELDS = {"InitVal", "WI", "Latency", "Cycles", "Repeats", "SkipFirst"}
EXPECTED_OPERAND_PORTS = {"data": 0, "address": 1, "enable": 2}
EXPECTED_ARTIFACTS = {
    "operations": ("artifacts/operations.json", EXPECTED_OPERATIONS_SHA),
    "adg": ("artifacts/adg.json", EXPECTED_ADG_SHA),
    "loop_index_contract": ("artifacts/loop_index_contract.json", EXPECTED_LOOP_SHA),
}


def fail(message: str) -> None:
    raise AssertionError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def load_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_manifest_artifact(manifest: dict, fixture: Path, name: str) -> Path:
    expected_path, expected_sha = EXPECTED_ARTIFACTS[name]
    artifact = manifest["artifacts"][name]
    relpath = artifact["path"]
    require(relpath == expected_path, f"manifest {name} path changed")
    require(artifact["sha256"] == expected_sha, f"manifest {name} sha256 changed")
    path = fixture / relpath
    require(path.is_file(), f"fixture {name} payload is missing")
    require(sha256(path) == expected_sha, f"fixture {name} payload hash mismatch")
    return path


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} <fixture-root>", file=sys.stderr)
        return 2

    fixture = Path(argv[1]).resolve()
    manifest = load_json(fixture / "manifest.json")

    require(manifest["source"]["commit"] == EXPECTED_VITRA_COMMIT,
            "fixture is not pinned to the final VITRA commit")
    require(manifest["source"]["remote_commit"] == EXPECTED_VITRA_COMMIT,
            "fixture remote provenance does not match the final VITRA commit")
    require(manifest["cstore_contract"]["operand_ports"] == EXPECTED_OPERAND_PORTS,
            "manifest CSTORE operand_ports changed")

    ops_path = check_manifest_artifact(manifest, fixture, "operations")
    adg_path = check_manifest_artifact(manifest, fixture, "adg")
    loop_path = check_manifest_artifact(manifest, fixture, "loop_index_contract")

    operations = load_json(ops_path)["Operations"]
    by_name = {entry["name"]: entry for entry in operations}
    require("CSTORE" in by_name, "CSTORE is missing from operations.json")
    require("ACC" in by_name, "ACC is missing from operations.json")
    require("FOR" not in by_name, "FOR must remain absent from operations.json")
    require("CLOAD" not in by_name, "CLOAD must remain absent from operations.json")
    require(by_name["CSTORE"]["numOperands"] == 3, "CSTORE must keep three logical operands")
    require(by_name["CSTORE"]["OPC"] == 4, "CSTORE opcode changed")
    require(by_name["STORE"]["numOperands"] == 2, "normal STORE must keep two logical operands")

    adg = load_json(adg_path)
    iob_modules = [module for module in adg["sub_modules"]
                   if isinstance(module, dict) and module.get("type") == "IOB"]
    require(len(iob_modules) == 1, "expected exactly one IOB module in the fixture ADG")
    attrs = iob_modules[0].get("attributes", iob_modules[0])
    require(set(attrs["operations"]) == {"INPUT", "OUTPUT", "LOAD", "STORE", "CSTORE"},
            "unexpected IOB capability set")
    require(attrs["num_operands"] == 3, "IOB must expose three logical operands")
    require(attrs["io_controller_cfg_id"]["UseEn"] == 19, "UseEn config id changed")

    loop = load_json(loop_path)
    require(loop["logical_source"] == "affine.for induction value", "unexpected loop logical source")
    require(loop["physical_operation"] == "ACC", "loop lowering must target physical ACC")
    require(loop["for_is_physical_operation"] is False, "physical FOR must remain disabled")
    require(loop["byte_address_scaling"] == "external-explicit-arithmetic",
            "unexpected byte-address scaling contract")
    require(loop["supported_loop_subset"]["minimum_trip_count"] == 1,
            "minimum supported trip count changed")
    require(loop["supported_loop_subset"]["maximum_trip_count"] == 4095,
            "maximum supported trip count changed")
    require(loop["supported_loop_subset"]["positive_step"] == "unsigned nonzero 16-bit value",
            "positive-step contract changed")
    require(loop["supported_loop_subset"]["negative_step"] is False,
            "negative-step support must remain disabled")
    require(set(loop["configuration_fields"]) == EXPECTED_CFG_FIELDS,
            "ACC loop-index configuration fields are incomplete")
    require(loop["configuration_formula"]["operation"] == "ACC",
            "loop-index configuration formula must target ACC")
    require(loop["configuration_formula"]["WI"] == 1, "ACC WI field changed")
    require(loop["configuration_formula"]["Latency"] == 0, "ACC Latency field changed")
    require(loop["configuration_formula"]["Repeats"] == 1, "ACC Repeats field changed")
    require(loop["configuration_formula"]["SkipFirst"] == 1, "ACC SkipFirst field changed")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
