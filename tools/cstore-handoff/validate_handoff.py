#!/usr/bin/env python3
"""Validate an immutable VITRA CSTORE handoff package without external tools."""
import argparse
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path


MISSING, HASH, SCHEMA, CONTRACT = 10, 11, 12, 13
SHA_LINE = re.compile(r"^([0-9a-f]{64})  ([^\n]+)$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")
UTC = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?Z$")
PLACEHOLDER = re.compile(r"^<[^<>]+>$")
BAD_WORDS = {"", "TBD", "TODO", "UNKNOWN", "N/A"}
REQUIRED_FILES = ("HANDOFF.md", "manifest.json", "SHA256SUMS",
                  "artifacts/operations.json", "artifacts/adg.json",
                  "evidence/hardware-validation.md")


class Invalid(Exception):
    def __init__(self, code, category, detail):
        self.code, self.category, self.detail = code, category, detail


def fail(code, category, detail):
    raise Invalid(code, category, detail)


class HandoffArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise Invalid(2, "usage", message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_file(package, value, code=SCHEMA):
    if not isinstance(value, str) or not value or "\\" in value:
        fail(code, "schema" if code == SCHEMA else "hash", "unsafe path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or any(part in ("", ".") for part in path.parts):
        fail(code, "schema" if code == SCHEMA else "hash", "unsafe path")
    candidate = package / path
    if candidate.is_symlink():
        fail(code, "schema" if code == SCHEMA else "hash", "path is not a regular file")
    try:
        candidate.resolve().relative_to(package.resolve())
    except ValueError:
        fail(code, "schema" if code == SCHEMA else "hash", "path escapes package")
    except (RuntimeError, OSError):
        fail(code, "schema" if code == SCHEMA else "hash", "path cannot be resolved")
    if not candidate.is_file():
        fail(code, "schema" if code == SCHEMA else "hash", "path is not a regular file")
    return candidate


def safe_repo_relative(value, label):
    if not isinstance(value, str) or not value or "\\" in value:
        fail(SCHEMA, "schema", f"unsafe source path: {label}")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or any(part in ("", ".") for part in path.parts):
        fail(SCHEMA, "schema", f"unsafe source path: {label}")
    return value


def nonplaceholder(value, name):
    if not isinstance(value, str) or value.strip().upper() in BAD_WORDS or PLACEHOLDER.fullmatch(value.strip()):
        fail(SCHEMA, "schema", f"invalid required string: {name}")
    return value


def require_dict(value, name):
    if not isinstance(value, dict): fail(SCHEMA, "schema", f"{name} must be an object")
    return value


def require_list(value, name):
    if not isinstance(value, list): fail(SCHEMA, "schema", f"{name} must be an array")
    return value


def required_string(obj, key, prefix):
    if key not in obj: fail(SCHEMA, "schema", f"missing {prefix}.{key}")
    return nonplaceholder(obj[key], f"{prefix}.{key}")


def recursively_no_placeholders(value, name="manifest"):
    if isinstance(value, str): nonplaceholder(value, name)
    elif isinstance(value, dict):
        for key, item in value.items(): recursively_no_placeholders(item, f"{name}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value): recursively_no_placeholders(item, f"{name}[{index}]")


def json_file(path, label):
    try:
        with path.open(encoding="utf-8") as stream: return json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        fail(SCHEMA, "schema", f"invalid JSON in {label}: {exc}")


def verify_layout_and_hashes(package):
    if not package.is_dir() or package.is_symlink(): fail(MISSING, "missing", "package directory missing")
    for rel in REQUIRED_FILES:
        if not (package / rel).is_file() or (package / rel).is_symlink():
            fail(MISSING, "missing", f"required file missing: {rel}")
    generator = package / "artifacts/generator-input"
    if not generator.is_dir() or generator.is_symlink() or not any(
            path.is_file() and not path.is_symlink() for path in generator.rglob("*")):
        fail(MISSING, "missing", "generator-input has no regular file")
    try:
        lines = (package / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        fail(MISSING, "missing", str(exc))
    except UnicodeError as exc:
        fail(HASH, "hash", f"invalid SHA256SUMS: {exc}")
    listed = {}
    for line in lines:
        match = SHA_LINE.fullmatch(line)
        if not match: fail(HASH, "hash", "malformed SHA256SUMS entry")
        digest, rel = match.groups()
        if rel in listed: fail(HASH, "hash", "duplicate SHA256SUMS path")
        safe_file(package, rel, HASH)
        listed[rel] = digest
    payload = set()
    for path in package.rglob("*"):
        if path.is_symlink(): fail(HASH, "hash", "symlink in package")
        if path.is_file() and path != package / "SHA256SUMS": payload.add(path.relative_to(package).as_posix())
    if set(listed) != payload: fail(HASH, "hash", "SHA256SUMS does not cover exactly the payload")
    for rel, expected in listed.items():
        if sha256(package / rel) != expected: fail(HASH, "hash", f"digest mismatch: {rel}")
    return listed


def validate_manifest(package, listed):
    manifest = json_file(package / "manifest.json", "manifest.json")
    require_dict(manifest, "manifest")
    recursively_no_placeholders(manifest)
    if manifest.get("schema_version") != 1 or manifest.get("contract") != "adora-cstore-v1" or manifest.get("status") != "ready":
        fail(SCHEMA, "schema", "unsupported manifest schema, contract, or status")
    source = require_dict(manifest.get("source"), "source")
    for key in ("repository", "branch", "commit", "remote_ref", "remote_commit", "source_snapshot"):
        required_string(source, key, "source")
    if not COMMIT.fullmatch(source["commit"]) or source["commit"] != source["remote_commit"] or not COMMIT.fullmatch(source["remote_commit"]):
        fail(SCHEMA, "schema", "source commits must be equal lowercase hashes")
    if source.get("working_tree_clean") is not True: fail(SCHEMA, "schema", "source working tree is not clean")
    generation = require_dict(manifest.get("generation"), "generation")
    for key in ("entrypoint", "command", "generation_id", "timestamp_utc", "artifact_audit"):
        required_string(generation, key, "generation")
    if not UTC.fullmatch(generation["timestamp_utc"]): fail(SCHEMA, "schema", "timestamp must be RFC3339 UTC")
    try: dt.datetime.fromisoformat(generation["timestamp_utc"].replace("Z", "+00:00"))
    except ValueError: fail(SCHEMA, "schema", "invalid timestamp")
    versions = require_dict(generation.get("toolchain_versions"), "generation.toolchain_versions")
    if not versions or any(not isinstance(k, str) or not isinstance(v, str) or not v for k, v in versions.items()):
        fail(SCHEMA, "schema", "invalid toolchain versions")
    if type(generation.get("exit_code")) is not int or generation["exit_code"] != 0 or type(generation.get("elapsed_seconds")) not in (int, float) or generation["elapsed_seconds"] <= 0:
        fail(SCHEMA, "schema", "invalid generation result")
    source_files = require_list(generation.get("source_files"), "generation.source_files")
    roles = set()
    if not source_files:
        fail(SCHEMA, "schema", "generation.source_files is empty")
    for index, entry in enumerate(source_files):
        entry = require_dict(entry, f"generation.source_files[{index}]")
        role = required_string(entry, "role", f"generation.source_files[{index}]")
        safe_repo_relative(entry.get("path"), f"generation.source_files[{index}]")
        blob = entry.get("blob")
        if not isinstance(blob, str) or not COMMIT.fullmatch(blob):
            fail(SCHEMA, "schema", "source file blob must be a lowercase Git hash")
        roles.add(role)
    if not {"generator", "test"}.issubset(roles):
        fail(SCHEMA, "schema", "source_files must include generator and test roles")
    rtl_hash = generation.get("rtl_sha256")
    if not isinstance(rtl_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", rtl_hash):
        fail(SCHEMA, "schema", "generation.rtl_sha256 must be lowercase SHA-256")
    resolved = require_dict(generation.get("resolved_input"), "generation.resolved_input")
    check_artifact(package, listed, resolved, "generation.resolved_input")
    required_string(resolved, "generated_source_path", "generation.resolved_input")
    delivery = require_dict(manifest.get("delivery_target"), "delivery_target")
    for key in ("repository", "branch", "commit_before_delivery"): required_string(delivery, key, "delivery_target")
    artifacts = require_dict(manifest.get("artifacts"), "artifacts")
    for key in ("operations", "adg"):
        artifact = require_dict(artifacts.get(key), f"artifacts.{key}")
        check_artifact(package, listed, artifact, f"artifacts.{key}")
        required_string(artifact, "generated_source_path", f"artifacts.{key}")
    return manifest


def check_artifact(package, listed, artifact, label):
    path = safe_file(package, artifact.get("path"), SCHEMA)
    expected = artifact.get("sha256")
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected): fail(SCHEMA, "schema", f"invalid hash: {label}")
    rel = path.relative_to(package).as_posix()
    if sha256(path) != expected or listed.get(rel) != expected: fail(SCHEMA, "schema", f"manifest hash mismatch: {label}")


def require_contract(manifest):
    contract = require_dict(manifest.get("cstore_contract"), "cstore_contract")
    expected = {"operation": "CSTORE", "opcode": 4, "latency": 1, "address_unit": "byte",
                "predicate_encoding": "uniform-width operand 2, word bit 0; 0 and 2 are false, 1 is true",
                "production_add_reg_sram": 2, "cload": "out_of_scope_and_not_advertised"}
    if any(contract.get(k) != v for k, v in expected.items()) or contract.get("operand_ports") != {"data": 0, "address": 1, "enable": 2}:
        fail(CONTRACT, "contract", "manifest CSTORE contract mismatch")
    use_en = contract.get("use_en")
    expected_use_en = {"kind": "static mode bit", "config_id": 19, "aggregate_bit": 127,
                       "zero": "normal STORE; operand 2 ignored", "one": "write permitted only when operand 2 bit 0 is one"}
    if use_en != expected_use_en: fail(CONTRACT, "contract", "manifest UseEn contract mismatch")
    validation = require_dict(manifest.get("validation"), "validation")
    true = require_dict(validation.get("true_write"), "validation.true_write")
    false = require_dict(validation.get("false_zero_write"), "validation.false_zero_write")
    alternating = require_dict(validation.get("alternating"), "validation.alternating")
    normal = require_dict(validation.get("normal_store"), "validation.normal_store")
    regression = require_dict(validation.get("full_regression"), "validation.full_regression")
    if type(true.get("write_count")) is not int or true["write_count"] != 1 or type(false.get("write_count")) is not int or false["write_count"] != 0 or alternating.get("predicates") != [1, 0, 1, 0] or type(alternating.get("writes")) is not int or alternating["writes"] <= 0 or type(alternating.get("suppressions")) is not int or alternating["suppressions"] <= 0 or normal.get("passed") is not True:
        fail(CONTRACT, "contract", "validation evidence does not prove conditional writes")
    if any(type(regression.get(key)) is not int for key in ("tests", "suites", "succeeded", "failed")) or regression["tests"] != regression["succeeded"] or regression["tests"] <= 0 or regression["suites"] <= 0 or regression["failed"] != 0:
        fail(CONTRACT, "contract", "full regression evidence is invalid")
    return contract


def edges(value, label):
    raw = value.get("connections")
    values = raw.values() if isinstance(raw, dict) else raw if isinstance(raw, list) else None
    if values is None: fail(CONTRACT, "contract", f"{label} connections missing")
    result = []
    for edge in values:
        if not isinstance(edge, list) or len(edge) != 6: fail(CONTRACT, "contract", f"invalid edge in {label}")
        result.append(tuple(edge))
    return result


def validate_operations(data, contract):
    operations = require_list(require_dict(data, "operations.json").get("Operations"), "Operations")
    cstores = [op for op in operations if isinstance(op, dict) and op.get("name") == "CSTORE"]
    if len(cstores) != 1 or any(isinstance(op, dict) and op.get("name") == "CLOAD" for op in operations):
        fail(CONTRACT, "contract", "CSTORE count or CLOAD contract violated")
    cstore = cstores[0]
    if cstore.get("numOperands") != 3 or cstore.get("numRes") != 0 or type(cstore.get("OPC")) is not int or type(cstore.get("latency")) is not int or cstore["OPC"] != contract["opcode"] or cstore["latency"] != contract["latency"] or not any(isinstance(op, dict) and op.get("name") == "STORE" for op in operations):
        fail(CONTRACT, "contract", "operations CSTORE/STORE contract violated")


def validate_iob(module, top_edges):
    attrs = module.get("attributes", module)
    if not isinstance(attrs, dict): fail(CONTRACT, "contract", "IOB attributes missing")
    if attrs.get("iob_mode") != 2 or attrs.get("num_operands") != 3 or attrs.get("num_input") != 6 or set(attrs.get("operations", [])) != {"INPUT", "OUTPUT", "LOAD", "STORE", "CSTORE"}:
        fail(CONTRACT, "contract", "conditional IOB capabilities invalid")
    config = require_dict(attrs.get("configuration"), "IOB configuration")
    ids = require_dict(attrs.get("io_controller_cfg_id"), "IOB io_controller_cfg_id")
    fields = {}
    for name in ("IsStore", "UseAddr", "UseEn"):
        identifier = ids.get(name)
        item = config.get(str(identifier), config.get(identifier))
        if not isinstance(identifier, int) or not isinstance(item, list) or len(item) != 3 or item[0] != name or not all(isinstance(v, int) for v in item[1:]):
            fail(CONTRACT, "contract", f"missing IOB configuration {name}")
        fields[name] = item
    aggregate = next((value for value in config.values() if isinstance(value, list) and len(value) == 3 and value[0] == "This"), None)
    if aggregate is None or aggregate[1] < aggregate[2]: fail(CONTRACT, "contract", "invalid aggregate configuration")
    intervals = []
    for value in config.values():
        if not isinstance(value, list) or len(value) != 3 or not isinstance(value[1], int) or not isinstance(value[2], int) or value[1] < value[2] or value[2] < aggregate[2] or value[1] > aggregate[1]:
            fail(CONTRACT, "contract", "configuration tuple out of range")
        if value[0] != "This": intervals.append((value[2], value[1]))
    for index, (low, high) in enumerate(intervals):
        if any(not (high < other_low or low > other_high)
               for other_index, (other_low, other_high) in enumerate(intervals)
               if other_index != index):
            fail(CONTRACT, "contract", "overlapping IOB configuration ranges")
    if ids["UseEn"] != 19 or fields["UseEn"] != ["UseEn", 127, 127] or any(item[1] != item[2] for item in (fields["IsStore"], fields["UseAddr"])):
        fail(CONTRACT, "contract", "IOB control bit layout invalid")
    instances = require_list(attrs.get("instances"), "IOB instances")
    instance_types = {item.get("id"): item.get("type") for item in instances if isinstance(item, dict)}
    local = edges(attrs, "IOB")
    for operand in range(3):
        source_ports = {2 * operand, 2 * operand + 1}
        muxes = {dst_id for src_id, src_type, _, dst_id, dst_type, _ in local
                 if instance_types.get(src_id) == "This" and src_type == "This" and dst_type == "Muxn"}
        muxes = {mux for mux in muxes if
                 {src_port for src_id, src_type, src_port, dst_id, dst_type, _ in local
                  if instance_types.get(src_id) == "This" and src_type == "This" and dst_id == mux and dst_type == "Muxn"} == source_ports and
                 {dst_port for src_id, src_type, _, dst_id, dst_type, dst_port in local
                  if instance_types.get(src_id) == "This" and src_type == "This" and dst_id == mux and dst_type == "Muxn"} == {0, 1}}
        if not any(instance_types.get(mux) == "Muxn" and any(
                instance_types.get(delay_pipe) == "DelayPipe" and any(
                    sid == delay_pipe and st == "DelayPipe" and sp == operand and
                    instance_types.get(did) == "IOController" and dtp == "IOController" and dp == operand
                    for sid, st, sp, did, dtp, dp in local)
                for delay_pipe in {did for sid, st, _, did, dtp, dp in local
                                   if sid == mux and st == "Muxn" and
                                   dtp == "DelayPipe" and dp == operand})
                   for mux in muxes):
            fail(CONTRACT, "contract", f"IOB operand {operand} path missing")
    module_id = module.get("id")
    # Every top-level instance of this IOB module must receive all six ports.
    # `top_edges` is checked by caller with its matching top-level instances.
    return module_id


def validate_adg(data):
    adg = require_dict(data, "adg.json")
    modules = [module for module in require_list(adg.get("sub_modules"), "sub_modules") if isinstance(module, dict) and module.get("type") == "IOB"]
    if not modules: fail(CONTRACT, "contract", "ADG has no IOB module")
    top_edges = edges(adg, "ADG")
    instances = require_list(adg.get("instances"), "ADG instances")
    for module in modules:
        module_id = validate_iob(module, top_edges)
        matching = [item for item in instances if isinstance(item, dict) and item.get("module_id") == module_id]
        if not matching or any(item.get("type") != "IOB" for item in matching):
            fail(CONTRACT, "contract", "IOB module is not instantiated as IOB")
        for item in matching:
            ports = {dst_port for _, _, _, dst_id, dst_type, dst_port in top_edges if dst_id == item.get("id") and dst_type == "IOB"}
            if ports != {0, 1, 2, 3, 4, 5}: fail(CONTRACT, "contract", "IOB top-level inputs are incomplete")


def default_package():
    incoming = Path(__file__).resolve().parents[2] / ".cstore-handoff/incoming"
    if not incoming.is_dir(): fail(MISSING, "missing", "incoming directory missing")
    candidates = [path for path in incoming.iterdir() if path.is_dir() and not path.is_symlink()]
    expected = [path for path in candidates if path.name == "vitra-cstore-hardware"]
    if len(candidates) != 1 or len(expected) != 1: fail(MISSING, "layout", "incoming must contain exactly vitra-cstore-hardware")
    return expected[0]


def validate(package):
    listed = verify_layout_and_hashes(package)
    manifest = validate_manifest(package, listed)
    contract = require_contract(manifest)
    operations = json_file(package / "artifacts/operations.json", "operations.json")
    adg = json_file(package / "artifacts/adg.json", "adg.json")
    try:
        validate_operations(operations, contract)
        validate_adg(adg)
    except Invalid as exc:
        if exc.code == SCHEMA:
            fail(CONTRACT, "contract", "malformed operations or ADG contract data")
        raise
    except (AttributeError, KeyError, TypeError, ValueError):
        fail(CONTRACT, "contract", "malformed operations or ADG contract data")
    return manifest


def main(argv=None):
    parser = HandoffArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path)
    try:
        args = parser.parse_args(argv)
        package = args.package if args.package is not None else default_package()
        manifest = validate(package)
    except Invalid as exc:
        print(f"HANDOFF PACKAGE INVALID category={exc.category} detail={exc.detail}", file=sys.stderr)
        return exc.code
    print("HANDOFF PACKAGE VALID "
          f"vitra_commit={manifest['source']['commit']} "
          f"manifest_sha256={sha256(package / 'manifest.json')} "
          f"operations_sha256={sha256(package / 'artifacts/operations.json')} "
          f"adg_sha256={sha256(package / 'artifacts/adg.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
