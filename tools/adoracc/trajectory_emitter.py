from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def emit_trajectory(
    *,
    trajectory_root: Path,
    input_path: Path,
    dirs: dict[str, Path],
    adg_path: Path | None,
    pipeline_metadata: dict[str, Any],
    policy_id: str = "compiler_logged",
) -> Path:
    """Emit one coarse-grained replayable episode for a real ADORA run."""
    data_root = trajectory_root / "data"
    episodes_dir = data_root / "episodes"
    observations_dir = data_root / "observations"
    metrics_dir = data_root / "metrics"
    for directory in (episodes_dir, observations_dir, metrics_dir):
        directory.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = input_path.stem.replace("-", "_")
    episode_id = f"{base_name}_{timestamp}"
    session_id = f"session_{base_name}_{timestamp}"

    kernel_opt = Path(pipeline_metadata["kernel_opt"])
    normalized = Path(pipeline_metadata["normalized"])
    kernel_mlir = Path(pipeline_metadata["kernel_mlir"])
    dfg_files = [Path(path) for path in pipeline_metadata.get("dfg_files", [])]
    adg_summary = _load_adg_summary(adg_path)
    dfg_summary = _summarize_dfgs(dfg_files)
    resource_summary = _build_resource_summary(adg_summary, dfg_summary)
    candidates = _build_region_candidates(adg_summary, dfg_summary, resource_summary)
    chosen_region = min(
        candidates,
        key=lambda item: (
            float(item["payload"]["route_pressure"]),
            float(item["payload"]["utilization"]),
            int(item["payload"]["region_id"]),
        ),
    )

    frontend_candidates_ref = _write_json(
        observations_dir / f"{episode_id}_frontend_candidates.json",
        [
            {"action_type": "no_op", "payload": {}},
            {"action_type": "try_fusion_hint", "payload": {"hint_id": "compiler_safe_skip", "region_id": 0}},
        ],
        trajectory_root,
    )
    backend_candidates_ref = _write_json(
        observations_dir / f"{episode_id}_backend_candidates.json",
        candidates,
        trajectory_root,
    )
    terminal_candidates_ref = _write_json(
        observations_dir / f"{episode_id}_terminal_candidates.json",
        [],
        trajectory_root,
    )

    frontend_notes_ref = _write_json(
        observations_dir / f"{episode_id}_frontend_notes.json",
        {"notes": "compiler emitted no safe frontend hint for this coarse-grained episode"},
        trajectory_root,
    )
    backend_notes_ref = _write_json(
        observations_dir / f"{episode_id}_backend_notes.json",
        {"notes": "prefer lower route pressure and utilization among bounded legal regions"},
        trajectory_root,
    )
    terminal_notes_ref = _write_json(
        observations_dir / f"{episode_id}_terminal_notes.json",
        {"notes": "terminal compile outcome"},
        trajectory_root,
    )

    region_summary_ref = _write_json(
        observations_dir / f"{episode_id}_region_summary.json",
        {
            "kernel_mlir": str(kernel_mlir),
            "kernel_opt_mlir": str(kernel_opt),
            "locality": "mixed",
            "kernel_present": _contains_token(kernel_mlir, "ADORA.kernel"),
        },
        trajectory_root,
    )
    routing_summary_ref = _write_json(
        observations_dir / f"{episode_id}_routing_summary.json",
        {
            "dfg_node_count": dfg_summary["node_count"],
            "dfg_edge_count": dfg_summary["edge_count"],
            "candidate_count": len(candidates),
            "adg_rows": adg_summary["num_row"],
            "adg_cols": adg_summary["num_colum"],
        },
        trajectory_root,
    )
    spatial_resource_summary_ref = _write_json(
        observations_dir / f"{episode_id}_spatial_resource_summary.json",
        resource_summary,
        trajectory_root,
    )
    candidate_regions_ref = _write_json(
        observations_dir / f"{episode_id}_candidate_regions.json",
        [
            {
                "region_id": c["payload"]["region_id"],
                "occupancy": c["payload"]["utilization"],
                "route_pressure": c["payload"]["route_pressure"],
                "memory_pressure": c["payload"]["memory_pressure"],
                "compute_pressure": c["payload"]["compute_pressure"],
                "overlap_score": c["payload"]["overlap_score"],
            }
            for c in candidates
        ],
        trajectory_root,
    )

    frontend_observation_ref = _write_json(
        observations_dir / f"{episode_id}_frontend_observation.json",
        {
            "session_id": session_id,
            "episode_id": episode_id,
            "step_id": 1,
            "phase": "frontend_hint",
            "graph": {
                "graph_hash": _hash_path(normalized),
                "graph_kind": "ir_region",
                "node_count": dfg_summary["node_count"],
                "edge_count": dfg_summary["edge_count"],
            },
            "summaries": {
                "region_summary_ref": region_summary_ref,
                "spatial_resource_summary_ref": spatial_resource_summary_ref,
            },
            "legal_actions": {
                "candidate_list_ref": frontend_candidates_ref,
                "action_mask_ref": None,
            },
            "compiler_signals": {
                "compile_cost_proxy": float(pipeline_metadata.get("normalize_ms", 0.0)),
                "route_pressure_proxy": dfg_summary["route_pressure_proxy"],
                "utilization_proxy": dfg_summary["utilization_proxy"],
                "backtrack_count": 0,
            },
            "diagnostics": {"notes_ref": frontend_notes_ref},
        },
        trajectory_root,
    )

    backend_observation_ref = _write_json(
        observations_dir / f"{episode_id}_backend_observation.json",
        {
            "session_id": session_id,
            "episode_id": episode_id,
            "step_id": 2,
            "phase": "backend_mapping",
            "graph": {
                "graph_hash": _hash_paths(dfg_files),
                "graph_kind": "dfg",
                "node_count": dfg_summary["node_count"],
                "edge_count": dfg_summary["edge_count"],
            },
            "summaries": {
                "routing_summary_ref": routing_summary_ref,
                "spatial_resource_summary_ref": spatial_resource_summary_ref,
            },
            "partial_state": {"candidate_regions_ref": candidate_regions_ref},
            "legal_actions": {
                "candidate_list_ref": backend_candidates_ref,
                "action_mask_ref": None,
            },
            "compiler_signals": {
                "compile_cost_proxy": float(pipeline_metadata.get("kernel_opt_ms", 0.0)),
                "route_pressure_proxy": dfg_summary["route_pressure_proxy"],
                "utilization_proxy": dfg_summary["utilization_proxy"],
                "backtrack_count": 0,
            },
            "diagnostics": {"notes_ref": backend_notes_ref},
        },
        trajectory_root,
    )

    terminal_observation_ref = _write_json(
        observations_dir / f"{episode_id}_terminal_observation.json",
        {
            "session_id": session_id,
            "episode_id": episode_id,
            "step_id": 3,
            "phase": "terminal",
            "graph": {
                "graph_hash": _hash_path(kernel_opt),
                "graph_kind": "dfg",
                "node_count": dfg_summary["node_count"],
                "edge_count": dfg_summary["edge_count"],
            },
            "legal_actions": {
                "candidate_list_ref": terminal_candidates_ref,
                "action_mask_ref": None,
            },
            "compiler_signals": {
                "compile_cost_proxy": float(pipeline_metadata.get("total_ms", 0.0)),
                "route_pressure_proxy": dfg_summary["route_pressure_proxy"],
                "utilization_proxy": dfg_summary["utilization_proxy"],
                "backtrack_count": 0,
            },
            "diagnostics": {"notes_ref": terminal_notes_ref},
        },
        trajectory_root,
    )

    frontend_action_ref = _write_json(
        observations_dir / f"{episode_id}_frontend_action.json",
        {},
        trajectory_root,
    )
    backend_action_ref = _write_json(
        observations_dir / f"{episode_id}_backend_action.json",
        chosen_region["payload"],
        trajectory_root,
    )

    diagnostics_step1_ref = _write_json(metrics_dir / f"{episode_id}_diag_1.json", {"status": "ok"}, trajectory_root)
    diagnostics_step2_ref = _write_json(metrics_dir / f"{episode_id}_diag_2.json", {"status": "ok"}, trajectory_root)
    diagnostics_step3_ref = _write_json(metrics_dir / f"{episode_id}_diag_3.json", {"status": "completed"}, trajectory_root)

    terminal_metrics_ref = _write_json(
        metrics_dir / f"{episode_id}_terminal_metrics.json",
        {
            "compile_time_ms": float(pipeline_metadata.get("total_ms", 0.0)),
            "mapping_success": bool(dfg_files),
            "utilization": dfg_summary["utilization_proxy"],
            "route_pressure": dfg_summary["route_pressure_proxy"],
            "memory_pressure": resource_summary["memory"]["memory_bank_pressure"],
            "compute_pressure": resource_summary["compute"]["pe_utilization"],
            "overlap_score": resource_summary["overlap"]["communication_compute_overlap_opportunity"],
            "hint_usefulness": 0.0,
            "backtrack_count": 0,
            "dfg_node_count": dfg_summary["node_count"],
            "dfg_edge_count": dfg_summary["edge_count"],
        },
        trajectory_root,
    )

    event_log_path = episodes_dir / f"{episode_id}.events.jsonl"
    event_rows = [
        {
            "schema_version": "v0",
            "session_id": session_id,
            "episode_id": episode_id,
            "step_id": 1,
            "phase": "frontend_hint",
            "observation": {
                "observation_hash": _hash_ref(trajectory_root / frontend_observation_ref),
                "observation_ref": frontend_observation_ref,
                "legal_action_mask_hash": "",
            },
            "action": {
                "action_type": "no_op",
                "action_payload_ref": frontend_action_ref,
                "action_idempotency_key": f"{episode_id}-front-1",
            },
            "validation": {"status": "ok", "rejection_reason": "", "repaired_action_ref": ""},
            "transition": {
                "accepted": True,
                "state_delta_ref": "",
                "next_observation_hash": _hash_ref(trajectory_root / backend_observation_ref),
            },
            "metrics": {
                "compile_time_ms": float(pipeline_metadata.get("normalize_ms", 0.0) + pipeline_metadata.get("kernel_extract_ms", 0.0)),
                "reward": 0.0,
                "mapping_success": True,
                "utilization": dfg_summary["utilization_proxy"],
                "route_pressure": dfg_summary["route_pressure_proxy"],
                "memory_pressure": resource_summary["memory"]["memory_bank_pressure"],
                "compute_pressure": resource_summary["compute"]["pe_utilization"],
                "overlap_score": resource_summary["overlap"]["communication_compute_overlap_opportunity"],
                "hint_usefulness": 0.0,
                "backtrack_count": 0,
            },
            "diagnostics_ref": diagnostics_step1_ref,
            "wall_clock_ms": float(pipeline_metadata.get("normalize_ms", 0.0)),
        },
        {
            "schema_version": "v0",
            "session_id": session_id,
            "episode_id": episode_id,
            "step_id": 2,
            "phase": "backend_mapping",
            "observation": {
                "observation_hash": _hash_ref(trajectory_root / backend_observation_ref),
                "observation_ref": backend_observation_ref,
                "legal_action_mask_hash": "",
            },
            "action": {
                "action_type": "choose_region",
                "action_payload_ref": backend_action_ref,
                "action_idempotency_key": f"{episode_id}-back-1",
            },
            "validation": {"status": "ok", "rejection_reason": "", "repaired_action_ref": ""},
            "transition": {
                "accepted": True,
                "state_delta_ref": "",
                "next_observation_hash": _hash_ref(trajectory_root / terminal_observation_ref),
            },
            "metrics": {
                "compile_time_ms": float(pipeline_metadata.get("kernel_opt_ms", 0.0) + pipeline_metadata.get("dfg_gen_ms", 0.0)),
                "reward": 0.0,
                "mapping_success": bool(dfg_files),
                "utilization": chosen_region["payload"]["utilization"],
                "route_pressure": chosen_region["payload"]["route_pressure"],
                "memory_pressure": chosen_region["payload"]["memory_pressure"],
                "compute_pressure": chosen_region["payload"]["compute_pressure"],
                "overlap_score": chosen_region["payload"]["overlap_score"],
                "hint_usefulness": 0.0,
                "backtrack_count": 0,
            },
            "diagnostics_ref": diagnostics_step2_ref,
            "wall_clock_ms": float(pipeline_metadata.get("kernel_opt_ms", 0.0)),
        },
        {
            "schema_version": "v0",
            "session_id": session_id,
            "episode_id": episode_id,
            "step_id": 3,
            "phase": "terminal",
            "observation": {
                "observation_hash": _hash_ref(trajectory_root / terminal_observation_ref),
                "observation_ref": terminal_observation_ref,
                "legal_action_mask_hash": "",
            },
            "action": {
                "action_type": "",
                "action_payload_ref": "",
                "action_idempotency_key": f"{episode_id}-term-1",
            },
            "validation": {"status": "ok", "rejection_reason": "", "repaired_action_ref": ""},
            "transition": {"accepted": True, "state_delta_ref": "", "next_observation_hash": ""},
            "metrics": {
                "compile_time_ms": float(pipeline_metadata.get("total_ms", 0.0)),
                "reward": 1.0 if dfg_files else 0.0,
                "mapping_success": bool(dfg_files),
                "utilization": chosen_region["payload"]["utilization"],
                "route_pressure": chosen_region["payload"]["route_pressure"],
                "memory_pressure": chosen_region["payload"]["memory_pressure"],
                "compute_pressure": chosen_region["payload"]["compute_pressure"],
                "overlap_score": chosen_region["payload"]["overlap_score"],
                "hint_usefulness": 0.0,
                "backtrack_count": 0,
            },
            "diagnostics_ref": diagnostics_step3_ref,
            "wall_clock_ms": float(pipeline_metadata.get("total_ms", 0.0)),
        },
    ]

    with event_log_path.open("w", encoding="utf-8") as handle:
        for row in event_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    episode_summary = {
        "schema_version": "v0",
        "session_id": session_id,
        "episode_id": episode_id,
        "benchmark_id": input_path.stem,
        "architecture_id": adg_path.name if adg_path else "unknown_architecture",
        "policy_id": policy_id,
        "policy_family": "heuristic",
        "seed": 0,
        "status": "completed",
        "step_count": 3,
        "terminal_metrics_ref": _relative_to_root(terminal_metrics_ref),
        "event_log_ref": _relative_to_root(event_log_path.relative_to(trajectory_root)),
    }

    summary_path = episodes_dir / f"{episode_id}.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(episode_summary, handle, indent=2, sort_keys=True)
    return summary_path


def _write_json(path: Path, payload: Any, root: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return _relative_to_root(path.relative_to(root))


def _relative_to_root(path: Path | str) -> str:
    return str(path).replace("\\", "/")


def _hash_ref(path: Path) -> str:
    return _hash_bytes(path.read_bytes())


def _hash_path(path: Path) -> str:
    return _hash_bytes(path.read_bytes())


def _hash_paths(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _hash_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _contains_token(path: Path, token: str) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return False
    return token in text


def _load_adg_summary(adg_path: Path | None) -> dict[str, Any]:
    if adg_path is None or not adg_path.exists():
        return {"num_row": 1, "num_colum": 1}
    with adg_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {
        "num_row": int(data.get("num_row", 1)),
        "num_colum": int(data.get("num_colum", 1)),
    }


def _summarize_dfgs(dfg_files: list[Path]) -> dict[str, Any]:
    node_count = 0
    edge_count = 0
    for dot_file in dfg_files:
        text = dot_file.read_text(encoding="utf-8")
        edge_count += text.count("->")
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("digraph") or stripped in {"{", "}"} or "->" in stripped:
                continue
            if "[" in stripped and "]" in stripped:
                node_count += 1
    if node_count == 0:
        node_count = max(len(dfg_files), 1)
    route_pressure = min(0.95, 0.1 + edge_count / max(node_count * 8, 1))
    utilization = min(0.95, 0.1 + node_count / max((node_count + 4), 1))
    return {
        "node_count": node_count,
        "edge_count": edge_count,
        "route_pressure_proxy": round(route_pressure, 3),
        "utilization_proxy": round(utilization, 3),
    }


def _build_resource_summary(adg_summary: dict[str, Any], dfg_summary: dict[str, Any]) -> dict[str, Any]:
    rows = max(int(adg_summary.get("num_row", 1)), 1)
    cols = max(int(adg_summary.get("num_colum", 1)), 1)
    node_count = int(dfg_summary["node_count"])
    edge_count = int(dfg_summary["edge_count"])
    route_pressure = float(dfg_summary["route_pressure_proxy"])
    utilization = float(dfg_summary["utilization_proxy"])
    memory_pressure = round(min(0.95, 0.15 + edge_count * 0.02 + node_count / max(rows * cols * 20, 1)), 3)
    compute_pressure = round(min(0.95, utilization + node_count / max(rows * cols * 25, 1)), 3)
    overlap_score = round(max(0.0, min(0.95, 1.0 - abs(memory_pressure - compute_pressure))), 3)
    communication_pressure = round(min(0.95, 0.1 + route_pressure * 0.75 + edge_count / max(node_count * 16, 1)), 3)
    return {
        "schema_version": "v0",
        "graph": {
            "node_count": node_count,
            "edge_count": edge_count,
            "fanout_proxy": round(edge_count / max(node_count, 1), 3),
            "criticality_proxy": round(min(0.95, (node_count + edge_count) / 64), 3),
        },
        "spatial": {
            "rows": rows,
            "cols": cols,
            "occupancy": utilization,
            "route_pressure": route_pressure,
            "neighbor_pressure": round(min(0.95, (route_pressure + utilization) / 2), 3),
        },
        "communication": {
            "producer_consumer_distance_proxy": round(edge_count / max(node_count + 1, 1), 3),
            "edge_cut_proxy": edge_count,
            "route_reuse_proxy": round(max(0.0, 1.0 - route_pressure), 3),
            "fanout_pressure": communication_pressure,
        },
        "memory": {
            "load_store_count_proxy": edge_count * 2,
            "memory_bank_pressure": memory_pressure,
            "local_buffer_fit": round(max(0.0, 1.0 - memory_pressure), 3),
            "reuse_locality_score": round(max(0.0, 1.0 - route_pressure), 3),
            "data_movement_volume_proxy": edge_count * 4,
        },
        "compute": {
            "op_count_proxy": node_count,
            "compute_intensity": round(node_count / max(edge_count + 1, 1), 3),
            "pe_utilization": compute_pressure,
            "critical_path_proxy": max(1, int(node_count + edge_count / 2)),
            "pipeline_imbalance": round(abs(compute_pressure - memory_pressure), 3),
        },
        "overlap": {
            "independent_task_groups_proxy": max(1, min(rows, 4)),
            "producer_consumer_slack": round(max(0.0, 1.0 - communication_pressure), 3),
            "pipeline_stage_pressure": round(min(0.95, (compute_pressure + memory_pressure) / 2), 3),
            "communication_compute_overlap_opportunity": overlap_score,
        },
    }


def _build_region_candidates(
    adg_summary: dict[str, Any],
    dfg_summary: dict[str, Any],
    resource_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = max(int(adg_summary.get("num_row", 1)), 1)
    cols = max(int(adg_summary.get("num_colum", 1)), 1)
    count = min(max(rows, 1), 4)
    candidates = []
    base_memory = float(resource_summary["memory"]["memory_bank_pressure"])
    base_compute = float(resource_summary["compute"]["pe_utilization"])
    base_overlap = float(resource_summary["overlap"]["communication_compute_overlap_opportunity"])
    edge_count = int(dfg_summary["edge_count"])
    for idx in range(count):
        region_id = idx + 1
        route_pressure = round(0.18 + idx * 0.11, 3)
        utilization = round(min(0.85, 0.35 + idx * 0.12), 3)
        memory_pressure = round(min(0.95, base_memory + idx * 0.04), 3)
        compute_pressure = round(min(0.95, base_compute + idx * 0.02), 3)
        overlap_score = round(max(0.0, base_overlap - idx * 0.03), 3)
        candidates.append(
            {
                "action_type": "choose_region",
                "payload": {
                    "region_id": region_id,
                    "route_pressure": route_pressure,
                    "utilization": utilization,
                    "memory_pressure": memory_pressure,
                    "compute_pressure": compute_pressure,
                    "overlap_score": overlap_score,
                    "distance_to_hotspot": idx,
                    "communication_edges_proxy": edge_count,
                    "legal_reason": "compiler_emitted_region_candidate",
                    "rows": rows,
                    "cols": cols,
                },
            }
        )
    return candidates
