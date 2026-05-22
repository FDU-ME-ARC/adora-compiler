#!/usr/bin/env python3
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Fudan University.

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import re
import time
from datetime import datetime
from pathlib import Path

from trajectory_emitter import emit_trajectory

PIPELINE_LOG_NAME = "pipeline.log"


def require_tool(tool: str) -> str:
    path = shutil.which(tool)
    if not path:
        raise FileNotFoundError(
            f"Required tool '{tool}' not found in PATH. "
            "Please ensure it is available before running."
        )
    return path


def require_tool_prefer_local(tool: str) -> str:
    script_dir = Path(__file__).resolve().parent
    local_tool = script_dir / tool
    if local_tool.is_file() and os.access(local_tool, os.X_OK):
        return str(local_tool)
    return require_tool(tool)


def run_command(
    args: list[str],
    cwd: Path | None = None,
    log_dir: Path | None = None,
) -> float:
    print("+", " ".join(args))
    start = time.perf_counter()
    if log_dir is None:
        subprocess.run(args, cwd=cwd, check=True)
        return (time.perf_counter() - start) * 1000.0

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / PIPELINE_LOG_NAME
    with open(log_path, "a", encoding="utf-8") as logf:
        logf.write("\n" + "=" * 72 + "\n")
        logf.write(datetime.now().isoformat(timespec="seconds") + "\n")
        logf.write(f"cwd: {cwd!s}\n")
        logf.write("+ " + " ".join(args) + "\n")
        logf.write("-" * 72 + "\n")
        logf.flush()
        subprocess.run(
            args,
            cwd=cwd,
            check=True,
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return (time.perf_counter() - start) * 1000.0


def prepare_ir_dirs(root: Path) -> dict[str, Path]:
    ir_dir = root / "adora-cc-ir"
    if ir_dir.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = root / f"adora-cc-ir-backup-{timestamp}"
        ir_dir.rename(backup_dir)

    kernels_dir = ir_dir / "0_kernels"
    kernels_opt_dir = ir_dir / "1_kernels_opt"
    dfgs_dir = ir_dir / "2_dfgs"
    tempfiles_dir = ir_dir / "tempfiles"
    temp_dfg_dir = tempfiles_dir / "DFGs"

    for directory in (kernels_dir, kernels_opt_dir, dfgs_dir, temp_dfg_dir):
        directory.mkdir(parents=True, exist_ok=True)

    return {
        "ir": ir_dir,
        "kernels": kernels_dir,
        "kernels_opt": kernels_opt_dir,
        "dfgs": dfgs_dir,
        "tempfiles": tempfiles_dir,
        "temp_dfg": temp_dfg_dir,
    }

def strip_module_attrs(text: str) -> str:
    """
    Remove the content of:
        module attributes { ... } 
    in MLIR, leaving only empty {}.
    Supports multi-line and nested braces.
    """
    TOKEN = "module attributes "

    out = []
    i = 0
    n = len(text)

    while True:
        pos = text.find(TOKEN, i)
        if pos == -1:
            out.append(text[i:])
            break

        out.append(text[i:pos])
        out.append(TOKEN)

        brace_start = pos + len(TOKEN)

        if brace_start >= n or text[brace_start] != '{':
            # Format does not match, skip
            i = brace_start
            continue

        # Write empty {}
        out.append("{}")

        # Skip the original {...}
        j = brace_start + 1
        brace_depth = 1
        while j < n and brace_depth > 0:
            if text[j] == '{':
                brace_depth += 1
            elif text[j] == '}':
                brace_depth -= 1
            j += 1

        i = j  # Continue scanning after the closing brace

    return "".join(out)


def has_adora_kernel(text: str) -> bool:
    """Return True if the MLIR text already contains ADORA.kernel (skip extract pass)."""
    return "ADORA.kernel" in text


def build_pipeline(
    input_path: Path,
    tools: dict[str, str],
    dirs: dict[str, Path],
    enable_unroll: bool,
    adg_path: Path | None,
    output_path: Path | None,   # NEW
) -> dict[str, object]:
    base_name = input_path.stem
    mlir_input = input_path

    log_dir = dirs["tempfiles"]
    stage_metrics_ms: dict[str, float] = {
        "cgeist_ms": 0.0,
        "normalize_ms": 0.0,
        "kernel_extract_ms": 0.0,
        "kernel_opt_ms": 0.0,
        "dfg_gen_ms": 0.0,
    }

    if input_path.suffix.upper() == ".C":
        cgeist_output = dirs["ir"] / f"{base_name}.mlir"
        stage_metrics_ms["cgeist_ms"] = run_command(
            [
                tools["cgeist"],
                "-O2",
                "--raise-scf-to-affine",  # Bug1 fix: lift scf.for→affine.for, elim index_cast
                str(input_path),
                "-S",
                "-o",
                str(cgeist_output),
            ],
            log_dir=log_dir,
        )
        mlir_input = cgeist_output

    # Read
    with open(mlir_input, "r", encoding="utf-8") as f:
        text = f.read()

    # Clean
    cleaned = strip_module_attrs(text)

    # Overwrite in place
    with open(mlir_input, "w", encoding="utf-8") as f:
        f.write(cleaned)

    normalized = dirs["temp_dfg"] / f"{base_name}_normalized.mlir"
    stage_metrics_ms["normalize_ms"] = run_command(
        [
            tools["cgra-opt"],
            "--allow-unregistered-dialect",
            "--affine-loop-normalize",
            "--affine-simplify-structures",
            "--normalize-memrefs",
            # "--force-specialization",
            # "--bufferization-bufferize",
            str(mlir_input),
            "-o",
            str(normalized),
        ],
        log_dir=log_dir,
    )

    kernel_mlir = dirs["kernels"] / f"{base_name}_kernel.mlir"
    with open(normalized, "r", encoding="utf-8") as f:
        normalized_text = f.read()
    skip_extract = has_adora_kernel(normalized_text)
    kernel_passes = [
        "--canonicalize",
        "-reconcile-unrealized-casts",
        "--affine-loop-fusion",
    ]
    if not skip_extract:
        kernel_passes.append("--adora-extract-affine-for-to-kernel")
    kernel_passes.extend(
        [
            "--arith-expand",
            "--memref-expand",
            "-cse",
            str(normalized),
            "-o",
            str(kernel_mlir),
        ]
    )
    stage_metrics_ms["kernel_extract_ms"] = run_command([tools["cgra-opt"]] + kernel_passes, log_dir=log_dir)

    kernel_opt = dirs["kernels_opt"] / f"{base_name}_opt.mlir"
    kernel_opt_cmd = [
        tools["cgra-opt"],
        "--adora-simplify-affine-loop-levels",
        "--canonicalize",
        "-cse",
        "--adora-simplify-loadstore",
        "--adora-math-rewrite",
        (
            '--adora-adjust-kernel-mem-footprint='
            'cachesize=128 singlearraysize=8 disable-remainder-block explicit-datablock'
        ),
    ]
    if enable_unroll:
        if not adg_path:
            raise ValueError("Unroll enabled but no ADG path provided.")
        kernel_opt_cmd.append(f"--adora-auto-unroll=cgra-adg={adg_path}")
    kernel_opt_cmd.extend([str(kernel_mlir), "-o", str(kernel_opt)])
    # Run with cwd=dirs["tempfiles"] when unroll is enabled so AutoUnroll creates
    # DesignSpace under adora-cc-ir/tempfiles/DesignSpace
    stage_metrics_ms["kernel_opt_ms"] = run_command(
        kernel_opt_cmd,
        cwd=dirs["tempfiles"] if enable_unroll else None,
        log_dir=log_dir,
    )

    # NEW: export kernel_opt result
    with open(kernel_opt, "r", encoding="utf-8") as f:
        kernel_opt_text = f.read()

    if output_path is None:
        # no -o: print to stdout
        sys.stdout.write(kernel_opt_text)
        if not kernel_opt_text.endswith("\n"):
            sys.stdout.write("\n")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(kernel_opt_text)

    stage_metrics_ms["dfg_gen_ms"] = run_command(
        [
            tools["cgra-opt"],
            "--adora-kernel-dfg-gen",
            str(kernel_opt),
        ],
        cwd=dirs["temp_dfg"],
        log_dir=log_dir,
    )

    copied_dfgs: list[str] = []
    for dot_file in dirs["temp_dfg"].glob("*_CDFG.dot"):
        shutil.copy(dot_file, dirs["dfgs"] / dot_file.name)
        copied_dfgs.append(str((dirs["dfgs"] / dot_file.name).resolve()))

    total_ms = sum(stage_metrics_ms.values())
    return {
        "mlir_input": str(Path(mlir_input).resolve()),
        "normalized": str(normalized.resolve()),
        "kernel_mlir": str(kernel_mlir.resolve()),
        "kernel_opt": str(kernel_opt.resolve()),
        "dfg_files": copied_dfgs,
        "total_ms": total_ms,
        **stage_metrics_ms,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile C/MLIR to CDFG using cgeist/mlir-opt/cgra-opt."
    )
    parser.add_argument("input", type=Path, help="Input .C or .MLIR file")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path.cwd(),
        help="Working directory for IR outputs (default: current directory)",
    )
    parser.add_argument(
        "--enable-unroll",
        action="store_true",
        help="Enable auto unroll during kernel optimization.",
    )
    parser.add_argument(
        "--adg-path",
        type=Path,
        help="Path to the CGRA .adg file used for unroll.",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write the optimized kernel MLIR (after kernel optimization) to this file. "
             "If not provided, print to stdout.",
    )
    parser.add_argument(
        "--emit-trajectory",
        action="store_true",
        help="Emit one schema-shaped trajectory episode for this compile invocation.",
    )
    parser.add_argument(
        "--trajectory-root",
        type=Path,
        default=None,
        help="Root directory where data/episodes, data/observations, and data/metrics are written. Defaults to --work-dir.",
    )
    parser.add_argument(
        "--trajectory-policy-id",
        type=str,
        default="compiler_logged",
        help="Policy identifier recorded in the emitted episode summary.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()

    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    suffix = input_path.suffix.upper()
    if suffix not in {".C", ".MLIR"}:
        print("Only .C or .MLIR inputs are supported.", file=sys.stderr)
        return 1
    
    output_path: Path | None = args.output.resolve() if args.output else None

    tools: dict[str, str] = {
        # "mlir-opt": require_tool("mlir-opt"),
        "cgra-opt": require_tool_prefer_local("cgra-opt"),
    }
    if suffix == ".C":
        tools["cgeist"] = require_tool("cgeist")
    else:
        tools["cgeist"] = ""

    if args.enable_unroll and not args.adg_path:
        print("Unroll enabled but --adg-path was not provided.", file=sys.stderr)
        return 1

    adg_path = args.adg_path.resolve() if args.adg_path else None
    if adg_path and not adg_path.exists():
        print(f"ADG file not found: {adg_path}", file=sys.stderr)
        return 1

    dirs = prepare_ir_dirs(args.work_dir.resolve())

    try:
        pipeline_metadata = build_pipeline(input_path, tools, dirs, args.enable_unroll, adg_path, output_path)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"Command failed with exit code {exc.returncode}", file=sys.stderr)
        print(
            f"See subprocess log: {dirs['tempfiles'] / PIPELINE_LOG_NAME}",
            file=sys.stderr,
        )
        return exc.returncode

    if args.emit_trajectory:
        trajectory_root = args.trajectory_root.resolve() if args.trajectory_root else args.work_dir.resolve()
        summary_path = emit_trajectory(
            trajectory_root=trajectory_root,
            input_path=input_path,
            dirs=dirs,
            adg_path=adg_path,
            pipeline_metadata=pipeline_metadata,
            policy_id=args.trajectory_policy_id,
        )
        print(f"Trajectory summary: {summary_path}", file=sys.stderr)

    print(f"Final optimal mlir file: {dirs['kernels_opt']}", file=sys.stderr)
    print(f"CDFG output directory: {dirs['dfgs']}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
