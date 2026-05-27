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
from datetime import datetime
from pathlib import Path

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
) -> None:
    print("+", " ".join(args))
    if log_dir is None:
        subprocess.run(args, cwd=cwd, check=True)
        return

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


def prepare_ir_dirs(root: Path) -> dict[str, Path]:
    ir_dir = root / "adora-cc-ir"
    if ir_dir.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = root / f"adora-cc-ir-backup-{timestamp}"
        ir_dir.rename(backup_dir)

    frontend_dir    = ir_dir / "1_frontend"
    normalize_dir   = ir_dir / "2_normalize"
    kernels_dir     = ir_dir / "3_kernel-extract"
    kernels_opt_dir = ir_dir / "4_kernel-opt"
    schedule_dir    = ir_dir / "5_task-schedule"
    dfgs_dir        = ir_dir / "6_dfg"

    for directory in (frontend_dir, normalize_dir, kernels_dir,
                      kernels_opt_dir, schedule_dir, dfgs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    return {
        "ir": ir_dir,
        "frontend": frontend_dir,
        "normalize": normalize_dir,
        "kernels": kernels_dir,
        "kernels_opt": kernels_opt_dir,
        "schedule": schedule_dir,
        "dfgs": dfgs_dir,
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


def _append_to_log(log_dir: Path, label: str, content: str) -> None:
    """Append a labelled block to pipeline.log."""
    log_file = log_dir / PIPELINE_LOG_NAME
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n=== {label} ===\n{content}\n")


def build_pipeline(
    input_path: Path,
    tools: dict[str, str],
    dirs: dict[str, Path],
    enable_unroll: bool,
    adg_path: Path | None,
    output_path: Path | None,
    schedule_tasks: bool = True,   # NEW: run --adora-schedule-tasks after kernel_opt
) -> None:
    base_name = input_path.stem
    mlir_input = input_path

    log_dir = dirs["ir"]

    if input_path.suffix.upper() == ".C":
        cgeist_output = dirs["frontend"] / f"{base_name}.mlir"
        run_command(
            [
                tools["cgeist"],
                "-O2",
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

    normalized = dirs["normalize"] / f"{base_name}_normalized.mlir"
    run_command(
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
    run_command([tools["cgra-opt"]] + kernel_passes, log_dir=log_dir)

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
    # Run with cwd=dirs["ir"] when unroll is enabled so AutoUnroll creates
    # DesignSpace under adora-cc-ir/DesignSpace
    run_command(
        kernel_opt_cmd,
        cwd=dirs["ir"] if enable_unroll else None,
        log_dir=log_dir,
    )

    # --- adora-schedule-tasks (default enabled) ---
    kernel_sched = kernel_opt  # fallback if scheduling skipped or fails
    if schedule_tasks:
        sched_pre  = dirs["schedule"] / f"{base_name}.pre.mlir"
        sched_post = dirs["schedule"] / f"{base_name}.post.mlir"
        shutil.copy(kernel_opt, sched_pre)
        sched_cmd = [
            tools["cgra-opt"],
            "--adora-schedule-tasks",
            str(sched_pre),
            "-o",
            str(sched_post),
        ]
        print("+", " ".join(sched_cmd))
        result = subprocess.run(sched_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            sched_failed = dirs["schedule"] / f"{base_name}.post.failed.mlir"
            sched_post.rename(sched_failed) if sched_post.exists() else None
            _append_to_log(log_dir, "schedule-tasks FAILED", result.stderr)
            print(
                f"[adoracc] Warning: adora-schedule-tasks failed on "
                f"{kernel_opt.name}; proceeding without task scheduling.\n"
                + result.stderr,
                file=sys.stderr,
            )
        else:
            _append_to_log(log_dir, "schedule-tasks OK", result.stdout)
            kernel_sched = sched_post

    # --- export final kernel IR ---
    with open(kernel_sched, "r", encoding="utf-8") as f:
        kernel_final_text = f.read()

    if output_path is None:
        sys.stdout.write(kernel_final_text)
        if not kernel_final_text.endswith("\n"):
            sys.stdout.write("\n")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(kernel_final_text)

    # DFG-gen looks for lib/DFG/Documents/GeneralOpName.txt relative to CWD.
    # The file lives in the adora-compiler source root, which is two levels
    # above the build/bin/ directory containing cgra-opt.
    adora_compiler_root = Path(tools["cgra-opt"]).parent.parent.parent
    run_command(
        [
            tools["cgra-opt"],
            "--adora-kernel-dfg-gen",
            str(kernel_sched),
        ],
        cwd=adora_compiler_root,
        log_dir=log_dir,
    )

    for dot_file in dirs["normalize"].glob("*_CDFG.dot"):
        shutil.copy(dot_file, dirs["dfgs"] / dot_file.name)


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
        "--disable-schedule-tasks",
        dest="schedule_tasks",
        action="store_false",
        help="Disable the adora-schedule-tasks pass (default: enabled).",
    )
    parser.set_defaults(schedule_tasks=True)
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
        build_pipeline(input_path, tools, dirs, args.enable_unroll, adg_path, output_path,
                       args.schedule_tasks)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"Command failed with exit code {exc.returncode}", file=sys.stderr)
        print(
            f"See subprocess log: {dirs['ir'] / PIPELINE_LOG_NAME}",
            file=sys.stderr,
        )
        return exc.returncode

    print(f"[adoracc] Pipeline complete. Output directories:", file=sys.stderr)
    print(f"  1_frontend    : {dirs['frontend']}", file=sys.stderr)
    print(f"  2_normalize   : {dirs['normalize']}", file=sys.stderr)
    print(f"  3_kernel-extract: {dirs['kernels']}", file=sys.stderr)
    print(f"  4_kernel-opt  : {dirs['kernels_opt']}", file=sys.stderr)
    print(f"  5_task-schedule: {dirs['schedule']}", file=sys.stderr)
    print(f"  6_dfg         : {dirs['dfgs']}", file=sys.stderr)
    print(f"  pipeline.log  : {dirs['ir'] / PIPELINE_LOG_NAME}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
