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
from datetime import datetime
from pathlib import Path


def require_tool(tool: str) -> str:
    path = shutil.which(tool)
    if not path:
        raise FileNotFoundError(
            f"Required tool '{tool}' not found in PATH. "
            "Please ensure it is available before running."
        )
    return path


def run_command(args: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(args))
    subprocess.run(args, cwd=cwd, check=True)


def prepare_ir_dirs(root: Path) -> dict[str, Path]:
    ir_dir = root / "adora-cc-ir"
    if ir_dir.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = root / f"adora-cc-ir-backup-{timestamp}"
        ir_dir.rename(backup_dir)

    kernels_dir = ir_dir / "0_kernels"
    kernels_opt_dir = ir_dir / "1_kernels_opt"
    dfgs_dir = ir_dir / "2_dfgs"
    temp_dir = ir_dir / "tempfiles"
    temp_dfg_dir = temp_dir / "DFGs"

    for directory in (kernels_dir, kernels_opt_dir, dfgs_dir, temp_dfg_dir):
        directory.mkdir(parents=True, exist_ok=True)

    return {
        "ir": ir_dir,
        "kernels": kernels_dir,
        "kernels_opt": kernels_opt_dir,
        "dfgs": dfgs_dir,
        "temp_dfg": temp_dfg_dir,
    }


def build_pipeline(
    input_path: Path,
    tools: dict[str, str],
    dirs: dict[str, Path],
    enable_unroll: bool,
    adg_path: Path | None,
) -> None:
    base_name = input_path.stem
    mlir_input = input_path

    if input_path.suffix.upper() == ".C":
        cgeist_output = dirs["ir"] / f"{base_name}.mlir"
        run_command(
            [
                tools["cgeist"],
                "-O2",
                str(input_path),
                "-S",
                "-o",
                str(cgeist_output),
            ]
        )
        mlir_input = cgeist_output

    normalized = dirs["ir"] / f"{base_name}_normalized.mlir"
    run_command(
        [
            tools["mlir-opt"],
            "--allow-unregistered-dialect",
            "--affine-loop-normalize",
            "--affine-simplify-structures",
            "--normalize-memrefs",
            "--force-specialization",
            "--bufferization-bufferize",
            str(mlir_input),
            "-o",
            str(normalized),
        ]
    )

    kernel_mlir = dirs["kernels"] / f"{base_name}_kernel.mlir"
    run_command(
        [
            tools["adora-opt"],
            "--canonicalize",
            "-reconcile-unrealized-casts",
            "--affine-loop-fusion",
            "--adora-extract-affine-for-to-kernel",
            "--arith-expand",
            "--memref-expand",
            "-cse",
            str(normalized),
            "-o",
            str(kernel_mlir),
        ]
    )

    kernel_opt = dirs["kernels_opt"] / f"{base_name}_opt.mlir"
    kernel_opt_cmd = [
        tools["adora-opt"],
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
    run_command(
        kernel_opt_cmd
    )

    run_command(
        [
            tools["adora-opt"],
            "--adora-kernel-dfg-gen",
            str(kernel_opt),
        ],
        cwd=dirs["temp_dfg"],
    )

    for dot_file in dirs["temp_dfg"].glob("*_CDFG.dot"):
        shutil.copy(dot_file, dirs["dfgs"] / dot_file.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile C/MLIR to CDFG using cgeist/mlir-opt/adora-opt."
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

    tools: dict[str, str] = {
        "mlir-opt": require_tool("mlir-opt"),
        "adora-opt": require_tool("adora-opt"),
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
        build_pipeline(input_path, tools, dirs, args.enable_unroll, adg_path)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"Command failed with exit code {exc.returncode}", file=sys.stderr)
        return exc.returncode

    print(f"CDFG output directory: {dirs['dfgs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
