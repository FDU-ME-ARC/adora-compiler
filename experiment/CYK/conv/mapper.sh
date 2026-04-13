#!/bin/bash
# ==============================================================================
# CGRA Mapper Execution Script (Cocotb-Pytest Target)
# ==============================================================================

set -e

echo "========================================================"
echo "              CGRA Mapping Initialization               "
echo "========================================================"

# 1. 自动加载全局环境变量
ENV_SCRIPT="/home/ykchen/projects/CGRVOPT/cgra-opt/env.sh"
if [ -f "$ENV_SCRIPT" ]; then
    echo "[INFO] Sourcing $ENV_SCRIPT to setup environment..."
    eval "$(conda shell.bash hook)"
    source "$ENV_SCRIPT"
fi

WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_MLIR="$WORK_DIR/affine.mlir"
KERNEL_MLIR="$WORK_DIR/kernel.mlir"
OUTPUT_PY="$WORK_DIR/conv_config.py"

# ADG_PATH="/home/jhlou/CGRVOPT/MatrixMeld/vitrartl/spec/vitra_cgra_adg.json"
# OP_PATH="/home/jhlou/CGRVOPT/MatrixMeld/rtl/spec/operations.json"

# 更大的CGRA
ADG_PATH="/home/jhlou/CGRVOPT/MatrixMeld/vitrartl_8x16/spec/vitra_cgra_adg.json"
OP_PATH="/home/jhlou/CGRVOPT/MatrixMeld/vitrartl_8x16/spec/operations.json"

TENSOR_LOG="$WORK_DIR/tensor_debug.log"
MAPPER_LOG="$WORK_DIR/mapper_debug.log"

if [ ! -f "$INPUT_MLIR" ]; then
    echo "[ERROR] Input file not found: $INPUT_MLIR"
    exit 1
fi

echo "--------------------------------------------------------"
echo "[INFO] Working Dir : $WORK_DIR"
echo "[INFO] Input MLIR  : $INPUT_MLIR"
echo "[INFO] Tensor Log  : $TENSOR_LOG"
echo "[INFO] Mapper Log  : $MAPPER_LOG"
echo "--------------------------------------------------------"

# ==============================================================================
# Step 1: 策略半自动探索 (Strategy Decision)
# 修复：锁死 algorithm-kind=conv_im2col，防止代价模型选择后端未实现的 Winograd。
#       但仍允许编译器自动探索最优的 tile_size 和 stationary_kind！
# ==============================================================================
echo "[INFO] Step 1: Running Strategy Exploration (locked to Im2Col)..."
tensor-opt \
  --adora-tensor-op-strategy-decision="adg-fn=$ADG_PATH bus-bandwidth=16 algorithm-kind=conv_direct" \
  $INPUT_MLIR -o $KERNEL_MLIR > $TENSOR_LOG 2>&1
  # --adora-tensor-op-strategy-decision="adg-fn=$ADG_PATH bus-bandwidth=16 algorithm-kind=conv_im2col" \

echo "[INFO] Strategy successfully injected! Check $TENSOR_LOG for exploration details."

# ==============================================================================
# Step 2: 硬件映射与代码生成 (CGRA Mapping)
# 作用: 捕获全部输出到 mapper_debug.log，终端只保持清爽的进度
# ==============================================================================
echo "[INFO] Step 2: Running CGRA Hardware Mapping (Check $MAPPER_LOG for details)..."
cgra-mapper \
  --adg=$ADG_PATH \
  --op-file=$OP_PATH \
  --output=$OUTPUT_PY \
  --output-type=pytest \
  --obj-opt=false \
  $KERNEL_MLIR > $MAPPER_LOG 2>&1

echo "========================================================"
echo "[SUCCESS] Pipeline Complete! Pytest Configuration generated at $OUTPUT_PY!"
echo "========================================================"