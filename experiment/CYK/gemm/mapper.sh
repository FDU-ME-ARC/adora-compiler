#!/bin/bash
# ==============================================================================
# CGRA Mapper Execution Script (Cocotb-Pytest Target)
# Script Path: /home/ykchen/projects/CGRVOPT/cgra-opt/experiment/AIModels/To_CYK/test_adora/conv/mapper.sh
# Description: 自动加载全局环境 env.sh，并将 final.mlir 映射为 Python 仿真配置流
# ==============================================================================

# 遇到错误立即退出
set -e

echo "========================================================"
echo "              CGRA Mapping Initialization               "
echo "========================================================"

# 1. 自动加载全局环境变量 (使用你提供的绝对路径)
ENV_SCRIPT="/home/ykchen/projects/CGRVOPT/cgra-opt/env.sh"

if [ -f "$ENV_SCRIPT" ]; then
    echo "[INFO] Sourcing $ENV_SCRIPT to setup environment..."
    eval "$(conda shell.bash hook)"
    source "$ENV_SCRIPT"
else
    echo "[ERROR] Global env script not found at: $ENV_SCRIPT"
    echo "Please check the path and try again."
    exit 1
fi

# 2. 定义工作区和输入输出文件 
# 动态获取当前脚本所在的绝对路径，确保在任何目录下执行该脚本都不会找不到文件
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_MLIR="$WORK_DIR/gemm_affine.mlir"
OUTPUT_PY="$WORK_DIR/conv_config.py"
LOG_FILE="$WORK_DIR/mapper_debug.log"

# 3. 智能判定硬件架构路径
# 注意：之前你在 scripts.sh 中使用的是 jhlou 目录下的定制架构。
# 这里保留该路径的探测，如果存在定制文件就用定制的，否则用 env.sh 里的默认值。
# (如果你在 ykchen 目录下也有专门的 MatrixMeld 路径，可以把下面两行改成你的路径)
# VITRA_ADG="/home/jhlou/CGRVOPT/MatrixMeld/rtl/spec/vitra_cgra_adg.json"
VITRA_ADG="/home/jhlou/CGRVOPT/MatrixMeld/vitrartl/spec/vitra_cgra_adg.json"
VITRA_OP="/home/jhlou/CGRVOPT/MatrixMeld/vitrartl/spec/operations.json"

if [ -f "$VITRA_ADG" ] && [ -f "$VITRA_OP" ]; then
    ADG_PATH="$VITRA_ADG"
    OP_PATH="$VITRA_OP"
    echo "[INFO] Detected MatrixMeld custom architecture."
else
    # 回退到 env.sh 中配置的默认目录
    ADG_PATH="${CGRA_ADG_PATH}/cgra_adg.json"
    OP_PATH="${CGRA_OP_FILE_PATH}/operations.json"
    echo "[INFO] Using default architecture from env.sh: $ADG_PATH"
fi

# 4. 执行前检查
if [ ! -f "$INPUT_MLIR" ]; then
    echo "[ERROR] Input file not found: $INPUT_MLIR"
    echo "Make sure final.mlir is in: $WORK_DIR"
    exit 1
fi

if [ ! -f "$ADG_PATH" ]; then
    echo "[ERROR] ADG file not found: $ADG_PATH"
    exit 1
fi

# 5. 执行 cgra-mapper
echo "--------------------------------------------------------"
echo "[INFO] Working Dir : $WORK_DIR"
echo "[INFO] Input MLIR  : $INPUT_MLIR"
echo "[INFO] Target ADG  : $ADG_PATH"
echo "[INFO] Output Type : pytest (Targeting Cocotb Simulator)"
echo "[INFO] Debug Log   : $LOG_FILE"
echo "--------------------------------------------------------"
echo "[INFO] Running cgra-mapper..."

cgra-mapper \
  --adg="$ADG_PATH" \
  --op-file="$OP_PATH" \
  --output="$OUTPUT_PY" \
  --output-type="pytest" \
  --obj-opt=false \
  "$INPUT_MLIR" > "$LOG_FILE" 2>&1

echo "--------------------------------------------------------"
echo "[SUCCESS] Mapping completed flawlessly!"
echo "[SUCCESS] Python configuration stream saved to: $OUTPUT_PY"
echo "========================================================"