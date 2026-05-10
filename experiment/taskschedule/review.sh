#!/bin/bash
# review.sh — run all taskschedule experiments and show token graph + IR
#
# Usage:
#   ./review.sh              # run all 4 examples
#   ./review.sh 01           # run only example 01
#   ./review.sh 03 --no-dot  # run example 03, skip dot generation
#
# Output per example:
#   <N>/output.mlir          — scheduled IR (stdout of cgra-opt)
#   <N>/tokens.dot           — token dep graph (Graphviz DOT)
#   <N>/tokens.png           — rendered PNG (only if 'dot' is in PATH)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CGRA_OPT="${REPO_ROOT}/build/bin/cgra-opt"
EXAMPLES_DIR="${SCRIPT_DIR}"

# --- arg parsing ---
FILTER=""
GEN_DOT=1
for arg in "$@"; do
  case "$arg" in
    --no-dot) GEN_DOT=0 ;;
    *)        FILTER="$arg" ;;
  esac
done

if [[ ! -x "${CGRA_OPT}" ]]; then
  echo "ERROR: cgra-opt not found at ${CGRA_OPT}"
  echo "       Run: ninja -C ${REPO_ROOT}/build cgra-opt"
  exit 1
fi

run_example() {
  local dir="$1"
  local name
  name="$(basename "${dir}")"
  local input="${dir}/input.mlir"

  if [[ ! -f "${input}" ]]; then
    echo "  [skip] no input.mlir in ${name}"
    return
  fi

  echo ""
  echo "══════════════════════════════════════════"
  echo "  Example: ${name}"
  echo "══════════════════════════════════════════"

  # --- 1. emit-token=false: plain scheduled IR ---
  echo ""
  echo "▶ Pass 1: adora-schedule-tasks (no token)"
  "${CGRA_OPT}" "${input}" \
    --adora-schedule-tasks="emit-token=false" \
    2>/dev/null \
    > "${dir}/output_no_token.mlir" || true
  echo "  → ${dir}/output_no_token.mlir"

  # --- 2. emit-token=true: async token IR + dot ---
  echo ""
  echo "▶ Pass 2: adora-schedule-tasks emit-token=true"
  DOT_ARG=""
  if [[ "${GEN_DOT}" -eq 1 ]]; then
    DOT_ARG=" dump-token-graph=${dir}/tokens.dot"
  fi

  "${CGRA_OPT}" "${input}" \
    --adora-schedule-tasks="emit-token=true${DOT_ARG}" \
    2>"${dir}/stderr.txt" \
    > "${dir}/output_token.mlir" || true

  echo "  → ${dir}/output_token.mlir"

  # Show token chain from IR
  echo ""
  echo "▶ Token ops in output:"
  grep -E "async|!ADORA.token" "${dir}/output_token.mlir" | sed 's/^/    /' || echo "    (none)"

  # --- 3. render dot to png ---
  if [[ "${GEN_DOT}" -eq 1 && -f "${dir}/tokens.dot" ]]; then
    echo ""
    echo "▶ Token graph DOT: ${dir}/tokens.dot"
    cat "${dir}/tokens.dot" | sed 's/^/    /'
    if command -v dot &>/dev/null; then
      dot -Tpng "${dir}/tokens.dot" -o "${dir}/tokens.png" 2>/dev/null
      echo "  → rendered: ${dir}/tokens.png"
    else
      echo "  (install graphviz to render PNG: apt install graphviz)"
    fi
  fi

  # --- 4. assign streams (if available) ---
  echo ""
  echo "▶ Pass 3: adora-assign-streams"
  "${CGRA_OPT}" "${dir}/output_token.mlir" \
    --adora-assign-streams \
    2>/dev/null \
    > "${dir}/output_streams.mlir" || echo "  (pass not available or failed)"
  if [[ -s "${dir}/output_streams.mlir" ]]; then
    echo "  → ${dir}/output_streams.mlir"
    echo "▶ Stream assignments:"
    grep -E "stream\s*=" "${dir}/output_streams.mlir" | sed 's/^/    /' || echo "    (none)"
  fi
}

# --- main ---
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  ADORA Task Schedule Experiment Review   ║"
echo "╚══════════════════════════════════════════╝"
echo "  cgra-opt: ${CGRA_OPT}"
echo "  examples: ${EXAMPLES_DIR}"

for dir in "${EXAMPLES_DIR}"/*/; do
  name="$(basename "${dir}")"
  if [[ -n "${FILTER}" && "${name}" != *"${FILTER}"* ]]; then
    continue
  fi
  run_example "${dir}"
done

echo ""
echo "══════════════════════════════════════════"
echo "  Done."
echo "══════════════════════════════════════════"
