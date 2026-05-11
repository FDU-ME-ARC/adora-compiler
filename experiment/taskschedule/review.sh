#!/bin/bash
# review.sh — run all taskschedule experiments
#
# Usage:
#   ./review.sh              # run all 4 examples
#   ./review.sh 01           # run only example matching "01"
#   ./review.sh 03 04        # run examples 03 and 04

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FILTER=("$@")

run_example() {
    local dir="$1"
    local name
    name="$(basename "${dir}")"

    # Apply filter if specified
    if [[ ${#FILTER[@]} -gt 0 ]]; then
        local match=0
        for f in "${FILTER[@]}"; do
            [[ "${name}" == *"${f}"* ]] && match=1 && break
        done
        [[ $match -eq 0 ]] && return
    fi

    local script="${dir}/run.sh"
    if [[ ! -f "${script}" ]]; then
        echo "[skip] no run.sh in ${name}"
        return
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    bash "${script}"
}

echo "╔══════════════════════════════════════════╗"
echo "║  ADORA Task Schedule Experiments         ║"
echo "╚══════════════════════════════════════════╝"

for dir in "${SCRIPT_DIR}"/*/; do
    run_example "${dir}"
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "All done."
