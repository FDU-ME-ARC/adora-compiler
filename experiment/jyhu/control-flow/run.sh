#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

CASES=(
  if_simple
  if_else
  if_elseif_else
  nested_if
  if_compute
  if_load
  if_store
  if_else_store
  loop_if
)

ADORACC="${ADORA_ADORACC:-${ROOT}/build/bin/adoracc.py}"
MAPPER="${ADORA_CGRA_MAPPER:-${ROOT}/build/bin/cgra-mapper}"
CGEIST="${ADORA_CGEIST:-$(command -v cgeist 2>/dev/null || true)}"
ADG="${ADORA_CGRA_ADG:-${ROOT}/test/spec/cgra_fp32/cgra_adg_fp32.json}"
OP_FILE="${ADORA_CGRA_OP_FILE:-${ROOT}/test/spec/cgra_fp32/operations_fp32.json}"
OP_NAME_FILE="${ADORA_OP_NAME_FILE:-${ROOT}/lib/DFG/Documents/GeneralOpName.txt}"
OUT_ROOT="${ADORA_CONTROL_FLOW_OUT:-${SCRIPT_DIR}/out}"
SUMMARY="${OUT_ROOT}/summary.tsv"

usage() {
  cat <<EOF
Usage: $0 all|<case>

Cases:
  ${CASES[*]}

Environment overrides:
  ADORA_ADORACC, ADORA_CGRA_MAPPER, ADORA_CGEIST
  ADORA_CGRA_ADG, ADORA_CGRA_OP_FILE, ADORA_OP_NAME_FILE
  ADORA_CONTROL_FLOW_OUT
EOF
}

contains_case() {
  local requested="$1"
  local case_name
  for case_name in "${CASES[@]}"; do
    [[ "${case_name}" == "${requested}" ]] && return 0
  done
  return 1
}

require_executable() {
  local path="$1"
  local label="$2"
  if [[ ! -x "${path}" ]]; then
    echo "error: ${label} is missing or not executable: ${path}" >&2
    return 1
  fi
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "${path}" ]]; then
    echo "error: ${label} is missing: ${path}" >&2
    return 1
  fi
}

infrastructure_error() {
  echo "error: $*" >&2
  return 2
}

error_signature() {
  local log_file="$1"
  local line=""
  if [[ -s "${log_file}" ]]; then
    line="$(grep -Eim1 'error|failed|failure|unsupported|not supported|assert|abort|segmentation' "${log_file}" 2>/dev/null || true)"
    [[ -n "${line}" ]] || line="$(tail -n 1 "${log_file}" 2>/dev/null || true)"
  fi
  [[ -n "${line}" ]] || line="no diagnostic"
  line="${line//$'\t'/ }"
  line="${line//$'\r'/ }"
  line="${line//$'\n'/ }"
  printf '%.160s' "${line}"
}

stage_failure() {
  local stage="$1"
  local rc="$2"
  local log_file="$3"
  printf 'FAIL(%s,rc=%s,%s)' "${stage}" "${rc}" "$(error_signature "${log_file}")"
}

first_file() {
  local directory="$1"
  local pattern="$2"
  find "${directory}" -type f -name "${pattern}" -print -quit 2>/dev/null || true
}

run_case() {
  local case_name="$1"
  local case_dir="${SCRIPT_DIR}/${case_name}"
  local source_c="${case_dir}/source.c"
  local input_mlir="${case_dir}/input.mlir"
  local case_out="${OUT_ROOT}/${case_name}"
  local pipeline_dir="${case_out}/pipeline"
  local copied_input="${case_out}/input.mlir"
  local final_mlir="${case_out}/final.mlir"
  local compiler_log="${case_out}/adoracc.log"
  local mapper_log="${case_out}/mapper.log"
  local mapper_output="${case_out}/mapper.py"

  if [[ -d "${case_out}" ]]; then
    if ! rm -rf -- "${case_out}"; then
      infrastructure_error "cannot clean output directory: ${case_out}"
      return 2
    fi
  fi
  if ! mkdir -p "${case_out}" "${pipeline_dir}"; then
    infrastructure_error "cannot create output directory: ${case_out}"
    return 2
  fi
  if ! cp "${input_mlir}" "${copied_input}"; then
    infrastructure_error "cannot copy testcase input to: ${copied_input}"
    return 2
  fi

  local log_file
  for log_file in "${case_out}/cgeist.log" "${compiler_log}" "${mapper_log}"; do
    if ! : >"${log_file}"; then
      infrastructure_error "cannot initialize log file: ${log_file}"
      return 2
    fi
  done

  local frontend_status
  if [[ -n "${CGEIST}" && -x "${CGEIST}" ]]; then
    if "${CGEIST}" -O2 "${source_c}" -S -o "${case_out}/frontend.mlir" \
        >"${case_out}/cgeist.log" 2>&1; then
      frontend_status="PASS"
    else
      local frontend_rc=$?
      frontend_status="$(stage_failure cgeist "${frontend_rc}" "${case_out}/cgeist.log")"
    fi
  else
    frontend_status="SKIP(cgeist unavailable)"
    if ! printf 'cgeist unavailable; fixed MLIR remains authoritative for this run.\n' \
        >"${case_out}/cgeist.log"; then
      infrastructure_error "cannot write frontend log: ${case_out}/cgeist.log"
      return 2
    fi
  fi

  "${ADORACC}" "${copied_input}" --work-dir "${pipeline_dir}" \
    -o "${final_mlir}" --disable-schedule-tasks >"${compiler_log}" 2>&1
  local compiler_rc=$?

  local ir_root="${pipeline_dir}/adora-cc-ir"
  local diagnostic_log="${compiler_log}"
  if [[ -s "${ir_root}/pipeline.log" ]]; then
    diagnostic_log="${ir_root}/pipeline.log"
  fi
  local normalized
  local kernel_opt
  local cdfg_dot
  normalized="$(first_file "${ir_root}/temp/normalize" '*_normalized.mlir')"
  kernel_opt="$(first_file "${ir_root}/2_kernel-opt" '*_opt.mlir')"
  cdfg_dot="$(first_file "${ir_root}/temp/dfg" '*_CDFG.dot')"

  local normalize_status="PASS"
  local kernel_status="PASS"
  local cdfg_status="PASS"
  [[ -n "${normalized}" && -s "${normalized}" ]] \
    || normalize_status="$(stage_failure normalize "${compiler_rc}" "${diagnostic_log}")"
  [[ -n "${kernel_opt}" && -s "${kernel_opt}" ]] \
    || kernel_status="$(stage_failure kernel-opt "${compiler_rc}" "${diagnostic_log}")"
  [[ "${compiler_rc}" -eq 0 && -n "${cdfg_dot}" && -s "${cdfg_dot}" ]] \
    || cdfg_status="$(stage_failure cdfg "${compiler_rc}" "${diagnostic_log}")"

  local mapper_status
  if [[ -s "${final_mlir}" ]]; then
    "${MAPPER}" \
      --adg="${ADG}" \
      --op-file="${OP_FILE}" \
      --op-name-file="${OP_NAME_FILE}" \
      --output-type=pytest \
      --obj-opt=true \
      --max-iters=1 \
      --timeout=10000 \
      "${final_mlir}" \
      --output="${mapper_output}" >"${mapper_log}" 2>&1
    local mapper_rc=$?
    if [[ "${mapper_rc}" -eq 0 && -s "${mapper_output}" ]]; then
      mapper_status="PASS"
    else
      mapper_status="$(stage_failure mapper "${mapper_rc}" "${mapper_log}")"
    fi
  else
    mapper_status="SKIP(no final MLIR)"
    if ! printf 'mapper skipped because the compiler did not produce final MLIR.\n' \
        >"${mapper_log}"; then
      infrastructure_error "cannot write mapper log: ${mapper_log}"
      return 2
    fi
  fi

  local overall="PASS"
  local status
  for status in "${frontend_status}" "${normalize_status}" "${kernel_status}" "${cdfg_status}" "${mapper_status}"; do
    if [[ "${status}" == FAIL\(* ]]; then
      overall="${status}"
      break
    fi
    if [[ "${status}" == SKIP\(* && "${status}" != "SKIP(cgeist unavailable)" ]]; then
      overall="${status}"
    fi
  done

  if ! printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${case_name}" "${frontend_status}" "${normalize_status}" \
      "${kernel_status}" "${cdfg_status}" "${mapper_status}" "${overall}" \
      >>"${SUMMARY}"; then
    infrastructure_error "cannot append summary: ${SUMMARY}"
    return 2
  fi

  printf '%-18s normalize=%-5s kernel=%-5s cdfg=%-5s mapper=%s\n' \
    "${case_name}" "${normalize_status}" "${kernel_status}" \
    "${cdfg_status}" "${mapper_status}"
}

main() {
  if [[ $# -ne 1 ]]; then
    usage >&2
    return 2
  fi

  local requested="$1"
  local selected=()
  if [[ "${requested}" == "all" ]]; then
    selected=("${CASES[@]}")
  elif contains_case "${requested}"; then
    selected=("${requested}")
  else
    echo "error: unknown control-flow case: ${requested}" >&2
    usage >&2
    return 2
  fi

  require_executable "${ADORACC}" adoracc.py || return 2
  require_executable "${MAPPER}" cgra-mapper || return 2
  require_file "${ADG}" 'CGRA ADG specification' || return 2
  require_file "${OP_FILE}" 'CGRA operation specification' || return 2
  require_file "${OP_NAME_FILE}" 'MLIR operation-name mapping' || return 2

  local case_name
  for case_name in "${selected[@]}"; do
    require_file "${SCRIPT_DIR}/${case_name}/source.c" "${case_name} C source" || return 2
    require_file "${SCRIPT_DIR}/${case_name}/input.mlir" "${case_name} MLIR input" || return 2
  done

  if ! mkdir -p "${OUT_ROOT}"; then
    infrastructure_error "cannot create output root: ${OUT_ROOT}"
    return 2
  fi
  if ! printf 'Case\tFrontend\tNormalize\tKernelOpt\tCDFG\tMapper\tOverall\n' >"${SUMMARY}"; then
    infrastructure_error "cannot initialize summary: ${SUMMARY}"
    return 2
  fi

  local case_rc
  for case_name in "${selected[@]}"; do
    run_case "${case_name}"
    case_rc=$?
    if [[ "${case_rc}" -ne 0 ]]; then
      return "${case_rc}"
    fi
  done

  echo "Summary: ${SUMMARY}"
}

main "$@"
