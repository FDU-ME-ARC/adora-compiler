#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/run.sh"

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

[[ -x "${RUNNER}" ]] || fail "runner is missing or not executable: ${RUNNER}"

tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT
fake_bin="${tmp_dir}/bin"
out_dir="${tmp_dir}/out"
mkdir -p "${fake_bin}"

cat >"${fake_bin}/adoracc.py" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
input="$1"
shift
work_dir=""
output=""
while (($#)); do
  case "$1" in
    --work-dir) work_dir="$2"; shift 2 ;;
    -o|--output) output="$2"; shift 2 ;;
    --disable-schedule-tasks) shift ;;
    *) shift ;;
  esac
done
base="$(basename "${input}" .mlir)"
ir_dir="${work_dir}/adora-cc-ir"
mkdir -p "${ir_dir}/temp/normalize" "${ir_dir}/temp/kernel-extract" \
  "${ir_dir}/2_kernel-opt" "${ir_dir}/temp/dfg"
cp "${input}" "${ir_dir}/temp/normalize/${base}_normalized.mlir"
if [[ -n "${FAKE_ADORACC_FAIL_CASE:-}" && "${work_dir}" == *"/${FAKE_ADORACC_FAIL_CASE}/"* ]]; then
  printf 'error: operand does not dominate this use\n' >"${ir_dir}/pipeline.log"
  exit 7
fi
cp "${input}" "${ir_dir}/temp/kernel-extract/${base}_kernel.mlir"
cp "${input}" "${ir_dir}/2_kernel-opt/${base}_opt.mlir"
cp "${input}" "${output}"
printf 'Digraph G {}\n' >"${ir_dir}/temp/dfg/kernel_0_CDFG.dot"
printf 'fake pipeline\n' >"${ir_dir}/pipeline.log"
EOF
chmod +x "${fake_bin}/adoracc.py"

cat >"${fake_bin}/cgra-mapper" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
output=""
while (($#)); do
  case "$1" in
    --output=*) output="${1#--output=}"; shift ;;
    --output) output="$2"; shift 2 ;;
    *) shift ;;
  esac
done
[[ -n "${output}" ]]
printf '# fake mapper output\n' >"${output}"
EOF
chmod +x "${fake_bin}/cgra-mapper"

cat >"${fake_bin}/failing-cgeist" <<'EOF'
#!/usr/bin/env bash
echo 'error: frontend conversion failed' >&2
exit 9
EOF
chmod +x "${fake_bin}/failing-cgeist"

before="$(find "${SCRIPT_DIR}" -mindepth 2 -maxdepth 2 \
  -type f \( -name source.c -o -name input.mlir \) -print0 \
  | sort -z | xargs -0 sha256sum)"

ADORA_ADORACC="${fake_bin}/adoracc.py" \
ADORA_CGRA_MAPPER="${fake_bin}/cgra-mapper" \
ADORA_CGEIST="${tmp_dir}/missing-cgeist" \
ADORA_CONTROL_FLOW_OUT="${out_dir}" \
  "${RUNNER}" all

summary="${out_dir}/summary.tsv"
[[ -s "${summary}" ]] || fail "summary.tsv was not created"
[[ "$(awk 'END { print NR }' "${summary}")" -eq 10 ]] \
  || fail "summary must contain one header and nine cases"
[[ "$(awk -F '\t' 'NR > 1 && $2 ~ /^SKIP\(cgeist unavailable\)$/ { n++ } END { print n+0 }' "${summary}")" -eq 9 ]] \
  || fail "all frontend stages must be skipped when cgeist is unavailable"
[[ "$(awk -F '\t' 'NR > 1 && $3 == "PASS" && $4 == "PASS" && $5 == "PASS" && $6 == "PASS" { n++ } END { print n+0 }' "${summary}")" -eq 9 ]] \
  || fail "all fixed-MLIR stages must pass with successful tools"
[[ "$(awk -F '\t' 'NR > 1 && $7 == "PASS" { n++ } END { print n+0 }' "${summary}")" -eq 9 ]] \
  || fail "missing optional cgeist must not downgrade Overall"

after="$(find "${SCRIPT_DIR}" -mindepth 2 -maxdepth 2 \
  -type f \( -name source.c -o -name input.mlir \) -print0 \
  | sort -z | xargs -0 sha256sum)"
[[ "${before}" == "${after}" ]] || fail "runner modified tracked testcase inputs"

ADORA_ADORACC="${fake_bin}/adoracc.py" \
ADORA_CGRA_MAPPER="${fake_bin}/cgra-mapper" \
ADORA_CGEIST="${tmp_dir}/missing-cgeist" \
ADORA_CONTROL_FLOW_OUT="${out_dir}" \
  "${RUNNER}" if_else

[[ "$(awk 'END { print NR }' "${summary}")" -eq 2 ]] \
  || fail "single-case run must replace the summary with one result"

FAKE_ADORACC_FAIL_CASE=if_load \
ADORA_ADORACC="${fake_bin}/adoracc.py" \
ADORA_CGRA_MAPPER="${fake_bin}/cgra-mapper" \
ADORA_CGEIST="${tmp_dir}/missing-cgeist" \
ADORA_CONTROL_FLOW_OUT="${out_dir}" \
  "${RUNNER}" all

[[ "$(awk 'END { print NR }' "${summary}")" -eq 10 ]] \
  || fail "all must retain nine rows when a middle case fails"
awk -F '\t' 'NR > 1 && $1 == "if_load" && $3 == "PASS" && $4 ~ /^FAIL\(kernel-opt,rc=7,.*operand does not dominate/ && $5 ~ /^FAIL\(cdfg,rc=7,.*operand does not dominate/ && $6 == "SKIP(no final MLIR)" && $7 ~ /^FAIL\(kernel-opt,rc=7,/ { found=1 } END { exit !found }' \
  "${summary}" || fail "compiler failures must report detailed stage and overall diagnostics"
awk -F '\t' 'NR > 1 && $1 == "loop_if" && $3 == "PASS" && $4 == "PASS" && $5 == "PASS" && $6 == "PASS" { found=1 } END { exit !found }' \
  "${summary}" || fail "all must continue through cases after a middle failure"
[[ ! -e "${out_dir}/if_load/final.mlir" && ! -e "${out_dir}/if_load/mapper.py" ]] \
  || fail "failed reruns must not retain stale compiler or mapper outputs"

ADORA_ADORACC="${fake_bin}/adoracc.py" \
ADORA_CGRA_MAPPER="${fake_bin}/cgra-mapper" \
ADORA_CGEIST="${fake_bin}/failing-cgeist" \
ADORA_CONTROL_FLOW_OUT="${out_dir}" \
  "${RUNNER}" if_else

awk -F '\t' 'NR == 2 && $2 ~ /^FAIL\(cgeist,rc=9,/ && $3 == "PASS" && $4 == "PASS" && $5 == "PASS" && $6 == "PASS" && $7 ~ /^FAIL\(cgeist,rc=9,/ { found=1 } END { exit !found }' \
  "${summary}" || fail "an executable cgeist failure must affect Overall"

if "${RUNNER}" does_not_exist >/dev/null 2>&1; then
  fail "unknown cases must return non-zero"
fi

printf 'not a directory\n' >"${tmp_dir}/not-directory"
if ADORA_ADORACC="${fake_bin}/adoracc.py" \
   ADORA_CGRA_MAPPER="${fake_bin}/cgra-mapper" \
   ADORA_CGEIST="${tmp_dir}/missing-cgeist" \
   ADORA_CONTROL_FLOW_OUT="${tmp_dir}/not-directory/out" \
     "${RUNNER}" if_simple >/dev/null 2>&1; then
  fail "an unusable output path must return non-zero"
fi

echo "PASS: control-flow runner contract"
