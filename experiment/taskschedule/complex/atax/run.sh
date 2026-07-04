source ../../../../../env.sh 

adoracc.py atax.c

cgra-opt \
  "$ADORA_COMPILER/.../atax/_gantt/_cc/adora-cc-ir/2_kernel-opt/atax_opt.mlir" \
  --adora-llm-pipeline-schedule \
  --adora-llm-pipeline-schedule-ranker-cmd="python3 $LLM_RANKER --backend dryrule" \
  --adora-llm-pipeline-schedule-num-tiles=2 \
  --adora-llm-pipeline-schedule-pe-per-tile=16 \
  -o atax_sched.mlir