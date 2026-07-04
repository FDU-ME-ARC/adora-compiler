source ../../../../../env.sh

adoracc.py attn.c

# 输入 = adoracc 产出的 kernel-opt(还没调度)那份。
# 注意:LLM 调度 pass 依赖 --adora-schedule-tasks 写的 dep_summary,
# 而 adoracc 默认已跑过 schedule-tasks,所以这份 _opt.mlir 已带 dep_summary。
IN=adora-cc-ir/2_kernel-opt/attn_opt.mlir

# ===========================================================================
# (A) dryrule:纯 Python 规则模拟 LLM,不联网、无需 key。CI / 离线复现用。
# ===========================================================================
cgra-opt "$IN" \
  --adora-llm-pipeline-schedule \
  --adora-llm-pipeline-schedule-ranker-cmd="python3 $LLM_RANKER --backend dryrule" \
  --adora-llm-pipeline-schedule-num-tiles=2 \
  --adora-llm-pipeline-schedule-pe-per-tile=16 \
  -o attn_sched_dryrule.mlir

# ===========================================================================
# (B) 真 LLM —— OpenAI 官方端点 (默认 model: gpt-4o-mini)
#     把 sk-XXXXXXXX 换成你的 OpenAI key。model 可用 --model 覆盖。
# ===========================================================================
OPENAI_KEY="sk-XXXXXXXX"          # <-- 填你的 OpenAI API key
OPENAI_MODEL="gpt-4o-mini"        # 可选: gpt-4o / gpt-4o-mini / gpt-4-turbo ...
cgra-opt "$IN" \
  --adora-llm-pipeline-schedule \
  --adora-llm-pipeline-schedule-ranker-cmd="python3 $LLM_RANKER --backend openai --api-key $OPENAI_KEY --model $OPENAI_MODEL" \
  --adora-llm-pipeline-schedule-num-tiles=2 \
  --adora-llm-pipeline-schedule-pe-per-tile=16 \
  --adora-llm-pipeline-schedule-ranker-log=attn_ranker_openai.ndjson \
  -o attn_sched_openai.mlir

# ===========================================================================
# (C) 真 LLM —— 火山引擎 Ark (OpenAI 兼容; 实验里用过 doubao)
#     base-url 固定为 Ark; model 填你在 Ark 开通的 doubao 接入点 (ep-xxx) 或模型名。
#     key 用 Ark 的 API Key (不是 OpenAI 的)。
# ===========================================================================
ARK_KEY="ARK_API_KEY_XXXXXXXX"                                  # <-- 填你的火山 Ark API key
ARK_URL="https://ark.cn-beijing.volces.com/api/v3"              # 火山 Ark OpenAI 兼容端点
ARK_MODEL="ep-XXXXXXXX"                                         # <-- 填 Ark 接入点 ep-xxx 或模型名(如 doubao-pro-32k)
cgra-opt "$IN" \
  --adora-llm-pipeline-schedule \
  --adora-llm-pipeline-schedule-ranker-cmd="python3 $LLM_RANKER --backend openai --base-url $ARK_URL --api-key $ARK_KEY --model $ARK_MODEL" \
  --adora-llm-pipeline-schedule-num-tiles=2 \
  --adora-llm-pipeline-schedule-pe-per-tile=16 \
  --adora-llm-pipeline-schedule-ranker-timeout=60000 \
  --adora-llm-pipeline-schedule-ranker-log=attn_ranker_ark.ndjson \
  -o attn_sched_ark.mlir

# ===========================================================================
# (D) 本地 OpenAI 兼容服务 (如 vLLM / ollama-openai; 默认 model: llama3)
#     默认 url http://localhost:8000/v1 ; 用 --local-url 改端口。
# ===========================================================================
LOCAL_URL="http://localhost:8000/v1"
LOCAL_MODEL="llama3"
cgra-opt "$IN" \
  --adora-llm-pipeline-schedule \
  --adora-llm-pipeline-schedule-ranker-cmd="python3 $LLM_RANKER --backend local --local-url $LOCAL_URL --model $LOCAL_MODEL" \
  --adora-llm-pipeline-schedule-num-tiles=2 \
  --adora-llm-pipeline-schedule-pe-per-tile=16 \
  --adora-llm-pipeline-schedule-ranker-log=attn_ranker_local.ndjson \
  -o attn_sched_local.mlir
