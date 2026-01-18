#!/bin/bash

# Usage: ./scripts/test_long_context.sh [1k|2k|4k|8k]

# Drop caches
/bin/sync
echo 3 | sudo tee /proc/sys/vm/drop_caches

# Parameters
CONTEXT_SIZE=${1:-1k}
CS=${CS:-32}
MODEL_TYPE=${MODEL_TYPE:-Qwen3-0.6B}
LAZY_LOAD=${LAZY_LOAD:-true}

# Paths
MODEL_BASE="/home/xiaolong/llms/${MODEL_TYPE}"
PROMPT_FILE="./scripts/prompts/prompt_${CONTEXT_SIZE}.txt"

# Validate prompt file exists
if [[ ! -f "${PROMPT_FILE}" ]]; then
    echo "Error: Prompt file not found: ${PROMPT_FILE}"
    echo "Available: 1k, 2k, 4k, 8k"
    exit 1
fi

# State/trace paths
TRACE_FOLDER="./data/trace/${MODEL_TYPE}-${CS}"
STATE_PATH="./data/state/${MODEL_TYPE}-${CS}"
mkdir -p "${TRACE_FOLDER}"
mkdir -p "${STATE_PATH}"

rm -rf ${STATE_PATH}/qwen3-runner-${CONTEXT_SIZE}

if [ "$DEBUG" ]; then
  APP="gdb --args ./build/bin/mllm-qwen3-runner"
else
  APP="./build/bin/mllm-qwen3-runner"
fi

# Run
CMD=
CMD="${APP} \
  -m "${MODEL_BASE}/model.mllm" \
  -c "${MODEL_BASE}/config.json" \
  -t "${MODEL_BASE}/tokenizer.json" \
  -i "${PROMPT_FILE}" \
  --state_path "${STATE_PATH}/qwen3-runner-${CONTEXT_SIZE}" \
  -cs "${CS}" \
  -tf "${TRACE_FOLDER}/${CONTEXT_SIZE}.csv" \
  $([ "$LAZY_LOAD" = "true" ] && echo "-ll")"

echo "${CMD}"
${CMD}
