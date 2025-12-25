#!/bin/bash

PERF=/usr/lib/linux-tools/5.15.0-1032-nvidia-tegra/perf
PERF_ARGS="record -e writeback:* -a"
CMD="./build/bin/mllm-qwen3-runner -m /home/xiaolong/llms/Qwen3-0.6B-w4a32kai/model.mllm  -c /home/xiaolong/llms/Qwen3-0.6B-w4a32kai/config.json -t /home/xiaolong/llms/Qwen3-0.6B-w4a32kai/tokenizer.json -mv v2 --cache_dir ./data/qwen3-runner"

if [[ $1 == "record" ]]; then
    sudo ${PERF} ${PERF_ARGS} -- ${CMD}
elif [[ $1 == "sleep" ]]; then
    sudo ${PERF} ${PERF_ARGS} -- sleep $2 &
    timeout $2 ${CMD}
elif [[ $1 == "script" ]]; then
    sudo ${PERF} script
else
    sudo ${PERF} $@
fi
