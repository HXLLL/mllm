#!/bin/bash

CS=${1:-1}
./build/bin/mllm-llm-benchmark \
    -n Qwen3-0.6B-w4a32kai \
    -m /home/xiaolong/llms/Qwen3-0.6B-w4a32kai/model.mllm \
    -c /home/xiaolong/llms/Qwen3-0.6B-w4a32kai/config.json \
    -pp 64 -tg 64 --trace_file ./data/cs${CS}.csv -cs ${CS}
