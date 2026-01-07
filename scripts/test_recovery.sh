#!/bin/bash
# Test fault-tolerant inference recovery

MODEL_DIR="/home/xiaolong/llms/Qwen3-0.6B-w4a32kai"
STATE_DIR="./data/qwen3-test-recovery"
RUNNER="./build/bin/mllm-qwen3-runner"
CHUNK_SIZE=4

PROMPT="You are a helpful assistant. Write a short poem about coding. "

run_cmd() {
    timeout --signal=INT "$1" "$RUNNER" \
        -m "$MODEL_DIR/model.mllm" \
        -c "$MODEL_DIR/config.json" \
        -t "$MODEL_DIR/tokenizer.json" \
        -mv v2 --state_path "$STATE_DIR" -cs "$CHUNK_SIZE"
}

echo "========================================="
echo "Test A: Prefill Resume"
echo "========================================="
rm -rf "$STATE_DIR"

echo "=== Run 1: Interrupt during prefill ==="
run_cmd 10 <<< "$PROMPT"
echo "Exit code: $?"
echo ""

echo "=== Run 2: Resume from prefill ==="
run_cmd 20
echo "Exit code: $?"
echo ""

echo "=== Run 3: Resume from decode ==="
run_cmd 30
echo "Exit code: $?"
echo ""

echo "=== Done ==="
