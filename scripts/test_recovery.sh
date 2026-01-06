#!/bin/bash
# Test fault-tolerant inference recovery

MODEL_DIR="/home/xiaolong/llms/Qwen3-0.6B-w4a32kai"
STATE_DIR="./data/qwen3-test-recovery"
RUNNER="./build/bin/mllm-qwen3-runner"

PROMPT="Explain the theory of relativity in simple terms. Start with special relativity and then move on to general relativity. Be thorough."

run_cmd() {
    timeout --signal=INT "$1" "$RUNNER" \
        -m "$MODEL_DIR/model.mllm" \
        -c "$MODEL_DIR/config.json" \
        -t "$MODEL_DIR/tokenizer.json" \
        -mv v2 --state_path "$STATE_DIR" -cs 32
}

# Clean state completely (don't pre-create, let the app create it)
rm -rf "$STATE_DIR"

echo "=== Test 1: Initial run (with input, interrupted) ==="
run_cmd 20 <<< "$PROMPT"
echo ""
echo "Exit code: $?"

echo ""
echo "=== Test 2: Recovery run (no input needed, resumes from saved state) ==="
run_cmd 15
echo ""
echo "Exit code: $?"

echo ""
echo "=== Test 3: Second recovery (continues from Test 2) ==="
run_cmd 15
echo ""
echo "Exit code: $?"

echo ""
echo "=== Verification ==="
echo "Check above output:"
echo "  - Run 1 should show 'Prefilling [X / Y]' logs"
echo "  - Run 2 should NOT show 'Prefilling' logs (prefill skipped)"
echo "  - Run 2 should replay Run 1's tokens then continue generating"
echo "=== Done ==="
