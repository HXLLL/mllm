#!/usr/bin/env python3
"""
Test script for interruption and resumption of qwen3_i model.

This script tests:
1. Normal prefill and decode
2. Interruption during prefill (simulated by running with small chunksize and killing)
3. Resumption after interruption
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# Configuration
MLLM_ROOT = Path(__file__).parent.parent
BUILD_DIR = MLLM_ROOT / "build"
BENCHMARK_BIN = BUILD_DIR / "bin" / "mllm-llm-benchmark"
TEST_CACHE_DIR = MLLM_ROOT / "data" / "test_cache"

# Model configuration (adjust these paths as needed)
MODEL_PATH = os.environ.get("MLLM_MODEL_PATH", "")
CONFIG_PATH = os.environ.get("MLLM_CONFIG_PATH", "")


def setup_test_cache():
    """Clean up and prepare test cache directory."""
    if TEST_CACHE_DIR.exists():
        shutil.rmtree(TEST_CACHE_DIR)
    TEST_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def read_cache_metadata():
    """Read and return the cache metadata."""
    metadata_path = TEST_CACHE_DIR / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return None


def run_benchmark(pp: int, tg: int, chunksize: int = 1, timeout: Optional[float] = None, 
                  interrupt_after: Optional[float] = None):
    """
    Run the benchmark with specified parameters.
    
    Args:
        pp: Prompt length (prefill tokens)
        tg: Generation length (decode tokens)
        chunksize: Chunk size for processing
        timeout: Maximum time to wait for completion
        interrupt_after: Send SIGTERM after this many seconds (simulates interruption)
    
    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    cmd = [
        str(BENCHMARK_BIN),
        "-n", "Qwen3_W4A32_KAI_i",
        "-m", MODEL_PATH,
        "-c", CONFIG_PATH,
        "-pp", str(pp),
        "-tg", str(tg),
        "-cs", str(chunksize),
        "-cl", "2048",
        "-i",  # intermittent mode
        "--cache_dir", str(TEST_CACHE_DIR)
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    if interrupt_after:
        time.sleep(interrupt_after)
        print(f"Sending SIGTERM after {interrupt_after}s...")
        proc.send_signal(signal.SIGTERM)
    
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return proc.returncode, stdout, stderr
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        return -1, stdout, stderr


def test_normal_execution():
    """Test normal execution without interruption."""
    print("\n" + "="*60)
    print("TEST: Normal execution")
    print("="*60)
    
    setup_test_cache()
    
    ret, stdout, stderr = run_benchmark(pp=32, tg=8, chunksize=1, timeout=300)
    
    print(f"Return code: {ret}")
    print(f"Stdout:\n{stdout}")
    if stderr:
        print(f"Stderr:\n{stderr}")
    
    metadata = read_cache_metadata()
    print(f"Cache metadata: {json.dumps(metadata, indent=2) if metadata else 'Not found'}")
    
    return ret == 0


def test_interruption_and_resume():
    """Test interruption during prefill and resumption."""
    print("\n" + "="*60)
    print("TEST: Interruption and Resume")
    print("="*60)
    
    setup_test_cache()
    
    # Run 1: Start with large prefill, interrupt after a short time
    print("\n--- Run 1: Start and interrupt during prefill ---")
    ret1, stdout1, stderr1 = run_benchmark(
        pp=128, tg=16, chunksize=8, 
        timeout=60, interrupt_after=2.0
    )
    
    print(f"Return code: {ret1}")
    print(f"Stdout (last 20 lines):\n{chr(10).join(stdout1.split(chr(10))[-20:])}")
    
    metadata1 = read_cache_metadata()
    print(f"Cache metadata after interruption: {json.dumps(metadata1, indent=2) if metadata1 else 'Not found'}")
    
    if metadata1 is None:
        print("FAIL: No cache metadata after interruption")
        return False
    
    # Check if we have partial progress
    processed_tokens = metadata1.get("model_state", {}).get("token_counter", 0)
    pending_tokens = metadata1.get("pending_tokens", [])
    
    print(f"Processed tokens: {processed_tokens}")
    print(f"Pending tokens count: {len(pending_tokens)}")
    
    if processed_tokens == 0:
        print("WARN: No tokens were processed before interruption")
    
    # Run 2: Resume from where we left off
    print("\n--- Run 2: Resume from interrupted state ---")
    ret2, stdout2, stderr2 = run_benchmark(
        pp=128, tg=16, chunksize=8, 
        timeout=120
    )
    
    print(f"Return code: {ret2}")
    print(f"Stdout (last 30 lines):\n{chr(10).join(stdout2.split(chr(10))[-30:])}")
    
    metadata2 = read_cache_metadata()
    print(f"Cache metadata after resume: {json.dumps(metadata2, indent=2) if metadata2 else 'Not found'}")
    
    # Check that resume detection message appears
    if "Resuming prefill" in stdout2:
        print("SUCCESS: Resume detected in output")
    else:
        print("INFO: No resume message (may be expected if first run completed)")
    
    return ret2 == 0


def test_fresh_start_after_completion():
    """Test that a fresh start works after previous completion."""
    print("\n" + "="*60)
    print("TEST: Fresh start after completion")
    print("="*60)
    
    # First run with small prefill (should complete)
    setup_test_cache()
    ret1, stdout1, stderr1 = run_benchmark(pp=16, tg=4, chunksize=1, timeout=120)
    
    metadata1 = read_cache_metadata()
    print(f"After first run - processed: {metadata1.get('model_state', {}).get('token_counter', 0) if metadata1 else 'N/A'}")
    
    # Second run with different input (should start fresh)
    print("\n--- Second run with different input ---")
    ret2, stdout2, stderr2 = run_benchmark(pp=24, tg=4, chunksize=1, timeout=120)
    
    if "starting fresh" in stdout2.lower() or "Input tokens don't match" in stdout2:
        print("SUCCESS: Fresh start detected when input changed")
    
    return ret2 == 0


def main():
    if not MODEL_PATH or not CONFIG_PATH:
        print("Error: Please set MLLM_MODEL_PATH and MLLM_CONFIG_PATH environment variables")
        print("Example:")
        print("  export MLLM_MODEL_PATH=/path/to/model.bin")
        print("  export MLLM_CONFIG_PATH=/path/to/config.json")
        sys.exit(1)
    
    if not BENCHMARK_BIN.exists():
        print(f"Error: Benchmark binary not found at {BENCHMARK_BIN}")
        print("Please build the project first.")
        sys.exit(1)
    
    print(f"Using model: {MODEL_PATH}")
    print(f"Using config: {CONFIG_PATH}")
    print(f"Test cache dir: {TEST_CACHE_DIR}")
    
    results = {}
    
    # Run tests
    try:
        results['normal'] = test_normal_execution()
    except Exception as e:
        print(f"Normal execution test failed with exception: {e}")
        results['normal'] = False
    
    try:
        results['interrupt_resume'] = test_interruption_and_resume()
    except Exception as e:
        print(f"Interrupt/resume test failed with exception: {e}")
        results['interrupt_resume'] = False
    
    try:
        results['fresh_start'] = test_fresh_start_after_completion()
    except Exception as e:
        print(f"Fresh start test failed with exception: {e}")
        results['fresh_start'] = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name}: {status}")
    
    all_passed = all(results.values())
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
