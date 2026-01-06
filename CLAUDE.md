# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current Work: Fault-Tolerant LLM Inference

**Goal**: Make inference engine resistant to power loss. Save/recover states (KV cache, etc.) to minimize redundant work on restart.

**Main files**:
- `mllm/models/qwen3_i/` - Model with state persistence (generation_state.hpp/cpp)
- `examples/qwen3/main.cpp` - Test application

**Build** (cross-compilation):
```bash
./scripts/remote_build.sh
```

**Test**:
```bash
./build/bin/mllm-qwen3-runner \
  -m /home/xiaolong/llms/Qwen3-0.6B-w4a32kai/model.mllm \
  -c /home/xiaolong/llms/Qwen3-0.6B-w4a32kai/config.json \
  -t /home/xiaolong/llms/Qwen3-0.6B-w4a32kai/tokenizer.json \
  -mv v2 --state_path ./data/qwen3-runner -cs 32
```
Use Ctrl-C to simulate power loss, then rerun. Second run should skip already-completed prefill.

## Coding Style Preferences

This is an experimental/academic project, not user-facing software. Optimize for developer productivity:

- **Short code is better**: Short code implies simple logic and is faster to review. Avoid verbose abstractions.
- **No excessive error handling**: Skip edge cases that won't happen in practice. Basic asserts are fine; elaborate recovery logic is not.
- **No backward-compatibility code**: State files, configs, etc. are deleted between tests. Don't add version checks, optional field handling, or migration logic.
- **No polished UIs**: Command-line tools with minimal output are sufficient.
- **Main purpose first**: Implement the core functionality. Don't add "nice-to-have" features, configurability, or defensive code unless explicitly needed.
- **Maintainability through simplicity**: Fewer lines = fewer bugs = easier to understand and modify.

## Architecture

**Core Framework** (`mllm/`):
- `core/` - Tensor operations, memory management
- `nn/` - Neural network modules with Pythonic API
- `backends/` - CPU, CUDA, OpenCL, QNN (NPU), Ascend
- `models/` - Model implementations (Qwen, LLaMA, DeepSeek, etc.)
- `engine/` - Inference engine, prefix caching

**Key Patterns**:
- Models inherit from `nn::Module` with `forward()` method
- Eager execution by default

## Code Standards

- C++20, UTF-8, LF line endings
- Google C++ Style Guide
- `TODO:` or `FIXME:` for annotations
