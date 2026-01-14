# AGENTS.md

Guidance for coding agents operating in this repo. Defer to CLAUDE.md if conflicts occur.

## Current Work: Fault-Tolerant LLM Inference

**Goal**: Make inference engine resistant to power loss. Save/recover states (KV cache, etc.) to minimize redundant work on restart.

**Main files**:
- `mllm/mllm/models/qwen3_i/` - Model with state persistence and task-based scheduler
  - `grid_scheduler.hpp/cpp` - Task-based scheduler (see `docs/task.md` for architecture)
  - `grid_task.hpp/cpp` - Task definitions (LoadParam, LoadKV, LoadH, Compute)
  - `generation_state.hpp/cpp` - State persistence with watermark mechanism
  - `modeling_qwen3_i.hpp/cpp` - Model implementation using scheduler
- `examples/qwen3/main.cpp` - Test application

## Commands

### Build
```bash
./scripts/remote_build.sh
```

### Test
Simple generation test:
```bash
./scripts/test_simple_generation.sh
```

Manual test with recovery validation:
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

### Naming Conventions
- Namespaces: snake_case (e.g., `mllm::models::qwen3_i`)
- Types/classes: PascalCase (e.g., `GenerationState`, `Tensor`)
- Functions: camelCase (e.g., `loadMetadata`, `updateKV`)
- Variables/parameters: snake_case (e.g., `layer_idx`, `token_id`)
- Members: snake_case_ with trailing underscore (e.g., `max_length_`)
- Constants/enums: kPascalCase (e.g., `kFloat32`)

### Types & APIs
- Integers: use fixed-width (`int32_t`, `int64_t`)
- Modern C++: prefer `std::span`, `std::optional`, `std::filesystem`
- Type aliases: use `using` instead of `typedef`

### Error Handling
- Use `MLLM_RT_ASSERT` / `MLLM_RT_ASSERT_EQ` for checks
- Use `MLLM_ERROR_EXIT` for unrecoverable errors
- Avoid empty catch blocks and defensive complexity

## State Persistence Rules
- State files are deleted between tests; do not add migrations or versioning
- Keep state format simple; minimal metadata only

## Safety & Hygiene
- Do not modify vendored/third_party unless requested
- Unless necessary, do not modify code other than examples/qwen3/ and mllm/mllm/models/qwen3_i. if you need to do so, confirm it with the user.
- Do not commit generated build outputs or large model files
- Avoid refactors during bug fixes; fix minimally
- Prefer existing libraries over new dependencies
- When in doubt, re-read CLAUDE.md
