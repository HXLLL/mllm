# Grid Scheduler Architecture

> **When to read**: Working on `mllm/models/qwen3_i/` scheduler code (grid_scheduler, grid_task, modeling_qwen3_i).

## Core Concept

LLM inference as a 2D grid: `[num_layers x num_chunks]`. Each cell computes one layer for one chunk of tokens.

```
         L0   L1   L2   ...  LN
Chunk0:  [0,0][1,0][2,0]...[N,0]  <- first chunk flows through all layers
Chunk1:  [0,1][1,1][2,1]...[N,1]  <- second chunk, needs Chunk0's KV for attention
```

## Task Types

- **LoadParamTask**: Load layer weights (per-layer)
- **LoadKVTask / LoadHTask**: Load from checkpoint (per-cell)
- **ComputeTask**: Run layer computation (per-cell)

## Dependencies

```
Compute[L][C] requires:
  1. LoadParam[L] done
  2. LoadH[L][C] done (hidden state from previous layer)
  3. All KV[L][0..C-1] ready (attention needs historical KV)
```

## Key Files

- `grid_scheduler.hpp/cpp`: Scheduler framework, wavefront management
- `grid_task.hpp/cpp`: Task definitions
- `generation_state.hpp/cpp`: Checkpoint save/load, watermark tracking
