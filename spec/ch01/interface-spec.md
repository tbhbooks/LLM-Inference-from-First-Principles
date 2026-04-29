# Chapter 1: Interface Specification

## Overview

This program computes KV cache memory requirements for transformer models and displays how memory constrains concurrent inference. No model loading, no GPU — pure arithmetic.

## Data Types

### ModelConfig

A structure holding transformer architecture parameters that determine KV cache size.

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Human-readable model name |
| `num_layers` | int | Number of transformer layers |
| `num_heads` | int | Number of KV attention heads (may differ from query heads in GQA) |
| `head_dim` | int | Dimension of each attention head |
| `dtype_bytes` | int | Bytes per element: 2 for float16/bfloat16, 4 for float32 |

### Methods on ModelConfig

Each method builds on the previous, from smallest unit to largest.

#### `kv_vector_bytes() -> int`

Bytes for ONE key (or value) vector for ONE token in ONE layer.

```
result = num_heads * head_dim * dtype_bytes
```

#### `kv_per_token_per_layer() -> int`

Bytes for the K+V pair for ONE token in ONE layer. Factor of 2 because we store both Key and Value.

```
result = 2 * kv_vector_bytes()
```

#### `kv_per_token() -> int`

Bytes for the K+V cache for ONE token across ALL layers.

```
result = kv_per_token_per_layer() * num_layers
```

#### `kv_for_sequence(seq_len: int) -> int`

Bytes for the full KV cache of a sequence of `seq_len` tokens.

```
result = kv_per_token() * seq_len
```

#### `max_concurrent_sequences(seq_len: int, memory_bytes: int) -> int`

How many concurrent sequences of `seq_len` tokens fit in `memory_bytes`. Uses integer division (floor).

```
per_seq = kv_for_sequence(seq_len)
if per_seq == 0: return 0
result = memory_bytes / per_seq   (integer division, floor)
```

## Helper Functions

### `format_bytes(bytes: int) -> string`

Converts a byte count to human-readable form using 1024-based (binary) units.

**Rules:**
- If >= 1 GB (1024^3 = 1,073,741,824): format as `X.XX GB`
- Else if >= 1 MB (1024^2 = 1,048,576): format as `X.XX MB`
- Else if >= 1 KB (1024): format as `X.XX KB`
- Else: format as `X B`
- Always 2 decimal places for KB/MB/GB

**Examples:**
| Input (bytes) | Output |
|---------------|--------|
| 1536 | `1.50 KB` |
| 3072 | `3.00 KB` |
| 16384 | `16.00 KB` |
| 32768 | `32.00 KB` |
| 36864 | `36.00 KB` |
| 524288 | `512.00 KB` |
| 2621440 | `2.50 MB` |
| 37748736 | `36.00 MB` |
| 150994944 | `144.00 MB` |
| 536870912 | `512.00 MB` |
| 2147483648 | `2.00 GB` |
| 2684354560 | `2.50 GB` |
| 10737418240 | `10.00 GB` |

### `separator()`

Prints a line of 78 `=` characters.

### `section(title: string)`

Prints a blank line, then separator, then `  {title}` (two-space indent), then separator, then a blank line.

## Preset Models

The `preset_models()` factory returns exactly these three configurations:

| name | num_layers | num_heads | head_dim | dtype_bytes |
|------|-----------|-----------|----------|-------------|
| `GPT-2 (124M)` | 12 | 12 | 64 | 2 |
| `LLaMA-7B` | 32 | 32 | 128 | 2 |
| `LLaMA-70B` | 80 | 64 | 128 | 2 |

## Derived Values Reference

All values below are deterministic from the formulas above.

### Per-token KV cache sizes

| Model | kv_vector_bytes | kv_per_token_per_layer | kv_per_token |
|-------|----------------|----------------------|-------------|
| GPT-2 (124M) | 1,536 | 3,072 | 36,864 |
| LLaMA-7B | 8,192 | 16,384 | 524,288 |
| LLaMA-70B | 16,384 | 32,768 | 2,621,440 |

### Per-sequence KV cache sizes

| Model | @ 1024 tokens | @ 4096 tokens |
|-------|--------------|--------------|
| GPT-2 (124M) | 37,748,736 | 150,994,944 |
| LLaMA-7B | 536,870,912 | 2,147,483,648 |
| LLaMA-70B | 2,684,354,560 | 10,737,418,240 |

### Max concurrent sequences (10 GB = 10,737,418,240 bytes)

| Model | @ 1024 tokens | @ 4096 tokens |
|-------|--------------|--------------|
| GPT-2 (124M) | 284 | 71 |
| LLaMA-7B | 20 | 5 |
| LLaMA-70B | 4 | 1 |

## Program Structure

The program prints 5 sections, each preceded by a `section()` call with the exact title below. After the final section, print a closing separator with "Chapter 1 complete. Next: vLLM architecture overview (ch02)".

### PART 1: KV Cache Memory — Where Does It All Go?

- Print introductory text (2 lines)
- Table with columns: Model (16-char left-aligned), Layers (8-char right), Heads (8-char right), HeadDim (8-char right), Per Token (14-char right), Per Tok/Layer (14-char right)
- Separator line: 72 dashes with 2-space indent
- One row per model

### PART 2: KV Cache vs Sequence Length

- Print introductory text (2 lines)
- Table with columns: Model (16-char left), 1024 tokens (16-char right), 4096 tokens (16-char right)
- Separator line: 50 dashes with 2-space indent
- One row per model

### PART 3: The Concurrency Ceiling (10 GB GPU Memory)

- Print "Given 10.00 GB of free GPU memory for KV cache," and "how many concurrent requests can we serve?"
- Table with columns: Model (16-char left), Max @ 1024 tokens (22-char right), Max @ 4096 tokens (22-char right)
- Separator line: 62 dashes with 2-space indent
- Values formatted as `{count} seq` (right-aligned within 18-char field, then ` seq`)
- After the table: two lines noting LLaMA-70B's limitation

### PART 4: The Memory Wall — Scaling Concurrent Requests

- Print introductory text (2 lines)
- `concurrent_counts = [1, 2, 4, 8, 16, 32]`, `seq_len = 1024`
- Header row: Model (16-char left), then for each count: `{n}req` in 9-char right field
- Separator line: 72 dashes with 2-space indent
- For each model, for each count:
  - Compute `needed = kv_for_sequence(1024) * n`
  - Compute `needed_gb = needed / 1024^3` (floating point)
  - If `n <= max_concurrent_sequences(1024, 10GB)`: print `{needed_gb:.1f}GB` right-aligned in 7-char field followed by `GB` (total 9 chars)
  - Else: print `OOM!!` right-aligned in 7-char field followed by `!!` (total 9 chars: `     OOM!!`)
- After the table: "OOM!! = Out of memory. The request cannot be served."

### PART 5: Prefill vs Decode — Two Very Different Phases

- Static text explaining:
  - PREFILL: parallel processing, compute-bound (FLOPS bottleneck)
  - DECODE: sequential generation, memory-bound (bandwidth bottleneck)
- ASCII box summarizing the key insight: "LLM serving is fundamentally a MEMORY problem."

## Constants

- `gpu_memory = 10 * 1024 * 1024 * 1024` (10 GB = 10,737,418,240 bytes)
- `concurrent_counts = [1, 2, 4, 8, 16, 32]`
- Sequence lengths tested: 1024 and 4096
