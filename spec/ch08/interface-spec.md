# Chapter 8 -- Interface Specification: Fit and Finish

This is a language-agnostic specification. It defines the contracts for
CLI polish, timing breakdown, memory estimates, and the MVP milestone.

---

## 1. Overview

This chapter polishes the CLI into a finished MVP:

- **Timing breakdown** — Separate prefill time vs decode time
- **KV cache memory estimate** — Report memory consumed by the KV cache
- **Polished `inspect`** — Consolidate ch04-06 info + quick prediction
- **Structured logging** — Use a logging framework (e.g., tracing) instead of prints
- **Better error messages** — Informative errors for common problems

---

## 2. Dependencies on Chapters 4–7

From Chapters 4–7: complete working GPT-2 pipeline with `inspect` and `generate`.

---

## 3. New Features

### 3.1 Timing Breakdown

During generation, measure and report separately:

```
Prefill time:  time to process all prompt tokens (one forward pass)
Decode time:   time for all decode steps (one forward pass per step)
Total time:    prefill_time + decode_time
```

Report format:
```
--- Stats ---
Tokens generated: N
Prefill: X.XXs (T prompt tokens)
Decode:  X.XXs (N tokens, X.XX tokens/sec)
Total:   X.XXs
```

### 3.2 KV Cache Memory Estimate

After generation (or in `inspect`), estimate and print the KV cache memory:

```
KV cache per token per layer = 2 * head_dim * n_head * sizeof(float32)
                             = 2 * 64 * 12 * 4
                             = 6144 bytes = 6 KB

Total KV cache = n_layer * seq_len * 6144
               = 12 * seq_len * 6 KB

For seq_len = 206 (6 prompt + 200 generated):
  Total = 12 * 206 * 6 KB ≈ 14.8 MB
```

Print in the stats section:
```
KV cache: X.X MB (N positions × 12 layers × 6 KB/pos/layer)
```

### 3.3 Polished `inspect` Subcommand

The `inspect` subcommand should consolidate all information:

```
rvllm inspect --model openai-community/gpt2
```

Output:
1. Model loading info (from ch04)
2. Configuration summary
3. Weight summary (count, total parameters)
4. Layer summary (from ch05)
5. Tokenizer check with round-trip (from ch04)
6. Quick forward pass: top-5 predictions for "What is AI?" (from ch06)
7. Parameter count and memory footprint estimate

### 3.4 Error Messages

Provide helpful errors for common problems:
- Model not found on HuggingFace → suggest checking the model ID
- Network error → suggest checking internet connection
- Out of memory → show model memory requirements
- Missing safetensors file → suggest the model format

### 3.5 Logging

Replace raw print statements with structured logging:
- Use the language's standard logging framework
- Log levels: INFO for user-visible output, DEBUG for internal details
- Loading steps logged at INFO
- Tensor stats and internal details at DEBUG

---

## 4. Output Formats

### 4.1 `generate` Output (Updated)

```
Loading model: openai-community/gpt2
  Tokenizer loaded (vocab_size: 50257)
  Model loaded (12 layers, 768 dim, 12 heads, ~124M params)

Prompt: "The future of artificial intelligence is"
Prompt tokens: 6
Generating up to 200 tokens...

--- Generated Text ---
The future of artificial intelligence is [... continuation ...]
--- End ---

--- Stats ---
Tokens generated: N
Prefill: X.XXs (6 prompt tokens)
Decode:  X.XXs (N tokens, X.XX tokens/sec)
Total:   X.XXs
KV cache: X.X MB (N positions × 12 layers × 6 KB/pos/layer)
```

### 4.2 `inspect` Output (Polished)

```
=== Model: openai-community/gpt2 ===

Config:
  Layers: 12, Hidden dim: 768, Heads: 12, Vocab: 50257
  Context window: 1024, Head dim: 64

Weights:
  148 tensors, ~124M parameters
  Memory: ~497 MB (float32)

Layers:
  2 embeddings, 24 LayerNorms, 12 MLPs, 12 attention blocks

Tokenizer:
  "What is AI?" → [tokens] → "What is AI?" ✓

Quick prediction (forward pass on "What is AI?"):
  1. "token"  (logit: XX.XX)
  2. "token"  (logit: XX.XX)
  3. "token"  (logit: XX.XX)
  4. "token"  (logit: XX.XX)
  5. "token"  (logit: XX.XX)
```

---

## 5. Validation Summary

| Test | What it checks |
|------|---------------|
| Prefill timing | Output reports prefill time separately |
| Decode timing | Output reports decode time separately |
| KV cache mention | Output mentions KV cache memory estimate |
| Inspect works | `inspect` subcommand runs and shows model info |
| Param count | Output mentions ~124M or parameter count |
| Memory estimate | Output mentions memory in MB or GB |
