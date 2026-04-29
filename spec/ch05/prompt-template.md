# Chapter 5 -- LLM Prompt Template: The Building Blocks

Copy and paste the prompt below into your LLM of choice, or use it as a
reference while implementing by hand. This builds on the project from
Chapters 1–4.

---

## The Prompt

```
I am building an LLM inference engine called "rvllm". This is Chapter 5.
I have an existing project from Chapters 1–4 with model loading and tokenizer
already working (from Chapter 4). The `inspect` subcommand prints model config,
weight names/shapes, parameter count, and tokenizer round-trip results.

Now implement GPT-2's non-attention building blocks and verify them with a
partial forward pass.

LANGUAGE/FRAMEWORK: [Rust with candle / Python with PyTorch / your choice]
An ML tensor library is required for matrix operations.

=== WHAT TO CREATE / MODIFY ===

Add or modify these files in the EXISTING project:

  NEW FILES:
    src/model/gpt2/embedding.rs    <-- Embedding lookup layer
    src/model/gpt2/layer_norm.rs   <-- LayerNorm implementation
    src/model/gpt2/linear.rs       <-- Linear layer with Conv1D transpose
    src/model/gpt2/mlp.rs          <-- MLP block (c_fc → GELU → c_proj)

  MODIFY:
    src/model/gpt2.rs (or mod.rs)  <-- Integrate new layer types, expose them
    src/main.rs                    <-- Extend `inspect` to show layer info + partial forward pass

  KEEP UNCHANGED:
    src/types.rs, src/error.rs, src/config.rs
    src/tokenizer/ (already working from ch04)
    src/engine/, src/scheduler/, src/memory/, src/api/ (still stubs)
    examples/ch01_*, examples/ch02_*

=== LAYER IMPLEMENTATIONS ===

1. EMBEDDING LOOKUP
   weight: [rows, 768]
   forward(indices) → weight[indices]  → [B, T, 768]

   Two instances: wte [50257, 768] for tokens, wpe [1024, 768] for positions.
   Combined: hidden = wte(token_ids) + wpe(position_ids)

2. LAYER NORMALIZATION
   weight: [768], bias: [768], epsilon: 1e-5
   forward(x):
     mean = mean(x, dim=-1, keepdim=True)
     var  = variance(x, dim=-1, keepdim=True)
     x_norm = (x - mean) / sqrt(var + epsilon)
     return x_norm * weight + bias

3. LINEAR LAYER
   weight: [out_features, in_features] (AFTER transpose from Conv1D format)
   bias: [out_features]
   forward(x) = x @ weight^T + bias

   CRITICAL: GPT-2 Conv1D weights are stored as [in, out] on disk.
   Standard linear expects [out, in]. TRANSPOSE when loading.

4. MLP BLOCK
   c_fc:   Linear [768 → 3072]
   c_proj: Linear [3072 → 768]
   forward(x):
     hidden = c_fc(x)          → [B, T, 3072]
     hidden = gelu(hidden)     → [B, T, 3072]
     output = c_proj(hidden)   → [B, T, 768]
     return output

   GELU (exact, erf-based):
     gelu(x) = x * 0.5 * (1.0 + erf(x / sqrt(2.0)))

=== WEIGHT NAMES FOR BLOCK 0 ===

LayerNorm (pre-attention):
  transformer.h.0.ln_1.weight             [768]
  transformer.h.0.ln_1.bias               [768]

LayerNorm (pre-MLP):
  transformer.h.0.ln_2.weight             [768]
  transformer.h.0.ln_2.bias               [768]

MLP:
  transformer.h.0.mlp.c_fc.weight         [768, 3072]   *** Conv1D ***
  transformer.h.0.mlp.c_fc.bias           [3072]
  transformer.h.0.mlp.c_proj.weight       [3072, 768]   *** Conv1D ***
  transformer.h.0.mlp.c_proj.bias         [768]

=== PARTIAL FORWARD PASS (for verification) ===

Run this on "What is AI?" and print tensor stats at each step:

1. Tokenize "What is AI?" → token_ids (should be ~4 tokens)
2. position_ids = [0, 1, 2, ..., len(token_ids)-1]
3. token_emb = wte(token_ids)         → print shape, mean, std
4. pos_emb = wpe(position_ids)        → print shape, mean, std
5. hidden = token_emb + pos_emb       → print shape, mean, std
6. normed = ln_1_block0(hidden)       → print shape, mean, std
7. mlp_out = mlp_block0(normed)       → print shape, mean, std

Print each step as:
  step_name: shape=[dims], mean=X.XXXX, std=X.XXXX

=== EXTENDED INSPECT OUTPUT ===

The `inspect` subcommand should now print:

1. [from ch04] Model loading info, config, weights summary, tokenizer check
2. [NEW] Layer summary: "Loaded: 2 embeddings, 12 LayerNorms (24 total), 12 MLP blocks"
3. [NEW] Embedding shapes: "Token embedding: [50257, 768], Position embedding: [1024, 768]"
4. [NEW] Partial forward pass with tensor stats (as above)

=== WHAT SUCCESS LOOKS LIKE ===

- All weights load correctly (148 tensors)
- Embedding lookup produces correct shapes
- LayerNorm output has mean ≈ 0 (normalized)
- MLP output has reasonable values (no NaN, non-zero std)
- Conv1D weights are transposed correctly (MLP output is not garbage)

=== COMMON FAILURE MODES ===

| Symptom                        | Likely cause                              |
|--------------------------------|-------------------------------------------|
| NaN in LayerNorm output        | Missing epsilon in sqrt(var + eps)        |
| All zeros from MLP             | Conv1D weights not transposed             |
| Shape mismatch at c_fc         | Using wrong dimension (768 vs 3072)       |
| MLP output all same value      | GELU not applied (linear only)            |
| Wrong embedding shape          | Indexing into wrong dimension              |

=== WHAT TO PRODUCE ===

Produce the complete set of NEW or MODIFIED source files. For each file,
show the full path relative to the project root and the complete file contents.

Do NOT recreate types.rs, error.rs, config.rs, or tokenizer files.
USE the existing types and error enum — build on them.

The `generate` subcommand remains stubbed. Only `inspect` is functional.
```

---

## Suggested Frameworks by Language

| Language | Tensor library | Tokenizer | Weight loader |
|----------|---------------|-----------|--------------|
| Rust     | `candle-core`, `candle-nn` | `tokenizers` crate | `candle-core` (built-in) |
| Python   | `torch` | `tokenizers` | `safetensors` |
| Python (minimal) | `numpy` | `tokenizers` | `safetensors` |
