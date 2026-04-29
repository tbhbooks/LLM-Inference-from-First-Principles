# Chapter 5 -- Interface Specification: The Building Blocks

This is a language-agnostic specification. It defines the contracts, shapes,
and behaviors for GPT-2's non-attention layers: Embedding, LayerNorm, MLP.

---

## 1. Overview

This chapter implements the foundational layers of GPT-2:

- **Embedding lookup** (token + position)
- **LayerNorm** (pre-norm style)
- **Linear** (with Conv1D transpose handling)
- **MLP** (up-project → GELU → down-project)

The goal is to load GPT-2 weights, wire up these layers, and verify them by
running a partial forward pass: embed "What is AI?" → LayerNorm → MLP block 0.
Print tensor statistics at each stage to prove correctness.

---

## 2. Dependencies on Chapter 4

This chapter builds on Chapter 4's model loader and tokenizer:

- HuggingFace Hub download / cache (already working)
- SafeTensors weight loading (already working)
- HuggingFace tokenizer (already working)
- `inspect` subcommand (extended with layer info)

---

## 3. Layer Contracts

### 3.1 Embedding Lookup

```
Input:  indices  [B, T]  (integer IDs, 0-based)
Weight: weight   [vocab_size, n_embd]  (or [block_size, n_embd] for position)

Step 1: output = weight[indices]   → [B, T, n_embd]

Output: [B, T, n_embd]
```

Two embedding tables:
- `wte` (token embedding): shape [50257, 768], indexed by token IDs
- `wpe` (position embedding): shape [1024, 768], indexed by position indices

Combined embedding: `hidden = wte(token_ids) + wpe(position_ids)`

### 3.2 Layer Normalization

```
Input:  x [B, T, 768]

Step 1: mean  = mean(x, dim=-1, keepdim=True)         → [B, T, 1]
Step 2: var   = variance(x, dim=-1, keepdim=True)      → [B, T, 1]
Step 3: x_norm = (x - mean) / sqrt(var + epsilon)      → [B, T, 768]
Step 4: output = x_norm * weight + bias                 → [B, T, 768]

Output: [B, T, 768]
```

Parameters: `weight` [768], `bias` [768], `epsilon` = 1e-5.

**Variance note:** Use the population variance (divide by N, not N-1). Most ML
frameworks' `var()` defaults to this for LayerNorm.

### 3.3 Linear Layer (with Conv1D handling)

```
Input:  x      [B, T, in_features]
Weight: weight [out_features, in_features]  (after transpose if Conv1D)
Bias:   bias   [out_features]

Step 1: output = x @ weight^T + bias   → [B, T, out_features]

Output: [B, T, out_features]
```

**Conv1D transpose rule:** GPT-2 stores weights as `[in, out]`. Standard linear
layers expect `[out, in]`. Transpose on load.

### 3.4 MLP

```
Input:  x [B, T, 768]

Step 1: hidden = linear(x, c_fc)            → [B, T, 3072]
Step 2: hidden = gelu(hidden)               → [B, T, 3072]
Step 3: output = linear(hidden, c_proj)     → [B, T, 768]

Output: [B, T, 768]
```

**GELU activation:** Use the **exact** (erf-based) form, not the tanh approximation.
```
gelu(x) = x * 0.5 * (1.0 + erf(x / sqrt(2.0)))
```

### 3.5 MLP Weight Names (Block 0)

| Weight name                          | Shape on disk  | After transpose |
|--------------------------------------|---------------|-----------------|
| `transformer.h.0.mlp.c_fc.weight`   | [768, 3072]   | [3072, 768]     |
| `transformer.h.0.mlp.c_fc.bias`     | [3072]        | —               |
| `transformer.h.0.mlp.c_proj.weight` | [3072, 768]   | [768, 3072]     |
| `transformer.h.0.mlp.c_proj.bias`   | [768]         | —               |

---

## 4. Program Behavior

### 4.1 `inspect` Subcommand (Extended)

The `inspect` subcommand from ch04 is extended to show layer details:

```
rvllm inspect --model openai-community/gpt2
```

Must print:
1. Everything from ch04 (loading, config, weight summary, tokenizer check)
2. **NEW:** Layer component summary showing loaded layer types
3. **NEW:** Embedding verification (shape confirmation)
4. **NEW:** Partial forward pass: embed → LayerNorm → MLP on "What is AI?"

### 4.2 Partial Forward Pass

Run this sequence and print tensor stats at each step:

```
1. Tokenize "What is AI?" → token_ids
2. Create position_ids [0, 1, 2, ..., len-1]
3. token_emb = wte(token_ids)           → print shape, mean, std
4. pos_emb = wpe(position_ids)          → print shape, mean, std
5. hidden = token_emb + pos_emb         → print shape, mean, std
6. normed = ln_1_block0(hidden)         → print shape, mean, std
7. mlp_out = mlp_block0(normed)         → print shape, mean, std
```

### 4.3 Tensor Stats Format

For each tensor checkpoint, print:
```
  step_name: shape=[dims], mean=X.XXXX, std=X.XXXX
```

Example:
```
  token_emb: shape=[1, 4, 768], mean=-0.0012, std=0.0345
```

---

## 5. Output Format

See `expected-output.txt` for the full template.

---

## 6. Validation Summary

| Test | What it checks |
|------|---------------|
| Layers loaded | Output mentions layer components (embedding, layernorm, mlp) |
| Embedding shape | Token embedding shape [50257, 768] appears |
| LayerNorm stats | LayerNorm output has mean ≈ 0 (within ±0.1) |
| MLP output | MLP output tensor has non-zero std |
| No NaN | No NaN or Inf values in any tensor stats |
| Conv1D evidence | Output mentions transpose or Conv1D handling |
| Exit 0 | Process exits successfully |
