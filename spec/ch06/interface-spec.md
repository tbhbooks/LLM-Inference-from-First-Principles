# Chapter 6 -- Interface Specification: Where the Model Learns to Look Back

This is a language-agnostic specification. It defines the contracts for
CausalSelfAttention, TransformerBlock, and the full Gpt2Model forward pass.

---

## 1. Overview

This chapter completes the GPT-2 model by implementing:

- **CausalSelfAttention** — QKV projection, multi-head split, scaled dot-product, causal masking, KV cache
- **TransformerBlock** — Pre-norm residual (LN → Attention → Add → LN → MLP → Add)
- **Gpt2Model** — Full forward pass (embed → 12 blocks → final LN → LM head)
- **Model trait** — Implement the `Model` trait for Gpt2Model

The chapter runs a single forward pass on "What is AI?" and prints top-5
next-token predictions. No generation loop yet — that's Chapter 7.

---

## 2. Dependencies on Chapters 4–5

From Chapter 4: model loading, tokenizer, weight access
From Chapter 5: Embedding, LayerNorm, Linear, MLP

---

## 3. Layer Contracts

### 3.1 Causal Self-Attention

```
Input:  x [B, T, 768]
State:  kv_cache = Option<(k_cache, v_cache)>
        k_cache, v_cache shape: [B, 12, S_prev, 64]  (S_prev = cached sequence length)

Step 1: QKV projection
        qkv = linear(x, c_attn)              → [B, T, 2304]
        q   = qkv[:, :, 0:768]               → [B, T, 768]
        k   = qkv[:, :, 768:1536]            → [B, T, 768]
        v   = qkv[:, :, 1536:2304]           → [B, T, 768]

Step 2: Reshape to multi-head
        q = reshape(q, [B, T, 12, 64])
            .transpose(1, 2)                  → [B, 12, T, 64]
        k = reshape(k, [B, T, 12, 64])
            .transpose(1, 2)                  → [B, 12, T, 64]
        v = reshape(v, [B, T, 12, 64])
            .transpose(1, 2)                  → [B, 12, T, 64]

Step 3: KV cache update
        if kv_cache exists:
            k = concat(k_cache, k, dim=2)     → [B, 12, S_prev+T, 64]
            v = concat(v_cache, v, dim=2)     → [B, 12, S_prev+T, 64]
        store (k, v) as new kv_cache
        S_total = k.shape[2]

Step 4: Scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) / sqrt(64)    → [B, 12, T, S_total]

Step 5: Causal masking (only when T > 1, i.e., prefill)
        For query position i (absolute position = offset + i):
            mask[i, j] = True  if j <= offset + i
            mask[i, j] = False if j > offset + i
        Set masked positions to -infinity before softmax.
        When T = 1 (decode): skip masking (single query attends to all cached keys).

Step 6: Softmax + value aggregation
        attn_weights = softmax(scores, dim=-1)            → [B, 12, T, S_total]
        attn_output  = attn_weights @ v                   → [B, 12, T, 64]

Step 7: Reshape + output projection
        attn_output = attn_output.transpose(1, 2)
                        .reshape([B, T, 768])             → [B, T, 768]
        output = linear(attn_output, c_proj)              → [B, T, 768]

Output: [B, T, 768]
```

### 3.2 Attention Weight Names (per layer i)

| Weight name                            | Shape on disk | After transpose | Component |
|----------------------------------------|--------------|-----------------|-----------|
| `transformer.h.{i}.attn.c_attn.weight` | [768, 2304]  | [2304, 768]     | QKV projection (Conv1D) |
| `transformer.h.{i}.attn.c_attn.bias`   | [2304]       | —               | QKV projection bias |
| `transformer.h.{i}.attn.c_proj.weight` | [768, 768]   | [768, 768]      | Output projection (Conv1D) |
| `transformer.h.{i}.attn.c_proj.bias`   | [768]        | —               | Output projection bias |

### 3.3 Transformer Block (Pre-Norm Residual)

```
Input:  x [B, T, 768]

Step 1: attn_out = attention(layer_norm_1(x))     → [B, T, 768]
Step 2: x        = x + attn_out                    → [B, T, 768]  (residual)
Step 3: mlp_out  = mlp(layer_norm_2(x))            → [B, T, 768]
Step 4: x        = x + mlp_out                     → [B, T, 768]  (residual)

Output: x [B, T, 768]
```

**Critical:** Pre-norm means LayerNorm BEFORE the sublayer. The residual adds
the ORIGINAL input (not the normalized version).

### 3.4 Full Model Forward Pass

```
Input:  input_ids [B, T], position_offset (integer)

Step 1: hidden = embedding(input_ids, position_offset)    → [B, T, 768]
Step 2: for each block in blocks[0..11]:
            hidden = block.forward(hidden)                 → [B, T, 768]
Step 3: hidden = layer_norm_f(hidden)                      → [B, T, 768]
Step 4: logits = hidden @ wte_weight.T                     → [B, T, 50257]

Output: logits [B, T, 50257]
```

The LM head reuses the token embedding weight (weight tying).

---

## 4. Program Behavior

### 4.1 `inspect` Subcommand (Extended)

Extends ch05's `inspect` to include:
1. All ch04 + ch05 info (loading, layers, partial forward pass)
2. **NEW:** Full model loaded confirmation
3. **NEW:** Single forward pass on "What is AI?" → top-5 next-token predictions

### 4.2 Forward Pass Demo

```
1. Tokenize "What is AI?" → token_ids
2. Run full model forward: logits = model.forward(token_ids, pos_offset=0)
3. Take logits from last position: logits[0, -1, :]  → [50257]
4. Find top-5 token IDs by logit value
5. Decode each token ID to text
6. Print top-5 predictions with logit values
```

### 4.3 Top-5 Output Format

```
Top-5 next token predictions for "What is AI?":
  1. " The"     (logit: 12.34)
  2. " It"      (logit: 11.56)
  3. " A"       (logit: 10.89)
  4. "\n"       (logit: 10.45)
  5. " What"    (logit: 10.12)
```

The `generate` subcommand remains stubbed.

---

## 5. Correctness Criteria

1. **Logits shape:** `[1, T, 50257]` where T is the number of input tokens
2. **Top predictions plausible:** The top-5 tokens should be common English words/tokens
3. **No NaN/Inf:** Logits must be finite
4. **Weight tying:** LM head uses `wte.weight`, not a separate weight matrix
5. **Causal masking:** Each position only attends to itself and prior positions
6. **KV cache initialized:** Cache should be populated after the forward pass

---

## 6. Common Failure Modes

| Symptom | Likely cause |
|---------|-------------|
| All same prediction for every position | Causal mask not applied during prefill |
| NaN in logits | LayerNorm epsilon missing, or attention scores overflow |
| Shape mismatch at c_attn | Not splitting 2304 into 3 × 768 correctly |
| Random/nonsense top-5 predictions | Conv1D weights not transposed |
| Logits shape [1, T, 768] instead of [1, T, 50257] | Missing LM head projection |
| Crash after attention | Multi-head reshape dimensions wrong |

---

## 7. Validation Summary

| Test | What it checks |
|------|---------------|
| Model loaded | Output confirms full model is loaded and ready |
| Logits shape | Output mentions 50257 (vocab size) in context of logits/predictions |
| Top predictions exist | Output shows numbered predictions (1. through 5.) |
| Predictions plausible | At least one top-5 token is a common English word |
| No NaN | No NaN or Inf in output |
| Running example | "What is AI?" appears in output |
| Exit 0 | Process exits successfully |
