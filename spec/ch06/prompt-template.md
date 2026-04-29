# Chapter 6 -- LLM Prompt Template: Where the Model Learns to Look Back

Copy and paste the prompt below into your LLM of choice, or use it as a
reference while implementing by hand. This builds on the project from
Chapters 1–5.

---

## The Prompt

```
I am building an LLM inference engine called "rvllm". This is Chapter 6.
I have an existing project from Chapters 1–5 with:
- Model loading and tokenizer (ch04)
- Embedding, LayerNorm, Linear, MLP layers (ch05)
- The `inspect` subcommand shows layer info and partial forward pass stats

Now implement CausalSelfAttention, TransformerBlock, and the full Gpt2Model.
Run a single forward pass and print top-5 next-token predictions.

LANGUAGE/FRAMEWORK: [Rust with candle / Python with PyTorch / your choice]

=== WHAT TO CREATE / MODIFY ===

  NEW FILES:
    src/model/gpt2/attention.rs    <-- CausalSelfAttention with KV cache
    src/model/gpt2/block.rs        <-- TransformerBlock (pre-norm residual)
    src/model/gpt2/model.rs        <-- Gpt2Model: full forward pass, Model trait impl

  MODIFY:
    src/model/gpt2.rs (or mod.rs)  <-- Integrate new components
    src/main.rs                    <-- Extend `inspect` to show top-5 predictions

  KEEP UNCHANGED:
    src/model/gpt2/embedding.rs, layer_norm.rs, linear.rs, mlp.rs (from ch05)
    src/types.rs, src/error.rs, src/config.rs
    src/tokenizer/ (from ch04)
    src/engine/, src/scheduler/, src/memory/, src/api/ (still stubs)

=== CAUSAL SELF-ATTENTION ===

Input: x [B, T, 768]
State: KV cache (optional, initially empty)

1. QKV PROJECTION
   qkv = linear(x, c_attn)                   [B, T, 2304]
   Split into q, k, v each [B, T, 768]

2. MULTI-HEAD RESHAPE
   q, k, v: [B, T, 768] → [B, 12, T, 64]
   (reshape to [B, T, 12, 64] then transpose dims 1,2)

3. KV CACHE UPDATE
   If cache exists: k = concat(cached_k, k, dim=2)
                    v = concat(cached_v, v, dim=2)
   Store updated k, v as new cache
   S_total = k.shape[2]  (total sequence so far)

4. SCALED DOT-PRODUCT
   scores = (q @ k^T) / sqrt(64)              [B, 12, T, S_total]

5. CAUSAL MASK (only when T > 1)
   Position i can only attend to positions 0..=(offset+i)
   Set future positions to -infinity
   When T=1 (decode step): no mask needed

6. SOFTMAX + AGGREGATE
   attn_weights = softmax(scores, dim=-1)
   attn_output = attn_weights @ v              [B, 12, T, 64]

7. OUTPUT PROJECTION
   Reshape: [B, 12, T, 64] → [B, T, 768]
   output = linear(attn_output, c_proj)

=== TRANSFORMER BLOCK (pre-norm residual) ===

  x_attn = attention(layer_norm_1(x))
  x = x + x_attn                               # residual
  x_mlp = mlp(layer_norm_2(x))
  x = x + x_mlp                                # residual

CRITICAL: LayerNorm BEFORE sublayer. Residual adds ORIGINAL x, not normalized.

=== FULL MODEL FORWARD PASS ===

  hidden = wte(input_ids) + wpe(positions)      [B, T, 768]
  for block in blocks[0..11]:
      hidden = block(hidden)                     [B, T, 768]
  hidden = ln_f(hidden)                          [B, T, 768]
  logits = hidden @ wte.weight^T                 [B, T, 50257]

LM head is weight-tied with token embedding (no separate weight).

=== MODEL TRAIT ===

Implement the Model trait for Gpt2Model:
  forward(input_ids, position_offset) → logits [B, T, 50257]
  reset_cache() → clear all KV caches in all 12 blocks

=== TOP-5 PREDICTIONS ===

After the forward pass on "What is AI?":
1. Take logits at last position: logits[0, -1, :]  → [50257]
2. Find top-5 indices (highest logit values)
3. Decode each index to its token string
4. Print formatted:

Top-5 next token predictions for "What is AI?":
  1. " The"     (logit: 12.34)
  2. " It"      (logit: 11.56)
  ...

=== ATTENTION WEIGHT NAMES ===

Per layer i:
  transformer.h.{i}.attn.c_attn.weight   [768, 2304]  *** Conv1D ***
  transformer.h.{i}.attn.c_attn.bias     [2304]
  transformer.h.{i}.attn.c_proj.weight   [768, 768]   *** Conv1D ***
  transformer.h.{i}.attn.c_proj.bias     [768]

Remember: Conv1D weights need transpose for standard linear layers.

=== WHAT SUCCESS LOOKS LIKE ===

- Full model loads all 148 weights
- Forward pass produces logits shape [1, T, 50257]
- Top-5 predictions are common English words (e.g., "The", "It", "A", newline)
- No NaN or Inf in logits
- KV cache is populated after the forward pass

=== COMMON FAILURE MODES ===

| Symptom                               | Likely cause                        |
|---------------------------------------|-------------------------------------|
| Same prediction for every position    | Causal mask not applied             |
| NaN in logits                         | Attention overflow or missing eps   |
| Shape mismatch at c_attn             | Not splitting 2304 into 3 × 768    |
| Random top-5 predictions             | Conv1D weights not transposed       |
| Logits shape [B, T, 768]             | Missing LM head (wte.weight^T)     |

=== WHAT TO PRODUCE ===

Produce the complete set of NEW or MODIFIED source files.
Do NOT recreate ch04/ch05 files that are unchanged.

The `generate` subcommand remains stubbed. Only `inspect` is functional.
```

---

## Suggested Frameworks by Language

| Language | Tensor library | Notes |
|----------|---------------|-------|
| Rust     | `candle-core`, `candle-nn` | Use `Tensor::matmul`, `softmax` |
| Python   | `torch` | Use `torch.nn.functional.scaled_dot_product_attention` or manual |
| Python (minimal) | `numpy` | Manual softmax, careful with broadcasting |
