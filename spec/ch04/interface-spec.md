# Chapter 4 -- Interface Specification: Downloading a Brain

This is a language-agnostic specification. It defines the contracts for
loading GPT-2 124M weights and tokenizer from HuggingFace Hub.

---

## 1. Overview

This chapter focuses on **loading and inspecting** GPT-2 124M:

- Download model files from HuggingFace Hub (safetensors + tokenizer)
- Load and parse model configuration
- Enumerate weight names and shapes
- Count total parameters
- Load and test the tokenizer (encode/decode round-trip)
- Wire the `inspect` subcommand; `generate` stays stubbed

No forward pass, no layers, no generation. Just getting the data into memory
and proving it's correct.

---

## 2. Configuration Constants

| Constant             | Value  | Notes |
|----------------------|--------|-------|
| `vocab_size`         | 50257  | BPE vocabulary (50000 merges + 256 bytes + 1 EOS) |
| `n_embd`             | 768    | Hidden dimension / embedding dimension |
| `n_head`             | 12     | Number of attention heads |
| `n_layer`            | 12     | Number of transformer blocks |
| `block_size`         | 1024   | Maximum sequence length (context window) |
| `head_dim`           | 64     | `n_embd / n_head` |
| `intermediate_size`  | 3072   | MLP hidden dim: `4 * n_embd` |
| `layer_norm_epsilon` | 1e-5   | LayerNorm stability constant |
| `eos_token_id`       | 50256  | `<|endoftext|>` token |

---

## 3. Model Source

| Field         | Value |
|---------------|-------|
| HuggingFace model ID | `openai-community/gpt2` |
| Required files | `model.safetensors`, `tokenizer.json`, `config.json` |
| Format | SafeTensors (preferred) or PyTorch `.bin` |
| Dtype | float32 |
| Total parameters | ~124M |

---

## 4. Weight Names and Tensor Shapes

### 4.1 Embedding Weights

| Weight name                 | Shape          | Component |
|-----------------------------|----------------|-----------|
| `transformer.wte.weight`    | [50257, 768]   | Token embedding |
| `transformer.wpe.weight`    | [1024, 768]    | Position embedding |

### 4.2 Per-Layer Weights (i = 0..11)

| Weight name                                  | Shape on disk     | Component           |
|----------------------------------------------|-------------------|---------------------|
| `transformer.h.{i}.ln_1.weight`              | [768]             | Pre-attention LayerNorm |
| `transformer.h.{i}.ln_1.bias`                | [768]             | Pre-attention LayerNorm |
| `transformer.h.{i}.attn.c_attn.weight`       | [768, 2304]       | QKV projection (Conv1D) |
| `transformer.h.{i}.attn.c_attn.bias`         | [2304]            | QKV projection bias |
| `transformer.h.{i}.attn.c_proj.weight`       | [768, 768]        | Attention output projection (Conv1D) |
| `transformer.h.{i}.attn.c_proj.bias`         | [768]             | Attention output projection bias |
| `transformer.h.{i}.ln_2.weight`              | [768]             | Pre-MLP LayerNorm |
| `transformer.h.{i}.ln_2.bias`                | [768]             | Pre-MLP LayerNorm |
| `transformer.h.{i}.mlp.c_fc.weight`          | [768, 3072]       | MLP up-projection (Conv1D) |
| `transformer.h.{i}.mlp.c_fc.bias`            | [3072]            | MLP up-projection bias |
| `transformer.h.{i}.mlp.c_proj.weight`        | [3072, 768]       | MLP down-projection (Conv1D) |
| `transformer.h.{i}.mlp.c_proj.bias`          | [768]             | MLP down-projection bias |

### 4.3 Final Weights

| Weight name                 | Shape          | Component |
|-----------------------------|----------------|-----------|
| `transformer.ln_f.weight`   | [768]          | Final LayerNorm |
| `transformer.ln_f.bias`     | [768]          | Final LayerNorm |
| `lm_head.weight`            | tied with `transformer.wte.weight` | LM head projection |

### 4.4 Conv1D Note

All weight matrices marked **Conv1D** are stored as `[in_features, out_features]`.
Standard linear layers expect `[out_features, in_features]`. This will matter in
Chapter 5 when we implement the layers — for now, just note the shapes as-is.

### 4.5 Weight Count

Total unique weight tensors: 148
- 2 embeddings
- 12 layers × 12 weights/layer = 144
- 2 final LayerNorm
- (lm_head is tied to wte, not counted separately)

Total parameters: ~124M

---

## 5. Tokenizer Contract

### 5.1 TokenizerBackend Trait

The tokenizer must implement:
```
encode(text: string) -> list<TokenId>
decode(ids: list<TokenId>) -> string
eos_token_id() -> TokenId    // returns 50256
```

### 5.2 Round-Trip Test

```
text = "What is AI?"
ids = tokenizer.encode(text)
decoded = tokenizer.decode(ids)
assert decoded == text   // or very close (whitespace may differ)
```

### 5.3 Running Example

"What is AI?" should tokenize to approximately 4 tokens.
The token IDs `[2061, 318, 9552, 30]` correspond to `["What", " is", " AI", "?"]`.

---

## 6. Program Behavior

### 6.1 CLI Command

```
rvllm inspect --model openai-community/gpt2
```

### 6.2 Output Sections

1. **Loading indicator** — Show model ID, download/cache status
2. **Model config** — layers, hidden dim, heads, vocab size
3. **Weight summary** — tensor count, total parameters, key weight shapes
4. **Tokenizer check** — round-trip test with "What is AI?"
5. **`generate` stub** — if user runs `generate`, print "not yet implemented"

---

## 7. Validation Summary

| Test | What it checks |
|------|---------------|
| Loading indicator | Output shows loading/downloaded/cached |
| Tokenizer mentioned | Output mentions tokenizer or vocab |
| Model dimensions | Output contains 768 and 12 (dim and layers/heads) |
| Weight count | Output mentions 148 or weight count |
| Parameter count | Output mentions ~124M or parameter count |
| Tokenizer round-trip | "What is AI?" appears in tokenizer output section |
| Exit 0 | Process exits successfully |
