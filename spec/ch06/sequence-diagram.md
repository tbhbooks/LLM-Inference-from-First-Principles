# Chapter 6 -- Sequence Diagram: Where the Model Learns to Look Back

## Diagram 1: Full Forward Pass

```mermaid
sequenceDiagram
    participant Main
    participant Tok as Tokenizer
    participant Model as Gpt2Model
    participant Emb as Embeddings
    participant Block as Block[0..11]
    participant Attn as Attention
    participant Cache as KV Cache
    participant MLP_mod as MLP
    participant LMHead as LM Head

    Main->>Tok: encode("What is AI?")
    Tok-->>Main: token_ids (4 tokens)

    Main->>Model: forward(token_ids, pos_offset=0)

    Model->>Emb: wte(token_ids) → [1, 4, 768]
    Model->>Emb: wpe([0,1,2,3]) → [1, 4, 768]
    Note over Model: hidden = token_emb + pos_emb

    loop layer = 0..11
        Note over Block: Pre-norm residual
        Block->>Block: ln_1(hidden) → [1, 4, 768]

        Block->>Attn: forward(normed)
        Attn->>Attn: c_attn → qkv [1, 4, 2304]
        Attn->>Attn: Split → q, k, v [1, 12, 4, 64]
        Attn->>Cache: Store k, v (first forward — cache empty)
        Attn->>Attn: scores = q @ k^T / √64 → [1, 12, 4, 4]
        Attn->>Attn: Causal mask (lower triangular)
        Attn->>Attn: softmax → @ v → [1, 12, 4, 64]
        Attn->>Attn: reshape + c_proj → [1, 4, 768]
        Attn-->>Block: attention output

        Note over Block: hidden = hidden + attn_out
        Block->>Block: ln_2(hidden)
        Block->>MLP_mod: c_fc → GELU → c_proj → [1, 4, 768]
        Note over Block: hidden = hidden + mlp_out
    end

    Model->>Model: ln_f(hidden) → [1, 4, 768]
    Model->>LMHead: hidden @ wte.weight^T → [1, 4, 50257]
    Model-->>Main: logits [1, 4, 50257]

    Main->>Main: logits[0, 3, :] → top-5 indices
    Main->>Tok: decode each top-5 index
    Main->>Main: Print predictions
```
**Figure 6.4** — Complete forward pass through all 12 blocks. The KV cache is populated during this first (prefill) pass.

## Diagram 2: Attention Detail — Prefill vs Decode

```mermaid
sequenceDiagram
    participant Q as Query
    participant K as Key Cache
    participant V as Value Cache
    participant Scores

    Note over Q,Scores: === PREFILL (T=4 tokens) ===
    Q->>Scores: Q [1,12,4,64] @ K^T [1,12,64,4]
    Note over Scores: scores [1,12,4,4]
    Note over Scores: Apply causal mask:<br>pos 0→[0], pos 1→[0,1],<br>pos 2→[0,1,2], pos 3→[0,1,2,3]
    Scores->>Scores: softmax → @ V → [1,12,4,64]

    Note over Q,Scores: === DECODE (T=1, position 4) ===
    Note over K: cache has [1,12,4,64]
    Q->>K: new k [1,12,1,64] → concat → [1,12,5,64]
    Q->>V: new v [1,12,1,64] → concat → [1,12,5,64]
    Q->>Scores: Q [1,12,1,64] @ K^T [1,12,64,5]
    Note over Scores: scores [1,12,1,5]
    Note over Scores: No mask needed (T=1)
    Scores->>Scores: softmax → @ V → [1,12,1,64]
```
**Figure 6.5** — Prefill processes all tokens with causal masking; decode processes one token reading from the growing KV cache.
