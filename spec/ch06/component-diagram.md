# Chapter 6 -- Component Diagram: Where the Model Learns to Look Back

## Full GPT-2 Architecture

```mermaid
classDiagram
    direction TB

    %% ── Configuration ─────────────────────────────────────────
    class Gpt2Config {
        vocab_size : 50257
        n_embd : 768
        n_head : 12
        n_layer : 12
        block_size : 1024
        head_dim : 64
        layer_norm_epsilon : 1e-5
    }

    %% ── Top-level Model ───────────────────────────────────────
    class Gpt2Model {
        wte : Embedding [50257, 768]
        wpe : Embedding [1024, 768]
        blocks : TransformerBlock[12]
        ln_f : LayerNorm
        +forward(input_ids, pos_offset) Tensor [B,T,50257]
        +reset_cache()
    }

    %% ── Model Trait ───────────────────────────────────────────
    class Model {
        <<trait>>
        +forward(input_ids, pos_offset) Tensor
        +reset_cache()
    }

    %% ── Transformer Block ─────────────────────────────────────
    class TransformerBlock {
        ln_1 : LayerNorm
        attn : CausalSelfAttention
        ln_2 : LayerNorm
        mlp : MLP
        +forward(x) Tensor [B,T,768]
    }

    %% ── Causal Self-Attention ─────────────────────────────────
    class CausalSelfAttention {
        c_attn : Linear [768 → 2304]
        c_proj : Linear [768 → 768]
        n_head : 12
        head_dim : 64
        kv_cache : Option~KvCache~
        +forward(x) Tensor [B,T,768]
        +clear_cache()
    }

    %% ── KV Cache ──────────────────────────────────────────────
    class KvCache {
        key : Tensor [B, 12, seq_so_far, 64]
        value : Tensor [B, 12, seq_so_far, 64]
    }

    %% ── From ch05 (unchanged) ────────────────────────────────
    class MLP {
        c_fc : Linear [768 → 3072]
        c_proj : Linear [3072 → 768]
        +forward(x) Tensor [B,T,768]
    }

    class Embedding {
        weight : Tensor
        +forward(indices) Tensor
    }

    class LayerNorm {
        weight, bias : Tensor
        epsilon : 1e-5
        +forward(x) Tensor
    }

    class Linear {
        weight, bias : Tensor
        +forward(x) Tensor
    }

    %% ── Relationships ─────────────────────────────────────────
    Gpt2Model ..|> Model : implements
    Gpt2Model --> Embedding : wte, wpe
    Gpt2Model --> TransformerBlock : blocks[0..11]
    Gpt2Model --> LayerNorm : ln_f
    Gpt2Model ..> Gpt2Config : configured by

    TransformerBlock --> LayerNorm : ln_1, ln_2
    TransformerBlock --> CausalSelfAttention : attn
    TransformerBlock --> MLP : mlp

    CausalSelfAttention --> Linear : c_attn, c_proj
    CausalSelfAttention --> KvCache : kv_cache

    MLP --> Linear : c_fc, c_proj
```
**Figure 6.1** — Complete GPT-2 architecture. New in this chapter: CausalSelfAttention, TransformerBlock, Gpt2Model (implements Model trait).

## Attention Data Flow

```mermaid
flowchart LR
    X["x<br>[B, T, 768]"] --> QKV["c_attn<br>[768→2304]"]
    QKV --> Split["Split 3-way"]
    Split --> Q["Q [B,12,T,64]"]
    Split --> K["K [B,12,T,64]"]
    Split --> V["V [B,12,T,64]"]

    K --> Cache["KV Cache<br>concat"]
    V --> Cache

    Q --> Scores["Q @ K^T / √64"]
    Cache --> Scores
    Scores --> Mask["Causal Mask<br>(prefill only)"]
    Mask --> Soft["Softmax"]
    Soft --> Agg["@ V"]
    Cache --> Agg
    Agg --> Reshape["Reshape<br>[B,T,768]"]
    Reshape --> Proj["c_proj<br>[768→768]"]
    Proj --> Out["output<br>[B,T,768]"]
```
**Figure 6.2** — CausalSelfAttention data flow showing QKV split, KV cache, masking, and output projection.

## Pre-Norm Residual Block

```mermaid
flowchart TD
    X["x [B,T,768]"] --> LN1["LayerNorm 1"]
    LN1 --> ATTN["Attention"]
    ATTN --> ADD1["+"]
    X --> ADD1

    ADD1 --> Y["x' [B,T,768]"]
    Y --> LN2["LayerNorm 2"]
    LN2 --> MLP_box["MLP"]
    MLP_box --> ADD2["+"]
    Y --> ADD2

    ADD2 --> OUT["output [B,T,768]"]

```
**Figure 6.3** — Pre-norm residual pattern. LayerNorm precedes each sublayer; the residual adds the **original** (unnormalized) input.
