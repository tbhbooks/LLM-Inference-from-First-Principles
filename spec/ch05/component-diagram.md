# Chapter 5 -- Component Diagram: The Building Blocks

## Layer Components

```mermaid
classDiagram
    direction TB

    %% ── Embedding ────────────────────────────────────────────
    class Embedding {
        weight : Tensor [rows, 768]
        +forward(indices: [B, T]) Tensor [B, T, 768]
    }

    %% ── LayerNorm ────────────────────────────────────────────
    class LayerNorm {
        weight : Tensor [768]
        bias : Tensor [768]
        epsilon : 1e-5
        +forward(x: [B, T, 768]) Tensor [B, T, 768]
    }

    %% ── Linear ───────────────────────────────────────────────
    class Linear {
        weight : Tensor [out, in]
        bias : Tensor [out]
        +forward(x: [B, T, in]) Tensor [B, T, out]
    }

    %% ── MLP ──────────────────────────────────────────────────
    class MLP {
        c_fc : Linear [768 → 3072]
        c_proj : Linear [3072 → 768]
        +forward(x: [B, T, 768]) Tensor [B, T, 768]
    }

    %% ── Partial Model (ch05 scope) ──────────────────────────
    class Gpt2Layers {
        wte : Embedding [50257, 768]
        wpe : Embedding [1024, 768]
        ln_1 : LayerNorm[12]
        ln_2 : LayerNorm[12]
        mlp : MLP[12]
        +embed(token_ids, pos_ids) Tensor
        +layernorm_block0(x) Tensor
        +mlp_block0(x) Tensor
    }

    %% ── Relationships ────────────────────────────────────────
    Gpt2Layers --> Embedding : wte, wpe
    Gpt2Layers --> LayerNorm : ln_1[0..11], ln_2[0..11]
    Gpt2Layers --> MLP : mlp[0..11]
    MLP --> Linear : c_fc, c_proj
```
**Figure 5.1** — Layer components implemented in this chapter. Attention is deferred to Chapter 6.

## Data Flow: Partial Forward Pass

```mermaid
flowchart LR
    A["token_ids<br>[1, 4]"] --> B["wte lookup<br>[1, 4, 768]"]
    C["pos_ids<br>[0,1,2,3]"] --> D["wpe lookup<br>[1, 4, 768]"]
    B --> E["+ add"]
    D --> E
    E --> F["hidden<br>[1, 4, 768]"]
    F --> G["LayerNorm<br>ln_1 block 0"]
    G --> H["normed<br>[1, 4, 768]"]
    H --> I["MLP block 0"]
    I --> J["mlp_out<br>[1, 4, 768]"]

```
**Figure 5.2** — The partial forward pass verified in this chapter. Each box prints tensor stats.

## MLP Internal Data Flow

```mermaid
flowchart LR
    X["input<br>[B, T, 768]"] --> FC["c_fc (Linear)<br>[768 → 3072]"]
    FC --> ACT["GELU<br>(exact, erf)"]
    ACT --> PROJ["c_proj (Linear)<br>[3072 → 768]"]
    PROJ --> OUT["output<br>[B, T, 768]"]
```
**Figure 5.3** — MLP internal computation. The hidden dimension expands 4x then contracts back.

## Conv1D Transpose on Load

```
On disk (Conv1D format):        After transpose (standard Linear):
┌──────────────────────┐        ┌──────────────────────┐
│  [768, 3072]         │  ───►  │  [3072, 768]         │
│  (in_features first) │        │  (out_features first) │
└──────────────────────┘        └──────────────────────┘

Computation with standard Linear:
  output = input @ weight.T + bias
         = input @ [3072, 768].T + bias
         = input @ [768, 3072] + bias
         = [B, T, 3072]  ✓
```
