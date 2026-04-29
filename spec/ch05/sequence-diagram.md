# Chapter 5 -- Sequence Diagram: The Building Blocks

## Diagram 1: Extended Inspect Flow

```mermaid
sequenceDiagram
    participant Main
    participant HF as HuggingFace Hub
    participant Tok as Tokenizer
    participant Weights as SafeTensors
    participant Layers as Gpt2Layers

    Note over Main: === LOAD (from ch04) ===
    Main->>HF: Request model files (openai-community/gpt2)
    HF-->>Main: model.safetensors, tokenizer.json

    Main->>Tok: Load tokenizer
    Tok-->>Main: Ready (vocab_size: 50257)

    Main->>Weights: Memory-map safetensors
    Weights-->>Main: Weight accessor ready

    Note over Main: === BUILD LAYERS (ch05) ===
    Main->>Layers: Load wte [50257, 768]
    Main->>Layers: Load wpe [1024, 768]

    loop i = 0..11
        Main->>Layers: Load ln_1[i] (weight [768], bias [768])
        Main->>Layers: Load ln_2[i] (weight [768], bias [768])
        Main->>Layers: Load mlp[i].c_fc (weight [768,3072] → transpose → [3072,768])
        Main->>Layers: Load mlp[i].c_fc.bias [3072]
        Main->>Layers: Load mlp[i].c_proj (weight [3072,768] → transpose → [768,3072])
        Main->>Layers: Load mlp[i].c_proj.bias [768]
    end

    Layers-->>Main: All layers loaded

    Note over Main: === VERIFY (partial forward pass) ===
    Main->>Tok: encode("What is AI?")
    Tok-->>Main: token_ids (4 tokens)

    Main->>Layers: wte(token_ids) → [1, 4, 768]
    Note over Main: Print: token_emb stats

    Main->>Layers: wpe([0,1,2,3]) → [1, 4, 768]
    Note over Main: Print: pos_emb stats

    Main->>Main: hidden = token_emb + pos_emb
    Note over Main: Print: combined stats

    Main->>Layers: ln_1_block0(hidden) → [1, 4, 768]
    Note over Main: Print: ln_1 stats (mean ≈ 0)

    Main->>Layers: mlp_block0(normed) → [1, 4, 768]
    Note over Main: Print: mlp_out stats

    Main->>Main: Print summary + exit 0
```
**Figure 5.4** — The extended `inspect` flow. Chapters 4's loading is followed by layer construction and a verification forward pass.

## Diagram 2: MLP Forward Pass Detail

```mermaid
sequenceDiagram
    participant Caller
    participant MLP
    participant FC as c_fc Linear
    participant GELU as GELU Activation
    participant Proj as c_proj Linear

    Caller->>MLP: forward(x [1, 4, 768])
    MLP->>FC: forward(x) = x @ weight.T + bias
    FC-->>MLP: hidden [1, 4, 3072]
    MLP->>GELU: gelu(hidden)
    Note over GELU: x * 0.5 * (1 + erf(x/√2))
    GELU-->>MLP: activated [1, 4, 3072]
    MLP->>Proj: forward(activated) = x @ weight.T + bias
    Proj-->>MLP: output [1, 4, 768]
    MLP-->>Caller: output [1, 4, 768]
```
**Figure 5.5** — MLP computation detail. The hidden dimension expands 4x (768→3072) then contracts back.
