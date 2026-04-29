# Chapter 4 -- Sequence Diagram: Downloading a Brain

## Diagram 1: Model Loading and Inspection

```mermaid
sequenceDiagram
    participant CLI as CLI (inspect)
    participant HF as HuggingFace Hub
    participant Tok as Tokenizer
    participant Weights as SafeTensors

    Note over CLI: rvllm inspect --model openai-community/gpt2

    CLI->>HF: Request model files (openai-community/gpt2)
    HF-->>CLI: tokenizer.json, model.safetensors, config.json
    Note over CLI: Print: "Model files downloaded/cached"

    CLI->>CLI: Parse config.json → Gpt2Config
    Note over CLI: Print: "12 layers, 768 dim, 12 heads"

    CLI->>Weights: Memory-map model.safetensors
    Weights-->>CLI: Weight accessor ready

    CLI->>Weights: Enumerate all weight names + shapes
    Note over Weights: 148 tensors found
    CLI->>CLI: Sum parameter count → ~124M
    Note over CLI: Print: weight summary

    CLI->>Tok: Load tokenizer from tokenizer.json
    Tok-->>CLI: Tokenizer ready (vocab_size: 50257)
    Note over CLI: Print: "Tokenizer loaded"

    Note over CLI: === TOKENIZER ROUND-TRIP ===
    CLI->>Tok: encode("What is AI?")
    Tok-->>CLI: [2061, 318, 9552, 30]
    CLI->>Tok: decode([2061, 318, 9552, 30])
    Tok-->>CLI: "What is AI?"
    Note over CLI: Print: round-trip ✓

    CLI->>CLI: Exit 0
```
**Figure 4.3** — The `inspect` subcommand flow: download, parse config, enumerate weights, test tokenizer.

## Diagram 2: File Download and Caching

```mermaid
flowchart TD
    A["rvllm inspect --model openai-community/gpt2"] --> B{"Files in local cache?"}
    B -->|yes| C["Load from cache<br>(~/.cache/huggingface/)"]
    B -->|no| D["Download from HuggingFace Hub"]
    D --> E["Save to cache"]
    E --> C
    C --> F["config.json<br>→ Gpt2Config"]
    C --> G["model.safetensors<br>→ weight accessor"]
    C --> H["tokenizer.json<br>→ Tokenizer"]
    F --> I["Print config summary"]
    G --> J["Enumerate weights<br>148 tensors, ~124M params"]
    H --> K["Round-trip test<br>'What is AI?' → ids → text"]
```
**Figure 4.4** — File caching flow. First run downloads; subsequent runs load from cache.
