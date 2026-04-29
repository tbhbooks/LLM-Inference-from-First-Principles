# Chapter 8 -- Component Diagram: Fit and Finish

## MVP Architecture (Complete)

```mermaid
classDiagram
    direction TB

    class CLI {
        +generate(prompt, max_tokens, model)
        +inspect(model)
        -measure_prefill() Duration
        -measure_decode() Duration
        -estimate_kv_cache() Bytes
        -format_stats() String
    }

    class Model {
        <<trait>>
        +forward(input_ids, pos) Tensor
        +reset_cache()
        +cache_size_bytes() usize
    }

    class Sampler {
        <<trait>>
        +sample(logits) TokenId
    }

    class TokenizerBackend {
        <<trait>>
        +encode(text) list~TokenId~
        +decode(ids) string
        +eos_token_id() TokenId
    }

    class Gpt2Model {
        blocks : TransformerBlock[12]
        +forward() Tensor
        +reset_cache()
        +cache_size_bytes() usize
    }

    class GreedySampler {
        +sample(logits) TokenId
    }

    class HfTokenizer {
        +encode(text) list~TokenId~
        +decode(ids) string
    }

    class TimingStats {
        prefill_time : Duration
        decode_time : Duration
        tokens_generated : int
        +decode_speed() float
        +total_time() Duration
        +format() String
    }

    class KvCacheStats {
        n_layers : 12
        n_heads : 12
        head_dim : 64
        total_positions : int
        +memory_bytes() usize
        +format() String
    }

    CLI --> Model
    CLI --> Sampler
    CLI --> TokenizerBackend
    CLI --> TimingStats
    CLI --> KvCacheStats

    Gpt2Model ..|> Model
    GreedySampler ..|> Sampler
    HfTokenizer ..|> TokenizerBackend
```
**Figure 8.1** — MVP architecture with timing and KV cache stats. New in ch08: TimingStats, KvCacheStats, cache_size_bytes() on Model trait.

## CLI Subcommand Flow

```mermaid
flowchart TD
    Start["rvllm"] --> Parse["Parse CLI args"]
    Parse --> G{"Subcommand?"}
    G -->|generate| Load["Load model + tokenizer"]
    G -->|inspect| Inspect["Load model + tokenizer"]

    Load --> Prefill["Prefill<br>(timed)"]
    Prefill --> Decode["Decode loop<br>(timed)"]
    Decode --> Stats["Print text + stats<br>(prefill/decode/total/KV)"]

    Inspect --> Config["Print config"]
    Config --> Weights["Print weight summary"]
    Weights --> Layers["Print layer summary"]
    Layers --> TokCheck["Tokenizer round-trip"]
    TokCheck --> Predict["Top-5 prediction"]
    Predict --> Memory["Memory estimate"]
```
**Figure 8.2** — Both `generate` and `inspect` subcommands, showing the MVP's complete CLI surface.
