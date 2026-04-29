# Chapter 7 -- Component Diagram: The Skeleton Speaks

## Generation Pipeline

```mermaid
classDiagram
    direction TB

    class Model {
        <<trait>>
        +forward(input_ids, pos_offset) Tensor
        +reset_cache()
    }

    class Sampler {
        <<trait>>
        +sample(logits: Tensor) TokenId
    }

    class TokenizerBackend {
        <<trait>>
        +encode(text) list~TokenId~
        +decode(ids) string
        +eos_token_id() TokenId
    }

    class GreedySampler {
        +sample(logits) TokenId
    }
    note for GreedySampler "argmax(logits[0, -1, :])"

    class Gpt2Model {
        +forward(input_ids, pos) Tensor
        +reset_cache()
    }

    class HfTokenizer {
        +encode(text) list~TokenId~
        +decode(ids) string
        +eos_token_id() TokenId
    }

    class GenerationLoop {
        model : Model
        tokenizer : TokenizerBackend
        sampler : Sampler
        +generate(prompt, max_tokens) string
    }

    %% Relationships
    GreedySampler ..|> Sampler : implements
    Gpt2Model ..|> Model : implements
    HfTokenizer ..|> TokenizerBackend : implements

    GenerationLoop --> Model : uses
    GenerationLoop --> TokenizerBackend : encode/decode
    GenerationLoop --> Sampler : select next token
```
**Figure 7.1** — Generation pipeline components. GreedySampler is new in this chapter; Model and TokenizerBackend were implemented in ch04-06.

## Generation Loop State Machine

```mermaid
stateDiagram-v2
    [*] --> Encode : prompt text

    Encode --> Prefill : token_ids [B, T]
    note right of Prefill : Process all prompt tokens at once
    note right of Prefill : KV cache populated

    Prefill --> CheckEOS : first generated token

    CheckEOS --> Decode : not EOS & under max_tokens
    CheckEOS --> Done : EOS or max_tokens

    Decode --> CheckEOS : next token

    Done --> Output : decode token IDs → text
    Output --> [*]
```
**Figure 7.2** — Generation loop state machine. Prefill runs once; decode loops until EOS or max_tokens.

## Data Flow Through Generation

```mermaid
flowchart LR
    subgraph "Encode"
        A["prompt text"] --> B["Tokenizer.encode()"]
        B --> C["token_ids<br>[6 tokens]"]
    end

    subgraph "Prefill"
        C --> D["Model.forward()<br>pos_offset=0"]
        D --> E["logits<br>[1,6,50257]"]
        E --> F["Sampler.sample()"]
        F --> G["token₁"]
    end

    subgraph "Decode Loop"
        G --> H["Model.forward()<br>pos_offset=6"]
        H --> I["logits<br>[1,1,50257]"]
        I --> J["Sampler.sample()"]
        J --> K["token₂"]
        K --> L{"EOS?<br>max?"}
        L -->|no| H
        L -->|yes| M["generated tokens"]
    end

    subgraph "Output"
        M --> N["Tokenizer.decode()"]
        N --> O["text + stats"]
    end
```
**Figure 7.3** — End-to-end data flow from prompt text to generated output.
