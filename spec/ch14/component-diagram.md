# Chapter 14 -- Component Diagram: Sampling Strategies

## LogitsProcessor Pipeline

```mermaid
classDiagram
    direction TB

    class LogitsProcessor {
        <<trait>>
        +process(logits, token_ids_so_far) FloatArray
    }

    class TemperatureProcessor {
        temperature: float
        +process(logits, _) FloatArray
    }
    note for TemperatureProcessor "logits / temperature"

    class TopKProcessor {
        k: int
        +process(logits, _) FloatArray
    }
    note for TopKProcessor "keep top-k, rest → -inf"

    class TopPProcessor {
        p: float
        +process(logits, _) FloatArray
    }
    note for TopPProcessor "cumulative prob cutoff"

    class RepetitionPenaltyProcessor {
        penalty: float
        +process(logits, seen_tokens) FloatArray
    }
    note for RepetitionPenaltyProcessor "penalize seen tokens"

    class SamplingParams {
        temperature: float
        top_k: int
        top_p: float
        repetition_penalty: float
        max_tokens: int
        +build_pipeline() list~LogitsProcessor~
    }

    class PipelineSampler {
        processors: list~LogitsProcessor~
        temperature: float
        rng: RandomGenerator
        +sample(logits, token_ids_so_far) TokenId
    }

    class Sampler {
        <<trait>>
        +sample(logits, token_ids_so_far) TokenId
    }

    class GreedySampler {
        +sample(logits, _) TokenId
    }
    note for GreedySampler "argmax — ch07 fallback"

    %% Trait implementations
    TemperatureProcessor ..|> LogitsProcessor : implements
    TopKProcessor ..|> LogitsProcessor : implements
    TopPProcessor ..|> LogitsProcessor : implements
    RepetitionPenaltyProcessor ..|> LogitsProcessor : implements

    PipelineSampler ..|> Sampler : implements
    GreedySampler ..|> Sampler : implements

    %% Composition
    SamplingParams ..> PipelineSampler : builds
    PipelineSampler o-- LogitsProcessor : contains 0..*
```
**Figure 14.1** — Sampling pipeline components. LogitsProcessor is the new trait; each strategy implements it. PipelineSampler composes processors and implements the Sampler trait from ch07.

## Pipeline Data Flow

```mermaid
flowchart LR
    subgraph "Input"
        A["Raw logits<br/>[vocab_size]"]
        B["Tokens so far<br/>[t₁, t₂, ..., tₙ]"]
    end

    subgraph "Pipeline (in order)"
        C["RepetitionPenalty<br/>penalize seen tokens"]
        D["Temperature<br/>scale by 1/T"]
        E["TopK<br/>keep top-k"]
        F["TopP<br/>cumulative cutoff"]
    end

    subgraph "Selection"
        G{"temp == 0?"}
        H["argmax<br/>(greedy)"]
        I["softmax →<br/>multinomial"]
    end

    A --> C
    B --> C
    C --> D --> E --> F
    F --> G
    G -->|yes| H
    G -->|no| I

    H --> J["TokenId"]
    I --> J
```
**Figure 14.2** — Data flow through the sampling pipeline. Processors transform logits in sequence; final selection is either greedy (argmax) or stochastic (multinomial).

## Temperature Effect on Distribution

```mermaid
flowchart TB
    subgraph "Temperature = 0.5 (sharp)"
        A1["Token A: 0.94"]
        A2["Token B: 0.04"]
        A3["Token C: 0.02"]
    end

    subgraph "Temperature = 1.0 (baseline)"
        B1["Token A: 0.71"]
        B2["Token B: 0.10"]
        B3["Token C: 0.04"]
    end

    subgraph "Temperature = 2.0 (flat)"
        C1["Token A: 0.34"]
        C2["Token B: 0.18"]
        C3["Token C: 0.12"]
    end

```
**Figure 14.3** — Temperature controls the sharpness of the probability distribution. Lower temperature concentrates probability on the top token; higher temperature spreads it out.
