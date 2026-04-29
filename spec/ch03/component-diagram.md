# Chapter 3 -- Component Diagram

## Module Structure and Dependencies

```mermaid
classDiagram
    direction TB

    %% ── Support modules ──────────────────────────────────────
    class types {
        <<module>>
        TokenId(u32)
        RequestId(u64)
        SeqLen(usize)
        BlockId(u32)
    }

    class error {
        <<module>>
        RvllmError
        +ModelLoad(string)
        +Tokenizer(string)
        +Inference(string)
        +Config(string)
        +Io(IoError)
    }

    class config {
        <<module>>
        EngineConfig
        +model_id : string
        +max_tokens : uint
        +device : DeviceKind
    }

    class DeviceKind {
        <<enum>>
        Cpu
        Metal
        Cuda
    }

    config --> DeviceKind

    %% ── Core interfaces ──────────────────────────────────────────
    class Model {
        <<interface>>
        +forward(input_ids: Tensor, pos: int) Result~Tensor~
        +reset_cache()
    }

    class Sampler {
        <<interface>>
        +sample(logits: Tensor) Result~TokenId~
    }

    class TokenizerBackend {
        <<interface>>
        +encode(text: string) Result~list~TokenId~~
        +decode(ids: list~TokenId~) Result~string~
        +eos_token_id() TokenId
    }

    %% ── Feature modules (stubs) ──────────────────────────────
    class api {
        <<module / stub>>
        HTTP layer
    }

    class engine {
        <<module / stub>>
        Inference orchestration
    }

    class scheduler {
        <<module / stub>>
        Request batching
    }

    class model {
        <<module>>
        Model interface defined here
    }

    class memory {
        <<module / stub>>
        KV-cache management
    }

    class sampling {
        <<module>>
        Sampler interface defined here
    }

    class tokenizer {
        <<module>>
        TokenizerBackend interface defined here
    }

    %% ── CLI ──────────────────────────────────────────────────
    class CLI {
        <<binary>>
        +main()
    }

    class Commands {
        <<enum>>
        Generate(model, prompt, max_tokens)
        Inspect(model)
    }

    CLI --> Commands

    %% ── Interface ownership ──────────────────────────────────────
    model --> Model
    sampling --> Sampler
    tokenizer --> TokenizerBackend

    %% ── Module dependencies ──────────────────────────────────
    %% Interfaces depend on support types
    Model ..> types : uses TokenId
    Model ..> error : returns Result
    Sampler ..> types : returns TokenId
    Sampler ..> error : returns Result
    TokenizerBackend ..> types : uses TokenId
    TokenizerBackend ..> error : returns Result

    %% Feature modules depend on interfaces and support
    engine ..> model : uses Model interface
    engine ..> sampling : uses Sampler interface
    engine ..> tokenizer : uses TokenizerBackend interface
    engine ..> scheduler : queries scheduler
    engine ..> memory : manages KV cache
    engine ..> config : reads EngineConfig

    scheduler ..> types : uses RequestId, SeqLen
    memory ..> types : uses BlockId

    api ..> engine : submits requests

    %% CLI drives the engine
    CLI ..> engine : dispatches commands
    CLI ..> config : builds EngineConfig
```

## Reading Guide

The diagram above captures the full module layout at the end of Chapter 3.
Every box marked **stub** contains no business logic -- it exists only to
establish the module boundary and make the project compile.

**Key relationships:**

| Arrow style | Meaning |
|-------------|---------|
| Solid (`-->`) | "owns" or "contains" |
| Dashed (`..>`) | "depends on" or "uses" |

**Newtypes** (`TokenId`, `RequestId`, `SeqLen`, `BlockId`) live in `types` and
are imported by any module that needs them. They wrap a single primitive and
exist to prevent accidental mix-ups at module boundaries.

**Error** is a single enum (`RvllmError`) with one variant per failure domain.
All modules convert their internal errors into this enum so callers have one
type to match on.

**Interfaces** are the primary design tool. `Model`, `Sampler`, and
`TokenizerBackend` define the contracts that concrete implementations will
fill in later chapters. At this stage they have no implementors.
