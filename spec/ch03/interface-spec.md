# Chapter 3 -- Interface Specification

This spec is **language-agnostic**. It defines the contracts that a Chapter 3
implementation must satisfy. Types use pseudocode notation; map them to your
language's idioms (Rust `Result<T, E>`, Python exceptions, Go `(T, error)`,
etc.).

---

## 1. Modules

Each module corresponds to a source directory or file. Modules listed as
*stub* must exist (so the project compiles / imports resolve) but contain no
business logic.

| Module | Responsibility | Status |
|-----------|-----------------------------------------------|--------|
| `api` | HTTP server, request/response types | stub |
| `engine` | Inference loop orchestration | stub |
| `scheduler` | Request queuing, batching, preemption | stub |
| `model` | Forward-pass interface + model loading | interface defined, no impl |
| `memory` | KV-cache block management | stub |
| `sampling` | Token selection strategies | interface defined, no impl |
| `tokenizer`| Text-to-token and token-to-text conversion | interface defined, no impl |
| `config` | Engine configuration struct + defaults | implemented |
| `error` | Centralized error enum | implemented |
| `types` | Shared newtypes | implemented |

---

## 2. Newtypes

Newtypes wrap a single primitive to prevent accidental misuse across module
boundaries. Each must support equality comparison, hashing, copying, and
a human-readable display format.

| Name | Inner type | Display format | Purpose |
|-------------|------------|----------------|----------------------------------|
| `TokenId` | u32 | `tok:<value>` | A token in the model vocabulary |
| `RequestId` | u64 | `req:<value>` | Unique client request identifier |
| `SeqLen` | usize | `len:<value>` | Length of a token sequence |
| `BlockId` | u32 | `blk:<value>` | KV-cache block identifier |

Newtypes expose their inner value via a public field (e.g., `TokenId.0` or
`token_id.value`).

---

## 3. Error Enum

A single top-level error type that every module converts into. Callers match
on variants to decide how to handle failures.

```
enum RvllmError:
    ModelLoad(detail: string)    -- Weight file missing, architecture mismatch, etc.
    Tokenizer(detail: string)    -- Tokenizer file missing, encoding failure, etc.
    Inference(detail: string)    -- Forward pass NaN, shape mismatch, OOM, etc.
    Config(detail: string)       -- Missing field, invalid value, parse error, etc.
    Io(inner: IoError)           -- File system or network I/O failure.
```

A convenience type alias `Result<T>` is defined as `Result<T, RvllmError>`.

---

## 4. Interfaces

### 4.1 Model

Defined in the `model` module.

```
interface Model:
    forward(input_ids: Tensor, pos: int) -> Result<Tensor>
        -- Run one forward pass.
        -- input_ids: 1-D integer tensor of token IDs.
        -- pos: position offset (for KV-cache indexing).
        -- Returns: 1-D float tensor of logits over vocabulary.

    reset_cache() -> void
        -- Clear any cached key/value state.
        -- Called between independent requests.
```

### 4.2 Sampler

Defined in the `sampling` module.

```
interface Sampler:
    sample(logits: Tensor) -> Result<TokenId>
        -- Given a 1-D float tensor of logits over the vocabulary,
           select and return one token.
        -- Strategy (greedy, top-k, etc.) is determined by the
           concrete implementation.
```

### 4.3 TokenizerBackend

Defined in the `tokenizer` module.

```
interface TokenizerBackend:
    encode(text: string) -> Result<list<TokenId>>
        -- Convert a UTF-8 string into a sequence of token IDs.

    decode(ids: list<TokenId>) -> Result<string>
        -- Convert a sequence of token IDs back into a UTF-8 string.

    eos_token_id() -> TokenId
        -- Return the end-of-sequence token ID for this model's vocabulary.
```

---

## 5. Configuration

### 5.1 EngineConfig

Top-level configuration for the inference engine. Constructable from CLI
flags, a config file, or programmatically. Provides sensible defaults.

| Field | Type | Default | Description |
|-------------|------------|-------------------------------|--------------------------------------|
| `model_id` | string | `"openai-community/gpt2"` | HuggingFace model ID or local path |
| `max_tokens`| uint | `128` | Max tokens to generate per request |
| `device` | DeviceKind | `Cpu` | Hardware backend |

### 5.2 DeviceKind

```
enum DeviceKind:
    Cpu
    Metal
    Cuda
```

---

## 6. CLI Specification

### Binary name
`rvllm`

### Global structure
```
rvllm <COMMAND>

Commands:
  generate   Generate text from a prompt
  inspect    Inspect a model's configuration and architecture
```

### `generate` subcommand

```
rvllm generate [OPTIONS] --prompt <PROMPT>

Options:
  --model <MODEL>          HuggingFace model ID or local path
                           [default: openai-community/gpt2]
  --prompt <PROMPT>        Input prompt (required)
  --max-tokens <N>         Maximum number of tokens to generate
                           [default: 128]
```

**Current behavior (Chapter 3):** Prints `generate: not yet implemented`
and exits with code 0.

### `inspect` subcommand

```
rvllm inspect [OPTIONS]

Options:
  --model <MODEL>          HuggingFace model ID or local path
                           [default: openai-community/gpt2]
```

**Current behavior (Chapter 3):** Prints `inspect: not yet implemented`
and exits with code 0.

---

## 7. Constraints

These constraints define what Chapter 3 is and is not.

1. **No business logic.** Modules marked *stub* contain no algorithms, no I/O,
   no computation. They exist solely to establish boundaries.
2. **Interfaces are defined but not implemented.** No struct implements `Model`,
   `Sampler`, or `TokenizerBackend` yet.
3. **The project must compile.** All module references resolve, all imports
   are valid, the binary runs and produces the expected stub output.
4. **Newtypes are usable.** They can be constructed, compared, hashed, copied,
   and displayed.
5. **Errors are usable.** `RvllmError` variants can be constructed and
   matched. The `Result` alias works.
6. **Configuration has defaults.** `EngineConfig::default()` (or equivalent)
   returns a valid config.
7. **CLI parses correctly.** `--help` shows usage for the binary and each
   subcommand. Invalid flags produce an error. Valid flags dispatch to the
   correct stub handler.
