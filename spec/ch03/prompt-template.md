# Chapter 3 -- LLM Prompt Template

Copy and paste this prompt into your LLM of choice to generate a working
implementation. This builds on the project from Chapters 1–2.

---

## Prompt

```
I am building an LLM inference engine called "rvllm". This is Chapter 3.
I already have a project directory from Chapters 1–2 with this structure:

  rvllm/
  ├── Cargo.toml (or equivalent project file)
  ├── src/
  │   └── main.rs (minimal placeholder)
  └── examples/
      ├── ch01_inference_problem.[ext]   <-- Chapter 1 (KV cache calculator)
      └── ch02_architecture.[ext]        <-- Chapter 2 (architecture printer)

Now it's time to build the real project skeleton. Create the module structure,
core interfaces, shared types, error handling, configuration, and a CLI with
two stubbed subcommands. There is NO business logic yet — every module is a
placeholder that establishes its interface.

TARGET LANGUAGE: [Same language as Chapters 1–2]
Note: This spec uses 'interface' as the language-agnostic term. In Rust,
implement as traits. In Go/Java, as interfaces. In Python, as ABCs or protocols.

=== WHAT TO CREATE ===

Add these files to the EXISTING project. Do NOT delete the examples/ directory
or any Chapter 1–2 files.

  rvllm/
  ├── Cargo.toml              <-- UPDATE: add dependencies (see below)
  ├── src/
  │   ├── lib.rs              <-- NEW: re-exports all modules
  │   ├── main.rs             <-- REPLACE: CLI with subcommands
  │   ├── types.rs            <-- NEW: shared newtypes
  │   ├── error.rs            <-- NEW: centralized error enum
  │   ├── config.rs           <-- NEW: engine configuration
  │   ├── model/mod.rs        <-- NEW: Model interface (trait only)
  │   ├── sampling/mod.rs     <-- NEW: Sampler interface (trait only)
  │   ├── tokenizer/mod.rs    <-- NEW: TokenizerBackend interface (trait only)
  │   ├── engine/mod.rs       <-- NEW: stub (empty placeholder)
  │   ├── scheduler/mod.rs    <-- NEW: stub (empty placeholder)
  │   ├── memory/mod.rs       <-- NEW: stub (empty placeholder)
  │   └── api/mod.rs          <-- NEW: stub (empty placeholder)
  └── examples/
      ├── ch01_inference_problem.[ext]   <-- KEEP (from Chapter 1)
      └── ch02_architecture.[ext]        <-- KEEP (from Chapter 2)

=== DEPENDENCIES ===

For Rust, add to Cargo.toml:
  thiserror (error types), tracing + tracing-subscriber (structured logging),
  tokio (async runtime), serde + serde_json (config), clap with derive (CLI),
  candle-core + candle-nn (tensor ops — needed for trait signatures),
  anyhow (for binary error handling)

For Python: no external deps needed for this chapter.

=== MODULE STRUCTURE ===

Create these modules/packages:

1. api          -- (stub) HTTP server layer. Empty for now.
2. engine       -- (stub) Inference loop orchestration. Empty for now.
3. scheduler    -- (stub) Request queuing and batching. Empty for now.
4. model        -- Defines the Model interface (see below). No implementation.
5. memory       -- (stub) KV-cache block management. Empty for now.
6. sampling     -- Defines the Sampler interface (see below). No implementation.
7. tokenizer    -- Defines the TokenizerBackend interface (see below). No impl.
8. config       -- EngineConfig struct with defaults.
9. error        -- RvllmError enum (centralized error type).
10. types       -- Shared newtypes: TokenId, RequestId, SeqLen, BlockId.

A library entry point (lib.rs) re-exports all modules.
A binary entry point (main.rs) provides the CLI.

=== NEWTYPES (in types module) ===

TokenId(u32)    -- A token in the model vocabulary
RequestId(u64)  -- Unique client request identifier
SeqLen(usize)   -- Length of a token sequence
BlockId(u32)    -- KV-cache block identifier

Each newtype must support:
- Equality comparison
- Hashing
- Copying / cloning
- Display (format: "tok:42", "req:7", "len:128", "blk:3")
- Public access to inner value

=== ERROR ENUM (in error module) ===

enum RvllmError:
    ModelLoad(String)     -- model loading failures
    Tokenizer(String)     -- tokenizer failures
    Inference(String)     -- forward-pass failures
    Config(String)        -- configuration errors
    Io(IoError)           -- I/O errors (auto-convert from stdlib)

Define a type alias: Result<T> = std::Result<T, RvllmError>

=== INTERFACES ===

Model (in model module):
    forward(input_ids: Tensor, pos: int) -> Result<Tensor>
    reset_cache() -> void

Sampler (in sampling module):
    sample(logits: Tensor) -> Result<TokenId>

TokenizerBackend (in tokenizer module):
    encode(text: string) -> Result<list<TokenId>>
    decode(ids: list<TokenId>) -> Result<string>
    eos_token_id() -> TokenId

For the Tensor type: use your language's tensor library type, or define a
placeholder type alias if no tensor library is in scope.

=== CONFIGURATION (in config module) ===

struct EngineConfig:
    model_id: string       (default: "openai-community/gpt2")
    max_tokens: uint       (default: 128)
    device: DeviceKind     (default: Cpu)

enum DeviceKind: Cpu | Metal | Cuda

EngineConfig must provide a default constructor.

=== CLI (binary entry point) ===

Binary name: rvllm

Subcommands:
1. generate
   --model <MODEL>        (default: "openai-community/gpt2")
   --prompt <PROMPT>      (required)
   --max-tokens <N>       (default: 128)

   Behavior: print "generate: not yet implemented", exit 0

2. inspect
   --model <MODEL>        (default: "openai-community/gpt2")

   Behavior: print "inspect: not yet implemented", exit 0

Running without a subcommand prints help and exits non-zero.

=== CONSTRAINTS ===

- The project MUST compile / run without errors.
- Stub modules are empty or contain only a doc comment.
- Interfaces are defined but have ZERO implementations.
- No business logic anywhere. This is purely a skeleton.
- Use structured logging (tracing crate in Rust, logging module in Python).
- Initialize the logger in main before any other work.
- Do NOT delete or modify Chapter 1–2 example files.

=== WHAT TO PRODUCE ===

Produce the complete set of NEW or MODIFIED source files. For each file,
show the full path relative to the project root and the complete file contents.

After this chapter, the project directory looks like:

  rvllm/
  ├── Cargo.toml              (updated with dependencies)
  ├── src/
  │   ├── lib.rs              (re-exports all modules)
  │   ├── main.rs             (CLI with generate + inspect subcommands)
  │   ├── types.rs            (TokenId, RequestId, SeqLen, BlockId)
  │   ├── error.rs            (RvllmError enum)
  │   ├── config.rs           (EngineConfig, DeviceKind)
  │   ├── model/mod.rs        (Model trait, no impl)
  │   ├── sampling/mod.rs     (Sampler trait, no impl)
  │   ├── tokenizer/mod.rs    (TokenizerBackend trait, no impl)
  │   ├── engine/mod.rs       (stub)
  │   ├── scheduler/mod.rs    (stub)
  │   ├── memory/mod.rs       (stub)
  │   └── api/mod.rs          (stub)
  └── examples/
      ├── ch01_inference_problem.[ext]   (from Chapter 1)
      └── ch02_architecture.[ext]        (from Chapter 2)
```
