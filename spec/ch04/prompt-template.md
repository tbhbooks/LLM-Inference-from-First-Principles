# Chapter 4 -- LLM Prompt Template: Downloading a Brain

Copy and paste the prompt below into your LLM of choice, or use it as a
reference while implementing by hand. This builds on the project skeleton
from Chapters 1–3.

---

## The Prompt

```
I am building an LLM inference engine called "rvllm". This is Chapter 4.
I have an existing project from Chapters 1–3 with this structure:

  rvllm/
  ├── Cargo.toml
  ├── src/
  │   ├── lib.rs              (re-exports all modules)
  │   ├── main.rs             (CLI: generate + inspect subcommands, both stubbed)
  │   ├── types.rs            (TokenId, RequestId, SeqLen, BlockId newtypes)
  │   ├── error.rs            (RvllmError enum with ModelLoad, Tokenizer, etc.)
  │   ├── config.rs           (EngineConfig, DeviceKind)
  │   ├── model/mod.rs        (Model trait: forward, reset_cache — NO impl yet)
  │   ├── sampling/mod.rs     (Sampler trait: sample — NO impl yet)
  │   ├── tokenizer/mod.rs    (TokenizerBackend trait: encode, decode, eos — NO impl yet)
  │   ├── engine/mod.rs       (empty stub)
  │   ├── scheduler/mod.rs    (empty stub)
  │   ├── memory/mod.rs       (empty stub)
  │   └── api/mod.rs          (empty stub)
  └── examples/
      ├── ch01_inference_problem.[ext]
      └── ch02_architecture.[ext]

The traits that already exist:

  Model trait:
      forward(input_ids: Tensor, pos: int) -> Result<Tensor>
      reset_cache() -> void

  Sampler trait:
      sample(logits: Tensor) -> Result<TokenId>

  TokenizerBackend trait:
      encode(text: string) -> Result<list<TokenId>>
      decode(ids: list<TokenId>) -> Result<string>
      eos_token_id() -> TokenId

Now implement model loading and tokenizer. Download GPT-2 124M from HuggingFace,
load weights into memory, verify the tokenizer, and wire the `inspect` subcommand.

NO FORWARD PASS. NO GENERATION. Just loading and inspecting.

LANGUAGE/FRAMEWORK: [Rust with candle / Python with PyTorch / Python with numpy / your choice]
An ML tensor library is required for loading safetensors.

=== WHAT TO CREATE / MODIFY ===

Add or modify these files in the EXISTING project:

  NEW FILES:
    src/model/gpt2.rs          <-- GPT-2 config + weight loading (NO forward pass yet)
    src/tokenizer/hf.rs        <-- HuggingFace tokenizer impl of TokenizerBackend

  MODIFY:
    src/model/mod.rs            <-- add `pub mod gpt2;` and re-export
    src/tokenizer/mod.rs        <-- add `pub mod hf;` and re-export
    src/main.rs                 <-- wire `inspect` subcommand to real loading + output
    Cargo.toml                  <-- add dependencies (hf-hub, tokenizers, candle)

  KEEP UNCHANGED:
    src/types.rs, src/error.rs, src/config.rs (use them, don't rewrite them)
    src/sampling/mod.rs (still just the trait stub)
    src/engine/, src/scheduler/, src/memory/, src/api/ (still stubs)
    examples/ch01_*, examples/ch02_*

=== MODEL SOURCE ===

HuggingFace model ID: "openai-community/gpt2"
Files needed:
  - model.safetensors (weight data)
  - tokenizer.json (BPE tokenizer)
  - config.json (model configuration)
Format: SafeTensors, Dtype: float32
Total parameters: ~124M

=== WHAT TO LOAD ===

1. Download/cache model files from HuggingFace Hub
2. Parse config.json to extract: n_layer, n_embd, n_head, vocab_size, block_size
3. Load safetensors and enumerate all weight tensors:
   - Print each weight name and shape
   - Count total weights (should be 148)
   - Sum parameters (~124M)
4. Note Conv1D weights (stored as [in, out]) — just report them, don't transpose yet
5. Note weight tying (lm_head tied to wte)

=== TOKENIZER ===

1. Load tokenizer from tokenizer.json
2. Implement TokenizerBackend trait:
   - encode("What is AI?") → token IDs
   - decode(token_ids) → "What is AI?"
   - eos_token_id() → 50256
3. Print round-trip test result

=== INSPECT SUBCOMMAND ===

Wire `inspect` to print:

Loading model: openai-community/gpt2
  Model files downloaded/cached
  Tokenizer loaded (vocab_size: 50257)
  Model config: 12 layers, 768 dim, 12 heads

Weights: 148 tensors, ~124M parameters
  transformer.wte.weight: [50257, 768]
  transformer.wpe.weight: [1024, 768]
  transformer.h.0.ln_1.weight: [768]
  ... (all weights or a summary)
  transformer.ln_f.weight: [768]
  transformer.ln_f.bias: [768]

Tokenizer check:
  "What is AI?" → [2061, 318, 9552, 30] → "What is AI?" ✓

=== GENERATE SUBCOMMAND ===

Leave `generate` stubbed. Print: "generate: not yet implemented (coming in ch07)"

=== WHAT SUCCESS LOOKS LIKE ===

Running `rvllm inspect --model openai-community/gpt2` should:
- Download (or load from cache) the model files
- Print model configuration
- List weight names and shapes (or summary)
- Show total parameter count (~124M)
- Show tokenizer round-trip test passing
- Exit cleanly with code 0

=== WHAT TO PRODUCE ===

Produce the complete set of NEW or MODIFIED source files. For each file,
show the full path relative to the project root and the complete file contents.

Do NOT recreate types.rs, error.rs, config.rs, or any Chapter 1–2 files.
USE the existing traits, types, and error enum — build on them.

After this chapter, the project directory looks like:

  rvllm/
  ├── Cargo.toml                           (updated: +hf-hub, +tokenizers, +candle)
  ├── src/
  │   ├── lib.rs                           (unchanged)
  │   ├── main.rs                          (MODIFIED: inspect wired to real code)
  │   ├── types.rs                         (unchanged)
  │   ├── error.rs                         (unchanged)
  │   ├── config.rs                        (unchanged)
  │   ├── model/
  │   │   ├── mod.rs                       (MODIFIED: +pub mod gpt2)
  │   │   └── gpt2.rs                      (NEW: config + weight loading, no forward pass)
  │   ├── sampling/
  │   │   └── mod.rs                       (unchanged — trait stub only)
  │   ├── tokenizer/
  │   │   ├── mod.rs                       (MODIFIED: +pub mod hf)
  │   │   └── hf.rs                        (NEW: HuggingFace tokenizer wrapper)
  │   ├── engine/mod.rs                    (still stub)
  │   ├── scheduler/mod.rs                 (still stub)
  │   ├── memory/mod.rs                    (still stub)
  │   └── api/mod.rs                       (still stub)
  └── examples/
      ├── ch01_inference_problem.[ext]     (from Chapter 1)
      └── ch02_architecture.[ext]          (from Chapter 2)
```

---

## Suggested Frameworks by Language

| Language | Tensor library | Tokenizer | SafeTensors loader | HF Hub |
|----------|---------------|-----------|-------------------|--------|
| Rust     | `candle-core` | `tokenizers` crate | `candle-core` (built-in) | `hf-hub` |
| Python   | `torch` or `numpy` | `tokenizers` | `safetensors` | `huggingface_hub` |
| TypeScript | — | `@huggingface/tokenizers` | `@huggingface/safetensors` | — |

For Rust specifically, you will also need:
- `hf-hub` crate for downloading from HuggingFace
- `clap` for CLI argument parsing (already in Cargo.toml from ch03)
- `anyhow` for error handling in the binary (already in Cargo.toml from ch03)
