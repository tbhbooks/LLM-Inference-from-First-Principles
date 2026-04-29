# Chapter 8 -- LLM Prompt Template: Fit and Finish

Copy and paste the prompt below into your LLM of choice, or use it as a
reference while implementing by hand. This builds on the project from
Chapters 1–7.

---

## The Prompt

```
I am building an LLM inference engine called "rvllm". This is Chapter 8 —
the MVP milestone. I have a fully working GPT-2 pipeline from Chapters 1–7:
- Model loading + tokenizer (ch04)
- Embedding, LayerNorm, MLP (ch05)
- Attention, KV cache, full model (ch06)
- Generation loop with greedy decoding (ch07)

Both `inspect` and `generate` subcommands work. Now polish the CLI into
a finished MVP.

LANGUAGE/FRAMEWORK: [Rust with candle / Python with PyTorch / your choice]

=== WHAT TO MODIFY ===

  MODIFY (no new files needed):
    src/main.rs                    <-- Timing breakdown, KV cache estimate, polished output
    src/model/gpt2/model.rs        <-- Add method to report cache size
    (any other files as needed for logging/error improvements)

  KEEP UNCHANGED:
    All layer implementations (embedding, layernorm, linear, mlp, attention, block)
    src/types.rs, src/error.rs, src/config.rs
    src/tokenizer/

=== TIMING BREAKDOWN ===

Measure prefill and decode separately:

1. Start prefill timer
2. logits = model.forward(prompt_tokens, pos=0)     // prefill
3. Stop prefill timer

4. Start decode timer
5. [decode loop as before]
6. Stop decode timer

Report:
  Prefill: X.XXs (N prompt tokens)
  Decode:  X.XXs (N tokens, X.XX tokens/sec)
  Total:   X.XXs

Decode speed = generated_tokens / decode_time

=== KV CACHE MEMORY ESTIMATE ===

After generation, calculate and print:

  bytes_per_token_per_layer = 2 * n_head * head_dim * sizeof(float32)
                            = 2 * 12 * 64 * 4 = 6144 bytes

  total_positions = prompt_len + generated_tokens
  total_bytes = n_layer * total_positions * bytes_per_token_per_layer
              = 12 * total_positions * 6144

Print:
  KV cache: X.X MB (N positions × 12 layers × 6 KB/pos/layer)

=== POLISHED INSPECT ===

Consolidate all info into a clean format:

=== Model: openai-community/gpt2 ===

Config:
  Layers: 12, Hidden dim: 768, Heads: 12, Vocab: 50257
  Context window: 1024, Head dim: 64

Weights:
  148 tensors, ~124M parameters
  Memory: ~497 MB (float32)

Layers:
  2 embeddings, 24 LayerNorms, 12 MLPs, 12 attention blocks

Tokenizer:
  "What is AI?" → [token_ids] → "What is AI?" ✓

Quick prediction (forward pass on "What is AI?"):
  1. "token"  (logit: XX.XX)
  ...5 predictions...

=== BETTER ERROR MESSAGES ===

Wrap common failure points with helpful messages:
- Model download failure → "Failed to download model. Check model ID and network."
- Weight loading failure → "Failed to load weights. Expected safetensors format."
- Tokenizer failure → "Failed to load tokenizer. Check tokenizer.json exists."
- Shape mismatch → Include expected vs actual shapes in error message.

=== STRUCTURED LOGGING ===

Replace println/print with structured logging:
- Rust: `tracing` crate (info!, debug!, warn!, error!)
- Python: `logging` module
- TypeScript: structured console.log with levels

Levels:
- INFO: loading model, config summary, generation output, stats
- DEBUG: tensor shapes, per-layer timing, cache sizes
- WARN: slow generation, large model
- ERROR: failures

=== WHAT SUCCESS LOOKS LIKE ===

After this chapter:
- `rvllm generate` shows timing breakdown (prefill vs decode)
- `rvllm generate` shows KV cache memory estimate
- `rvllm inspect` shows a clean consolidated model report
- Error messages are helpful, not stack traces
- Debug logging available for troubleshooting

This is the MVP milestone — the engine works end to end.

=== WHAT TO PRODUCE ===

Produce ONLY the modified files. The core inference pipeline (model, layers,
tokenizer, sampler) should be unchanged from ch07.
```
