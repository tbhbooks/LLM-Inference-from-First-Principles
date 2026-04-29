# Chapter 7 -- LLM Prompt Template: The Skeleton Speaks

Copy and paste the prompt below into your LLM of choice, or use it as a
reference while implementing by hand. This builds on the project from
Chapters 1–6.

---

## The Prompt

```
I am building an LLM inference engine called "rvllm". This is Chapter 7.
I have an existing project from Chapters 1–6 with:
- Model loading and tokenizer (ch04)
- Embedding, LayerNorm, Linear, MLP layers (ch05)
- CausalSelfAttention, TransformerBlock, full Gpt2Model with KV cache (ch06)
- The `inspect` subcommand shows model info and top-5 predictions
- The `generate` subcommand is still stubbed

Now implement the generation loop with greedy decoding and wire up `generate`.

LANGUAGE/FRAMEWORK: [Rust with candle / Python with PyTorch / your choice]

=== WHAT TO CREATE / MODIFY ===

  NEW FILES:
    src/sampling/greedy.rs         <-- GreedySampler: Sampler trait impl (argmax)

  MODIFY:
    src/sampling/mod.rs            <-- add `pub mod greedy;` and re-export
    src/main.rs                    <-- Wire `generate` subcommand to real generation

  KEEP UNCHANGED:
    src/model/ (all files from ch04-06)
    src/tokenizer/ (from ch04)
    src/types.rs, src/error.rs, src/config.rs
    src/engine/, src/scheduler/, src/memory/, src/api/ (still stubs)

=== GREEDY SAMPLER ===

Implement the Sampler trait:

  trait Sampler:
      sample(logits: Tensor) -> Result<TokenId>

  GreedySampler.sample(logits):
      last_logits = logits[0, -1, :]    → [50257]
      return argmax(last_logits)

That's it. No temperature, no top-k, no randomness.
Greedy decoding always picks the most likely next token.

=== GENERATION LOOP ===

Wire this into the `generate` subcommand:

1. Load model (Gpt2Model, implements Model trait)
2. Load tokenizer (HuggingFace, implements TokenizerBackend trait)
3. Create sampler (GreedySampler, implements Sampler trait)
4. Start timer

5. ENCODE: token_ids = tokenizer.encode(prompt)
6. CLEAR CACHE: model.reset_cache()

7. PREFILL: logits = model.forward(token_ids, pos_offset=0)
            next_token = sampler.sample(logits)

8. DECODE LOOP:
   generated = [next_token]
   position = len(token_ids)
   for step in 1..max_tokens:
       if next_token == tokenizer.eos_token_id():
           break
       logits = model.forward([next_token], pos_offset=position)
       next_token = sampler.sample(logits)
       generated.append(next_token)
       position += 1

9. DECODE: text = tokenizer.decode(generated)
10. Stop timer

11. PRINT OUTPUT (see format below)

=== CLI ARGUMENTS ===

  rvllm generate --prompt "text" [--max-tokens 200] [--model openai-community/gpt2]

Defaults:
  prompt: "The future of artificial intelligence is"
  max_tokens: 200
  model: openai-community/gpt2

=== OUTPUT FORMAT ===

Loading model: openai-community/gpt2
  Model files downloaded/cached
  Tokenizer loaded (vocab_size: 50257)
  Model weights loaded (12 layers, 768 dim, 12 heads)

Prompt: "The future of artificial intelligence is"
Prompt tokens: 6
Generating up to 200 tokens...

--- Generated Text ---
The future of artificial intelligence is [... continuation ...]
--- End ---

--- Stats ---
Tokens generated: N
Time: X.XXs
Speed: X.XX tokens/sec

=== WHAT SUCCESS LOOKS LIKE ===

Running `rvllm generate --prompt "The future of artificial intelligence is"`
produces coherent, grammatically plausible English text continuing the prompt.
GPT-2 124M is small — expect reasonable but not brilliant prose.

The output includes:
- Loading info (model, tokenizer, config)
- The prompt and token count
- Generated text (delimited)
- Stats (count, time, speed)

=== COMMON FAILURE MODES ===

| Symptom                                | Likely cause                           |
|----------------------------------------|----------------------------------------|
| All outputs same token repeated        | Conv1D weights not transposed (ch05)   |
| Coherent first token, then degrades    | KV cache not appended correctly (ch06) |
| Crash on second decode step            | KV cache concat dimension wrong        |
| Empty generated text                   | Sampler returning EOS immediately      |
| Very slow (< 1 tok/sec)              | Not using KV cache                     |

=== WHAT TO PRODUCE ===

Produce the complete set of NEW or MODIFIED source files.
Do NOT recreate files from ch04-06 that are unchanged.

After this chapter, `generate` is fully functional.
`inspect` continues to work as before.
```
