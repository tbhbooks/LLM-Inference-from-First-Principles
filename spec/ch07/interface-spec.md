# Chapter 7 -- Interface Specification: The Skeleton Speaks

This is a language-agnostic specification. It defines the contracts for
greedy decoding, the generation loop, and the `generate` CLI subcommand.

---

## 1. Overview

This chapter brings the model to life with:

- **GreedySampler** — Implement the `Sampler` trait with argmax
- **Generation loop** — Prefill → decode with KV cache → EOS/max check
- **`generate` subcommand** — Wire real generation into the CLI

After this chapter, `rvllm generate --prompt "..."` produces real text.

---

## 2. Dependencies on Chapters 4–6

From Chapter 4: model loading, tokenizer
From Chapter 5: Embedding, LayerNorm, Linear, MLP
From Chapter 6: CausalSelfAttention, TransformerBlock, Gpt2Model (Model trait)

---

## 3. Greedy Sampler

```
Input:  logits [B, T, vocab_size]  (typically B=1, T=1 during decode)
Output: token_id (integer)

Step 1: last_logits = logits[0, -1, :]     → [vocab_size]
Step 2: token_id = argmax(last_logits)

Return: token_id
```

The sampler implements the `Sampler` trait:
```
trait Sampler:
    sample(logits: Tensor) -> Result<TokenId>
```

Greedy sampling (argmax) is deterministic — same model + prompt always produces
the same output.

---

## 4. Generation Loop

### 4.1 Algorithm

```
function generate(model, tokenizer, sampler, prompt, max_tokens):
    // Encode
    token_ids = tokenizer.encode(prompt)
    prompt_len = len(token_ids)

    // Clear any prior KV cache
    model.reset_cache()

    // Prefill: process all prompt tokens at once
    logits = model.forward(token_ids, position_offset=0)    → [1, prompt_len, 50257]
    next_token = sampler.sample(logits)

    // Decode loop
    generated = [next_token]
    position = prompt_len

    for step in 1..max_tokens:
        if next_token == eos_token_id:  // 50256
            break

        logits = model.forward([next_token], position_offset=position)  → [1, 1, 50257]
        next_token = sampler.sample(logits)

        generated.append(next_token)
        position += 1

    // Decode to text
    text = tokenizer.decode(generated)
    return text
```

### 4.2 Generation Parameters

| Parameter   | Value | Notes |
|-------------|-------|-------|
| Decoding    | Greedy (argmax) | No temperature, no sampling |
| Default prompt | `"The future of artificial intelligence is"` | |
| Default max_tokens | 200 | |
| Stop condition | EOS token (50256) **or** max_tokens reached | |
| Batch size  | 1 | Single sequence |

### 4.3 Timing

The implementation should measure and report:
- **Total time** — wall-clock from start of generation to end
- **Tokens generated** — count of generated tokens
- **Speed** — tokens per second (tokens_generated / total_time)

---

## 5. CLI: `generate` Subcommand

```
rvllm generate --prompt "The future of artificial intelligence is" [--max-tokens 200] [--model openai-community/gpt2]
```

### 5.1 Output Format

```
Loading model: openai-community/gpt2
  [loading details]

Prompt: "The future of artificial intelligence is"
Prompt tokens: 6
Generating up to 200 tokens...

--- Generated Text ---
The future of artificial intelligence is [continuation...]
--- End ---

--- Stats ---
Tokens generated: N
Time: X.XXs
Speed: X.XX tokens/sec
```

---

## 6. Correctness Criteria

1. **Coherent text:** Greedy generation from "The future of artificial intelligence is"
   produces grammatically plausible English for at least 20 tokens.
2. **No degenerate repetition:** No single word repeated 10+ times consecutively.
3. **Prompt echo:** Generated text starts with or is preceded by the prompt.
4. **KV cache works:** Prefill populates cache; decode extends it. The model
   should not re-process all tokens on each step.
5. **EOS handling:** Generation stops at EOS token (50256) or max_tokens.
6. **Statistics printed:** Speed, token count, and timing are reported.
7. **Deterministic:** Running twice with the same prompt produces the same text.

---

## 7. Common Failure Modes

| Symptom | Likely cause |
|---------|-------------|
| All outputs same token repeated | Conv1D weights not transposed (ch05 bug) |
| Coherent for first token, then degrades | KV cache not being appended correctly |
| Crash on second decode step | KV cache concat dimension wrong |
| Empty generated text | Sampler returning EOS immediately |
| Very slow generation | Not using KV cache (recomputing full sequence each step) |
| Output differs from HuggingFace | Using approximate GELU instead of exact |

---

## 8. Validation Summary

| Test | What it checks |
|------|---------------|
| Loading indicator | Model loading phase shown |
| Tokenizer loaded | Tokenizer mentioned in output |
| Generated text section | Delimited section exists and is non-empty |
| Minimum word count | Generated text has ≥ 20 words |
| No degenerate repetition | No word repeated 10+ times consecutively |
| Prompt echo | Prompt appears in output |
| No NaN | No NaN/Inf in generated text |
| Speed stat | Reports tokens/sec |
| Token count | Reports number of generated tokens |
| Timing | Reports wall-clock time |
| Exit 0 | Process exits successfully |
