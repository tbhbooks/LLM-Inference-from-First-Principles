# Chapter 14 -- Interface Specification: Sampling Strategies

This is a language-agnostic specification. It defines the contracts for
temperature scaling, top-k filtering, top-p (nucleus) sampling, repetition
penalty, and the composable LogitsProcessor pipeline.

---

## 1. Overview

Chapter 7 built the simplest possible sampler: greedy argmax. It always picks
the most probable token. Deterministic, fast, boring.

This chapter replaces greedy with a full sampling pipeline:

- **TemperatureProcessor** — Scale logits to control randomness
- **TopKProcessor** — Keep only the top-k most probable tokens
- **TopPProcessor** — Keep the smallest set whose cumulative probability >= p
- **RepetitionPenaltyProcessor** — Penalize tokens that already appeared
- **LogitsProcessor pipeline** — Compose processors in order
- **SamplingParams** — User-facing config that builds the pipeline

After this chapter, the engine supports creative, diverse, non-repetitive text
generation — not just greedy decoding.

---

## 2. Dependencies

From Chapter 7: `Sampler` trait, `GreedySampler`
From Chapter 13: Engine loop that calls `sampler.sample(logits)`

---

## 3. LogitsProcessor Trait

The core abstraction. Every sampling strategy implements this trait:

```
trait LogitsProcessor:
    process(logits: FloatArray, token_ids_so_far: list[TokenId]) -> FloatArray
```

- `logits` — Raw unnormalized scores from the LM head, shape `[vocab_size]`
  (already extracted to the last position before reaching the processor).
- `token_ids_so_far` — All tokens generated so far in this sequence
  (needed by repetition penalty to know which tokens to penalize).
- Returns modified logits (same shape). Never changes the input in place.

Processors are composable: the output of one is the input to the next.

---

## 4. TemperatureProcessor

```
struct TemperatureProcessor:
    temperature: float    // > 0.0; values < 1.0 sharpen, > 1.0 flatten

process(logits, token_ids_so_far):
    return logits / temperature    // divide every logit by temperature
```

**Behavior:**
- `temperature = 1.0` — No change (identity).
- `temperature < 1.0` (e.g., 0.5) — Sharpens the distribution. Makes the model
  more confident. In the limit (→ 0), approaches greedy.
- `temperature > 1.0` (e.g., 2.0) — Flattens the distribution. More random.
  All tokens become more equally probable.
- `temperature = 0.0` — Special case: skip the processor entirely and use
  argmax instead of sampling. This is handled at the pipeline level, not inside
  the processor.

**Invariant:** All logit values are scaled by the same factor. Relative ordering
is preserved (argmax of scaled logits == argmax of original logits).

---

## 5. TopKProcessor

```
struct TopKProcessor:
    k: int    // number of tokens to keep; -1 or 0 means disabled

process(logits, token_ids_so_far):
    if k <= 0 or k >= len(logits):
        return logits              // disabled — pass through

    // Find the k-th largest value
    threshold = k_th_largest(logits, k)

    // Set everything below the threshold to -infinity
    for i in 0..len(logits):
        if logits[i] < threshold:
            logits[i] = -infinity

    return logits
```

**Behavior:**
- Only the top-k tokens remain with finite logits.
- After softmax, only those k tokens have non-zero probability.
- `k = 1` is equivalent to greedy decoding.
- `k = 50` is a common default for creative generation.

---

## 6. TopPProcessor (Nucleus Sampling)

```
struct TopPProcessor:
    p: float    // cumulative probability threshold; 0.0 < p <= 1.0

process(logits, token_ids_so_far):
    if p >= 1.0:
        return logits              // disabled — pass through

    // Step 1: Convert to probabilities
    probs = softmax(logits)

    // Step 2: Sort by probability descending
    sorted_indices = argsort(probs, descending=true)
    sorted_probs = probs[sorted_indices]

    // Step 3: Compute cumulative sum
    cumsum = cumulative_sum(sorted_probs)

    // Step 4: Find the cutoff — smallest set with cumsum >= p
    // Keep all tokens up to and including the one that pushes cumsum past p
    cutoff_index = first_index_where(cumsum >= p)

    // Step 5: Zero out everything after the cutoff
    for i in (cutoff_index + 1)..len(sorted_indices):
        logits[sorted_indices[i]] = -infinity

    return logits
```

**Behavior:**
- Adapts dynamically: when the model is confident (one token has 0.95 prob),
  very few tokens survive. When uncertain, more tokens survive.
- `p = 0.9` is the standard nucleus sampling threshold.
- `p = 1.0` disables the filter (all tokens pass).
- `p = 0.0` would keep nothing — invalid, should be rejected at construction.

**Note:** TopP operates on probabilities (requires softmax), but returns
modified logits (with -infinity for filtered tokens). The final softmax
happens after the full pipeline.

---

## 7. RepetitionPenaltyProcessor

```
struct RepetitionPenaltyProcessor:
    penalty: float    // >= 1.0; 1.0 means no penalty

process(logits, token_ids_so_far):
    if penalty == 1.0:
        return logits              // disabled — pass through

    // Collect unique token IDs that appeared
    seen = unique(token_ids_so_far)

    for token_id in seen:
        if logits[token_id] > 0:
            logits[token_id] = logits[token_id] / penalty    // positive logits: divide
        else:
            logits[token_id] = logits[token_id] * penalty    // negative logits: multiply (makes more negative)

    return logits
```

**Behavior:**
- Only affects tokens that have already appeared in the generated sequence.
- Positive logits are divided by the penalty (reduced). Negative logits are
  multiplied by the penalty (pushed further negative). Both reduce the
  token's probability after softmax.
- `penalty = 1.2` is a common default.
- The asymmetric treatment (divide vs multiply) is how vLLM and HuggingFace
  implement it — it ensures the penalty always reduces probability regardless
  of logit sign.

---

## 8. SamplingParams

User-facing configuration that controls the full sampling pipeline:

```
struct SamplingParams:
    temperature: float       // default 1.0; 0.0 = greedy
    top_k: int               // default -1 (disabled)
    top_p: float             // default 1.0 (disabled)
    repetition_penalty: float // default 1.0 (disabled)
    max_tokens: int          // default 200
```

### 8.1 Validation Rules

| Parameter | Valid range | Error if violated |
|-----------|-------------|-------------------|
| `temperature` | >= 0.0 | "temperature must be non-negative" |
| `top_k` | -1, or >= 1 | "top_k must be -1 (disabled) or >= 1" |
| `top_p` | 0.0 < p <= 1.0 | "top_p must be in (0.0, 1.0]" |
| `repetition_penalty` | >= 1.0 | "repetition_penalty must be >= 1.0" |
| `max_tokens` | >= 1 | "max_tokens must be >= 1" |

### 8.2 Default SamplingParams

The default `SamplingParams` produces greedy decoding (equivalent to ch07):
- `temperature = 1.0`, `top_k = -1`, `top_p = 1.0`, `repetition_penalty = 1.0`
- With all processors disabled, no logit modification occurs, and sampling from
  `softmax(logits)` with temperature 1.0 is equivalent to multinomial sampling.
- For true greedy, set `temperature = 0.0`.

---

## 9. Pipeline Construction

Build the processor list from SamplingParams. Order matters:

```
function build_pipeline(params: SamplingParams) -> list[LogitsProcessor]:
    processors = []

    // Step 1: Repetition penalty first — operates on raw logits
    if params.repetition_penalty > 1.0:
        processors.append(RepetitionPenaltyProcessor(params.repetition_penalty))

    // Step 2: Temperature — scales logits before filtering
    if params.temperature != 0.0 and params.temperature != 1.0:
        processors.append(TemperatureProcessor(params.temperature))

    // Step 3: Top-K — coarse filter
    if params.top_k > 0:
        processors.append(TopKProcessor(params.top_k))

    // Step 4: Top-P — fine filter (applied after top-k narrows the field)
    if params.top_p < 1.0:
        processors.append(TopPProcessor(params.top_p))

    return processors
```

### 9.1 Pipeline Execution

```
function sample_with_pipeline(
    logits: FloatArray,          // [vocab_size], already extracted from last position
    token_ids_so_far: list[TokenId],
    processors: list[LogitsProcessor],
    temperature: float
) -> TokenId:

    // Apply all processors in order
    for processor in processors:
        logits = processor.process(logits, token_ids_so_far)

    // Final selection
    if temperature == 0.0:
        return argmax(logits)                        // greedy — deterministic
    else:
        probs = softmax(logits)                      // convert to probabilities
        return sample_from_distribution(probs)       // multinomial sampling
```

### 9.2 Pipeline Order Rationale

| Order | Processor | Why this position |
|-------|-----------|-------------------|
| 1 | RepetitionPenalty | Operates on raw logits; must see original scale |
| 2 | Temperature | Scales before filtering; affects which tokens survive top-k/top-p |
| 3 | Top-K | Coarse filter — removes obviously unlikely tokens |
| 4 | Top-P | Fine filter — adapts cutoff to remaining distribution |

---

## 10. Updated Sampler Trait

The Sampler trait from ch07 gains context:

```
trait Sampler:
    sample(logits: Tensor, token_ids_so_far: list[TokenId]) -> Result<TokenId>
```

The ch07 GreedySampler ignored `token_ids_so_far`. The new `PipelineSampler`
uses it for repetition penalty.

```
struct PipelineSampler:
    processors: list[LogitsProcessor]
    temperature: float
    rng: RandomGenerator

    sample(logits, token_ids_so_far):
        // Extract last position logits
        last_logits = logits[0, -1, :]    // [vocab_size]

        // Apply processor pipeline
        for proc in processors:
            last_logits = proc.process(last_logits, token_ids_so_far)

        // Select token
        if temperature == 0.0:
            return argmax(last_logits)
        else:
            probs = softmax(last_logits)
            return multinomial_sample(probs, rng)
```

---

## 11. Demo Program

The demo (`examples/ch14_sampling_strategies`) should:

1. Create a mock logits vector (10 tokens with known values)
2. Show the raw logits and their softmax probabilities
3. Apply each processor individually and show the effect:
   - Temperature 0.5 (sharpen)
   - Temperature 2.0 (flatten)
   - Top-K with k=3
   - Top-P with p=0.9
   - Repetition penalty 1.2 with some tokens marked as seen
4. Show the full pipeline with a realistic SamplingParams
5. Compare outputs: greedy vs. temperature vs. nucleus sampling

### 11.1 Expected Output Structure

```
=== PART 1: Beyond Argmax ===
[Show raw logits and why greedy always picks the same token]

=== PART 2: Temperature — Controlling Confidence ===
[Show temperature scaling at 0.5, 1.0, 2.0]
[Show probability distributions side by side]

=== PART 3: Top-K and Top-P — Trimming the Tail ===
[Show top-k=3 filtering]
[Show top-p=0.9 nucleus sampling]
[Show how top-p adapts to confidence]

=== PART 4: Repetition Penalty — Stop Saying the Same Thing ===
[Show penalty applied to seen tokens]
[Show before/after probabilities]

=== PART 5: The Pipeline — Composing Strategies ===
[Show pipeline order: repetition → temperature → top-k → top-p]
[Show logits at each stage]

=== PART 6: SamplingParams — The User's Control Panel ===
[Show different SamplingParams presets]
[Show how they produce different distributions]

Chapter 14 complete. Next: Building the API Server (ch15)
```

---

## 12. Correctness Criteria

1. **Temperature scales logits:** `logits / T` for all values.
2. **Top-K filters:** Exactly k tokens remain with finite logits.
3. **Top-P filters:** Smallest set with cumulative prob >= p survives.
4. **Repetition penalty:** Seen tokens have reduced probability.
5. **Pipeline composes:** Processors applied in order, output of one feeds next.
6. **Greedy preserved:** `temperature = 0.0` produces argmax (deterministic).
7. **Default params:** Default SamplingParams with `temperature = 0.0` matches
   ch07 GreedySampler behavior exactly.
8. **Probabilities sum to 1:** After softmax of processed logits, probabilities
   sum to 1.0 (within floating-point tolerance).

---

## 13. Validation Summary

| Test | What it checks |
|------|---------------|
| Part 1 present | "Beyond Argmax" section exists |
| Part 2 present | Temperature section with probability values |
| Part 3 present | Top-K and Top-P section |
| Part 4 present | Repetition penalty section |
| Part 5 present | Pipeline/processor composition section |
| Part 6 present | SamplingParams section |
| Temperature values | Shows probability values changing with temperature |
| Top-K filtering | Shows tokens being filtered |
| Top-P / nucleus | Mentions nucleus or top-p with cumulative probability |
| Repetition | Shows penalty applied to repeated tokens |
| Pipeline order | Shows processors applied in sequence |
| Completion marker | "Chapter 14 complete" appears |
