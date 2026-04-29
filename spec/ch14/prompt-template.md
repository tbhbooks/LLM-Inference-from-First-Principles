# Chapter 14 -- LLM Prompt Template: Sampling Strategies

Copy and paste the prompt below into your LLM of choice, or use it as a
reference while implementing by hand. This builds on the project from
Chapters 1-13.

---

## The Prompt

```
I am building an LLM inference engine called "rvllm". This is Chapter 14.
I have an existing project from Chapters 1-13 with:
- Full GPT-2 model with KV cache (ch04-06)
- Greedy generation loop (ch07)
- PagedAttention memory management (ch09-10)
- Continuous batching scheduler (ch11-12)
- Engine loop with Sampler trait (ch13)

Now implement the full sampling pipeline: temperature, top-k, top-p (nucleus),
repetition penalty, and a composable LogitsProcessor chain.

LANGUAGE/FRAMEWORK: [Rust with candle / Python with PyTorch / your choice]

=== WHAT TO CREATE / MODIFY ===

  NEW FILES:
    src/sampling/logits_processor.rs   <-- LogitsProcessor trait
    src/sampling/temperature.rs        <-- TemperatureProcessor
    src/sampling/top_k.rs              <-- TopKProcessor
    src/sampling/top_p.rs              <-- TopPProcessor
    src/sampling/repetition.rs         <-- RepetitionPenaltyProcessor
    examples/ch14_sampling_strategies  <-- Demo program

  MODIFY:
    src/sampling/mod.rs     <-- Re-export new processors, PipelineSampler
    src/types.rs            <-- Add SamplingParams struct
    src/main.rs             <-- Wire ch14 example

  KEEP UNCHANGED:
    src/model/ (all files)
    src/tokenizer/ (all files)
    src/engine/, src/scheduler/, src/memory/
    src/sampling/greedy.rs  <-- Keep as fallback

=== LOGITS PROCESSOR TRAIT ===

  trait LogitsProcessor:
      process(logits: FloatArray, token_ids_so_far: list[TokenId]) -> FloatArray

  Every sampling strategy implements this.
  Processors compose: output of one feeds into the next.

=== TEMPERATURE PROCESSOR ===

  TemperatureProcessor(temperature: float)

  process(logits, _):
      return logits / temperature

  - temperature < 1.0 sharpens (more deterministic)
  - temperature > 1.0 flattens (more random)
  - temperature = 0.0 is handled at the pipeline level (use argmax)

=== TOP-K PROCESSOR ===

  TopKProcessor(k: int)

  process(logits, _):
      if k <= 0 or k >= len(logits): return logits
      threshold = k_th_largest(logits, k)
      set all logits below threshold to -infinity
      return logits

=== TOP-P PROCESSOR (NUCLEUS) ===

  TopPProcessor(p: float)

  process(logits, _):
      if p >= 1.0: return logits
      probs = softmax(logits)
      sort descending by probability
      find cutoff where cumulative sum >= p
      set all tokens after cutoff to -infinity in logits
      return logits

=== REPETITION PENALTY PROCESSOR ===

  RepetitionPenaltyProcessor(penalty: float)

  process(logits, token_ids_so_far):
      if penalty == 1.0: return logits
      for each unique token in token_ids_so_far:
          if logits[token] > 0: logits[token] /= penalty
          else: logits[token] *= penalty
      return logits

=== SAMPLING PARAMS ===

  struct SamplingParams:
      temperature: float        // default 1.0; 0.0 = greedy
      top_k: int                // default -1 (disabled)
      top_p: float              // default 1.0 (disabled)
      repetition_penalty: float // default 1.0 (disabled)
      max_tokens: int           // default 200

  Validation:
      temperature >= 0.0
      top_k == -1 or top_k >= 1
      0.0 < top_p <= 1.0
      repetition_penalty >= 1.0
      max_tokens >= 1

=== PIPELINE CONSTRUCTION ===

  Build processors from SamplingParams in this order:
  1. RepetitionPenalty (if penalty > 1.0)
  2. Temperature (if != 0.0 and != 1.0)
  3. TopK (if k > 0)
  4. TopP (if p < 1.0)

  After all processors run:
  - If temperature == 0.0: argmax (greedy)
  - Else: softmax → multinomial sampling

=== PIPELINE SAMPLER ===

  struct PipelineSampler:
      processors: list[LogitsProcessor]
      temperature: float
      rng: RandomGenerator

  Implements the Sampler trait:

      sample(logits, token_ids_so_far):
          last_logits = logits[0, -1, :]
          for proc in processors:
              last_logits = proc.process(last_logits, token_ids_so_far)
          if temperature == 0.0:
              return argmax(last_logits)
          else:
              probs = softmax(last_logits)
              return multinomial_sample(probs, rng)

=== DEMO PROGRAM (examples/ch14_sampling_strategies) ===

The demo uses mock logits (no model needed). It should:

1. Create a logits vector for 10 tokens with known values
2. Show raw logits and their softmax probabilities
3. Apply each processor individually, printing before/after probabilities
4. Show the full pipeline composing multiple processors
5. Show how different SamplingParams produce different distributions

Output format: 6 parts (Beyond Argmax, Temperature, Top-K/Top-P,
Repetition Penalty, Pipeline, SamplingParams).

End with: "Chapter 14 complete. Next: Building the API Server (ch15)"

=== WHAT SUCCESS LOOKS LIKE ===

Running the demo produces clearly labeled output showing:
- How temperature changes probability distributions
- How top-k removes unlikely tokens
- How top-p adapts its cutoff to model confidence
- How repetition penalty reduces scores of seen tokens
- How the pipeline composes these in order
- How SamplingParams translates user intent into a processor chain

Each section shows probability values (numbers between 0 and 1).
The output demonstrates that different parameters produce different distributions.

=== WHAT TO PRODUCE ===

Produce the complete set of NEW or MODIFIED source files.
Do NOT recreate files from earlier chapters that are unchanged.

After this chapter, the engine can generate with temperature, top-k, top-p,
and repetition penalty — not just greedy argmax.
```
