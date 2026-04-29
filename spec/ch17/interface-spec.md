# Chapter 17 -- Interface Specification: Speculative Decoding

This is a language-agnostic specification. It defines the contracts for
draft-then-verify speculative decoding: a small model proposes K tokens,
the target model verifies all K in one forward pass.

---

## 1. Overview

Standard autoregressive decoding generates one token per forward pass. Each
pass underutilizes the GPU — the model is memory-bandwidth-bound during
decode, not compute-bound. The arithmetic units sit idle while weights stream
from memory.

Speculative decoding exploits this imbalance. A cheap draft model proposes K
candidate tokens. The expensive target model verifies all K in a single
forward pass (the same cost as verifying one). If the draft model guessed
correctly, we get K+1 tokens for the price of one target forward pass.

The key guarantee: **output is identical to what the target model would have
produced on its own.** Speculation never degrades quality — it only changes
speed. If a draft token is wrong, we reject it and use the target's token
instead. Worst case, we get exactly 1 token per step (same as no speculation).

After this chapter, the engine can produce multiple tokens per target forward
pass, with configurable draft length and acceptance rate tracking.

---

## 2. Dependencies

From Chapter 13: Engine loop, `Sampler` trait
From Chapter 14: Sampling pipeline (used for final token selection)

The speculative decoder wraps around the engine's decode step — it does not
replace the model or sampler, but orchestrates how many forward passes happen
and how tokens are accepted.

---

## 3. SpeculativeConfig

```
struct SpeculativeConfig:
    draft_tokens: int       // K — number of tokens to draft per step
                            // default: 4
```

### 3.1 Validation Rules

| Parameter | Valid range | Error if violated |
|-----------|-------------|-------------------|
| `draft_tokens` | >= 1 | "draft_tokens must be >= 1" |

### 3.2 Typical Values

| Use case | K | Rationale |
|----------|---|-----------|
| Conservative | 2 | High acceptance, low wasted compute |
| Default | 4 | Good balance of speedup and acceptance |
| Aggressive | 8 | Higher potential speedup, lower acceptance |

---

## 4. Draft Phase

The draft model generates K tokens autoregressively, starting from the
current context.

### 4.1 DraftResult

```
struct DraftResult:
    proposed_tokens: list[TokenId]      // length K
    proposed_logits: list[FloatArray]    // length K, each [vocab_size]
```

### 4.2 Draft Algorithm

```
function draft(context_tokens: list[TokenId], K: int) -> DraftResult:
    tokens = []
    logits_list = []
    current_context = copy(context_tokens)

    for i in 0..K:
        logits = draft_model.forward(current_context)    // cheap forward pass
        token = argmax(logits)                           // draft uses greedy
        tokens.append(token)
        logits_list.append(logits)
        current_context.append(token)                    // extend context

    return DraftResult(tokens, logits_list)
```

**Key points:**
- The draft model is much smaller than the target (e.g., 125M vs 1.5B params).
- Draft always uses greedy selection (argmax). The quality of the draft
  doesn't matter as long as acceptance rate is reasonable.
- Each draft step is cheap — the goal is that K draft passes cost less than
  one target pass.
- The proposed logits are stored for potential use in advanced rejection
  sampling (not used in this chapter's greedy verification, but part of the
  interface for future extensibility).

---

## 5. Verify Phase

The target model processes all K draft tokens in one forward pass, producing
logits for each position.

### 5.1 VerifyResult

```
struct VerifyResult:
    accepted_tokens: list[TokenId]   // 0 to K accepted draft tokens
    bonus_token: Option[TokenId]     // extra token from target model
    num_accepted: int                // count of accepted draft tokens
```

**Invariants:**
- `num_accepted == len(accepted_tokens)`
- `num_accepted` is in range `[0, K]`
- `bonus_token` is always present (there is always at least one new token)
- Total new tokens = `num_accepted + 1` (the bonus)

### 5.2 Target Forward Pass

The target model receives the original context plus all K draft tokens and
produces logits for each position:

```
// Input to target model: [context..., d0, d1, ..., d_{K-1}]
// Output: logits for positions corresponding to d0..d_{K-1} and the next position
// target_logits[i] = target's logits at position of draft token i
// target_logits[K] = target's logits at position after last draft token
```

This is a single forward pass. The target model processes all K+1 positions
at once (like a prefill), not K+1 separate decode steps. That is the source
of the speedup.

---

## 6. Greedy Verification Algorithm

The core of speculative decoding. Compare each draft token against what the
target model would have chosen:

```
function verify(
    draft_result: DraftResult,
    target_logits: list[FloatArray]     // length K+1, each [vocab_size]
) -> VerifyResult:

    K = len(draft_result.proposed_tokens)
    accepted = []

    for i in 0..K:
        target_token = argmax(target_logits[i])

        if target_token == draft_result.proposed_tokens[i]:
            // Draft got it right — accept this token
            accepted.append(draft_result.proposed_tokens[i])
        else:
            // Draft got it wrong — reject this and all subsequent tokens
            // Use the target's choice as the bonus token
            bonus = target_token
            return VerifyResult(
                accepted_tokens = accepted,
                bonus_token = Some(bonus),
                num_accepted = len(accepted)
            )

    // All K draft tokens accepted — bonus is the next token from target
    bonus = argmax(target_logits[K])
    return VerifyResult(
        accepted_tokens = accepted,
        bonus_token = Some(bonus),
        num_accepted = K
    )
```

### 6.1 Verification Scenarios

**Scenario A: Full acceptance (best case)**

```
Draft tokens:  [d0, d1, d2, d3]     (K=4)
Target argmax: [d0, d1, d2, d3, d4] (all match + bonus)

Result: accepted=[d0,d1,d2,d3], bonus=d4, total=5 tokens
Speedup: 5x over standard decode
```

**Scenario B: Partial acceptance**

```
Draft tokens:  [d0, d1, d2, d3]     (K=4)
Target argmax: [d0, d1, t2, -, -]   (mismatch at position 2)

Result: accepted=[d0,d1], bonus=t2, total=3 tokens
Speedup: 3x over standard decode
```

**Scenario C: Immediate rejection (worst case)**

```
Draft tokens:  [d0, d1, d2, d3]     (K=4)
Target argmax: [t0, -, -, -, -]     (mismatch at position 0)

Result: accepted=[], bonus=t0, total=1 token
Speedup: 1x (no gain, same as standard decode)
```

### 6.2 Correctness Guarantee

Greedy verification ensures that the output sequence is identical to what
the target model would produce without speculation:

- Every accepted token matches `argmax(target_logits[i])`.
- The bonus token IS `argmax(target_logits[first_mismatch])` or
  `argmax(target_logits[K])` — exactly what the target would have output.
- No incorrect tokens ever appear in the final sequence.

---

## 7. Acceptance Rate Metrics

Track acceptance statistics across multiple speculative steps:

```
struct AcceptanceMetrics:
    total_drafted: int       // total draft tokens proposed across all steps
    total_accepted: int      // total draft tokens accepted across all steps

    acceptance_rate() -> float:
        if total_drafted == 0: return 0.0
        return total_accepted / total_drafted

    average_tokens_per_step() -> float:
        // Each step produces num_accepted + 1 tokens
        // Over N steps: sum(num_accepted_i + 1) / N
        // = (total_accepted + N) / N
        // Approximation using rate:
        // avg = K * acceptance_rate + 1
        return draft_tokens * acceptance_rate() + 1
```

### 7.1 Interpreting Acceptance Rate

| Rate | Meaning | Action |
|------|---------|--------|
| > 80% | Excellent — draft model closely matches target | Increase K |
| 50-80% | Good — decent speedup | K=4 is reasonable |
| 20-50% | Marginal — some speedup | Decrease K or improve draft model |
| < 20% | Poor — speculation overhead may exceed benefit | Disable or K=1 |

### 7.2 Speedup Calculation

```
tokens_per_step_without_speculation = 1
tokens_per_step_with_speculation = K * acceptance_rate + 1

speedup = tokens_per_step_with_speculation / tokens_per_step_without_speculation
        = K * acceptance_rate + 1
```

Example with K=4, acceptance rate 75%:
- `tokens_per_step = 4 * 0.75 + 1 = 4.0`
- Speedup: **4x** over standard decode

---

## 8. SpeculativeDecoder

The main orchestrating struct:

```
struct SpeculativeDecoder:
    config: SpeculativeConfig
    total_drafted: int        // running count
    total_accepted: int       // running count

    // Create a new decoder with the given config
    new(config: SpeculativeConfig) -> SpeculativeDecoder

    // Run one speculative step: draft K tokens, then verify
    draft(context_tokens: list[TokenId], K: int) -> DraftResult

    // Verify draft tokens against target logits
    verify(draft_result: DraftResult, target_logits: list[FloatArray]) -> VerifyResult

    // Get current acceptance rate
    acceptance_rate() -> float

    // Get average tokens produced per speculative step
    average_tokens_per_step() -> float
```

---

## 9. Integration with Engine Loop

Speculative decoding modifies the engine's decode step:

```
// Without speculation (ch13):
loop:
    logits = target_model.forward(context)      // 1 forward pass
    token = sampler.sample(logits)              // 1 token
    context.append(token)
    if token == eos: break

// With speculation (ch17):
loop:
    draft_result = decoder.draft(context, K)    // K cheap forward passes
    target_logits = target_model.forward(       // 1 expensive forward pass
        context + draft_result.proposed_tokens   // processes all K+1 positions
    )
    verify_result = decoder.verify(draft_result, target_logits)

    // Append all verified tokens at once
    context.extend(verify_result.accepted_tokens)
    if verify_result.bonus_token is not None:
        context.append(verify_result.bonus_token)

    if eos in verify_result.all_tokens(): break
```

The target model forward pass with K+1 tokens is similar in cost to a single
token decode (both are memory-bandwidth-bound), but produces up to K+1 tokens.

---

## 10. Demo Program

The demo (`examples/ch17_speculative_decoding`) uses mock models and should
demonstrate 4 scenarios with K=4.

### 10.1 Expected Output Structure

```
=== PART 1: The Problem — GPU Starvation During Decode ===
[Explain: decode is bandwidth-bound, GPU underutilized]
[Show: 1 token per forward pass, arithmetic units idle]
[Introduce speculative decoding: draft K, verify in 1 pass]

=== PART 2: The Draft Phase — Proposing Tokens Cheaply ===
[Show draft model generating K=4 tokens]
[Show: context → draft → [d0, d1, d2, d3]]
[Emphasize: draft is cheap, doesn't need to be perfect]

=== PART 3: The Verify Phase — Accept or Reject ===
[Scenario 1: Full accept — 4/4 accepted + bonus = 5 tokens]
[Scenario 2: Partial accept — 2/4 accepted + bonus = 3 tokens]
[Scenario 3: Immediate reject — 0/4 accepted + bonus = 1 token]
[Show token-by-token verification decisions]

=== PART 4: The Speedup — More Tokens Per Pass ===
[Compare: without speculation = 1 token/pass]
[With speculation: avg tokens/pass based on acceptance rate]
[Show speedup factor calculation]

=== PART 5: Acceptance Rate — Tracking Performance ===
[Run 20 speculative steps]
[Show per-step: drafted, accepted, bonus, total]
[Show running acceptance rate after each step]
[Final: "X/Y drafted tokens accepted (Z%)" ]
[Final: "Average tokens per target pass: N"]

Chapter 17 complete. Next: Structured Output (ch18)
```

---

## 11. Correctness Criteria

1. **Draft produces K tokens:** `len(draft_result.proposed_tokens) == K`
2. **Verify accepts 0 to K:** `0 <= verify_result.num_accepted <= K`
3. **Bonus always present:** `verify_result.bonus_token is not None`
4. **Total tokens = accepted + 1:** always at least 1 token per step
5. **Greedy match:** accepted tokens match `argmax(target_logits[i])`
   for all accepted positions
6. **Reject stops scanning:** once a mismatch is found, no further
   positions are checked
7. **Bonus correctness:** bonus is `argmax(target_logits[first_mismatch])`
   or `argmax(target_logits[K])` if all accepted
8. **Acceptance tracking:** `acceptance_rate = total_accepted / total_drafted`
9. **Full accept case:** when all K match, total tokens = K + 1
10. **Immediate reject case:** when first token mismatches, total tokens = 1

---

## 12. Validation Summary

| Test | What it checks |
|------|---------------|
| Part 1 present | "The Problem" or GPU underutilization section exists |
| Part 2 present | Draft phase section with proposed tokens |
| Part 3 present | Verify phase section with accept/reject decisions |
| Part 4 present | Speedup comparison section |
| Part 5 present | Acceptance rate tracking over multiple steps |
| Draft tokens shown | Output shows K tokens being drafted |
| Verify decisions | Shows accept/reject at each position |
| Full accept | Demonstrates all-accepted scenario with K+1 tokens |
| Partial accept | Demonstrates partial acceptance scenario |
| Immediate reject | Demonstrates 0-accepted scenario with 1 token |
| Acceptance rate | Shows percentage or ratio of accepted/drafted |
| Speedup shown | Shows tokens-per-pass improvement |
| Completion marker | "Chapter 17 complete" appears |
