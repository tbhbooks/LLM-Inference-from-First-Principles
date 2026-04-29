# Chapter 17 -- LLM Prompt Template: Speculative Decoding

Copy and paste the prompt below into your LLM of choice, or use it as a
reference while implementing by hand. This builds on the project from
Chapters 1-14.

---

## The Prompt

```
I am building an LLM inference engine called "rvllm". This is Chapter 17.
I have an existing project from Chapters 1-14 with:
- Full GPT-2 model with KV cache (ch04-06)
- Greedy generation loop (ch07)
- PagedAttention memory management (ch09-10)
- Continuous batching scheduler (ch11-12)
- Engine loop with Sampler trait (ch13)
- Sampling strategies pipeline (ch14)

Now implement speculative decoding: a small draft model proposes K tokens,
then the target model verifies all K tokens in a single forward pass.
Accepted tokens skip individual decode steps, producing 1 to K+1 tokens
per target forward pass.

LANGUAGE/FRAMEWORK: [Rust with candle / Python with PyTorch / your choice]

=== WHAT TO CREATE / MODIFY ===

  NEW FILES:
    src/engine/speculative.rs     <-- SpeculativeDecoder
    examples/ch17_speculative_decoding  <-- Demo program

  MODIFY:
    src/types.rs          <-- Add SpeculativeConfig
    src/engine/mod.rs     <-- Re-export speculative module

  KEEP UNCHANGED:
    src/model/ (all files)
    src/tokenizer/ (all files)
    src/memory/ (all files)
    src/api/ (all files)

=== SPECULATIVE CONFIG ===

  struct SpeculativeConfig:
      draft_tokens: int       // K — number of tokens to draft per step (default 4)

  Validation:
      draft_tokens >= 1

=== DRAFT RESULT ===

  struct DraftResult:
      proposed_tokens: list[TokenId]      // length K
      proposed_logits: list[FloatArray]    // length K, each [vocab_size]

  The draft model produces K tokens autoregressively.
  Each token comes with the logits the draft model assigned.

=== VERIFY RESULT ===

  struct VerifyResult:
      accepted_tokens: list[TokenId]   // 0 to K accepted draft tokens
      bonus_token: Option[TokenId]     // extra token from target model
      num_accepted: int                // count of accepted draft tokens

  Total tokens produced = num_accepted + (1 if bonus_token else 0)
  Minimum: 1 token (immediate reject, bonus only)
  Maximum: K + 1 tokens (all accepted + bonus)

=== SPECULATIVE DECODER ===

  struct SpeculativeDecoder:
      config: SpeculativeConfig
      total_drafted: int        // running count for acceptance rate
      total_accepted: int       // running count for acceptance rate

  Methods:

      draft(context_tokens, K) -> DraftResult:
          // Use draft model to generate K tokens autoregressively
          // For this chapter: use a mock draft model (lookup table or heuristic)
          tokens = []
          logits_list = []
          current_context = context_tokens
          for i in 0..K:
              logits = draft_model.forward(current_context)
              token = argmax(logits)    // draft always uses greedy
              tokens.append(token)
              logits_list.append(logits)
              current_context = current_context + [token]
          return DraftResult(tokens, logits_list)

      verify(draft_result, target_logits) -> VerifyResult:
          // target_logits: shape [K+1, vocab_size]
          //   position i: target model's logits for position of draft_token[i]
          //   position K: target model's logits for position after last draft token
          //
          // Greedy verification:
          accepted = []
          for i in 0..K:
              if argmax(target_logits[i]) == draft_result.proposed_tokens[i]:
                  accepted.append(draft_result.proposed_tokens[i])
              else:
                  // Reject — use target model's choice as bonus
                  bonus = argmax(target_logits[i])
                  update_acceptance_stats(len(accepted), K)
                  return VerifyResult(accepted, Some(bonus), len(accepted))
          // All K accepted — bonus is the next token from target
          bonus = argmax(target_logits[K])
          update_acceptance_stats(K, K)
          return VerifyResult(accepted, Some(bonus), K)

      acceptance_rate() -> float:
          if total_drafted == 0: return 0.0
          return total_accepted / total_drafted

=== GREEDY VERIFICATION ALGORITHM (detailed) ===

  Given:
    draft_tokens: [d0, d1, d2, d3]          // K=4 draft tokens
    target_logits: [L0, L1, L2, L3, L4]     // K+1 logit vectors from target

  Step-by-step:
    i=0: argmax(L0) == d0?  YES → accept d0
    i=1: argmax(L1) == d1?  YES → accept d1
    i=2: argmax(L2) == d2?  NO  → reject, bonus = argmax(L2)
         STOP. Return accepted=[d0, d1], bonus=argmax(L2), num_accepted=2

  If all match:
    i=0: argmax(L0) == d0?  YES → accept d0
    i=1: argmax(L1) == d1?  YES → accept d1
    i=2: argmax(L2) == d2?  YES → accept d2
    i=3: argmax(L3) == d3?  YES → accept d3
    ALL ACCEPTED. bonus = argmax(L4), num_accepted=4
    Return accepted=[d0,d1,d2,d3], bonus=argmax(L4), num_accepted=4

  Total tokens in output: num_accepted + 1 (bonus is always present)

=== MOCK DRAFT MODEL ===

  For this chapter, use a mock draft model:
  - Given a context, return predetermined logits from a lookup table
  - Or use a simple heuristic: the draft model always proposes a fixed
    sequence of tokens (e.g., token IDs cycling through a small set)

=== MOCK TARGET MODEL ===

  For this chapter, use a mock target model:
  - Given draft tokens, return predetermined target logits
  - Control which tokens match and which don't to demonstrate all scenarios

=== DEMO PROGRAM (examples/ch17_speculative_decoding) ===

The demo uses mock models (no real model loading). It should demonstrate
4 scenarios with K=4:

1. FULL ACCEPT: Draft 4 tokens, all accepted + bonus = 5 tokens in 1 pass
   - All argmax(target_logits[i]) == draft_tokens[i]
   - Show: "Drafted: [d0, d1, d2, d3], Accepted: 4/4, Bonus: d4, Total: 5"

2. PARTIAL ACCEPT: Draft 4, accept 2, reject at position 2, bonus = 3 tokens
   - argmax(target_logits[0]) == d0, argmax(target_logits[1]) == d1
   - argmax(target_logits[2]) != d2 → reject
   - Show: "Drafted: [d0, d1, d2, d3], Accepted: 2/4, Bonus: t2, Total: 3"

3. IMMEDIATE REJECT: Draft 4, reject at position 0, bonus only = 1 token
   - argmax(target_logits[0]) != d0 → immediate reject
   - Show: "Drafted: [d0, d1, d2, d3], Accepted: 0/4, Bonus: t0, Total: 1"
   - Same as normal decode — no speedup

4. ACCEPTANCE RATE: Run 20 speculative steps, track cumulative accept rate
   - Mix of full/partial/rejected across steps
   - Show per-step results and running acceptance rate
   - Final: "Acceptance rate: X/Y drafted tokens accepted (Z%)"
   - Show speedup: "Average tokens per target pass: N (vs 1 without speculation)"

Output format: 5 parts
  - PART 1: The Problem (decode is bandwidth-bound)
  - PART 2: Draft Phase
  - PART 3: Verify Phase (all 3 scenarios)
  - PART 4: Speedup Analysis
  - PART 5: Acceptance Rate (20-step run)

End with: "Chapter 17 complete. Next: Structured Output (ch18)"

=== WHAT SUCCESS LOOKS LIKE ===

Running the demo produces clearly labeled output showing:
- Why speculative decoding helps (GPU underutilized during decode)
- How the draft model proposes K tokens cheaply
- How verification accepts or rejects at each position
- The bonus token mechanism (always get at least 1 token)
- Speedup analysis comparing with and without speculation
- Acceptance rate tracking over multiple steps

Each section shows token IDs and acceptance decisions.
The output demonstrates that speculative decoding can produce
multiple tokens per target forward pass.

=== WHAT TO PRODUCE ===

Produce the complete set of NEW or MODIFIED source files.
Do NOT recreate files from earlier chapters that are unchanged.

After this chapter, the engine supports speculative decoding —
drafting K tokens with a cheap model and verifying with the target,
producing 1 to K+1 tokens per expensive forward pass.
```
