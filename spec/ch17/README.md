# Chapter 17: Speculative Decoding

Draft-then-verify speculative decoding: a small model proposes K tokens,
the target model verifies all K in one forward pass.

## Spec Artifacts

| File | Purpose |
|------|---------|
| [prompt-template.md](prompt-template.md) | LLM prompt for implementing this chapter |
| [interface-spec.md](interface-spec.md) | Language-agnostic contracts and algorithms |
| [expected-output.txt](expected-output.txt) | Expected demo output format |
| [component-diagram.md](component-diagram.md) | Class and data flow diagrams (Figures 17.1-17.3) |
| [sequence-diagram.md](sequence-diagram.md) | Interaction sequence diagrams (Figures 17.4-17.6) |
| [validation/](validation/) | Pytest test suite |

## Key Concepts

- **Draft phase:** Small model generates K tokens cheaply via greedy argmax
- **Verify phase:** Target model processes all K tokens in one forward pass
- **Greedy verification:** Accept while argmax(target) == draft; reject on first mismatch
- **Bonus token:** Always produced (target's choice at rejection point or position K+1)
- **Guarantee:** Output identical to standard decode -- speculation never degrades quality
- **Speedup:** 1 to K+1 tokens per target forward pass (avg = K * acceptance_rate + 1)

## Dependencies

- Chapter 13: Engine loop
- Chapter 14: Sampler trait and sampling pipeline

## Validation

```bash
RVLLM_CH17_BIN="cargo run --example ch17_speculative_decoding" pytest spec/ch17/validation/ -v
```

18 assertions across 6 test classes. 30-second timeout (mock models only).
