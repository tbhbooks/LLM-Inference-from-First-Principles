# Chapter 17 Validation Tests

Language-agnostic tests for Speculative Decoding (Chapter 17).

## Running

```bash
RVLLM_CH17_BIN="cargo run --example ch17_speculative_decoding" pytest spec/ch17/validation/ -v
RVLLM_CH17_BIN="uv run python -m rvllm ch17_demo" pytest spec/ch17/validation/ -v

# Rust reference implementation
source spec/runners/rust.env
pytest spec/ch17/validation/ -v
```

See [spec/runners/README.md](../../runners/README.md) for full details.

## What the Tests Check

| Test Class | Tests | What It Validates |
|------------|-------|-------------------|
| `TestStructure` | 5 | All 5 parts present (Problem, Draft, Verify, Speedup, Acceptance Rate) |
| `TestDraftPhase` | 2 | Draft mentioned, draft tokens shown |
| `TestVerifyPhase` | 6 | Accept/reject mentioned, bonus token, full accept, partial accept, immediate reject scenarios |
| `TestAcceptance` | 2 | Acceptance rate shown, multiple steps demonstrated |
| `TestSpeedup` | 2 | Speedup mentioned, comparison with standard decode |
| `TestCompletion` | 1 | "Chapter 17 complete" marker present |

**Total: 18 assertions across 6 test classes.**

## Timeout

30 seconds (no model loading -- uses mock models only).
