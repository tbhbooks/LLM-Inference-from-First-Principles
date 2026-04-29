# Chapter 18 Validation Tests

Language-agnostic tests for Structured Output (Chapter 18).

## Running

```bash
RVLLM_CH18_BIN="cargo run --example ch18_structured_output" pytest spec/ch18/validation/ -v
RVLLM_CH18_BIN="uv run python -m rvllm ch18_demo" pytest spec/ch18/validation/ -v

# Rust reference implementation
source spec/runners/rust.env
pytest spec/ch18/validation/ -v
```

See [spec/runners/README.md](../../runners/README.md) for full details.

## What the Tests Check

| Test Class | Tests | What It Validates |
|------------|-------|-------------------|
| `TestStructure` | 5 | All 5 parts present (The Problem, FSM Construction, Unconstrained vs Constrained, Token Masking, Pipeline Integration) |
| `TestFSM` | 3 | FSM/state machine mentioned, states shown, transitions shown |
| `TestConstrained` | 3 | Constrained/mask mentioned, tokens filtered, valid/invalid distinction |
| `TestUnconstrained` | 2 | Unconstrained mentioned, contrast with constrained shown |
| `TestPipelineIntegration` | 2 | LogitsProcessor/pipeline mentioned, integration demonstrated |
| `TestCompletion` | 1 | "Chapter 18 complete" marker present |

**Total: 16 assertions across 6 test classes.**

## Timeout

30 seconds (no model loading — uses mock logits only).
