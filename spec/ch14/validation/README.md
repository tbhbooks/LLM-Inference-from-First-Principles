# Chapter 14 Validation Tests

Language-agnostic tests for Sampling Strategies (Chapter 14).

## Running

```bash
RVLLM_CH14_BIN="cargo run --example ch14_sampling_strategies" pytest spec/ch14/validation/ -v
RVLLM_CH14_BIN="uv run python -m rvllm ch14_demo" pytest spec/ch14/validation/ -v

# Rust reference implementation
source spec/runners/rust.env
pytest spec/ch14/validation/ -v
```

See [spec/runners/README.md](../../runners/README.md) for full details.

## What the Tests Check

| Test Class | Tests | What It Validates |
|------------|-------|-------------------|
| `TestStructure` | 6 | All 6 parts present (Beyond Argmax, Temperature, Top-K/Top-P, Repetition, Pipeline, SamplingParams) |
| `TestTemperature` | 2 | Temperature mentioned, probability values shown |
| `TestTopK` | 2 | Top-k mentioned, filtering demonstrated |
| `TestTopP` | 2 | Top-p/nucleus mentioned, cumulative probability shown |
| `TestRepetitionPenalty` | 2 | Repetition mentioned, before/after effect shown |
| `TestPipeline` | 2 | Pipeline/processor mentioned, stages shown |
| `TestCompletion` | 1 | "Chapter 14 complete" marker present |

**Total: 17 assertions across 7 test classes.**

## Timeout

60 seconds (no model loading — uses mock logits only).
