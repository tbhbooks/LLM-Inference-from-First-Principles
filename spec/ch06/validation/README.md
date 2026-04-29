# Chapter 6 Validation Tests

Language-agnostic tests for Attention, KV Cache, and Full Forward Pass (Chapter 6).

## Running

```bash
RVLLM_CH06_BIN="cargo run -- inspect --model openai-community/gpt2" pytest spec/ch06/validation/ -v
RVLLM_CH06_BIN="uv run python -m rvllm inspect --model openai-community/gpt2" pytest spec/ch06/validation/ -v

# Rust reference implementation
source spec/runners/rust.env
pytest spec/ch06/validation/ -v
```

See [spec/runners/README.md](../../runners/README.md) for full details.

## What the Tests Check

| Test Class | Tests | What It Validates |
|------------|-------|-------------------|
| `TestModelLoaded` | 1 | Full model assembled/loaded |
| `TestLogitsShape` | 1 | Vocab size 50257 in logits context |
| `TestTopPredictions` | 3 | Numbered predictions, plausible tokens, logit values |
| `TestNoNaN` | 1 | No NaN/Inf in output |
| `TestRunningExample` | 1 | "What is AI?" appears in output |

**Total: 7 assertions across 5 test classes.**

## Timeout

3 minutes (model download + full forward pass).
