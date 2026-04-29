# Chapter 4 Validation Tests

Language-agnostic tests for model loading and inspection (Chapter 4).

## Running

```bash
# Set env var to your binary/command, then run pytest
RVLLM_CH04_BIN="cargo run -- inspect --model openai-community/gpt2" pytest spec/ch04/validation/ -v
RVLLM_CH04_BIN="uv run python -m rvllm inspect --model openai-community/gpt2" pytest spec/ch04/validation/ -v

# Rust reference implementation
source spec/runners/rust.env
pytest spec/ch04/validation/ -v
```

See [spec/runners/README.md](../../runners/README.md) for full details.

## What the Tests Check

| Test Class | Tests | What It Validates |
|------------|-------|-------------------|
| `TestModelLoading` | 2 | Model and tokenizer loaded |
| `TestModelConfig` | 2 | Hidden dim (768), layers/heads (12) |
| `TestWeights` | 2 | Weight count (148), param count (~124M) |
| `TestTokenizerRoundTrip` | 1 | "What is AI?" round-trip |

**Total: 7 assertions across 4 test classes.**

## Timeout

3 minutes (covers model download on first run).

## Scope

This chapter tests **loading and inspecting only**. No forward pass, no
generation. Those are tested in Chapters 5-7.
