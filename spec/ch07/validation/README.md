# Chapter 7 Validation Tests

Language-agnostic tests for the Generation Loop with Greedy Decoding (Chapter 7).

## Running

```bash
RVLLM_CH07_BIN="cargo run -- generate --prompt 'The future of artificial intelligence is'" pytest spec/ch07/validation/ -v
RVLLM_CH07_BIN="uv run python -m rvllm generate --prompt 'The future of artificial intelligence is'" pytest spec/ch07/validation/ -v

# Rust reference implementation
source spec/runners/rust.env
pytest spec/ch07/validation/ -v
```

See [spec/runners/README.md](../../runners/README.md) for full details.

## What the Tests Check

| Test Class | Tests | What It Validates |
|------------|-------|-------------------|
| `TestModelLoading` | 2 | Model and tokenizer loaded |
| `TestGeneratedText` | 6 | Coherent text, no repetition, no NaN, prompt echo |
| `TestStatistics` | 3 | Speed, token count, timing |

**Total: 11 assertions across 3 test classes.**

## Timeout

3 minutes (model download + generation of up to 200 tokens).
