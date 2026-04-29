# Chapter 5 Validation Tests

Language-agnostic tests for GPT-2 building blocks: Embedding, LayerNorm, MLP (Chapter 5).

## Running

```bash
# Set env var to your binary/command, then run pytest
RVLLM_CH05_BIN="cargo run -- inspect --model openai-community/gpt2" pytest spec/ch05/validation/ -v
RVLLM_CH05_BIN="uv run python -m rvllm inspect --model openai-community/gpt2" pytest spec/ch05/validation/ -v

# Rust reference implementation
source spec/runners/rust.env
pytest spec/ch05/validation/ -v
```

See [spec/runners/README.md](../../runners/README.md) for full details.

## What the Tests Check

| Test Class | Tests | What It Validates |
|------------|-------|-------------------|
| `TestLayersLoaded` | 3 | Output confirms embedding, LayerNorm, MLP loaded |
| `TestEmbeddingShape` | 1 | Token embedding dimensions [50257, 768] |
| `TestPartialForwardPass` | 3 | LayerNorm mean ≈ 0, MLP std > 0, no NaN |
| `TestConv1DHandling` | 1 | Mentions Conv1D transpose |

**Total: 8 assertions across 4 test classes.**

## Timeout

The test suite allows up to 3 minutes for the full run (model download + layer verification).
