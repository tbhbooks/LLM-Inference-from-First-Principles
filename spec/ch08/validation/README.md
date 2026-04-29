# Chapter 8 Validation Tests

Language-agnostic tests for CLI polish and MVP milestone (Chapter 8).

## Running

```bash
# Both subcommands are tested
RVLLM_CH08_GEN_BIN="cargo run -- generate --prompt 'The future of artificial intelligence is'" \
RVLLM_CH08_INSPECT_BIN="cargo run -- inspect --model openai-community/gpt2" \
pytest spec/ch08/validation/ -v

# Rust reference implementation
source spec/runners/rust.env
pytest spec/ch08/validation/ -v
```

See [spec/runners/README.md](../../runners/README.md) for full details.

## What the Tests Check

| Test Class | Tests | What It Validates |
|------------|-------|-------------------|
| `TestTimingBreakdown` | 2 | Prefill and decode timed separately |
| `TestKvCacheStats` | 1 | KV cache memory estimate shown |
| `TestInspect` | 3 | Model name, param count, memory estimate |

**Total: 6 assertions across 3 test classes.**

## Notes

- Chapter 8 is additive. All Chapter 7 `generate` tests should still pass.
- Run ch07 tests alongside ch08 for full MVP validation.
- The inspect tests use a separate env var (`RVLLM_CH08_INSPECT_BIN`).

## Timeout

3 minutes per subcommand.
