# Chapter 1 Validation Tests

Language-agnostic tests for the KV cache calculator (Chapter 1).

## Running

```bash
# Set env var to your binary/command, then run pytest
RVLLM_CH01_BIN=./your-binary pytest spec/ch01/validation/ -v
RVLLM_CH01_BIN="python3 my_ch01.py" pytest spec/ch01/validation/ -v

# Rust reference implementation
source spec/runners/rust.env
pytest spec/ch01/validation/ -v
```

See [spec/runners/README.md](../../runners/README.md) for full details.

## What the Tests Check

| Category | Tests | What's Checked |
|----------|-------|----------------|
| Section headers | 7 | All 5 PART headers and their titles present |
| Per-token sizes | 6 | KV bytes per token and per token/layer for all 3 models |
| Sequence sizes | 6 | KV bytes at 1024 and 4096 sequence lengths |
| Concurrency ceiling | 6 | Max concurrent sequences at both lengths |
| Memory wall | 2 | OOM appears, exactly 5 OOM!! occurrences (4 table + 1 legend) |
| Prefill/Decode | 5 | Key terms and concepts present |
| Closing | 2 | Chapter complete message and next chapter hook |

Total: 34 assertions.
