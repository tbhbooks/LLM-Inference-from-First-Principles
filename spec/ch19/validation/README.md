# Chapter 19 Validation Tests

Language-agnostic tests for Parallelism (Chapter 19).

## Running

```bash
RVLLM_CH19_BIN="cargo run --example ch19_parallelism" pytest spec/ch19/validation/ -v
RVLLM_CH19_BIN="uv run python -m rvllm ch19_demo" pytest spec/ch19/validation/ -v

# Rust reference implementation
source spec/runners/rust.env
pytest spec/ch19/validation/ -v
```

See [spec/runners/README.md](../../runners/README.md) for full details.

## What the Tests Check

| Test Class | Tests | What It Validates |
|------------|-------|-------------------|
| `TestStructure` | 5 | All 5 parts present (TP, Naive PP, batch_queue PP, Combined, Scaling) |
| `TestTensorParallelism` | 3 | TP/tensor parallel mentioned, AllReduce mentioned, communication volume shown |
| `TestPipelineParallelism` | 4 | Pipeline mentioned, bubble/utilization shown, timeline shown, percentage shown |
| `TestBatchQueue` | 2 | batch_queue/overlap mentioned, improved utilization demonstrated |
| `TestCombined` | 2 | TP+PP shown together, total GPU count shown |
| `TestCompletion` | 1 | "Chapter 19 complete" marker present |

**Total: 17 assertions across 6 test classes.**

## Timeout

30 seconds (pure simulator -- no model loading, no GPU needed).
