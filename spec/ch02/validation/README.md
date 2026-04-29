# Chapter 2 Validation Tests

Language-agnostic tests for the architecture overview program (Chapter 2).

## Running

```bash
# Set env var to your binary/command, then run pytest
RVLLM_CH02_BIN=./your-binary pytest spec/ch02/validation/ -v
RVLLM_CH02_BIN="python3 my_ch02.py" pytest spec/ch02/validation/ -v

# Rust reference implementation
source spec/runners/rust.env
pytest spec/ch02/validation/ -v
```

See [spec/runners/README.md](../../runners/README.md) for full details.

## What the Tests Check

| Test Class | What It Validates |
|------------|-------------------|
| `TestHeaderFooter` | Title, separators, "What's Next?" footer, chapter teasers |
| `TestVllmArchitecture` | All 6 vLLM layers present |
| `TestRvllmArchitecture` | All 7 rvllm modules present |
| `TestComparisonTable` | Mapping table with chapter references |
| `TestRequestLifecycle` | All 9 step headers, specific details per step, loop box |
| `TestKeySimplifications` | All 5 dropped features, goal statement |
| `TestStructure` | Section separators, no ANSI codes, exit code 0 |
