# Chapter 3 Validation Tests

Language-agnostic tests for the project skeleton CLI (Chapter 3).

## Running

```bash
# Set env var to your binary/command, then run pytest
RVLLM_CH03_BIN=./your-binary pytest spec/ch03/validation/ -v
RVLLM_CH03_BIN="python3 main.py" pytest spec/ch03/validation/ -v

# Rust reference implementation
source spec/runners/rust.env
pytest spec/ch03/validation/ -v
```

See [spec/runners/README.md](../../runners/README.md) for full details.

## What the Tests Check

| Test Class | What It Validates |
|------------|-------------------|
| `TestHelpOutput` | `--help` mentions expected subcommands and flags |
| `TestStubCommands` | `generate` and `inspect` run without crashing, produce output |
| `TestErrorHandling` | Invalid input handled gracefully (non-zero exit, no crash) |
