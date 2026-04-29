# Running Validation Tests

The validation tests are language-agnostic. They run your implementation as a subprocess
and check stdout for expected values. Any language works -- Rust, Python, Go, TypeScript, etc.

## Quick Start

Set the environment variable for the chapter you want to test:

    # Chapter 1
    export RVLLM_CH01_BIN="./your-binary"
    pytest spec/ch01/validation/ -v

    # Chapter 3
    export RVLLM_CH03_BIN="./your-binary"
    pytest spec/ch03/validation/ -v

## Language Examples

### Rust
    source spec/runners/rust.env
    pytest spec/ch01/validation/ -v

### Python
    export RVLLM_CH01_BIN="python3 my_ch01.py"
    pytest spec/ch01/validation/ -v

### Go
    export RVLLM_CH01_BIN="go run ./cmd/ch01"
    pytest spec/ch01/validation/ -v

### Node.js / TypeScript
    export RVLLM_CH01_BIN="npx ts-node ch01.ts"
    pytest spec/ch01/validation/ -v

## Environment Variables

| Variable | Chapter | Description |
|----------|---------|-------------|
| `RVLLM_CH01_BIN` | 1 | KV cache calculator |
| `RVLLM_CH02_BIN` | 2 | Architecture printer |
| `RVLLM_CH03_BIN` | 3 | Project skeleton CLI |
| `RVLLM_CH04_BIN` | 4 | GPT-2 text generator |

## Tips

- All tests use `pytest`. Install with `pip install pytest`.
- Tests skip automatically if the env var is not set.
- Commands can include arguments: `RVLLM_CH04_BIN="cargo run --example gpt2 --release"`
- Chapter 4 has a 3-minute timeout (model download + generation).
