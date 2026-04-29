# Chapter 15 Validation Tests

Language-agnostic tests for Building the API Server (Chapter 15).

## Running

```bash
RVLLM_CH15_BIN="cargo run --example ch15_api_server" pytest spec/ch15/validation/ -v
RVLLM_CH15_BIN="uv run python -m rvllm ch15_demo" pytest spec/ch15/validation/ -v

# Rust reference implementation
source spec/runners/rust.env
pytest spec/ch15/validation/ -v
```

See [spec/runners/README.md](../../runners/README.md) for full details.

## What the Tests Check

| Test Class | Tests | What It Validates |
|------------|-------|-------------------|
| `TestStructure` | 5 | All 5 parts present (Server Startup, Health Check, Completion Request, Streaming Response, Error Handling) |
| `TestServerStartup` | 2 | Listen address shown, endpoints listed |
| `TestHealthCheck` | 2 | /health endpoint shown, OK status returned |
| `TestCompletionRequest` | 4 | OpenAI response format (id, object, choices, usage, finish_reason) |
| `TestStreamingResponse` | 3 | SSE `data: ` prefix, `[DONE]` sentinel, multiple token events |
| `TestErrorHandling` | 4 | Error object with message, type, and HTTP status code |
| `TestCompletion` | 1 | "Chapter 15 complete" marker present |

**Total: 21 assertions across 7 test classes.**

## Timeout

30 seconds (no real server or model -- mock simulation only).
