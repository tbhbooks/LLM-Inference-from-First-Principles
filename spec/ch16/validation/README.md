# Chapter 16 Validation Tests

Language-agnostic tests for Prefix Caching (Chapter 16).

## Running

```bash
RVLLM_CH16_BIN="cargo run --example ch16_prefix_caching" pytest spec/ch16/validation/ -v
RVLLM_CH16_BIN="uv run python -m rvllm ch16_demo" pytest spec/ch16/validation/ -v

# Rust reference implementation
source spec/runners/rust.env
pytest spec/ch16/validation/ -v
```

See [spec/runners/README.md](../../runners/README.md) for full details.

## What the Tests Check

| Test Class | Tests | What It Validates |
|------------|-------|-------------------|
| `TestStructure` | 5 | All 5 parts present (Hash Chaining, Cache Miss, Cache Hit, Eviction, Hit Rate) |
| `TestHashChaining` | 2 | Hash and chaining mentioned, block hash values shown |
| `TestCacheMiss` | 2 | Miss mentioned, blocks allocated fresh |
| `TestCacheHit` | 3 | Hit mentioned, blocks reused, prefill savings shown |
| `TestEviction` | 2 | Eviction/LRU mentioned, ref_count values shown |
| `TestHitRate` | 2 | Hit rate mentioned, percentage value shown |
| `TestCompletion` | 1 | "Chapter 16 complete" marker present |

**Total: 17 assertions across 7 test classes.**

## Timeout

30 seconds (no model loading — uses mock data only).
