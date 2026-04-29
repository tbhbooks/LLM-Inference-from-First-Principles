"""
Chapter 16 validation tests: Prefix Caching.

Runs the reader's implementation as a subprocess and validates the output
against the interface spec (see ../interface-spec.md and ../expected-output.txt).
Language-agnostic -- works with any implementation.

Usage:
    RVLLM_CH16_BIN="cargo run --example ch16_prefix_caching" pytest test_ch16.py -v
    RVLLM_CH16_BIN="uv run python -m rvllm ch16_demo" pytest test_ch16.py -v
"""

import re

import pytest

from conftest import run_binary

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TIMEOUT_SECONDS = 30  # No model loading needed — mock data only


@pytest.fixture(scope="module")
def ch16_output() -> str:
    """Run the Chapter 16 binary once and cache its stdout for all tests."""
    result = run_binary("RVLLM_CH16_BIN", timeout=TIMEOUT_SECONDS)
    return result.stdout


def _lower(output: str) -> str:
    return output.lower()


# ---------------------------------------------------------------------------
# Tests: All 5 parts present
# ---------------------------------------------------------------------------


class TestStructure:
    """Verify all 5 required parts are present in the output."""

    def test_part1_hash_chaining(self, ch16_output: str):
        """Output must contain Part 1: Hash Chaining."""
        lower = _lower(ch16_output)
        assert "part 1" in lower or "hash chain" in lower, (
            "Output must contain 'PART 1' or 'Hash Chain'.\n"
            f"Got:\n{ch16_output[:500]}"
        )

    def test_part2_cache_miss(self, ch16_output: str):
        """Output must contain Part 2: Cache Miss."""
        lower = _lower(ch16_output)
        assert "part 2" in lower or "cache miss" in lower or "first request" in lower, (
            "Output must contain 'PART 2' or 'Cache Miss'.\n"
            f"Got:\n{ch16_output[:500]}"
        )

    def test_part3_cache_hit(self, ch16_output: str):
        """Output must contain Part 3: Cache Hit."""
        lower = _lower(ch16_output)
        assert "part 3" in lower or "cache hit" in lower or "second request" in lower, (
            "Output must contain 'PART 3' or 'Cache Hit'.\n"
            f"Got:\n{ch16_output[:500]}"
        )

    def test_part4_eviction(self, ch16_output: str):
        """Output must contain Part 4: Eviction."""
        lower = _lower(ch16_output)
        assert "part 4" in lower or "evict" in lower, (
            "Output must contain 'PART 4' or 'Eviction'.\n"
            f"Got:\n{ch16_output[:500]}"
        )

    def test_part5_hit_rate(self, ch16_output: str):
        """Output must contain Part 5: Hit Rate."""
        lower = _lower(ch16_output)
        assert "part 5" in lower or "hit rate" in lower, (
            "Output must contain 'PART 5' or 'Hit Rate'.\n"
            f"Got:\n{ch16_output[:500]}"
        )


# ---------------------------------------------------------------------------
# Tests: Hash Chaining
# ---------------------------------------------------------------------------


class TestHashChaining:
    """Verify hash chaining is demonstrated."""

    def test_hash_or_chain_mentioned(self, ch16_output: str):
        """Output must mention hash and chaining."""
        lower = _lower(ch16_output)
        assert "hash" in lower, (
            "Output must mention 'hash'.\n"
            f"Got:\n{ch16_output[:500]}"
        )
        assert "chain" in lower, (
            "Output must mention 'chain' or 'chaining'.\n"
            f"Got:\n{ch16_output[:500]}"
        )

    def test_block_hash_values_shown(self, ch16_output: str):
        """Output must show block hash values (hex or large integers)."""
        # Look for hex values like 0xa3f7... or large decimal integers
        hex_pattern = r"0x[0-9a-fA-F]{4,}"
        dec_pattern = r"\d{10,}"
        has_hex = re.search(hex_pattern, ch16_output)
        has_large_int = re.search(dec_pattern, ch16_output)
        assert has_hex or has_large_int, (
            "Output must show block hash values (hex like 0xa3f7... "
            "or large integers).\n"
            f"Got:\n{ch16_output[:1000]}"
        )


# ---------------------------------------------------------------------------
# Tests: Cache Miss
# ---------------------------------------------------------------------------


class TestCacheMiss:
    """Verify the first request demonstrates a cache miss."""

    def test_miss_mentioned(self, ch16_output: str):
        """Output must mention cache miss."""
        assert "miss" in _lower(ch16_output), (
            "Output must mention 'miss' (cache miss on first request).\n"
            f"Got:\n{ch16_output[:500]}"
        )

    def test_blocks_allocated(self, ch16_output: str):
        """Output must show blocks being allocated fresh."""
        lower = _lower(ch16_output)
        assert "alloc" in lower, (
            "Output must mention block allocation (e.g., 'allocated').\n"
            f"Got:\n{ch16_output[:1000]}"
        )


# ---------------------------------------------------------------------------
# Tests: Cache Hit
# ---------------------------------------------------------------------------


class TestCacheHit:
    """Verify the second request demonstrates a cache hit."""

    def test_hit_mentioned(self, ch16_output: str):
        """Output must mention cache hit."""
        assert "hit" in _lower(ch16_output), (
            "Output must mention 'hit' (cache hit on second request).\n"
            f"Got:\n{ch16_output[:500]}"
        )

    def test_blocks_reused(self, ch16_output: str):
        """Output must show blocks being reused or prefill skipped."""
        lower = _lower(ch16_output)
        has_reuse = (
            "reuse" in lower
            or "skip" in lower
            or "cached" in lower
            or "computed" in lower
        )
        assert has_reuse, (
            "Output must show blocks reused or prefill skipped "
            "(e.g., 'reused', 'skipped', 'cached').\n"
            f"Got:\n{ch16_output[:1000]}"
        )

    def test_prefill_savings(self, ch16_output: str):
        """Output must show tokens or computation being saved."""
        lower = _lower(ch16_output)
        has_savings = (
            "skip" in lower
            or "saving" in lower
            or "saved" in lower
            or "computed tokens" in lower
            or "prefill" in lower
        )
        assert has_savings, (
            "Output must show prefill savings (e.g., 'skipped', 'saved', 'prefill').\n"
            f"Got:\n{ch16_output[:1000]}"
        )


# ---------------------------------------------------------------------------
# Tests: Eviction
# ---------------------------------------------------------------------------


class TestEviction:
    """Verify eviction/LRU behavior is demonstrated."""

    def test_eviction_or_lru_mentioned(self, ch16_output: str):
        """Output must mention eviction or LRU."""
        lower = _lower(ch16_output)
        has_eviction = (
            "evict" in lower
            or "lru" in lower
            or "free_queue" in lower
            or "free queue" in lower
        )
        assert has_eviction, (
            "Output must mention 'evict', 'LRU', or 'free_queue'.\n"
            f"Got:\n{ch16_output[:500]}"
        )

    def test_ref_count_shown(self, ch16_output: str):
        """Output must show ref_count values changing."""
        lower = _lower(ch16_output)
        assert "ref_count" in lower or "ref count" in lower or "refcount" in lower, (
            "Output must show ref_count values.\n"
            f"Got:\n{ch16_output[:1000]}"
        )


# ---------------------------------------------------------------------------
# Tests: Hit Rate
# ---------------------------------------------------------------------------


class TestHitRate:
    """Verify hit rate statistics are shown."""

    def test_hit_rate_mentioned(self, ch16_output: str):
        """Output must mention hit rate."""
        lower = _lower(ch16_output)
        assert "hit rate" in lower or "hitrate" in lower or "hit ratio" in lower, (
            "Output must mention 'hit rate'.\n"
            f"Got:\n{ch16_output[:500]}"
        )

    def test_percentage_shown(self, ch16_output: str):
        """Output must show a percentage value."""
        pct_pattern = r"\d+\.?\d*%"
        assert re.search(pct_pattern, ch16_output), (
            "Output must show a percentage value (e.g., '88.9%').\n"
            f"Got:\n{ch16_output[-500:]}"
        )


# ---------------------------------------------------------------------------
# Tests: Completion
# ---------------------------------------------------------------------------


class TestCompletion:
    """Verify the chapter completion marker."""

    def test_chapter_16_complete(self, ch16_output: str):
        """Output must contain the completion marker."""
        assert "chapter 16 complete" in _lower(ch16_output), (
            "Output must contain 'Chapter 16 complete'.\n"
            f"Got:\n{ch16_output[-500:]}"
        )
