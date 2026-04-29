"""
Chapter 8 validation tests: Fit and Finish (MVP Milestone).

Tests run BOTH subcommands (generate and inspect) to verify the polished CLI.
This chapter is additive — all ch07 generate tests should still pass.

Usage:
    RVLLM_CH08_GEN_BIN="cargo run -- generate --prompt 'The future of artificial intelligence is'" \
    RVLLM_CH08_INSPECT_BIN="cargo run -- inspect --model openai-community/gpt2" \
    pytest test_ch08.py -v
"""

import re

import pytest

from conftest import run_binary

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TIMEOUT_SECONDS = 180


@pytest.fixture(scope="module")
def ch08_generate_output() -> str:
    """Run the generate subcommand and cache output."""
    result = run_binary("RVLLM_CH08_GEN_BIN", timeout=TIMEOUT_SECONDS)
    return result.stdout


@pytest.fixture(scope="module")
def ch08_inspect_output() -> str:
    """Run the inspect subcommand and cache output."""
    result = run_binary("RVLLM_CH08_INSPECT_BIN", timeout=TIMEOUT_SECONDS)
    return result.stdout


# ---------------------------------------------------------------------------
# Generate Tests (extends ch07)
# ---------------------------------------------------------------------------


class TestTimingBreakdown:
    """Verify prefill and decode timing are reported separately."""

    def test_prefill_timing(self, ch08_generate_output: str):
        """Output must report prefill time."""
        lower = ch08_generate_output.lower()
        has_prefill = bool(
            re.search(r"prefill.*\d+\.?\d*\s*(s|ms|sec)", lower)
        )
        assert has_prefill, (
            "Output should report prefill timing (e.g., 'Prefill: 0.12s').\n"
            f"Got:\n{ch08_generate_output[-500:]}"
        )

    def test_decode_timing(self, ch08_generate_output: str):
        """Output must report decode time."""
        lower = ch08_generate_output.lower()
        has_decode = bool(
            re.search(r"decode.*\d+\.?\d*\s*(s|ms|sec)", lower)
        )
        assert has_decode, (
            "Output should report decode timing (e.g., 'Decode: 5.67s').\n"
            f"Got:\n{ch08_generate_output[-500:]}"
        )


class TestKvCacheStats:
    """Verify KV cache memory estimate is shown."""

    def test_kv_cache_mentioned(self, ch08_generate_output: str):
        """Output must mention KV cache with a size."""
        lower = ch08_generate_output.lower()
        has_kv = "kv cache" in lower or "kv_cache" in lower or "cache" in lower
        has_size = bool(re.search(r"\d+\.?\d*\s*(kb|mb|gb|bytes)", lower))
        assert has_kv and has_size, (
            "Output should mention KV cache memory (e.g., 'KV cache: 14.8 MB').\n"
            f"Got:\n{ch08_generate_output[-500:]}"
        )


# ---------------------------------------------------------------------------
# Inspect Tests
# ---------------------------------------------------------------------------


class TestInspect:
    """Verify the polished inspect subcommand."""

    def test_inspect_shows_model(self, ch08_inspect_output: str):
        """Inspect should show the model name/ID."""
        lower = ch08_inspect_output.lower()
        assert "gpt2" in lower or "gpt-2" in lower, (
            "Inspect should mention the model (gpt2).\n"
            f"Got:\n{ch08_inspect_output[:500]}"
        )

    def test_param_count(self, ch08_inspect_output: str):
        """Inspect should report parameter count."""
        lower = ch08_inspect_output.lower()
        has_params = any(
            phrase in lower
            for phrase in ["124m", "124 m", "parameter", "params"]
        )
        assert has_params, (
            "Inspect should mention parameter count (~124M).\n"
            f"Got:\n{ch08_inspect_output[:500]}"
        )

    def test_memory_estimate(self, ch08_inspect_output: str):
        """Inspect should report memory footprint."""
        lower = ch08_inspect_output.lower()
        has_memory = bool(re.search(r"\d+\.?\d*\s*(mb|gb)", lower))
        assert has_memory, (
            "Inspect should report memory estimate (e.g., '~497 MB').\n"
            f"Got:\n{ch08_inspect_output[:500]}"
        )
