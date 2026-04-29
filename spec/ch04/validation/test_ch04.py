"""
Chapter 4 validation tests: Downloading a Brain (Load + Inspect).

Runs the reader's implementation as a subprocess and validates the output
against the interface spec (see ../interface-spec.md and ../expected-output.txt).
Language-agnostic -- works with any implementation.

Usage:
    RVLLM_CH04_BIN="cargo run -- inspect --model openai-community/gpt2" pytest test_ch04.py -v
    RVLLM_CH04_BIN="uv run python -m rvllm inspect --model openai-community/gpt2" pytest test_ch04.py -v

See spec/runners/README.md for more examples (Rust, Python, Go, etc.).
"""

import re

import pytest

from conftest import run_binary

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TIMEOUT_SECONDS = 180  # 3 minutes: generous for model download


@pytest.fixture(scope="module")
def ch04_output() -> str:
    """Run the Chapter 4 binary once and cache its stdout for all tests."""
    result = run_binary("RVLLM_CH04_BIN", timeout=TIMEOUT_SECONDS)
    return result.stdout


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestModelLoading:
    """Verify the output shows model loading occurred."""

    def test_loading_indicator(self, ch04_output: str):
        """Output must indicate the model was loaded."""
        lower = ch04_output.lower()
        has_loading = any(
            phrase in lower
            for phrase in [
                "loading model",
                "loaded model",
                "model loaded",
                "model files",
                "downloaded",
                "cached",
                "safetensors",
            ]
        )
        assert has_loading, (
            "Output should indicate model loading. "
            "Expected one of: 'loading model', 'loaded model', 'downloaded', etc.\n"
            f"Got:\n{ch04_output[:500]}"
        )

    def test_tokenizer_loaded(self, ch04_output: str):
        """Output should mention the tokenizer."""
        lower = ch04_output.lower()
        assert "tokenizer" in lower or "vocab" in lower, (
            "Output should mention tokenizer or vocabulary.\n"
            f"Got:\n{ch04_output[:500]}"
        )


class TestModelConfig:
    """Verify model configuration is displayed."""

    def test_hidden_dim(self, ch04_output: str):
        """Output must mention the hidden dimension (768)."""
        assert "768" in ch04_output, (
            "Output should mention hidden dimension 768.\n"
            f"Got:\n{ch04_output[:500]}"
        )

    def test_layers_or_heads(self, ch04_output: str):
        """Output must mention 12 (layers or heads)."""
        # Check for "12" in context of layers/heads/blocks
        has_12 = bool(re.search(r"12\s*(layer|head|block|transformer)", ch04_output.lower()))
        if not has_12:
            # Fallback: just check "12" appears
            has_12 = "12" in ch04_output
        assert has_12, (
            "Output should mention 12 layers or 12 heads.\n"
            f"Got:\n{ch04_output[:500]}"
        )


class TestWeights:
    """Verify weight summary information."""

    def test_weight_count(self, ch04_output: str):
        """Output should mention 148 weights or a weight count."""
        has_count = "148" in ch04_output
        if not has_count:
            has_count = bool(re.search(r"\d+\s*(tensor|weight)", ch04_output.lower()))
        assert has_count, (
            "Output should mention weight/tensor count (148).\n"
            f"Got:\n{ch04_output[:500]}"
        )

    def test_param_count(self, ch04_output: str):
        """Output should mention ~124M parameters."""
        lower = ch04_output.lower()
        has_params = any(
            phrase in lower
            for phrase in ["124m", "124 m", "~124", "124,", "124 million"]
        )
        if not has_params:
            has_params = bool(re.search(r"1[23]\d[,.]?\d*\s*m", lower))
        assert has_params, (
            "Output should mention parameter count (~124M).\n"
            f"Got:\n{ch04_output[:500]}"
        )


class TestTokenizerRoundTrip:
    """Verify tokenizer round-trip test."""

    def test_what_is_ai(self, ch04_output: str):
        """'What is AI?' must appear in the tokenizer check output."""
        assert "What is AI?" in ch04_output, (
            "Expected 'What is AI?' in tokenizer round-trip output.\n"
            f"Got:\n{ch04_output[:500]}"
        )
