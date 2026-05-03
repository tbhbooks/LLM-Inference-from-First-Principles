"""
Chapter 5 validation tests: The Building Blocks (Embedding, LayerNorm, MLP).

Runs the reader's implementation as a subprocess and validates the output
against the interface spec (see ../interface-spec.md and ../expected-output.txt).
Language-agnostic -- works with any implementation.

Usage:
    RVLLM_CH05_BIN="cargo run -- inspect --model openai-community/gpt2" pytest test_ch05.py -v
    RVLLM_CH05_BIN="python -m rvllm inspect --model openai-community/gpt2" pytest test_ch05.py -v
"""

import re

import pytest

from conftest import run_binary

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TIMEOUT_SECONDS = 180  # 3 minutes: generous for model download + layer tests


@pytest.fixture(scope="module")
def ch05_output() -> str:
    """Run the Chapter 5 binary once and cache its stdout for all tests."""
    result = run_binary("RVLLM_CH05_BIN", timeout=TIMEOUT_SECONDS)
    return result.stdout


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLayersLoaded:
    """Verify layer components are reported as loaded."""

    def test_embedding_mentioned(self, ch05_output: str):
        """Output must mention embedding layers."""
        lower = ch05_output.lower()
        assert "embedding" in lower, (
            "Output should mention embedding layers.\n"
            f"Got:\n{ch05_output[:500]}"
        )

    def test_layernorm_mentioned(self, ch05_output: str):
        """Output must mention LayerNorm layers."""
        lower = ch05_output.lower()
        assert "layernorm" in lower or "layer norm" in lower or "layer_norm" in lower, (
            "Output should mention LayerNorm layers.\n"
            f"Got:\n{ch05_output[:500]}"
        )

    def test_mlp_mentioned(self, ch05_output: str):
        """Output must mention MLP blocks."""
        lower = ch05_output.lower()
        assert "mlp" in lower, (
            "Output should mention MLP blocks.\n"
            f"Got:\n{ch05_output[:500]}"
        )


class TestEmbeddingShape:
    """Verify embedding dimensions are correct."""

    def test_token_embedding_shape(self, ch05_output: str):
        """Token embedding should be [50257, 768]."""
        assert "50257" in ch05_output and "768" in ch05_output, (
            "Output should show token embedding dimensions (50257 x 768).\n"
            f"Got:\n{ch05_output[:500]}"
        )


class TestPartialForwardPass:
    """Verify the partial forward pass produces valid tensor metrics."""

    def _parse_tensor_metric(self, output: str, step_name: str) -> dict:
        """Parse a tensor metric line like 'step_name: shape=[1, 4, 768], mean=-0.0012, std=0.0345'."""
        # Look for the step name followed by stats
        pattern = rf"{step_name}.*?mean\s*=\s*([-\d.eE+]+).*?std\s*=\s*([-\d.eE+]+)"
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            return {
                "mean": float(match.group(1)),
                "std": float(match.group(2)),
            }
        return {}

    def test_layernorm_mean_near_zero(self, ch05_output: str):
        """LayerNorm output should have mean close to 0."""
        stats = self._parse_tensor_metric(ch05_output, "ln_1")
        if not stats:
            # Try alternative names
            stats = self._parse_tensor_metric(ch05_output, "layernorm")
            if not stats:
                stats = self._parse_tensor_metric(ch05_output, "normed")
        assert stats, (
            "Could not find LayerNorm tensor metrics in output. "
            "Expected a line like 'ln_1_block0: shape=[1, 4, 768], mean=X.XXXX, std=X.XXXX'\n"
            f"Got:\n{ch05_output}"
        )
        assert abs(stats["mean"]) < 0.1, (
            f"LayerNorm mean should be close to 0, got {stats['mean']}. "
            "This suggests LayerNorm is not working correctly."
        )

    def test_mlp_output_nonzero(self, ch05_output: str):
        """MLP output should have non-zero standard deviation."""
        stats = self._parse_tensor_metric(ch05_output, "mlp")
        assert stats, (
            "Could not find MLP tensor metrics in output. "
            "Expected a line like 'mlp_block0: shape=[1, 4, 768], mean=X.XXXX, std=X.XXXX'\n"
            f"Got:\n{ch05_output}"
        )
        assert stats["std"] > 0.001, (
            f"MLP output std is {stats['std']} (expected > 0.001). "
            "This suggests the MLP is producing constant output — likely Conv1D weights not transposed."
        )

    def test_no_nan_in_metrics(self, ch05_output: str):
        """No tensor metric line should contain NaN or Inf."""
        nan_patterns = [r'\bnan\b', r'\bNaN\b', r'\bNAN\b', r'\binf\b', r'\bInf\b', r'\bINF\b']
        # Only check lines that look like tensor metrics
        metric_lines = [
            line for line in ch05_output.split("\n")
            if "mean" in line.lower() and "std" in line.lower()
        ]
        for line in metric_lines:
            for pattern in nan_patterns:
                match = re.search(pattern, line)
                assert match is None, (
                    f"Found '{match.group()}' in tensor metrics: {line}\n"
                    "This indicates broken computation (likely missing LayerNorm epsilon)."
                )


class TestConv1DHandling:
    """Verify Conv1D transpose is mentioned."""

    def test_conv1d_evidence(self, ch05_output: str):
        """Output should mention Conv1D or transpose handling."""
        lower = ch05_output.lower()
        has_evidence = any(
            phrase in lower
            for phrase in ["transpose", "conv1d", "transposed"]
        )
        assert has_evidence, (
            "Output should mention Conv1D weight transposing.\n"
            f"Got:\n{ch05_output[:500]}"
        )
