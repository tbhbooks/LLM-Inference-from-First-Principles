"""
Chapter 6 validation tests: Attention, KV Cache, Full Forward Pass.

Runs the reader's implementation as a subprocess and validates the output
against the interface spec (see ../interface-spec.md and ../expected-output.txt).
Language-agnostic -- works with any implementation.

Usage:
    RVLLM_CH06_BIN="cargo run -- inspect --model openai-community/gpt2" pytest test_ch06.py -v
    RVLLM_CH06_BIN="uv run python -m rvllm inspect --model openai-community/gpt2" pytest test_ch06.py -v
"""

import re

import pytest

from conftest import run_binary

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TIMEOUT_SECONDS = 180


@pytest.fixture(scope="module")
def ch06_output() -> str:
    """Run the Chapter 6 binary once and cache its stdout for all tests."""
    result = run_binary("RVLLM_CH06_BIN", timeout=TIMEOUT_SECONDS)
    return result.stdout


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestModelLoaded:
    """Verify the full model is assembled and ready."""

    def test_model_assembled(self, ch06_output: str):
        """Output must indicate the full model is loaded/assembled."""
        lower = ch06_output.lower()
        has_model = any(
            phrase in lower
            for phrase in [
                "model assembled",
                "model loaded",
                "model ready",
                "model built",
                "full model",
                "gpt2 model",
                "transformer blocks",
            ]
        )
        assert has_model, (
            "Output should indicate the full model is assembled.\n"
            f"Got:\n{ch06_output[:500]}"
        )


class TestLogitsShape:
    """Verify logits have the correct vocabulary dimension."""

    def test_vocab_size_in_logits(self, ch06_output: str):
        """Output must mention 50257 in context of logits or predictions."""
        assert "50257" in ch06_output, (
            "Output should mention vocab size 50257 (logits shape).\n"
            f"Got:\n{ch06_output[:500]}"
        )


class TestTopPredictions:
    """Verify top-5 next-token predictions are shown."""

    def test_has_numbered_predictions(self, ch06_output: str):
        """Output must show at least 5 numbered predictions."""
        # Look for patterns like "1." "2." ... "5."
        prediction_numbers = re.findall(r"^\s*(\d+)\.\s", ch06_output, re.MULTILINE)
        nums = [int(n) for n in prediction_numbers]
        has_five = all(i in nums for i in range(1, 6))
        assert has_five, (
            "Output should contain numbered predictions 1-5.\n"
            f"Found numbers: {nums}\n"
            f"Output:\n{ch06_output}"
        )

    def test_predictions_plausible(self, ch06_output: str):
        """At least one top-5 prediction should be a common English token."""
        # Common tokens GPT-2 might predict after "What is AI?"
        common_tokens = [
            "the", "it", "a", "an", "this", "that", "in", "what",
            "is", "and", "or", "but", "for", "to", "of", "we",
            "i", "you", "they", "he", "she", "how", "why",
        ]
        lower = ch06_output.lower()
        # Check if any common token appears in a prediction line
        found = False
        for line in ch06_output.split("\n"):
            # Lines that look like predictions (numbered with quoted tokens)
            if re.match(r"\s*\d+\.\s", line):
                line_lower = line.lower()
                for token in common_tokens:
                    # Check for the token as a word (with possible surrounding quotes/spaces)
                    if re.search(rf'["\s]{token}["\s,)]', line_lower):
                        found = True
                        break
            if found:
                break
        assert found, (
            "Expected at least one common English token in top-5 predictions.\n"
            f"Output:\n{ch06_output}"
        )

    def test_has_logit_values(self, ch06_output: str):
        """Predictions should include logit values."""
        # Look for "logit" followed by a number, or parenthesized numbers
        has_logits = bool(
            re.search(r"logit[s]?\s*[:=]\s*[-\d.]+", ch06_output, re.IGNORECASE)
        ) or bool(
            re.search(r"\(\s*[-\d.]+\s*\)", ch06_output)
        )
        assert has_logits, (
            "Predictions should include logit values.\n"
            f"Output:\n{ch06_output}"
        )


class TestNoNaN:
    """Verify no NaN or Inf in output."""

    def test_no_nan_in_predictions(self, ch06_output: str):
        """Logit values and predictions should not contain NaN."""
        nan_patterns = [r'\bnan\b', r'\bNaN\b', r'\bNAN\b', r'\binf\b', r'\bInf\b']
        for pattern in nan_patterns:
            match = re.search(pattern, ch06_output)
            assert match is None, (
                f"Found '{match.group()}' in output. "
                f"This indicates broken computation.\n"
                f"Output:\n{ch06_output[:500]}"
            )


class TestRunningExample:
    """Verify the running example is used."""

    def test_what_is_ai(self, ch06_output: str):
        """'What is AI?' must appear in the output."""
        assert "What is AI?" in ch06_output or "what is ai?" in ch06_output.lower(), (
            "Expected 'What is AI?' to appear in the output.\n"
            f"Got:\n{ch06_output[:500]}"
        )
