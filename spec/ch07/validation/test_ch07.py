"""
Chapter 7 validation tests: Generation Loop with Greedy Decoding.

Runs the reader's implementation as a subprocess and validates the output
against the interface spec (see ../interface-spec.md and ../expected-output.txt).
Language-agnostic -- works with any implementation.

Usage:
    RVLLM_CH07_BIN="cargo run -- generate --prompt 'The future of artificial intelligence is'" pytest test_ch07.py -v
    RVLLM_CH07_BIN="uv run python -m rvllm generate --prompt 'The future of artificial intelligence is'" pytest test_ch07.py -v
"""

import re

import pytest

from conftest import run_binary

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TIMEOUT_SECONDS = 180  # 3 minutes: generous for model download + generation


@pytest.fixture(scope="module")
def ch07_output() -> str:
    """Run the Chapter 7 binary once and cache its stdout for all tests."""
    result = run_binary("RVLLM_CH07_BIN", timeout=TIMEOUT_SECONDS)
    return result.stdout


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestModelLoading:
    """Verify the output shows model loading occurred."""

    def test_loading_indicator(self, ch07_output: str):
        """Output must indicate the model was loaded."""
        lower = ch07_output.lower()
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
            f"Got:\n{ch07_output[:500]}"
        )

    def test_tokenizer_loaded(self, ch07_output: str):
        """Output should mention the tokenizer."""
        lower = ch07_output.lower()
        assert "tokenizer" in lower or "vocab" in lower, (
            "Output should mention tokenizer or vocabulary.\n"
            f"Got:\n{ch07_output[:500]}"
        )


class TestGeneratedText:
    """Verify coherent text was generated."""

    def _extract_generated_section(self, output: str) -> str:
        """Extract text between generation markers, or fall back to full output."""
        patterns = [
            r"---\s*Generated Text\s*---\s*\n(.*?)\n\s*---",
            r"Generated Text[:\s]*\n(.*?)(?:\n---|\n\n\n|\Z)",
        ]
        for pattern in patterns:
            match = re.search(pattern, output, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        idx = output.lower().find("generated text")
        if idx >= 0:
            return output[idx:].strip()
        return output

    def test_has_generated_text_section(self, ch07_output: str):
        """Output must contain a generated text section."""
        lower = ch07_output.lower()
        assert "generated text" in lower or "generation" in lower, (
            "Output should contain a 'Generated Text' section.\n"
            f"Got:\n{ch07_output[:500]}"
        )

    def test_generated_text_nonempty(self, ch07_output: str):
        """The generated text must not be empty."""
        text = self._extract_generated_section(ch07_output)
        assert len(text.strip()) > 0, "Generated text section is empty."

    def test_minimum_word_count(self, ch07_output: str):
        """Generated text must contain at least 20 words (basic coherence)."""
        text = self._extract_generated_section(ch07_output)
        words = text.split()
        assert len(words) >= 20, (
            f"Generated text has only {len(words)} words (expected >= 20). "
            f"Text: {text[:200]}"
        )

    def test_no_degenerate_repetition(self, ch07_output: str):
        """No single word should repeat 10+ times consecutively."""
        text = self._extract_generated_section(ch07_output)
        words = text.split()

        if len(words) < 10:
            return

        max_repeat = 1
        current_repeat = 1
        for i in range(1, len(words)):
            if words[i].lower() == words[i - 1].lower():
                current_repeat += 1
                max_repeat = max(max_repeat, current_repeat)
            else:
                current_repeat = 1

        assert max_repeat < 10, (
            f"Degenerate repetition detected: a word repeated {max_repeat} times "
            f"consecutively. This usually means broken attention or weight loading.\n"
            f"Text: {text[:300]}"
        )

    def test_prompt_appears_in_output(self, ch07_output: str):
        """The prompt should appear somewhere in the output."""
        prompt = "The future of artificial intelligence is"
        assert prompt.lower() in ch07_output.lower() or "future of artificial" in ch07_output.lower(), (
            f"Expected the prompt '{prompt}' to appear in the output.\n"
            f"Got:\n{ch07_output[:500]}"
        )

    def test_no_nan_in_output(self, ch07_output: str):
        """Output should not contain NaN values."""
        nan_patterns = [r'\bnan\b', r'\bNaN\b', r'\bNAN\b', r'\binf\b', r'\bInf\b']
        text = self._extract_generated_section(ch07_output)
        for pattern in nan_patterns:
            match = re.search(pattern, text)
            assert match is None, (
                f"Found '{match.group()}' in generated text. "
                f"This indicates broken computation.\nText: {text[:200]}"
            )


class TestStatistics:
    """Verify performance statistics are printed."""

    def test_has_speed_stat(self, ch07_output: str):
        """Output must report generation speed."""
        lower = ch07_output.lower()
        has_speed = any(
            phrase in lower
            for phrase in [
                "tokens/sec",
                "tokens per sec",
                "tok/s",
                "token/s",
            ]
        )
        if not has_speed:
            has_speed = bool(re.search(r"speed[:\s]+[\d.]+", lower))
        assert has_speed, (
            "Output should report generation speed (e.g., 'X.XX tokens/sec').\n"
            f"Got:\n{ch07_output[-500:]}"
        )

    def test_has_token_count(self, ch07_output: str):
        """Output must report how many tokens were generated."""
        lower = ch07_output.lower()
        has_count = bool(
            re.search(r"(generated|produced|output)\s+\d+\s+token", lower)
        )
        if not has_count:
            has_count = bool(re.search(r"\d+\s+tokens?\s+(generated|in\s)", lower))
        assert has_count, (
            "Output should report token count (e.g., 'Generated 50 tokens').\n"
            f"Got:\n{ch07_output[-500:]}"
        )

    def test_has_timing(self, ch07_output: str):
        """Output must report wall-clock time."""
        lower = ch07_output.lower()
        has_time = bool(
            re.search(r"\d+\.?\d*\s*(s|ms|sec|seconds|millis)", lower)
        )
        assert has_time, (
            "Output should report timing (e.g., 'Time: 2.34s').\n"
            f"Got:\n{ch07_output[-500:]}"
        )
