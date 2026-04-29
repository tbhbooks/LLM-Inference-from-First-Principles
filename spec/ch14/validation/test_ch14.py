"""
Chapter 14 validation tests: Sampling Strategies.

Runs the reader's implementation as a subprocess and validates the output
against the interface spec (see ../interface-spec.md and ../expected-output.txt).
Language-agnostic -- works with any implementation.

Usage:
    RVLLM_CH14_BIN="cargo run --example ch14_sampling_strategies" pytest test_ch14.py -v
    RVLLM_CH14_BIN="uv run python -m rvllm ch14_demo" pytest test_ch14.py -v
"""

import re

import pytest

from conftest import run_binary

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TIMEOUT_SECONDS = 60  # No model loading needed — mock logits only


@pytest.fixture(scope="module")
def ch14_output() -> str:
    """Run the Chapter 14 binary once and cache its stdout for all tests."""
    result = run_binary("RVLLM_CH14_BIN", timeout=TIMEOUT_SECONDS)
    return result.stdout


def _lower(output: str) -> str:
    return output.lower()


# ---------------------------------------------------------------------------
# Tests: All 6 parts present
# ---------------------------------------------------------------------------


class TestStructure:
    """Verify all 6 required parts are present in the output."""

    def test_part1_beyond_argmax(self, ch14_output: str):
        """Output must contain Part 1: Beyond Argmax."""
        lower = _lower(ch14_output)
        assert "part 1" in lower or "beyond argmax" in lower, (
            "Output must contain 'PART 1' or 'Beyond Argmax'.\n"
            f"Got:\n{ch14_output[:500]}"
        )

    def test_part2_temperature(self, ch14_output: str):
        """Output must contain Part 2: Temperature."""
        lower = _lower(ch14_output)
        assert "part 2" in lower or "controlling confidence" in lower, (
            "Output must contain 'PART 2' or 'Controlling Confidence'.\n"
            f"Got:\n{ch14_output[:500]}"
        )

    def test_part3_topk_topp(self, ch14_output: str):
        """Output must contain Part 3: Top-K and Top-P."""
        lower = _lower(ch14_output)
        assert "part 3" in lower or "trimming the tail" in lower, (
            "Output must contain 'PART 3' or 'Trimming the Tail'.\n"
            f"Got:\n{ch14_output[:500]}"
        )

    def test_part4_repetition(self, ch14_output: str):
        """Output must contain Part 4: Repetition Penalty."""
        lower = _lower(ch14_output)
        assert "part 4" in lower or "stop saying the same" in lower or "repetition" in lower, (
            "Output must contain 'PART 4' or 'repetition'.\n"
            f"Got:\n{ch14_output[:500]}"
        )

    def test_part5_pipeline(self, ch14_output: str):
        """Output must contain Part 5: The Pipeline."""
        lower = _lower(ch14_output)
        assert "part 5" in lower or "composing strategies" in lower, (
            "Output must contain 'PART 5' or 'Composing Strategies'.\n"
            f"Got:\n{ch14_output[:500]}"
        )

    def test_part6_sampling_params(self, ch14_output: str):
        """Output must contain Part 6: SamplingParams."""
        lower = _lower(ch14_output)
        assert "part 6" in lower or "control panel" in lower or "samplingparams" in lower, (
            "Output must contain 'PART 6' or 'SamplingParams'.\n"
            f"Got:\n{ch14_output[:500]}"
        )


# ---------------------------------------------------------------------------
# Tests: Concept coverage
# ---------------------------------------------------------------------------


class TestTemperature:
    """Verify temperature scaling is demonstrated."""

    def test_temperature_mentioned(self, ch14_output: str):
        """Output must mention temperature."""
        assert "temperature" in _lower(ch14_output), (
            "Output must mention 'temperature'.\n"
            f"Got:\n{ch14_output[:500]}"
        )

    def test_temperature_shows_probabilities(self, ch14_output: str):
        """Temperature section must show probability values."""
        # Look for decimal numbers like 0.1234 in the output
        prob_pattern = r"0\.\d{2,}"
        matches = re.findall(prob_pattern, ch14_output)
        assert len(matches) >= 3, (
            f"Expected at least 3 probability values (e.g., 0.7054), "
            f"found {len(matches)}.\n"
            f"Got:\n{ch14_output[:500]}"
        )


class TestTopK:
    """Verify top-k filtering is demonstrated."""

    def test_topk_mentioned(self, ch14_output: str):
        """Output must mention top-k."""
        lower = _lower(ch14_output)
        assert "top-k" in lower or "top_k" in lower or "topk" in lower, (
            "Output must mention 'top-k', 'top_k', or 'topk'.\n"
            f"Got:\n{ch14_output[:500]}"
        )

    def test_topk_shows_filtering(self, ch14_output: str):
        """Top-k section must show tokens being filtered."""
        lower = _lower(ch14_output)
        has_filtering = (
            "-inf" in lower
            or "0.0000" in ch14_output
            or "filtered" in lower
            or "removed" in lower
            or "= 0" in lower
        )
        assert has_filtering, (
            "Top-k section must show tokens being filtered "
            "(e.g., -inf, 0.0000, 'filtered').\n"
            f"Got:\n{ch14_output[:1000]}"
        )


class TestTopP:
    """Verify top-p (nucleus) sampling is demonstrated."""

    def test_topp_or_nucleus_mentioned(self, ch14_output: str):
        """Output must mention top-p or nucleus."""
        lower = _lower(ch14_output)
        assert (
            "top-p" in lower
            or "top_p" in lower
            or "topp" in lower
            or "nucleus" in lower
        ), (
            "Output must mention 'top-p', 'top_p', 'topp', or 'nucleus'.\n"
            f"Got:\n{ch14_output[:500]}"
        )

    def test_topp_shows_cumulative(self, ch14_output: str):
        """Top-p section must reference cumulative probability."""
        lower = _lower(ch14_output)
        assert "cumul" in lower or "cumsum" in lower or "cum" in lower, (
            "Top-p section must mention cumulative probability.\n"
            f"Got:\n{ch14_output[:1000]}"
        )


class TestRepetitionPenalty:
    """Verify repetition penalty is demonstrated."""

    def test_repetition_mentioned(self, ch14_output: str):
        """Output must mention repetition."""
        assert "repetition" in _lower(ch14_output), (
            "Output must mention 'repetition'.\n"
            f"Got:\n{ch14_output[:500]}"
        )

    def test_penalty_shows_change(self, ch14_output: str):
        """Repetition section must show before/after or reduced values."""
        lower = _lower(ch14_output)
        has_change = (
            "before" in lower
            or "after" in lower
            or "reduced" in lower
            or "penalized" in lower
            or "penalty" in lower
        )
        assert has_change, (
            "Repetition section must show penalty effect "
            "(e.g., 'before'/'after', 'reduced', 'penalized').\n"
            f"Got:\n{ch14_output[:1000]}"
        )


class TestPipeline:
    """Verify the composable pipeline is demonstrated."""

    def test_pipeline_or_processor_mentioned(self, ch14_output: str):
        """Output must mention pipeline or processor."""
        lower = _lower(ch14_output)
        assert "pipeline" in lower or "processor" in lower, (
            "Output must mention 'pipeline' or 'processor'.\n"
            f"Got:\n{ch14_output[:500]}"
        )

    def test_pipeline_shows_stages(self, ch14_output: str):
        """Pipeline section must show multiple stages applied in sequence."""
        lower = _lower(ch14_output)
        has_stages = (
            "stage" in lower
            or "step" in lower
            or ("after" in lower and ("temperature" in lower or "top" in lower))
        )
        assert has_stages, (
            "Pipeline section must show stages applied in sequence.\n"
            f"Got:\n{ch14_output[:1000]}"
        )


# ---------------------------------------------------------------------------
# Tests: Completion
# ---------------------------------------------------------------------------


class TestCompletion:
    """Verify the chapter completion marker."""

    def test_chapter_14_complete(self, ch14_output: str):
        """Output must contain the completion marker."""
        assert "chapter 14 complete" in _lower(ch14_output), (
            "Output must contain 'Chapter 14 complete'.\n"
            f"Got:\n{ch14_output[-500:]}"
        )
