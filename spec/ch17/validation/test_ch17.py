"""
Chapter 17 validation tests: Speculative Decoding.

Runs the reader's implementation as a subprocess and validates the output
against the interface spec (see ../interface-spec.md and ../expected-output.txt).
Language-agnostic -- works with any implementation.

Usage:
    RVLLM_CH17_BIN="cargo run --example ch17_speculative_decoding" pytest test_ch17.py -v
    RVLLM_CH17_BIN="uv run python -m rvllm ch17_demo" pytest test_ch17.py -v
"""

import re

import pytest

from conftest import run_binary

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TIMEOUT_SECONDS = 30  # No model loading needed — mock models only


@pytest.fixture(scope="module")
def ch17_output() -> str:
    """Run the Chapter 17 binary once and cache its stdout for all tests."""
    result = run_binary("RVLLM_CH17_BIN", timeout=TIMEOUT_SECONDS)
    return result.stdout


def _lower(output: str) -> str:
    return output.lower()


# ---------------------------------------------------------------------------
# Tests: All 5 parts present
# ---------------------------------------------------------------------------


class TestStructure:
    """Verify all 5 required parts are present in the output."""

    def test_part1_the_problem(self, ch17_output: str):
        """Output must contain Part 1: The Problem."""
        lower = _lower(ch17_output)
        assert "part 1" in lower or "the problem" in lower or "gpu starvation" in lower, (
            "Output must contain 'PART 1' or 'The Problem' or 'GPU Starvation'.\n"
            f"Got:\n{ch17_output[:500]}"
        )

    def test_part2_draft_phase(self, ch17_output: str):
        """Output must contain Part 2: Draft Phase."""
        lower = _lower(ch17_output)
        assert "part 2" in lower or "draft phase" in lower or "proposing tokens" in lower, (
            "Output must contain 'PART 2' or 'Draft Phase'.\n"
            f"Got:\n{ch17_output[:500]}"
        )

    def test_part3_verify_phase(self, ch17_output: str):
        """Output must contain Part 3: Verify Phase."""
        lower = _lower(ch17_output)
        assert "part 3" in lower or "verify phase" in lower or "accept or reject" in lower, (
            "Output must contain 'PART 3' or 'Verify Phase'.\n"
            f"Got:\n{ch17_output[:500]}"
        )

    def test_part4_speedup(self, ch17_output: str):
        """Output must contain Part 4: Speedup."""
        lower = _lower(ch17_output)
        assert "part 4" in lower or "speedup" in lower or "tokens per pass" in lower, (
            "Output must contain 'PART 4' or 'Speedup'.\n"
            f"Got:\n{ch17_output[:500]}"
        )

    def test_part5_acceptance_rate(self, ch17_output: str):
        """Output must contain Part 5: Acceptance Rate."""
        lower = _lower(ch17_output)
        assert "part 5" in lower or "acceptance rate" in lower or "tracking performance" in lower, (
            "Output must contain 'PART 5' or 'Acceptance Rate'.\n"
            f"Got:\n{ch17_output[:500]}"
        )


# ---------------------------------------------------------------------------
# Tests: Draft Phase
# ---------------------------------------------------------------------------


class TestDraftPhase:
    """Verify draft phase is demonstrated."""

    def test_draft_mentioned(self, ch17_output: str):
        """Output must mention draft/drafting."""
        assert "draft" in _lower(ch17_output), (
            "Output must mention 'draft'.\n"
            f"Got:\n{ch17_output[:500]}"
        )

    def test_draft_tokens_shown(self, ch17_output: str):
        """Output must show draft tokens being proposed."""
        lower = _lower(ch17_output)
        has_draft_tokens = (
            "proposed" in lower
            or "drafted" in lower
            or "draft token" in lower
            or "candidate" in lower
        )
        assert has_draft_tokens, (
            "Output must show draft tokens being proposed "
            "(e.g., 'proposed', 'drafted', 'draft token', 'candidate').\n"
            f"Got:\n{ch17_output[:1000]}"
        )


# ---------------------------------------------------------------------------
# Tests: Verify Phase
# ---------------------------------------------------------------------------


class TestVerifyPhase:
    """Verify that the verification phase is demonstrated with all scenarios."""

    def test_accept_mentioned(self, ch17_output: str):
        """Output must mention accepting tokens."""
        assert "accept" in _lower(ch17_output), (
            "Output must mention 'accept'.\n"
            f"Got:\n{ch17_output[:500]}"
        )

    def test_reject_mentioned(self, ch17_output: str):
        """Output must mention rejecting tokens."""
        assert "reject" in _lower(ch17_output), (
            "Output must mention 'reject'.\n"
            f"Got:\n{ch17_output[:500]}"
        )

    def test_bonus_token_mentioned(self, ch17_output: str):
        """Output must mention the bonus token."""
        assert "bonus" in _lower(ch17_output), (
            "Output must mention 'bonus' token.\n"
            f"Got:\n{ch17_output[:500]}"
        )

    def test_full_accept_scenario(self, ch17_output: str):
        """Output must show a scenario where all K tokens are accepted."""
        lower = _lower(ch17_output)
        has_full_accept = (
            "all accepted" in lower
            or "4/4" in lower
            or "all match" in lower
            or "full accept" in lower
            or "total=5" in lower
            or "total: 5" in lower
        )
        assert has_full_accept, (
            "Output must show a full accept scenario "
            "(e.g., 'all accepted', '4/4', 'total=5').\n"
            f"Got:\n{ch17_output[:2000]}"
        )

    def test_partial_accept_scenario(self, ch17_output: str):
        """Output must show a scenario where some tokens are rejected."""
        lower = _lower(ch17_output)
        has_partial = (
            "2/4" in lower
            or "partial" in lower
            or "mismatch" in lower
            or ("accept" in lower and "reject" in lower)
        )
        assert has_partial, (
            "Output must show a partial accept scenario "
            "(e.g., '2/4', 'partial', or showing both accept and reject).\n"
            f"Got:\n{ch17_output[:2000]}"
        )

    def test_immediate_reject_scenario(self, ch17_output: str):
        """Output must show a scenario where the first token is rejected."""
        lower = _lower(ch17_output)
        has_immediate = (
            "0/4" in lower
            or "immediate reject" in lower
            or "total=1" in lower
            or "total: 1" in lower
            or ("accepted=0" in lower or "accepted: 0" in lower)
        )
        assert has_immediate, (
            "Output must show an immediate reject scenario "
            "(e.g., '0/4', 'immediate reject', 'total=1').\n"
            f"Got:\n{ch17_output[:2000]}"
        )


# ---------------------------------------------------------------------------
# Tests: Acceptance Rate
# ---------------------------------------------------------------------------


class TestAcceptance:
    """Verify acceptance rate tracking is demonstrated."""

    def test_acceptance_rate_shown(self, ch17_output: str):
        """Output must show acceptance rate as a percentage or ratio."""
        lower = _lower(ch17_output)
        has_rate = (
            "acceptance rate" in lower
            or "accept rate" in lower
            or re.search(r"\d+\.\d+%", ch17_output) is not None
            or re.search(r"\d+/\d+", ch17_output) is not None
        )
        assert has_rate, (
            "Output must show acceptance rate "
            "(e.g., 'acceptance rate', '70.0%', '56/80').\n"
            f"Got:\n{ch17_output[-1000:]}"
        )

    def test_multiple_steps_shown(self, ch17_output: str):
        """Output must show results from multiple speculative steps."""
        lower = _lower(ch17_output)
        has_multi_step = (
            "step" in lower
            and (
                re.search(r"step\s+\d+", lower) is not None
                or "steps" in lower
            )
        )
        assert has_multi_step, (
            "Output must show multiple speculative steps "
            "(e.g., 'Step 1', 'Step 2', '20 steps').\n"
            f"Got:\n{ch17_output[-1000:]}"
        )


# ---------------------------------------------------------------------------
# Tests: Speedup
# ---------------------------------------------------------------------------


class TestSpeedup:
    """Verify speedup analysis is demonstrated."""

    def test_speedup_mentioned(self, ch17_output: str):
        """Output must mention speedup or tokens per pass."""
        lower = _lower(ch17_output)
        has_speedup = (
            "speedup" in lower
            or "tokens per" in lower
            or "times faster" in lower
            or re.search(r"\d+\.?\d*x", lower) is not None
        )
        assert has_speedup, (
            "Output must mention speedup or tokens per pass.\n"
            f"Got:\n{ch17_output[-1000:]}"
        )

    def test_comparison_shown(self, ch17_output: str):
        """Output must compare with and without speculation."""
        lower = _lower(ch17_output)
        has_comparison = (
            "without" in lower
            or "standard decode" in lower
            or "vs" in lower
            or "compared" in lower
            or "1 token" in lower
        )
        assert has_comparison, (
            "Output must compare speculative vs standard decode.\n"
            f"Got:\n{ch17_output[-1000:]}"
        )


# ---------------------------------------------------------------------------
# Tests: Completion
# ---------------------------------------------------------------------------


class TestCompletion:
    """Verify the chapter completion marker."""

    def test_chapter_17_complete(self, ch17_output: str):
        """Output must contain the completion marker."""
        assert "chapter 17 complete" in _lower(ch17_output), (
            "Output must contain 'Chapter 17 complete'.\n"
            f"Got:\n{ch17_output[-500:]}"
        )
