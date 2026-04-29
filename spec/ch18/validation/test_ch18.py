"""
Chapter 18 validation tests: Structured Output.

Runs the reader's implementation as a subprocess and validates the output
against the interface spec (see ../interface-spec.md and ../expected-output.txt).
Language-agnostic -- works with any implementation.

Usage:
    RVLLM_CH18_BIN="cargo run --example ch18_structured_output" pytest test_ch18.py -v
    RVLLM_CH18_BIN="uv run python -m rvllm ch18_demo" pytest test_ch18.py -v
"""

import re

import pytest

from conftest import run_binary

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TIMEOUT_SECONDS = 30  # No model loading needed — mock logits only


@pytest.fixture(scope="module")
def ch18_output() -> str:
    """Run the Chapter 18 binary once and cache its stdout for all tests."""
    result = run_binary("RVLLM_CH18_BIN", timeout=TIMEOUT_SECONDS)
    return result.stdout


def _lower(output: str) -> str:
    return output.lower()


# ---------------------------------------------------------------------------
# Tests: All 5 parts present
# ---------------------------------------------------------------------------


class TestStructure:
    """Verify all 5 required parts are present in the output."""

    def test_part1_the_problem(self, ch18_output: str):
        """Output must contain Part 1: The Problem."""
        lower = _lower(ch18_output)
        assert "part 1" in lower or "the problem" in lower, (
            "Output must contain 'PART 1' or 'The Problem'.\n"
            f"Got:\n{ch18_output[:500]}"
        )

    def test_part2_fsm_construction(self, ch18_output: str):
        """Output must contain Part 2: FSM Construction."""
        lower = _lower(ch18_output)
        assert "part 2" in lower or "fsm construction" in lower, (
            "Output must contain 'PART 2' or 'FSM Construction'.\n"
            f"Got:\n{ch18_output[:500]}"
        )

    def test_part3_unconstrained_vs_constrained(self, ch18_output: str):
        """Output must contain Part 3: Unconstrained vs Constrained."""
        lower = _lower(ch18_output)
        assert (
            "part 3" in lower
            or "unconstrained vs constrained" in lower
            or ("unconstrained" in lower and "constrained" in lower)
        ), (
            "Output must contain 'PART 3' or 'Unconstrained vs Constrained'.\n"
            f"Got:\n{ch18_output[:500]}"
        )

    def test_part4_token_masking(self, ch18_output: str):
        """Output must contain Part 4: Token Masking."""
        lower = _lower(ch18_output)
        assert "part 4" in lower or "token masking" in lower, (
            "Output must contain 'PART 4' or 'Token Masking'.\n"
            f"Got:\n{ch18_output[:500]}"
        )

    def test_part5_pipeline_integration(self, ch18_output: str):
        """Output must contain Part 5: Pipeline Integration."""
        lower = _lower(ch18_output)
        assert "part 5" in lower or "pipeline integration" in lower, (
            "Output must contain 'PART 5' or 'Pipeline Integration'.\n"
            f"Got:\n{ch18_output[:500]}"
        )


# ---------------------------------------------------------------------------
# Tests: FSM / state machine concepts
# ---------------------------------------------------------------------------


class TestFSM:
    """Verify FSM / state machine concepts are demonstrated."""

    def test_fsm_mentioned(self, ch18_output: str):
        """Output must mention FSM or state machine."""
        lower = _lower(ch18_output)
        assert (
            "fsm" in lower
            or "state machine" in lower
            or "finite state" in lower
        ), (
            "Output must mention 'FSM', 'state machine', or 'finite state'.\n"
            f"Got:\n{ch18_output[:500]}"
        )

    def test_states_shown(self, ch18_output: str):
        """Output must show FSM states."""
        lower = _lower(ch18_output)
        assert "state 0" in lower or "state0" in lower or "initial" in lower, (
            "Output must show FSM states (e.g., 'State 0', 'initial').\n"
            f"Got:\n{ch18_output[:1000]}"
        )

    def test_transitions_shown(self, ch18_output: str):
        """Output must show FSM transitions."""
        lower = _lower(ch18_output)
        assert "transition" in lower or "->" in ch18_output or "advance" in lower, (
            "Output must show FSM transitions "
            "(e.g., 'transition', '->', 'advance').\n"
            f"Got:\n{ch18_output[:1000]}"
        )


# ---------------------------------------------------------------------------
# Tests: Constrained output
# ---------------------------------------------------------------------------


class TestConstrained:
    """Verify constrained/masked output is demonstrated."""

    def test_constrained_mentioned(self, ch18_output: str):
        """Output must mention constrained or mask."""
        lower = _lower(ch18_output)
        assert "constrained" in lower or "mask" in lower, (
            "Output must mention 'constrained' or 'mask'.\n"
            f"Got:\n{ch18_output[:500]}"
        )

    def test_tokens_filtered(self, ch18_output: str):
        """Output must show tokens being filtered/masked."""
        lower = _lower(ch18_output)
        has_filtering = (
            "-inf" in lower
            or "infinity" in lower
            or "masked" in lower
            or "invalid" in lower
            or "filtered" in lower
        )
        assert has_filtering, (
            "Output must show tokens being filtered "
            "(e.g., '-inf', 'masked', 'invalid', 'filtered').\n"
            f"Got:\n{ch18_output[:1000]}"
        )

    def test_valid_invalid_distinction(self, ch18_output: str):
        """Output must distinguish valid from invalid tokens."""
        lower = _lower(ch18_output)
        has_valid = "valid" in lower
        has_invalid = "invalid" in lower or "-inf" in lower or "masked" in lower
        assert has_valid and has_invalid, (
            "Output must show both valid and invalid tokens.\n"
            f"Got:\n{ch18_output[:1000]}"
        )


# ---------------------------------------------------------------------------
# Tests: Unconstrained comparison
# ---------------------------------------------------------------------------


class TestUnconstrained:
    """Verify unconstrained output is shown for comparison."""

    def test_unconstrained_mentioned(self, ch18_output: str):
        """Output must mention unconstrained generation."""
        lower = _lower(ch18_output)
        assert (
            "unconstrained" in lower
            or "without constraint" in lower
            or "no constraint" in lower
            or "no grammar" in lower
        ), (
            "Output must mention unconstrained generation for comparison.\n"
            f"Got:\n{ch18_output[:500]}"
        )

    def test_comparison_shown(self, ch18_output: str):
        """Output must show the contrast between constrained and unconstrained."""
        lower = _lower(ch18_output)
        # Should show that unconstrained picks a different (word) token
        has_contrast = (
            ("forty" in lower or "word" in lower or "not a number" in lower)
            or ("different" in lower and ("output" in lower or "result" in lower))
        )
        assert has_contrast, (
            "Output must show the contrast between unconstrained and constrained "
            "(e.g., unconstrained picks a word token like 'forty').\n"
            f"Got:\n{ch18_output[:1000]}"
        )


# ---------------------------------------------------------------------------
# Tests: Pipeline integration
# ---------------------------------------------------------------------------


class TestPipelineIntegration:
    """Verify GrammarConstraint integrates with LogitsProcessor pipeline."""

    def test_logitsprocessor_or_pipeline_mentioned(self, ch18_output: str):
        """Output must mention LogitsProcessor or pipeline."""
        lower = _lower(ch18_output)
        assert "logitsprocessor" in lower or "pipeline" in lower or "processor" in lower, (
            "Output must mention 'LogitsProcessor' or 'pipeline'.\n"
            f"Got:\n{ch18_output[:500]}"
        )

    def test_integration_demonstrated(self, ch18_output: str):
        """Output must show GrammarConstraint used as a processor in pipeline."""
        lower = _lower(ch18_output)
        has_integration = (
            ("grammar" in lower and ("pipeline" in lower or "processor" in lower))
            or "grammarconstraint" in lower
            or ("constraint" in lower and "pipeline" in lower)
        )
        assert has_integration, (
            "Output must show GrammarConstraint integrated with the pipeline.\n"
            f"Got:\n{ch18_output[:1000]}"
        )


# ---------------------------------------------------------------------------
# Tests: Completion
# ---------------------------------------------------------------------------


class TestCompletion:
    """Verify the chapter completion marker."""

    def test_chapter_18_complete(self, ch18_output: str):
        """Output must contain the completion marker."""
        assert "chapter 18 complete" in _lower(ch18_output), (
            "Output must contain 'Chapter 18 complete'.\n"
            f"Got:\n{ch18_output[-500:]}"
        )
