"""
Validation tests for Chapter 13: The Engine Loop.

Runs the ch13 binary and asserts that engine orchestration concepts
appear in the output. Language-agnostic -- works with any implementation.

Usage:
    RVLLM_CH13_BIN=/path/to/your/binary pytest test_ch13.py -v
    RVLLM_CH13_BIN="python3 ch13.py" pytest test_ch13.py -v
"""

import pytest

from conftest import run_binary  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def ch13_output():
    """Run the ch13 binary once and return its stdout."""
    result = run_binary("RVLLM_CH13_BIN")
    return result.stdout


# ---------------------------------------------------------------------------
# Structure tests — all 6 sections present
# ---------------------------------------------------------------------------

class TestSectionHeaders:
    """Verify all 6 parts are present in output."""

    @pytest.mark.parametrize("part_num", [1, 2, 3, 4, 5, 6])
    def test_part_header_present(self, ch13_output, part_num):
        assert f"PART {part_num}" in ch13_output

    def test_part1_title(self, ch13_output):
        assert "Conductor" in ch13_output or "Engine" in ch13_output

    def test_part2_title(self, ch13_output):
        assert "Step" in ch13_output or "step" in ch13_output

    def test_part5_title(self, ch13_output):
        assert "Journey" in ch13_output or "Lifecycle" in ch13_output or "journey" in ch13_output

    def test_part6_title(self, ch13_output):
        assert "Pipeline" in ch13_output or "pipeline" in ch13_output


# ---------------------------------------------------------------------------
# Engine concepts — step phases
# ---------------------------------------------------------------------------

class TestEngineConcepts:
    """Verify core engine loop concepts appear."""

    def test_step_mentioned(self, ch13_output):
        """The word 'step' should appear (the core method)."""
        assert "step" in ch13_output.lower()

    def test_schedule_phase(self, ch13_output):
        """Schedule phase should be shown."""
        assert "schedule" in ch13_output.lower()

    def test_forward_phase(self, ch13_output):
        """Forward pass phase should be shown."""
        assert "forward" in ch13_output.lower()

    def test_sample_phase(self, ch13_output):
        """Sampling phase should be shown."""
        assert "sample" in ch13_output.lower()

    def test_update_phase(self, ch13_output):
        """Update phase should be shown."""
        assert "update" in ch13_output.lower()


# ---------------------------------------------------------------------------
# Request lifecycle
# ---------------------------------------------------------------------------

class TestRequestLifecycle:
    """Verify request status transitions are demonstrated."""

    def test_waiting_status(self, ch13_output):
        """Waiting status should appear."""
        assert "Waiting" in ch13_output or "waiting" in ch13_output

    def test_running_status(self, ch13_output):
        """Running status should appear."""
        assert "Running" in ch13_output or "running" in ch13_output

    def test_finished_status(self, ch13_output):
        """Finished status should appear."""
        output = ch13_output.upper()
        assert "FINISHED" in output or "COMPLETED" in output or "COMPLETE" in output


# ---------------------------------------------------------------------------
# Block tracking
# ---------------------------------------------------------------------------

class TestBlockTracking:
    """Verify block allocation is tracked across steps."""

    def test_free_blocks_shown(self, ch13_output):
        """Free block count should be visible."""
        assert "free" in ch13_output.lower()

    def test_block_allocation(self, ch13_output):
        """Block allocation should be shown."""
        output_lower = ch13_output.lower()
        assert "allocat" in output_lower  # allocate, allocated, allocation

    def test_block_freeing(self, ch13_output):
        """Block freeing should be shown when requests complete."""
        output_lower = ch13_output.lower()
        assert "free" in output_lower or "freed" in output_lower


# ---------------------------------------------------------------------------
# Multiple requests / batching
# ---------------------------------------------------------------------------

class TestBatching:
    """Verify multiple requests run in the same batch."""

    def test_multiple_requests(self, ch13_output):
        """Multiple request IDs should appear."""
        output = ch13_output
        # At least two different request identifiers
        has_a_and_b = ("A" in output and "B" in output)
        has_req_0_and_1 = ("request 0" in output.lower() and "request 1" in output.lower())
        assert has_a_and_b or has_req_0_and_1

    def test_batch_shown(self, ch13_output):
        """Batch composition should be visible."""
        output_lower = ch13_output.lower()
        assert ("batch" in output_lower or
                "scheduled" in output_lower or
                "sequence" in output_lower)


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------

class TestInputPreparation:
    """Verify prefill vs decode distinction."""

    def test_prefill_mentioned(self, ch13_output):
        """Prefill mode should be mentioned."""
        assert "prefill" in ch13_output.lower()

    def test_decode_mentioned(self, ch13_output):
        """Decode mode should be mentioned."""
        assert "decode" in ch13_output.lower()

    def test_positions_shown(self, ch13_output):
        """Token positions should be shown."""
        assert "position" in ch13_output.lower()


# ---------------------------------------------------------------------------
# Closing
# ---------------------------------------------------------------------------

class TestClosing:
    """Verify the program ends correctly."""

    def test_chapter_complete(self, ch13_output):
        assert "Chapter 13 complete" in ch13_output

    def test_next_chapter_hook(self, ch13_output):
        assert "ch14" in ch13_output or "chapter 14" in ch13_output.lower()
