"""
Validation tests for Chapter 11: Continuous Batching.

Runs the ch11 binary and asserts that static vs continuous batching concepts
appear in the output. Language-agnostic -- works with any implementation.

Usage:
    RVLLM_CH11_BIN=/path/to/your/binary pytest test_ch11.py -v
    RVLLM_CH11_BIN="python3 ch11.py" pytest test_ch11.py -v
"""

import pytest

from conftest import run_binary  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def ch11_output():
    """Run the ch11 binary once and return its stdout."""
    result = run_binary("RVLLM_CH11_BIN")
    return result.stdout


# ---------------------------------------------------------------------------
# Structure tests — all 6 sections present
# ---------------------------------------------------------------------------

class TestSectionHeaders:
    """Verify all 6 parts are present in output."""

    @pytest.mark.parametrize("part_num", [1, 2, 3, 4, 5, 6])
    def test_part_header_present(self, ch11_output, part_num):
        assert f"PART {part_num}" in ch11_output

    def test_part1_title(self, ch11_output):
        output_lower = ch11_output.lower()
        assert "waiting" in output_lower or "static" in output_lower

    def test_part2_title(self, ch11_output):
        output_lower = ch11_output.lower()
        assert "continuous" in output_lower

    def test_part4_title(self, ch11_output):
        output_lower = ch11_output.lower()
        assert "prefill" in output_lower and "decode" in output_lower

    def test_part5_title(self, ch11_output):
        output_lower = ch11_output.lower()
        assert "sequencestatus" in output_lower or "sequence" in output_lower

    def test_part6_title(self, ch11_output):
        output_lower = ch11_output.lower()
        assert "throughput" in output_lower


# ---------------------------------------------------------------------------
# Batching concepts
# ---------------------------------------------------------------------------

class TestBatchingConcepts:
    """Verify core batching concepts appear."""

    def test_static_batching_mentioned(self, ch11_output):
        """Static batching should be discussed as the baseline."""
        assert "static" in ch11_output.lower()

    def test_continuous_batching_mentioned(self, ch11_output):
        """Continuous batching should be discussed as the improvement."""
        assert "continuous" in ch11_output.lower()

    def test_iteration_level(self, ch11_output):
        """Iteration-level scheduling should be mentioned."""
        assert "iteration" in ch11_output.lower()

    def test_batch_size(self, ch11_output):
        """Batch size should be shown."""
        assert "batch" in ch11_output.lower()


# ---------------------------------------------------------------------------
# Mixed batches
# ---------------------------------------------------------------------------

class TestMixedBatches:
    """Verify prefill + decode mixed batch handling."""

    def test_prefill_mentioned(self, ch11_output):
        """Prefill phase should be mentioned."""
        assert "prefill" in ch11_output.lower()

    def test_decode_mentioned(self, ch11_output):
        """Decode phase should be mentioned."""
        assert "decode" in ch11_output.lower()


# ---------------------------------------------------------------------------
# New types
# ---------------------------------------------------------------------------

class TestNewTypes:
    """Verify SequenceStatus and SequenceGroup types are demonstrated."""

    def test_waiting_status(self, ch11_output):
        """Waiting status should appear."""
        assert "Waiting" in ch11_output

    def test_running_status(self, ch11_output):
        """Running status should appear."""
        assert "Running" in ch11_output

    def test_finished_status(self, ch11_output):
        """Finished status should appear."""
        assert "Finished" in ch11_output

    def test_sequence_status_enum(self, ch11_output):
        """SequenceStatus should be named."""
        assert "SequenceStatus" in ch11_output or "sequence_status" in ch11_output.lower()


# ---------------------------------------------------------------------------
# Throughput comparison
# ---------------------------------------------------------------------------

class TestThroughputComparison:
    """Verify the head-to-head comparison."""

    def test_utilization_shown(self, ch11_output):
        """Utilization percentages should be shown."""
        assert "%" in ch11_output

    def test_idle_slots_shown(self, ch11_output):
        """Idle GPU slots should be counted."""
        assert "idle" in ch11_output.lower()

    def test_throughput_shown(self, ch11_output):
        """Throughput metric should appear."""
        output_lower = ch11_output.lower()
        assert "throughput" in output_lower or "tok/iter" in output_lower


# ---------------------------------------------------------------------------
# Closing
# ---------------------------------------------------------------------------

class TestClosing:
    """Verify the program ends correctly."""

    def test_chapter_complete(self, ch11_output):
        assert "Chapter 11 complete" in ch11_output

    def test_next_chapter_hook(self, ch11_output):
        assert "ch12" in ch11_output or "chapter 12" in ch11_output.lower()
