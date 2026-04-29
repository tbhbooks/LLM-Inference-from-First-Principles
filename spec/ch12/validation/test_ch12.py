"""
Validation tests for Chapter 12: The Scheduler.

Runs the ch12 binary and asserts that scheduler concepts
appear in the output. Language-agnostic -- works with any implementation.

Usage:
    RVLLM_CH12_BIN=/path/to/your/binary pytest test_ch12.py -v
    RVLLM_CH12_BIN="python3 ch12.py" pytest test_ch12.py -v
"""

import pytest

from conftest import run_binary  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def ch12_output():
    """Run the ch12 binary once and return its stdout."""
    result = run_binary("RVLLM_CH12_BIN")
    return result.stdout


# ---------------------------------------------------------------------------
# Structure tests — all 6 sections present
# ---------------------------------------------------------------------------

class TestSectionHeaders:
    """Verify all 6 parts are present in output."""

    @pytest.mark.parametrize("part_num", [1, 2, 3, 4, 5, 6])
    def test_part_header_present(self, ch12_output, part_num):
        assert f"PART {part_num}" in ch12_output

    def test_part1_title(self, ch12_output):
        output_lower = ch12_output.lower()
        assert "queue" in output_lower or "gpu" in output_lower

    def test_part2_title(self, ch12_output):
        output_lower = ch12_output.lower()
        assert "schedule" in output_lower

    def test_part3_title(self, ch12_output):
        output_lower = ch12_output.lower()
        assert "memory" in output_lower or "admission" in output_lower

    def test_part4_title(self, ch12_output):
        output_lower = ch12_output.lower()
        assert "preempt" in output_lower or "memory" in output_lower

    def test_part5_title(self, ch12_output):
        output_lower = ch12_output.lower()
        assert "scheduler" in output_lower or "simulation" in output_lower

    def test_part6_title(self, ch12_output):
        output_lower = ch12_output.lower()
        assert "contract" in output_lower or "scheduleroutput" in output_lower


# ---------------------------------------------------------------------------
# Three queues
# ---------------------------------------------------------------------------

class TestThreeQueues:
    """Verify the three-queue architecture is demonstrated."""

    def test_waiting_queue(self, ch12_output):
        """Waiting queue should be shown."""
        assert "waiting" in ch12_output.lower()

    def test_running_queue(self, ch12_output):
        """Running set should be shown."""
        assert "running" in ch12_output.lower()

    def test_swapped_queue(self, ch12_output):
        """Swapped queue should be shown."""
        assert "swapped" in ch12_output.lower()


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------

class TestScheduling:
    """Verify core scheduling concepts."""

    def test_schedule_method(self, ch12_output):
        """The schedule() method should be called."""
        assert "schedule" in ch12_output.lower()

    def test_fcfs_or_fifo(self, ch12_output):
        """FCFS or FIFO ordering should be mentioned."""
        output_lower = ch12_output.lower()
        assert "fcfs" in output_lower or "fifo" in output_lower or "first" in output_lower

    def test_batch_limit(self, ch12_output):
        """Batch size limit should be enforced."""
        output_lower = ch12_output.lower()
        assert "max" in output_lower or "budget" in output_lower or "limit" in output_lower


# ---------------------------------------------------------------------------
# Preemption
# ---------------------------------------------------------------------------

class TestPreemption:
    """Verify preemption is demonstrated."""

    def test_preemption_mentioned(self, ch12_output):
        """Preemption should be discussed."""
        output_lower = ch12_output.lower()
        assert "preempt" in output_lower

    def test_preempted_ids(self, ch12_output):
        """Preempted request IDs should be shown."""
        output_lower = ch12_output.lower()
        assert "preempted" in output_lower


# ---------------------------------------------------------------------------
# Memory awareness
# ---------------------------------------------------------------------------

class TestMemoryAwareness:
    """Verify memory-aware admission."""

    def test_block_mentioned(self, ch12_output):
        """Blocks should be mentioned (memory unit)."""
        assert "block" in ch12_output.lower()

    def test_memory_check(self, ch12_output):
        """Memory availability should be checked."""
        output_lower = ch12_output.lower()
        assert "can_allocate" in output_lower or "free" in output_lower or "memory" in output_lower

    def test_allocate_mentioned(self, ch12_output):
        """Block allocation should be shown."""
        output_lower = ch12_output.lower()
        assert "allocat" in output_lower


# ---------------------------------------------------------------------------
# SchedulerOutput
# ---------------------------------------------------------------------------

class TestSchedulerOutput:
    """Verify SchedulerOutput structure is shown."""

    def test_new_or_prefill(self, ch12_output):
        """New requests or prefill should appear."""
        output_lower = ch12_output.lower()
        assert "new_requests" in output_lower or "prefill" in output_lower

    def test_running_or_decode(self, ch12_output):
        """Running requests or decode should appear."""
        output_lower = ch12_output.lower()
        assert "running" in output_lower or "decode" in output_lower

    def test_token_counts(self, ch12_output):
        """Token counts should be shown."""
        output_lower = ch12_output.lower()
        assert "token" in output_lower


# ---------------------------------------------------------------------------
# Multi-step simulation
# ---------------------------------------------------------------------------

class TestMultiStepSimulation:
    """Verify multi-step simulation shows queue evolution."""

    def test_multiple_steps(self, ch12_output):
        """Multiple steps should be shown."""
        assert "Step 1" in ch12_output or "step 1" in ch12_output.lower()
        assert "Step 2" in ch12_output or "step 2" in ch12_output.lower()

    def test_queue_state_changes(self, ch12_output):
        """Queue states should change across steps."""
        # At least one admission and one preemption/finish event
        output_lower = ch12_output.lower()
        assert "admit" in output_lower or "enters" in output_lower or "admitted" in output_lower

    def test_finish_event(self, ch12_output):
        """At least one request should finish."""
        output_lower = ch12_output.lower()
        assert "finish" in output_lower or "done" in output_lower or "complete" in output_lower


# ---------------------------------------------------------------------------
# Closing
# ---------------------------------------------------------------------------

class TestClosing:
    """Verify the program ends correctly."""

    def test_chapter_complete(self, ch12_output):
        assert "Chapter 12 complete" in ch12_output

    def test_next_chapter_hook(self, ch12_output):
        assert "ch13" in ch12_output or "chapter 13" in ch12_output.lower()
