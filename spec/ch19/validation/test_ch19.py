"""
Chapter 19 validation tests: Parallelism.

Runs the reader's implementation as a subprocess and validates the output
against the interface spec (see ../interface-spec.md and ../expected-output.txt).
Language-agnostic -- works with any implementation.

Usage:
    RVLLM_CH19_BIN="cargo run --example ch19_parallelism" pytest test_ch19.py -v
    RVLLM_CH19_BIN="uv run python -m rvllm ch19_demo" pytest test_ch19.py -v
"""

import re

import pytest

from conftest import run_binary

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TIMEOUT_SECONDS = 30  # Pure simulator — no model loading


@pytest.fixture(scope="module")
def ch19_output() -> str:
    """Run the Chapter 19 binary once and cache its stdout for all tests."""
    result = run_binary("RVLLM_CH19_BIN", timeout=TIMEOUT_SECONDS)
    return result.stdout


def _lower(output: str) -> str:
    return output.lower()


# ---------------------------------------------------------------------------
# Tests: All 5 parts present
# ---------------------------------------------------------------------------


class TestStructure:
    """Verify all 5 required parts are present in the output."""

    def test_part1_tensor_parallelism(self, ch19_output: str):
        """Output must contain Part 1: Tensor Parallelism."""
        lower = _lower(ch19_output)
        assert "part 1" in lower or "tensor parallelism" in lower, (
            "Output must contain 'PART 1' or 'Tensor Parallelism'.\n"
            f"Got:\n{ch19_output[:500]}"
        )

    def test_part2_naive_pipeline(self, ch19_output: str):
        """Output must contain Part 2: Naive Pipeline Parallelism."""
        lower = _lower(ch19_output)
        assert "part 2" in lower or "naive" in lower or "no overlap" in lower, (
            "Output must contain 'PART 2' or 'naive' or 'no overlap'.\n"
            f"Got:\n{ch19_output[:500]}"
        )

    def test_part3_batch_queue(self, ch19_output: str):
        """Output must contain Part 3: batch_queue Pipeline Parallelism."""
        lower = _lower(ch19_output)
        assert "part 3" in lower or "batch_queue" in lower or "micro-batch" in lower or "overlap" in lower, (
            "Output must contain 'PART 3' or 'batch_queue' or 'overlap'.\n"
            f"Got:\n{ch19_output[:500]}"
        )

    def test_part4_combined(self, ch19_output: str):
        """Output must contain Part 4: Combined TP+PP."""
        lower = _lower(ch19_output)
        assert "part 4" in lower or "combined" in lower, (
            "Output must contain 'PART 4' or 'combined'.\n"
            f"Got:\n{ch19_output[:500]}"
        )

    def test_part5_scaling(self, ch19_output: str):
        """Output must contain Part 5: Scaling Guide."""
        lower = _lower(ch19_output)
        assert "part 5" in lower or "scaling" in lower or "guide" in lower, (
            "Output must contain 'PART 5' or 'scaling'.\n"
            f"Got:\n{ch19_output[:500]}"
        )


# ---------------------------------------------------------------------------
# Tests: Tensor Parallelism concepts
# ---------------------------------------------------------------------------


class TestTensorParallelism:
    """Verify tensor parallelism concepts are demonstrated."""

    def test_tp_mentioned(self, ch19_output: str):
        """Output must mention tensor parallelism or TP."""
        lower = _lower(ch19_output)
        assert "tensor parallel" in lower or "tp=" in lower or "tp =" in lower or "tp:" in lower, (
            "Output must mention 'tensor parallelism' or 'TP'.\n"
            f"Got:\n{ch19_output[:500]}"
        )

    def test_allreduce_mentioned(self, ch19_output: str):
        """Output must mention AllReduce communication."""
        lower = _lower(ch19_output)
        assert "allreduce" in lower or "all-reduce" in lower or "all_reduce" in lower, (
            "Output must mention 'AllReduce' or 'all-reduce'.\n"
            f"Got:\n{ch19_output[:500]}"
        )

    def test_communication_volume_shown(self, ch19_output: str):
        """Output must show communication volume numbers (bytes or MB/GB)."""
        lower = _lower(ch19_output)
        has_volume = (
            "bytes" in lower
            or " mb" in lower
            or " gb" in lower
            or "volume" in lower
            or "communication" in lower
        )
        assert has_volume, (
            "Output must show communication volume (bytes, MB, GB).\n"
            f"Got:\n{ch19_output[:1000]}"
        )


# ---------------------------------------------------------------------------
# Tests: Pipeline Parallelism concepts
# ---------------------------------------------------------------------------


class TestPipelineParallelism:
    """Verify pipeline parallelism concepts are demonstrated."""

    def test_pipeline_mentioned(self, ch19_output: str):
        """Output must mention pipeline parallelism or PP."""
        lower = _lower(ch19_output)
        assert "pipeline" in lower, (
            "Output must mention 'pipeline'.\n"
            f"Got:\n{ch19_output[:500]}"
        )

    def test_bubble_or_utilization_shown(self, ch19_output: str):
        """Output must mention bubbles or utilization."""
        lower = _lower(ch19_output)
        assert "bubble" in lower or "utilization" in lower or "idle" in lower, (
            "Output must mention 'bubble', 'utilization', or 'idle'.\n"
            f"Got:\n{ch19_output[:1000]}"
        )

    def test_timeline_shown(self, ch19_output: str):
        """Output must show a timeline or schedule grid."""
        lower = _lower(ch19_output)
        has_timeline = (
            "timeline" in lower
            or "timestep" in lower
            or "stage" in lower
            or "schedule" in lower
        )
        assert has_timeline, (
            "Output must show a timeline or schedule.\n"
            f"Got:\n{ch19_output[:1000]}"
        )

    def test_utilization_percentage(self, ch19_output: str):
        """Output must show a utilization percentage."""
        # Look for patterns like "25.0%" or "57.1%" or "25%"
        pct_pattern = r"\d+\.?\d*\s*%"
        matches = re.findall(pct_pattern, ch19_output)
        assert len(matches) >= 1, (
            "Output must show at least one utilization percentage.\n"
            f"Got:\n{ch19_output[:1000]}"
        )


# ---------------------------------------------------------------------------
# Tests: batch_queue / overlap
# ---------------------------------------------------------------------------


class TestBatchQueue:
    """Verify batch_queue or micro-batch overlap is demonstrated."""

    def test_batch_queue_or_overlap_mentioned(self, ch19_output: str):
        """Output must mention batch_queue, micro-batch, or overlap."""
        lower = _lower(ch19_output)
        assert (
            "batch_queue" in lower
            or "micro-batch" in lower
            or "micro_batch" in lower
            or "microbatch" in lower
            or "overlap" in lower
        ), (
            "Output must mention 'batch_queue', 'micro-batch', or 'overlap'.\n"
            f"Got:\n{ch19_output[:500]}"
        )

    def test_improved_utilization(self, ch19_output: str):
        """Output must show improved utilization with batch_queue vs naive."""
        lower = _lower(ch19_output)
        # Check that output discusses improvement or comparison
        has_comparison = (
            "improved" in lower
            or "better" in lower
            or "comparison" in lower
            or "vs" in lower
            or "naive" in lower
        )
        # Also accept showing multiple utilization percentages
        pct_pattern = r"\d+\.?\d*\s*%"
        pct_matches = re.findall(pct_pattern, ch19_output)

        assert has_comparison or len(pct_matches) >= 2, (
            "Output must show improved utilization with batch_queue.\n"
            "Expected comparison terms or multiple percentage values.\n"
            f"Got:\n{ch19_output[:1000]}"
        )


# ---------------------------------------------------------------------------
# Tests: Combined TP+PP
# ---------------------------------------------------------------------------


class TestCombined:
    """Verify combined TP+PP is demonstrated."""

    def test_combined_shown(self, ch19_output: str):
        """Output must show TP and PP used together."""
        lower = _lower(ch19_output)
        has_tp = "tp" in lower or "tensor parallel" in lower
        has_pp = "pp" in lower or "pipeline parallel" in lower or "pipeline" in lower
        assert has_tp and has_pp, (
            "Output must show both TP and PP.\n"
            f"Got:\n{ch19_output[:500]}"
        )

    def test_total_gpu_count(self, ch19_output: str):
        """Output must show total GPU count for combined config."""
        lower = _lower(ch19_output)
        has_gpu_count = (
            "8 gpu" in lower
            or "8gpu" in lower
            or "total gpu" in lower
            or "total: 8" in lower
            or "= 8" in lower
        )
        assert has_gpu_count, (
            "Output must show total GPU count (e.g., '8 GPUs').\n"
            f"Got:\n{ch19_output[:1000]}"
        )


# ---------------------------------------------------------------------------
# Tests: Completion
# ---------------------------------------------------------------------------


class TestCompletion:
    """Verify the chapter completion marker."""

    def test_chapter_19_complete(self, ch19_output: str):
        """Output must contain the completion marker."""
        assert "chapter 19 complete" in _lower(ch19_output), (
            "Output must contain 'Chapter 19 complete'.\n"
            f"Got:\n{ch19_output[-500:]}"
        )
