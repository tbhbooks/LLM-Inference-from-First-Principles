"""
Validation tests for Chapter 9: The Memory Problem.

Runs the ch09 binary and asserts that key concepts and numbers appear
in the output. Language-agnostic -- works with any implementation.

Usage:
    RVLLM_CH09_BIN=/path/to/your/binary pytest test_ch09.py -v
    RVLLM_CH09_BIN="python3 ch09.py" pytest test_ch09.py -v
"""

import pytest

from conftest import run_binary  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def ch09_output():
    """Run the ch09 binary once and return its stdout."""
    result = run_binary("RVLLM_CH09_BIN")
    return result.stdout


# ---------------------------------------------------------------------------
# Structure tests — all 6 sections present
# ---------------------------------------------------------------------------

class TestSectionHeaders:
    """Verify all 6 parts are present in output."""

    @pytest.mark.parametrize("part_num", [1, 2, 3, 4, 5, 6])
    def test_part_header_present(self, ch09_output, part_num):
        assert f"PART {part_num}" in ch09_output

    def test_part1_title(self, ch09_output):
        assert "Internal Fragmentation" in ch09_output

    def test_part2_title(self, ch09_output):
        assert "External Fragmentation" in ch09_output

    def test_part3_title(self, ch09_output):
        assert "Scaling Wall" in ch09_output

    def test_part4_title(self, ch09_output):
        assert "Analogy" in ch09_output


# ---------------------------------------------------------------------------
# Internal fragmentation (Scenario 1)
# ---------------------------------------------------------------------------

class TestInternalFragmentation:
    """Verify scenario 1: waste from over-allocation."""

    def test_waste_percentage(self, ch09_output):
        """47 wasted out of 90 allocated = 52.2%"""
        assert "52.2%" in ch09_output

    def test_total_allocated(self, ch09_output):
        """90 slots allocated total."""
        assert "90" in ch09_output

    def test_tokens_used(self, ch09_output):
        """43 tokens actually used."""
        assert "43" in ch09_output

    def test_waste_count(self, ch09_output):
        """47 slots wasted."""
        assert "47" in ch09_output


# ---------------------------------------------------------------------------
# External fragmentation (Scenario 2)
# ---------------------------------------------------------------------------

class TestExternalFragmentation:
    """Verify scenario 2: gaps that prevent allocation."""

    def test_allocation_fails(self, ch09_output):
        """Request F allocation should fail."""
        output_upper = ch09_output.upper()
        assert "FAIL" in output_upper, (
            "Expected 'FAIL' or 'FAILED' in output.\n"
            f"Got:\n{ch09_output[:500]}"
        )

    def test_free_slots_count(self, ch09_output):
        """45 free slots after deallocating B and D."""
        assert "45" in ch09_output

    def test_largest_block(self, ch09_output):
        """Largest contiguous block should be 15."""
        assert "15" in ch09_output


# ---------------------------------------------------------------------------
# Scaling wall (Scenario 3)
# ---------------------------------------------------------------------------

class TestScalingWall:
    """Verify scenario 3: how many requests fit."""

    def test_three_requests_fit(self, ch09_output):
        """floor(100/30) = 3 requests."""
        assert "3" in ch09_output


# ---------------------------------------------------------------------------
# OS analogy (Scenario 4)
# ---------------------------------------------------------------------------

class TestOSAnalogy:
    """Verify the OS virtual memory comparison."""

    def test_page_mentioned(self, ch09_output):
        """OS paging concept referenced."""
        output_lower = ch09_output.lower()
        assert "page" in output_lower

    def test_block_mentioned(self, ch09_output):
        """Block-based allocation concept referenced."""
        output_lower = ch09_output.lower()
        assert "block" in output_lower

    def test_page_table_or_block_table(self, ch09_output):
        """The key mapping concept referenced."""
        output_lower = ch09_output.lower()
        assert "page table" in output_lower or "block table" in output_lower


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

class TestVisualization:
    """Verify ASCII memory visualization is present."""

    def test_free_slot_dots(self, ch09_output):
        """Free slots shown as '.' characters."""
        assert ".." in ch09_output, (
            "Expected '..' (free slots) in memory visualization.\n"
            f"Got:\n{ch09_output[:500]}"
        )


# ---------------------------------------------------------------------------
# Key insight and closing
# ---------------------------------------------------------------------------

class TestKeyInsight:
    """Verify the chapter's central message."""

    def test_paged_attention_teased(self, ch09_output):
        """PagedAttention or paging concept named."""
        output_lower = ch09_output.lower()
        assert "pagedattention" in output_lower or "paged attention" in output_lower or "page table" in output_lower


class TestClosing:
    """Verify the program ends correctly."""

    def test_chapter_complete(self, ch09_output):
        assert "Chapter 9 complete" in ch09_output

    def test_next_chapter_hook(self, ch09_output):
        assert "ch10" in ch09_output or "chapter 10" in ch09_output.lower()
