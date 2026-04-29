"""
Validation tests for Chapter 10: PagedAttention.

Runs the ch10 binary and asserts that block-based allocation concepts
appear in the output. Language-agnostic -- works with any implementation.

Usage:
    RVLLM_CH10_BIN=/path/to/your/binary pytest test_ch10.py -v
    RVLLM_CH10_BIN="python3 ch10.py" pytest test_ch10.py -v
"""

import pytest

from conftest import run_binary  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def ch10_output():
    """Run the ch10 binary once and return its stdout."""
    result = run_binary("RVLLM_CH10_BIN")
    return result.stdout


# ---------------------------------------------------------------------------
# Structure tests — all 6 sections present
# ---------------------------------------------------------------------------

class TestSectionHeaders:
    """Verify all 6 parts are present in output."""

    @pytest.mark.parametrize("part_num", [1, 2, 3, 4, 5, 6])
    def test_part_header_present(self, ch10_output, part_num):
        assert f"PART {part_num}" in ch10_output

    def test_part1_title(self, ch10_output):
        assert "Block Allocation" in ch10_output

    def test_part4_title(self, ch10_output):
        assert "Slot Mapping" in ch10_output

    def test_part5_title(self, ch10_output):
        assert "Head-to-Head" in ch10_output or "Contiguous vs Paged" in ch10_output


# ---------------------------------------------------------------------------
# Block concepts
# ---------------------------------------------------------------------------

class TestBlockConcepts:
    """Verify core PagedAttention concepts appear."""

    def test_block_mentioned(self, ch10_output):
        """The word 'block' should appear (the allocation unit)."""
        assert "block" in ch10_output.lower()

    def test_block_size(self, ch10_output):
        """Block size of 16 should be mentioned."""
        assert "16" in ch10_output

    def test_slot_mentioned(self, ch10_output):
        """Physical slot mapping should appear."""
        assert "slot" in ch10_output.lower()

    def test_block_ids_shown(self, ch10_output):
        """Block IDs should be visible (numeric)."""
        # At least "block 0" or "block_id" or similar
        output_lower = ch10_output.lower()
        assert ("block 0" in output_lower or
                "block_id" in output_lower or
                "blocks: [" in output_lower or
                "block 1" in output_lower)


# ---------------------------------------------------------------------------
# Slot mapping
# ---------------------------------------------------------------------------

class TestSlotMapping:
    """Verify virtual-to-physical slot mapping is demonstrated."""

    def test_offset_shown(self, ch10_output):
        """Offset within a block should be shown."""
        assert "offset" in ch10_output.lower()

    def test_physical_slot_shown(self, ch10_output):
        """Physical slot index should be computed."""
        output_lower = ch10_output.lower()
        assert "physical" in output_lower or "slot" in output_lower


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

class TestComparison:
    """Verify contiguous vs paged comparison."""

    def test_contiguous_mentioned(self, ch10_output):
        """Contiguous allocation referenced for comparison."""
        output_lower = ch10_output.lower()
        assert "contiguous" in output_lower

    def test_paged_mentioned(self, ch10_output):
        """Paged allocation referenced."""
        output_lower = ch10_output.lower()
        assert "paged" in output_lower or "block" in output_lower

    def test_waste_or_fragmentation(self, ch10_output):
        """Comparison terms used."""
        output_lower = ch10_output.lower()
        assert "waste" in output_lower or "fragmentation" in output_lower

    def test_paged_serves_more(self, ch10_output):
        """Paged should serve more requests than contiguous."""
        # The comparison table should show paged serving 5 vs contiguous serving 4
        assert "5" in ch10_output and "4" in ch10_output


# ---------------------------------------------------------------------------
# Key benefits
# ---------------------------------------------------------------------------

class TestBenefits:
    """Verify Part 6 summarizes the wins."""

    def test_zero_fragmentation(self, ch10_output):
        """External fragmentation elimination mentioned."""
        output_lower = ch10_output.lower()
        assert "fragmentation" in output_lower

    def test_dynamic_growth(self, ch10_output):
        """Dynamic allocation mentioned."""
        output_lower = ch10_output.lower()
        assert "dynamic" in output_lower or "grow" in output_lower


# ---------------------------------------------------------------------------
# Closing
# ---------------------------------------------------------------------------

class TestClosing:
    """Verify the program ends correctly."""

    def test_chapter_complete(self, ch10_output):
        assert "Chapter 10 complete" in ch10_output

    def test_next_chapter_hook(self, ch10_output):
        assert "ch11" in ch10_output or "chapter 11" in ch10_output.lower()
