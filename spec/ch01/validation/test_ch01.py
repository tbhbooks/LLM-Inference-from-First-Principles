"""
Validation tests for Chapter 1: The LLM Inference Problem.

Runs the ch01 binary and asserts that key numbers and structure appear
in the output. Language-agnostic -- works with any implementation.

Usage:
    RVLLM_CH01_BIN=/path/to/your/binary pytest test_ch01.py -v
    RVLLM_CH01_BIN="python3 ch01.py" pytest test_ch01.py -v

See spec/runners/README.md for more examples (Rust, Python, Go, etc.).
"""

import pytest

from conftest import run_binary  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def ch01_output():
    """Run the ch01 binary once and return its stdout."""
    result = run_binary("RVLLM_CH01_BIN")
    return result.stdout


# ---------------------------------------------------------------------------
# Structure tests — all 5 sections present
# ---------------------------------------------------------------------------

class TestSectionHeaders:
    """Verify all 5 parts are present in output."""

    @pytest.mark.parametrize("part_num", [1, 2, 3, 4, 5])
    def test_part_header_present(self, ch01_output, part_num):
        assert f"PART {part_num}" in ch01_output

    def test_part1_title(self, ch01_output):
        assert "KV Cache Memory" in ch01_output

    def test_part2_title(self, ch01_output):
        assert "Sequence Length" in ch01_output

    def test_part3_title(self, ch01_output):
        assert "Concurrency Ceiling" in ch01_output

    def test_part4_title(self, ch01_output):
        assert "Memory Wall" in ch01_output

    def test_part5_title(self, ch01_output):
        assert "Prefill vs Decode" in ch01_output


# ---------------------------------------------------------------------------
# KV cache per-token sizes (Part 1)
# ---------------------------------------------------------------------------

class TestPerTokenSizes:
    """Verify per-token KV cache sizes are correct."""

    def test_gpt2_per_token(self, ch01_output):
        """GPT-2: 12 heads * 64 dim * 2 bytes * 2 (K+V) * 12 layers = 36,864 = 36.00 KB"""
        assert "36.00 KB" in ch01_output

    def test_llama7b_per_token(self, ch01_output):
        """LLaMA-7B: 32 heads * 128 dim * 2 bytes * 2 (K+V) * 32 layers = 524,288 = 512.00 KB"""
        assert "512.00 KB" in ch01_output

    def test_llama70b_per_token(self, ch01_output):
        """LLaMA-70B: 64 heads * 128 dim * 2 bytes * 2 (K+V) * 80 layers = 2,621,440 = 2.50 MB"""
        assert "2.50 MB" in ch01_output

    def test_gpt2_per_token_per_layer(self, ch01_output):
        """GPT-2: 3,072 bytes = 3.00 KB per token per layer"""
        assert "3.00 KB" in ch01_output

    def test_llama7b_per_token_per_layer(self, ch01_output):
        """LLaMA-7B: 16,384 bytes = 16.00 KB per token per layer"""
        assert "16.00 KB" in ch01_output

    def test_llama70b_per_token_per_layer(self, ch01_output):
        """LLaMA-70B: 32,768 bytes = 32.00 KB per token per layer"""
        assert "32.00 KB" in ch01_output


# ---------------------------------------------------------------------------
# Sequence-level sizes (Part 2)
# ---------------------------------------------------------------------------

class TestSequenceSizes:
    """Verify per-sequence KV cache sizes."""

    def test_gpt2_1024(self, ch01_output):
        assert "36.00 MB" in ch01_output

    def test_gpt2_4096(self, ch01_output):
        assert "144.00 MB" in ch01_output

    def test_llama7b_1024(self, ch01_output):
        assert "512.00 MB" in ch01_output

    def test_llama7b_4096(self, ch01_output):
        assert "2.00 GB" in ch01_output

    def test_llama70b_1024(self, ch01_output):
        assert "2.50 GB" in ch01_output

    def test_llama70b_4096(self, ch01_output):
        assert "10.00 GB" in ch01_output


# ---------------------------------------------------------------------------
# Concurrency ceiling (Part 3)
# ---------------------------------------------------------------------------

class TestConcurrencyCeiling:
    """Verify max concurrent sequence counts."""

    def test_gpt2_max_1024(self, ch01_output):
        """10 GB / 36 MB = 284 sequences"""
        assert "284" in ch01_output

    def test_gpt2_max_4096(self, ch01_output):
        """10 GB / 144 MB = 71 sequences"""
        assert "71" in ch01_output

    def test_llama7b_max_1024(self, ch01_output):
        """10 GB / 512 MB = 20 sequences"""
        assert "20" in ch01_output

    def test_llama7b_max_4096(self, ch01_output):
        """10 GB / 2 GB = 5 sequences"""
        assert "5" in ch01_output

    def test_llama70b_max_1024(self, ch01_output):
        """10 GB / 2.5 GB = 4 sequences"""
        assert "4" in ch01_output

    def test_llama70b_max_4096(self, ch01_output):
        """10 GB / 10 GB = 1 sequence"""
        assert "1 seq" in ch01_output


# ---------------------------------------------------------------------------
# Memory wall / OOM (Part 4)
# ---------------------------------------------------------------------------

class TestMemoryWall:
    """Verify the memory wall section shows OOM conditions."""

    def test_oom_appears(self, ch01_output):
        """At least one OOM should appear (LLaMA-7B @ 32 req, LLaMA-70B @ 8+)."""
        assert "OOM" in ch01_output

    def test_oom_count(self, ch01_output):
        """Should have exactly 4 OOM table entries + 1 legend line = 5 total:
        LLaMA-7B: 32req (1)
        LLaMA-70B: 8req, 16req, 32req (3)
        Legend: "OOM!! = Out of memory..." (1)
        """
        assert ch01_output.count("OOM!!") == 5


# ---------------------------------------------------------------------------
# Prefill vs Decode (Part 5)
# ---------------------------------------------------------------------------

class TestPrefillDecode:
    """Verify Part 5 content."""

    def test_prefill_mentioned(self, ch01_output):
        assert "PREFILL" in ch01_output

    def test_decode_mentioned(self, ch01_output):
        assert "DECODE" in ch01_output

    def test_compute_bound(self, ch01_output):
        assert "compute-bound" in ch01_output.lower() or "COMPUTE" in ch01_output

    def test_memory_bound(self, ch01_output):
        assert "memory-bound" in ch01_output.lower() or "MEMORY BANDWIDTH" in ch01_output

    def test_memory_problem(self, ch01_output):
        assert "MEMORY problem" in ch01_output


# ---------------------------------------------------------------------------
# Closing
# ---------------------------------------------------------------------------

class TestClosing:
    """Verify the program ends correctly."""

    def test_chapter_complete(self, ch01_output):
        assert "Chapter 1 complete" in ch01_output

    def test_next_chapter_hook(self, ch01_output):
        assert "ch02" in ch01_output or "chapter 2" in ch01_output.lower()
