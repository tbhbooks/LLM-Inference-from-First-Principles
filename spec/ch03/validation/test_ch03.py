"""
Chapter 3 validation tests.

Runs the ch03 binary and checks CLI help, stub commands, and error handling.
Language-agnostic -- works with any implementation.

Usage:
    RVLLM_CH03_BIN=/path/to/your/binary pytest test_ch03.py -v
    RVLLM_CH03_BIN="python3 main.py" pytest test_ch03.py -v

See spec/runners/README.md for more examples (Rust, Python, Go, etc.).
"""

import pytest

from conftest import run_binary


def run(args: list[str], expect_success: bool = True):
    """Run the ch03 binary with the given arguments and return the result."""
    return run_binary("RVLLM_CH03_BIN", args=args, timeout=10, expect_success=expect_success)


# ---------------------------------------------------------------------------
# Help output
# ---------------------------------------------------------------------------

class TestHelpOutput:
    """Verify that --help mentions the expected subcommands and flags."""

    def test_top_level_help_mentions_generate(self):
        result = run(["--help"])
        assert "generate" in result.stdout.lower()

    def test_top_level_help_mentions_inspect(self):
        result = run(["--help"])
        assert "inspect" in result.stdout.lower()

    def test_generate_help_mentions_prompt(self):
        result = run(["generate", "--help"])
        assert "--prompt" in result.stdout

    def test_generate_help_mentions_model(self):
        result = run(["generate", "--help"])
        assert "--model" in result.stdout

    def test_generate_help_mentions_max_tokens(self):
        result = run(["generate", "--help"])
        assert "max-tokens" in result.stdout.lower() or "max_tokens" in result.stdout.lower()

    def test_inspect_help_mentions_model(self):
        result = run(["inspect", "--help"])
        assert "--model" in result.stdout


# ---------------------------------------------------------------------------
# Stub behavior
# ---------------------------------------------------------------------------

class TestStubCommands:
    """Verify that stub commands run without crashing and produce output."""

    def test_generate_exits_zero(self):
        result = run(["generate", "--prompt", "Hello"])
        assert result.returncode == 0

    def test_generate_produces_output(self):
        result = run(["generate", "--prompt", "Hello"])
        # Should print something indicating it is not yet implemented.
        combined = result.stdout + result.stderr
        assert len(combined.strip()) > 0

    def test_generate_with_all_flags(self):
        result = run([
            "generate",
            "--prompt", "Hello world",
            "--model", "openai-community/gpt2",
            "--max-tokens", "64",
        ])
        assert result.returncode == 0

    def test_inspect_exits_zero(self):
        result = run(["inspect"])
        assert result.returncode == 0

    def test_inspect_produces_output(self):
        result = run(["inspect"])
        combined = result.stdout + result.stderr
        assert len(combined.strip()) > 0

    def test_inspect_with_model_flag(self):
        result = run(["inspect", "--model", "openai-community/gpt2"])
        assert result.returncode == 0


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """Verify that invalid input is handled gracefully (non-zero exit, no crash)."""

    def test_no_subcommand_exits_nonzero(self):
        result = run([], expect_success=False)
        assert result.returncode != 0

    def test_generate_missing_prompt_exits_nonzero(self):
        result = run(["generate"], expect_success=False)
        assert result.returncode != 0

    def test_unknown_subcommand_exits_nonzero(self):
        result = run(["frobnicate"], expect_success=False)
        assert result.returncode != 0

    def test_unknown_flag_exits_nonzero(self):
        result = run(["generate", "--nonexistent-flag", "value"], expect_success=False)
        assert result.returncode != 0
