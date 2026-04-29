"""
Chapter 2 validation tests.

Runs the ch02 binary and checks that all required content is present.
Language-agnostic -- works with any implementation.

Usage:
    RVLLM_CH02_BIN=/path/to/your/binary pytest spec/ch02/validation/test_ch02.py -v
    RVLLM_CH02_BIN="python3 ch02.py" pytest spec/ch02/validation/test_ch02.py -v

See spec/runners/README.md for more examples (Rust, Python, Go, etc.).
"""

import pytest

from conftest import run_binary


@pytest.fixture(scope="module")
def ch02_output():
    """Run the ch02 binary and capture its stdout."""
    result = run_binary("RVLLM_CH02_BIN", timeout=10)
    return result.stdout


# ---------------------------------------------------------------------------
# Header and footer
# ---------------------------------------------------------------------------


class TestHeaderFooter:
    def test_title_present(self, ch02_output):
        assert "Chapter 2: vLLM Architecture Overview" in ch02_output

    def test_header_separator(self, ch02_output):
        assert "=" * 40 in ch02_output

    def test_whats_next_footer(self, ch02_output):
        assert "What's Next?" in ch02_output

    def test_paged_attention_hook(self, ch02_output):
        assert "PagedAttention" in ch02_output

    def test_chapter_3_teaser(self, ch02_output):
        assert "Chapter 3" in ch02_output

    def test_chapter_6_teaser(self, ch02_output):
        assert "Chapter 6" in ch02_output


# ---------------------------------------------------------------------------
# Section 1: vLLM six-layer architecture
# ---------------------------------------------------------------------------


class TestVllmArchitecture:
    def test_section_header(self, ch02_output):
        assert "vLLM's Six-Layer Architecture" in ch02_output

    def test_fastapi(self, ch02_output):
        assert "FastAPI" in ch02_output

    def test_async_llm(self, ch02_output):
        assert "AsyncLLM" in ch02_output

    def test_engine_core_client(self, ch02_output):
        assert "EngineCoreClient" in ch02_output

    def test_engine_core(self, ch02_output):
        assert "EngineCore" in ch02_output

    def test_multiproc_executor(self, ch02_output):
        assert "MultiprocExecutor" in ch02_output

    def test_gpu_worker(self, ch02_output):
        assert "GPUWorker" in ch02_output

    def test_zmq_mention(self, ch02_output):
        assert "ZMQ" in ch02_output


# ---------------------------------------------------------------------------
# Section 2: rvllm seven-module architecture
# ---------------------------------------------------------------------------


class TestRvllmArchitecture:
    def test_section_header(self, ch02_output):
        assert "rvllm's Seven-Module Architecture" in ch02_output

    def test_api_module(self, ch02_output):
        assert "api/" in ch02_output

    def test_engine_module(self, ch02_output):
        assert "engine/" in ch02_output

    def test_scheduler_module(self, ch02_output):
        assert "scheduler/" in ch02_output

    def test_memory_module(self, ch02_output):
        assert "memory/" in ch02_output

    def test_model_module(self, ch02_output):
        assert "model/" in ch02_output

    def test_sampling_module(self, ch02_output):
        assert "sampling/" in ch02_output

    def test_tokenizer_module(self, ch02_output):
        assert "tokenizer/" in ch02_output

    def test_seven_modules_statement(self, ch02_output):
        assert "Seven modules" in ch02_output


# ---------------------------------------------------------------------------
# Section 3: Mapping table
# ---------------------------------------------------------------------------


class TestComparisonTable:
    def test_section_header(self, ch02_output):
        assert "Mapping:" in ch02_output

    def test_removed_entries(self, ch02_output):
        assert "(removed)" in ch02_output

    def test_chapter_references(self, ch02_output):
        for ch in [
            "Ch 3: Transformer Primer",
            "Ch 4: Tokenization",
            "Ch 5: The Scheduler",
            "Ch 6: Memory",
            "Ch 9: Sampling",
            "Ch 10: API",
            "Ch 11: Putting It Together",
        ]:
            assert ch in ch02_output, f"Missing chapter reference: {ch}"

    def test_two_layers_vanish(self, ch02_output):
        assert "Two vLLM layers vanish" in ch02_output


# ---------------------------------------------------------------------------
# Section 4: Request lifecycle (9 steps)
# ---------------------------------------------------------------------------


class TestRequestLifecycle:
    def test_section_header(self, ch02_output):
        assert "Request Lifecycle" in ch02_output

    def test_example_prompt(self, ch02_output):
        assert "Explain PagedAttention" in ch02_output

    @pytest.mark.parametrize(
        "step_name",
        ["ARRIVE", "TOKENIZE", "ENQUEUE", "SCHEDULE", "ALLOCATE", "EXECUTE", "SAMPLE", "UPDATE", "RESPOND"],
    )
    def test_step_present(self, ch02_output, step_name):
        assert f"Step" in ch02_output
        assert step_name in ch02_output

    def test_step_1_arrive(self, ch02_output):
        assert "POST /v1/completions" in ch02_output

    def test_step_2_tokenize(self, ch02_output):
        assert "50872" in ch02_output
        assert "7873" in ch02_output
        assert "58662" in ch02_output

    def test_step_3_enqueue(self, ch02_output):
        assert "WAITING" in ch02_output

    def test_step_4_schedule(self, ch02_output):
        assert "RUNNING" in ch02_output

    def test_step_5_allocate(self, ch02_output):
        assert "block_size=16" in ch02_output
        assert "254" in ch02_output

    def test_step_6_execute(self, ch02_output):
        assert "forward pass" in ch02_output

    def test_step_7_sample(self, ch02_output):
        assert "argmax" in ch02_output
        assert "1334" in ch02_output

    def test_step_8_update(self, ch02_output):
        assert "EOS" in ch02_output
        assert "max_tokens" in ch02_output

    def test_step_9_respond(self, ch02_output):
        assert "streams" in ch02_output or "stream" in ch02_output

    def test_loop_box(self, ch02_output):
        assert "Steps 4-8 repeat" in ch02_output

    def test_loop_termination_conditions(self, ch02_output):
        assert "end-of-sequence" in ch02_output or "EOS" in ch02_output
        assert "max_tokens" in ch02_output
        assert "disconnects" in ch02_output


# ---------------------------------------------------------------------------
# Section 5: Key simplifications
# ---------------------------------------------------------------------------


class TestKeySimplifications:
    def test_section_header(self, ch02_output):
        assert "Key Simplifications" in ch02_output

    def test_dropped_zmq(self, ch02_output):
        assert "ZMQ IPC" in ch02_output

    def test_dropped_multi_gpu(self, ch02_output):
        assert "Multi-GPU" in ch02_output

    def test_dropped_cuda_graphs(self, ch02_output):
        assert "CUDA graphs" in ch02_output

    def test_dropped_speculative(self, ch02_output):
        assert "Speculative decoding" in ch02_output

    def test_dropped_lora(self, ch02_output):
        assert "LoRA" in ch02_output

    def test_goal_statement(self, ch02_output):
        assert "understand every line" in ch02_output


# ---------------------------------------------------------------------------
# Structural checks
# ---------------------------------------------------------------------------


class TestStructure:
    def test_section_separators(self, ch02_output):
        """At least 5 section separators (sections 1-5)."""
        separator_count = ch02_output.count("\u2500" * 20)
        assert separator_count >= 5, f"Expected at least 5 section separators, found {separator_count}"

    def test_header_footer_separators(self, ch02_output):
        """Header and footer use '=' separators."""
        eq_count = ch02_output.count("=" * 40)
        assert eq_count >= 3, f"Expected at least 3 '====' separators (header, footer open, footer close), found {eq_count}"

    def test_no_ansi_codes(self, ch02_output):
        """Output should not contain ANSI escape codes."""
        assert "\033[" not in ch02_output
        assert "\x1b[" not in ch02_output

    def test_exit_code_zero(self):
        """Binary should exit with code 0 (covered by fixture, but explicit)."""
        result = run_binary("RVLLM_CH02_BIN", timeout=10)
        assert result.returncode == 0
