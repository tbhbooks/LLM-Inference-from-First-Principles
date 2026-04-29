"""
Chapter 15 validation tests: Building the API Server.

Runs the reader's implementation as a subprocess and validates the output
against the interface spec (see ../interface-spec.md and ../expected-output.txt).
Language-agnostic -- works with any implementation.

Usage:
    RVLLM_CH15_BIN="cargo run --example ch15_api_server" pytest test_ch15.py -v
    RVLLM_CH15_BIN="uv run python -m rvllm ch15_demo" pytest test_ch15.py -v
"""

import re

import pytest

from conftest import run_binary

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TIMEOUT_SECONDS = 30  # No real server or model — mock simulation only


@pytest.fixture(scope="module")
def ch15_output() -> str:
    """Run the Chapter 15 binary once and cache its stdout for all tests."""
    result = run_binary("RVLLM_CH15_BIN", timeout=TIMEOUT_SECONDS)
    return result.stdout


def _lower(output: str) -> str:
    return output.lower()


# ---------------------------------------------------------------------------
# Tests: All 5 parts present
# ---------------------------------------------------------------------------


class TestStructure:
    """Verify all 5 required parts are present in the output."""

    def test_part1_server_startup(self, ch15_output: str):
        """Output must contain Part 1: Server Startup."""
        lower = _lower(ch15_output)
        assert "part 1" in lower or "server startup" in lower, (
            "Output must contain 'PART 1' or 'Server Startup'.\n"
            f"Got:\n{ch15_output[:500]}"
        )

    def test_part2_health_check(self, ch15_output: str):
        """Output must contain Part 2: Health Check."""
        lower = _lower(ch15_output)
        assert "part 2" in lower or "health check" in lower, (
            "Output must contain 'PART 2' or 'Health Check'.\n"
            f"Got:\n{ch15_output[:500]}"
        )

    def test_part3_completion_request(self, ch15_output: str):
        """Output must contain Part 3: Completion Request."""
        lower = _lower(ch15_output)
        assert "part 3" in lower or "completion request" in lower, (
            "Output must contain 'PART 3' or 'Completion Request'.\n"
            f"Got:\n{ch15_output[:500]}"
        )

    def test_part4_streaming_response(self, ch15_output: str):
        """Output must contain Part 4: Streaming Response."""
        lower = _lower(ch15_output)
        assert "part 4" in lower or "streaming response" in lower or "streaming" in lower, (
            "Output must contain 'PART 4' or 'Streaming Response'.\n"
            f"Got:\n{ch15_output[:500]}"
        )

    def test_part5_error_handling(self, ch15_output: str):
        """Output must contain Part 5: Error Handling."""
        lower = _lower(ch15_output)
        assert "part 5" in lower or "error handling" in lower, (
            "Output must contain 'PART 5' or 'Error Handling'.\n"
            f"Got:\n{ch15_output[:500]}"
        )


# ---------------------------------------------------------------------------
# Tests: Server Startup
# ---------------------------------------------------------------------------


class TestServerStartup:
    """Verify server configuration is displayed."""

    def test_shows_listen_address(self, ch15_output: str):
        """Startup must show the listen address with host and port."""
        lower = _lower(ch15_output)
        has_address = (
            "127.0.0.1" in lower
            or "0.0.0.0" in lower
            or "localhost" in lower
            or ":8080" in lower
            or ":8000" in lower
        )
        assert has_address, (
            "Server startup must show a listen address (e.g., 127.0.0.1:8080).\n"
            f"Got:\n{ch15_output[:500]}"
        )

    def test_shows_endpoints(self, ch15_output: str):
        """Startup must list the API endpoints."""
        lower = _lower(ch15_output)
        assert "/v1/completions" in lower, (
            "Server startup must list the /v1/completions endpoint.\n"
            f"Got:\n{ch15_output[:500]}"
        )


# ---------------------------------------------------------------------------
# Tests: Health Check
# ---------------------------------------------------------------------------


class TestHealthCheck:
    """Verify health check endpoint is demonstrated."""

    def test_health_endpoint_shown(self, ch15_output: str):
        """Output must show the /health endpoint."""
        lower = _lower(ch15_output)
        assert "/health" in lower or "health" in lower, (
            "Output must show the /health endpoint.\n"
            f"Got:\n{ch15_output[:500]}"
        )

    def test_health_ok_response(self, ch15_output: str):
        """Health check must show an OK status response."""
        lower = _lower(ch15_output)
        has_ok = '"status"' in lower or '"ok"' in lower or "200" in lower
        assert has_ok, (
            "Health check must show an OK status (e.g., {\"status\": \"ok\"}).\n"
            f"Got:\n{ch15_output[:1000]}"
        )


# ---------------------------------------------------------------------------
# Tests: Completion Request
# ---------------------------------------------------------------------------


class TestCompletionRequest:
    """Verify non-streaming completion is demonstrated."""

    def test_shows_openai_response_format(self, ch15_output: str):
        """Response must include OpenAI format fields."""
        lower = _lower(ch15_output)
        has_id = '"id"' in lower or "cmpl-" in lower
        has_object = "text_completion" in lower
        assert has_id and has_object, (
            "Completion response must include OpenAI format fields "
            "(id, object='text_completion').\n"
            f"Got:\n{ch15_output[:1000]}"
        )

    def test_shows_choices(self, ch15_output: str):
        """Response must include choices array."""
        lower = _lower(ch15_output)
        assert '"choices"' in lower or "choices" in lower, (
            "Completion response must include 'choices'.\n"
            f"Got:\n{ch15_output[:1000]}"
        )

    def test_shows_usage(self, ch15_output: str):
        """Response must include usage statistics."""
        lower = _lower(ch15_output)
        assert "usage" in lower or "prompt_tokens" in lower or "total_tokens" in lower, (
            "Completion response must include usage statistics.\n"
            f"Got:\n{ch15_output[:1000]}"
        )

    def test_shows_finish_reason(self, ch15_output: str):
        """Response must include finish_reason."""
        lower = _lower(ch15_output)
        assert "finish_reason" in lower, (
            "Completion response must include 'finish_reason'.\n"
            f"Got:\n{ch15_output[:1000]}"
        )


# ---------------------------------------------------------------------------
# Tests: Streaming Response
# ---------------------------------------------------------------------------


class TestStreamingResponse:
    """Verify SSE streaming is demonstrated."""

    def test_sse_data_prefix(self, ch15_output: str):
        """Streaming output must show SSE 'data: ' prefix."""
        assert "data: " in ch15_output, (
            "Streaming section must show SSE events with 'data: ' prefix.\n"
            f"Got:\n{ch15_output[:1000]}"
        )

    def test_sse_done_sentinel(self, ch15_output: str):
        """Streaming output must show the [DONE] sentinel."""
        assert "[DONE]" in ch15_output, (
            "Streaming section must show the '[DONE]' sentinel.\n"
            f"Got:\n{ch15_output[:1000]}"
        )

    def test_sse_shows_individual_tokens(self, ch15_output: str):
        """Streaming section must show multiple SSE events (multiple tokens)."""
        data_lines = [line for line in ch15_output.splitlines() if line.strip().startswith("data: {")]
        assert len(data_lines) >= 2, (
            f"Expected at least 2 SSE data events with JSON payloads, "
            f"found {len(data_lines)}.\n"
            f"Got:\n{ch15_output[:1000]}"
        )


# ---------------------------------------------------------------------------
# Tests: Error Handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Verify error responses match OpenAI format."""

    def test_error_response_shown(self, ch15_output: str):
        """Output must show an error response."""
        lower = _lower(ch15_output)
        assert '"error"' in lower, (
            "Error handling section must show an error response with '\"error\"' field.\n"
            f"Got:\n{ch15_output[-1000:]}"
        )

    def test_error_has_message(self, ch15_output: str):
        """Error response must include a message field."""
        lower = _lower(ch15_output)
        assert '"message"' in lower, (
            "Error response must include '\"message\"' field.\n"
            f"Got:\n{ch15_output[-1000:]}"
        )

    def test_error_has_type(self, ch15_output: str):
        """Error response must include a type field."""
        lower = _lower(ch15_output)
        assert '"type"' in lower or "invalid_request_error" in lower, (
            "Error response must include '\"type\"' field or 'invalid_request_error'.\n"
            f"Got:\n{ch15_output[-1000:]}"
        )

    def test_error_status_code(self, ch15_output: str):
        """Error section must mention HTTP status code."""
        lower = _lower(ch15_output)
        has_status = "400" in lower or "bad request" in lower
        assert has_status, (
            "Error section must mention HTTP 400 or 'Bad Request'.\n"
            f"Got:\n{ch15_output[-1000:]}"
        )


# ---------------------------------------------------------------------------
# Tests: Completion
# ---------------------------------------------------------------------------


class TestCompletion:
    """Verify the chapter completion marker."""

    def test_chapter_15_complete(self, ch15_output: str):
        """Output must contain the completion marker."""
        assert "chapter 15 complete" in _lower(ch15_output), (
            "Output must contain 'Chapter 15 complete'.\n"
            f"Got:\n{ch15_output[-500:]}"
        )
