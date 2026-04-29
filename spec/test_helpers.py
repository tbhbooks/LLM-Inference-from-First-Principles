"""Shared test infrastructure for all chapter validations."""
import os
import subprocess
import shlex

import pytest


def run_binary(env_var: str, args: list[str] = None, timeout: int = 30, expect_success: bool = True) -> subprocess.CompletedProcess:
    """Run a chapter binary and return the result.

    The binary is located via the environment variable `env_var`.
    Supports both direct binary paths and compound commands like "python3 script.py".

    Args:
        env_var: Environment variable name pointing to the binary/command
        args: Additional CLI arguments
        timeout: Seconds before killing the process
        expect_success: If True, assert exit code 0
    """
    raw = os.environ.get(env_var)
    if not raw:
        pytest.skip(
            f"Set {env_var} to run these tests.\n"
            f"  For Rust:   {env_var}=./target/debug/rvllm pytest ...\n"
            f"  For Python: {env_var}='python3 my_impl.py' pytest ...\n"
            f"  For Go:     {env_var}=./rvllm pytest ..."
        )

    cmd = shlex.split(raw)
    if args:
        cmd.extend(args)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    if expect_success:
        assert result.returncode == 0, (
            f"Expected exit code 0, got {result.returncode}.\n"
            f"stdout: {result.stdout[:500]}\n"
            f"stderr: {result.stderr[:500]}"
        )
    return result
