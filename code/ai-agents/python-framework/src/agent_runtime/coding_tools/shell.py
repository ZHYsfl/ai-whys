import os
import signal
import subprocess
from threading import Event

from agent_runtime.errors import ToolExecutionError


def execute_command(
    command: str,
    workdir: str = "",
    cancel_event: Event | None = None,
) -> str:
    if cancel_event is not None and cancel_event.is_set():
        raise ToolExecutionError("command cancelled")

    try:
        process = subprocess.Popen(
            ["sh", "-c", command],
            cwd=workdir or None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
    except OSError as exc:
        raise ToolExecutionError(f"command failed: {exc}") from exc

    while True:
        try:
            stdout, stderr = process.communicate(timeout=0.1)
            break
        except subprocess.TimeoutExpired:
            if cancel_event is not None and cancel_event.is_set():
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                stdout, stderr = process.communicate()
                result = _format_output(stdout, stderr)
                raise ToolExecutionError("command cancelled", result=result)

    result = _format_output(stdout, stderr)
    if process.returncode != 0:
        raise ToolExecutionError(
            f"command failed: exit status {process.returncode}",
            result=result,
        )
    return result


def _format_output(stdout: str, stderr: str) -> str:
    result = ""
    if stdout:
        result = "stdout:" + stdout
    if stderr:
        if result:
            result += "\n"
        result += "stderr:" + stderr
    return result
