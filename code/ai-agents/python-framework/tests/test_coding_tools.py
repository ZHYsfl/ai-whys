import threading
import time

import pytest

from agent_runtime.coding_tools import (
    append_file,
    edit_file,
    execute_command,
    read_file,
    write_file,
)
from agent_runtime.errors import ToolExecutionError


def test_file_tools_edit_read_write_append_and_truncate(tmp_path) -> None:
    path = tmp_path / "notes.txt"
    assert write_file(path, "零\n一\n一\n三") == "successfully wrote file"
    assert append_file(path, "\n四") == "successfully appended to file"
    assert edit_file(path, "一", "二") == "successfully edited file"
    assert path.read_text() == "零\n二\n一\n三\n四"
    assert read_file(path, start_line=1, end_line=3) == "二\n一"
    assert read_file(path, start_line=99, end_line=100) == ""
    assert read_file(path, max_response_len=3) == "零\n二\n[truncated by max_response_len]"

    with pytest.raises(ToolExecutionError, match="old_string not found"):
        edit_file(path, "missing", "value")


def test_execute_command_formats_stdout_stderr_and_failure(tmp_path) -> None:
    result = execute_command("printf out; printf err >&2", str(tmp_path))
    assert result == "stdout:out\nstderr:err"
    assert execute_command("pwd", str(tmp_path)) == f"stdout:{tmp_path}\n"

    with pytest.raises(ToolExecutionError, match="exit status 7") as caught:
        execute_command("printf partial; exit 7", str(tmp_path))
    assert caught.value.result == "stdout:partial"


def test_execute_command_can_be_cancelled() -> None:
    cancel_event = threading.Event()
    timer = threading.Timer(0.1, cancel_event.set)
    timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(ToolExecutionError, match="command cancelled"):
            execute_command("printf before; sleep 5", cancel_event=cancel_event)
    finally:
        timer.cancel()
    assert time.monotonic() - started < 2
