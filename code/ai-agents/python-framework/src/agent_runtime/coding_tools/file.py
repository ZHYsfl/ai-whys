from os import PathLike
from pathlib import Path

from agent_runtime.errors import ToolExecutionError


DEFAULT_MAX_RESPONSE_LEN = 10_000


def edit_file(path: str | PathLike[str], old_string: str, new_string: str) -> str:
    file_path = Path(path)
    try:
        content = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        message = f"failed to read file: {exc}"
        raise ToolExecutionError(message) from exc

    replaced = content.replace(old_string, new_string, 1)
    if replaced == content:
        raise ToolExecutionError("old_string not found in file")

    try:
        file_path.write_text(replaced, encoding="utf-8")
    except OSError as exc:
        message = f"failed to write file: {exc}"
        raise ToolExecutionError(message) from exc
    return "successfully edited file"


def read_file(
    path: str | PathLike[str],
    start_line: int = 0,
    end_line: int = 0,
    max_response_len: int = 0,
) -> str:
    try:
        selected = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise ToolExecutionError(f"failed to read file: {exc}") from exc

    if start_line > 0 or end_line > 0:
        lines = selected.split("\n")
        start_line = max(start_line, 0)
        if end_line <= 0 or end_line > len(lines):
            end_line = len(lines)
        if start_line > end_line:
            start_line = end_line
        selected = "\n".join(lines[start_line:end_line])

    if max_response_len <= 0:
        max_response_len = DEFAULT_MAX_RESPONSE_LEN
    if len(selected) > max_response_len:
        selected = selected[:max_response_len] + "\n[truncated by max_response_len]"
    return selected


def write_file(path: str | PathLike[str], content: str) -> str:
    try:
        Path(path).write_text(content, encoding="utf-8")
    except OSError as exc:
        message = f"failed to write file: {exc}"
        raise ToolExecutionError(message) from exc
    return "successfully wrote file"


def append_file(path: str | PathLike[str], content: str) -> str:
    try:
        with Path(path).open("a", encoding="utf-8") as file:
            file.write(content)
    except OSError as exc:
        message = f"failed to append: {exc}"
        raise ToolExecutionError(message) from exc
    return "successfully appended to file"
