import logging
import queue
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable

from .estimate import estimate_messages


logger = logging.getLogger(__name__)

COMPRESS_TRIGGER_RATIO = 0.8
SUMMARY_FOLD_AT = 5
COMPRESS_KEEP_RECENT_MESSAGES = 40
SUMMARIZE_ATTEMPTS = 3
SUMMARIZE_ATTEMPT_TIMEOUT = 60.0
SUMMARIZE_RETRY_GAP = 0.5
SUMMARY_OPEN_TAG = "<conversation_summary>"
SUMMARY_CLOSE_TAG = "</conversation_summary>"

Summarizer = Callable[[list[dict[str, Any]]], str]


class MemoryMode(IntEnum):
    NONE = 0
    SLIDE_WINDOW = 1
    COMPRESS = 2


@dataclass
class MemoryConfig:
    mode: MemoryMode = MemoryMode.NONE
    max: int = 0
    keep_recent: int = 0
    summarizer: Summarizer | None = None


class MemoryManager(ABC):
    @abstractmethod
    def prepare(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        pass


class DefaultMemory(MemoryManager):
    def prepare(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return messages


class SlideWindowMemory(MemoryManager):
    def __init__(self, max_messages: int) -> None:
        self.max_messages = max_messages

    def prepare(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.max_messages <= 0:
            return messages
        system, non_system = split_system(messages)
        if len(non_system) > self.max_messages:
            non_system = non_system[-self.max_messages :]
        return join_system(system, drop_leading_orphan_tools(non_system))


class CompressMemory(MemoryManager):
    def __init__(
        self,
        max_input_tokens: int,
        keep_recent: int = 0,
        summarizer: Summarizer | None = None,
    ) -> None:
        self.max_input_tokens = max_input_tokens
        self.keep_recent = keep_recent or COMPRESS_KEEP_RECENT_MESSAGES
        if self.keep_recent <= 0:
            self.keep_recent = COMPRESS_KEEP_RECENT_MESSAGES
        self.summarizer = summarizer
        self._last_prompt_tokens = 0
        self._usage_lock = threading.Lock()

    @property
    def last_prompt_tokens(self) -> int:
        with self._usage_lock:
            return self._last_prompt_tokens

    def record_usage(self, prompt_tokens: int) -> None:
        if prompt_tokens > 0:
            with self._usage_lock:
                self._last_prompt_tokens = prompt_tokens

    def prepare(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.max_input_tokens <= 0 or not self._over_threshold(messages):
            return messages
        return self._compress(messages)

    def _over_threshold(self, messages: list[dict[str, Any]]) -> bool:
        threshold = int(self.max_input_tokens * COMPRESS_TRIGGER_RATIO)
        return (
            self.last_prompt_tokens >= threshold
            or estimate_messages(messages) >= threshold
        )

    def _compress(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        system, non_system = split_system(messages)
        keep = self.keep_recent
        if keep <= 0:
            keep = COMPRESS_KEEP_RECENT_MESSAGES

        summary_count = count_leading_summaries(non_system)
        region_start = summary_count
        if summary_count >= SUMMARY_FOLD_AT:
            region_start = 0

        cut = len(non_system) - keep
        cut = tool_group_safe_cut(non_system, region_start, cut)
        region = non_system[region_start:cut]
        if not region:
            return messages

        tail = drop_leading_orphan_tools(non_system[cut:])
        with self._usage_lock:
            self._last_prompt_tokens = 0

        summary = self._summarize_region(system, region)
        if summary:
            output: list[dict[str, Any]] = []
            if system is not None:
                output.append(system)
            output.extend(non_system[:region_start])
            output.append(summary_message(summary))
            output.extend(tail)
            return output

        output = []
        if system is not None:
            output.append(system)
        output.extend(non_system[:summary_count])
        output.extend(tail)
        return output

    def _summarize_region(
        self,
        system: dict[str, Any] | None,
        region: list[dict[str, Any]],
    ) -> str | None:
        if (
            self.summarizer is None
            or system is None
            or not isinstance(system.get("content"), str)
        ):
            return None

        request = [system, *region]
        error: Exception | None = None
        for attempt in range(1, SUMMARIZE_ATTEMPTS + 1):
            if attempt > 1:
                time.sleep(SUMMARIZE_RETRY_GAP)
            try:
                got = _call_with_timeout(
                    self.summarizer,
                    request,
                    SUMMARIZE_ATTEMPT_TIMEOUT,
                )
                if got:
                    return got
            except Exception as exc:
                error = exc

        logger.error(
            "memory: summarize failed %d times (%s), falling back to slide window",
            SUMMARIZE_ATTEMPTS,
            error,
        )
        return None


def _call_with_timeout(
    summarizer: Summarizer,
    messages: list[dict[str, Any]],
    timeout: float,
) -> str:
    outcome: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def run() -> None:
        try:
            outcome.put((True, summarizer(messages)))
        except Exception as exc:
            outcome.put((False, exc))

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        succeeded, value = outcome.get(timeout=timeout)
    except queue.Empty as exc:
        raise TimeoutError("summarizer timed out") from exc
    if not succeeded:
        raise value
    return value


def new_memory(config: MemoryConfig) -> MemoryManager:
    if config.mode == MemoryMode.SLIDE_WINDOW:
        return SlideWindowMemory(config.max)
    if config.mode == MemoryMode.COMPRESS:
        return CompressMemory(
            max_input_tokens=config.max,
            keep_recent=config.keep_recent,
            summarizer=config.summarizer,
        )
    return DefaultMemory()


def split_system(
    messages: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    system = None
    non_system = []
    for message in messages:
        if message.get("role") == "system":
            system = message
        else:
            non_system.append(message)
    return system, non_system


def join_system(
    system: dict[str, Any] | None,
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = [system] if system is not None else []
    output.extend(messages)
    return output


def drop_leading_orphan_tools(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    index = 0
    while index < len(messages) and messages[index].get("role") == "tool":
        index += 1
    return messages[index:]


def tool_group_safe_cut(
    messages: list[dict[str, Any]],
    low: int,
    cut: int,
) -> int:
    if cut <= low:
        return low
    index = cut - 1
    while index > low and messages[index].get("role") == "tool":
        index -= 1
    message = messages[index]
    if message.get("role") == "assistant" and message.get("tool_calls"):
        return index
    if index < cut - 1:
        return index + 1
    return cut


def summary_message(text: str) -> dict[str, Any]:
    return {
        "role": "user",
        "content": f"{SUMMARY_OPEN_TAG}\n{text}\n{SUMMARY_CLOSE_TAG}",
    }


def is_summary_message(message: dict[str, Any]) -> bool:
    return (
        message.get("role") == "user"
        and isinstance(message.get("content"), str)
        and message["content"].startswith(SUMMARY_OPEN_TAG)
    )


def count_leading_summaries(messages: list[dict[str, Any]]) -> int:
    index = 0
    while index < len(messages) and is_summary_message(messages[index]):
        index += 1
    return index
