# Copyright (c) Kyutai, all rights reserved.
# Derived from kyutai-labs/moshi; see NOTICE and LICENSES/MIT-moshi.txt.

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import torch
from torch import nn


@dataclass
class State:
    batch_size: int
    device: torch.device

    def reset(self) -> None:
        pass


class StreamingModule(nn.Module):
    """A module whose forward pass can carry explicit state across chunks."""

    def __init__(self) -> None:
        super().__init__()
        self._streaming_state: Any = None # 槽位：None 代表非流式moshi；非 None 代表流式中
        self._streaming_detached = False # "别带我玩"标记：True 代表流式状态不传递给子模块；False 代表流式状态传递给子模块

    @property
    def is_streaming(self) -> bool: # 槽位即模式
        return self._streaming_state is not None

    def set_streaming_detached(self, detached: bool) -> None:
        self._streaming_detached = detached

    def _init_streaming_state(self, batch_size: int) -> State: # 模板方法
        raise NotImplementedError

    def _modules_for_streaming(self, root: bool = True) -> Iterator["StreamingModule"]: # 树形收编
        if not root and self._streaming_detached: # 被标记的模块：整个子树跳过
            return
        yield self # 先收自己
        for child in self.children():
            yield from _descendant_streaming_modules(child) # 再递归孩子

    @contextmanager
    def streaming(self, batch_size: int) -> Iterator["StreamingModule"]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        modules = list(self._modules_for_streaming())
        if any(module.is_streaming for module in modules):
            raise RuntimeError("streaming state is already active")
        started = []
        try:
            for module in modules:
                module._streaming_state = module._init_streaming_state(batch_size) # 基类只定义什么时候造 state，造什么交给子类实现（流程归父类，内容归子类）卷积子类造"接力棒"+"首棒标记"，transformer 子类造 KV Cache
                started.append(module)
            yield self
        finally:
            for module in reversed(started):
                module._streaming_state = None

    def reset_streaming(self) -> None:
        if not self.is_streaming:
            raise RuntimeError("streaming is not active")
        for module in self._modules_for_streaming():
            module._streaming_state.reset()


def _descendant_streaming_modules(module: nn.Module) -> Iterator[StreamingModule]:
    if isinstance(module, StreamingModule):
        yield from module._modules_for_streaming(root=False)
    else:
        for child in module.children():
            yield from _descendant_streaming_modules(child)


class StreamingContainer(StreamingModule):
    def _init_streaming_state(self, batch_size: int) -> State:
        device = next(parameter.device for parameter in self.parameters())
        return State(batch_size, device)
