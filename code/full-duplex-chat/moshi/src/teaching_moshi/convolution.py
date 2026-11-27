# Copyright (c) Kyutai, all rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Derived from kyutai-labs/moshi and AudioCraft; see NOTICE and LICENSES/.

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm

from .streaming import State, StreamingModule


def get_extra_padding_for_conv1d(
    x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0
) -> int:
    # x.shape is (batch_size, 1, time_steps=sample_rate*seconds)
    # kernel_size = frame_size, stride = frame_size
    length = x.shape[-1] # time_steps
    frames = (length - kernel_size + padding_total) / stride + 1 # 卷积形状变化公式 ((l-f)/f)+1 = l/f
    ideal_length = (math.ceil(frames) - 1) * stride + kernel_size - padding_total #  第二行做反解：找到一个最短的"理想长度" ≥ 当前 length，使 frames 变成整数——ideal_length = (ceil(frames) − 1)·s + f − P，代回公式恰得 ceil(frames) 帧、一个采样点都不剩。返回值 ideal_length − length 就是差额。  
    return ideal_length - length


def pad_for_conv1d(
    x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0
) -> torch.Tensor:
    # x.shape is (batch_size, 1, time_steps=sample_rate*seconds)
    # kernel_size = frame_size, stride = frame_size
    padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
    return F.pad(x, (0, padding)) # 对最后一维（时间轴）右补 padding 个零——只在句尾加，不动句首


def unpad1d(x: torch.Tensor, paddings: tuple[int, int]) -> torch.Tensor:
    # x.shape is (batch_size, output_dimension, (num_frames+1)*stride)
    left, right = paddings
    end = x.shape[-1] - right if right else x.shape[-1]
    return x[..., left:end]


class NormConv1d(nn.Module):
    def __init__(self, *args, norm: str = "none", **kwargs) -> None:
        super().__init__()
        self.conv = nn.Conv1d(*args, **kwargs)
        if norm == "weight_norm":
            self.conv = weight_norm(self.conv)
        elif norm != "none":
            raise ValueError(f"unknown convolution norm: {norm}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class NormConvTranspose1d(nn.Module):
    def __init__(self, *args, norm: str = "none", **kwargs) -> None:
        super().__init__()
        self.convtr = nn.ConvTranspose1d(*args, **kwargs)
        if norm == "weight_norm":
            self.convtr = weight_norm(self.convtr)
        elif norm != "none":
            raise ValueError(f"unknown convolution norm: {norm}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convtr(x)


@dataclass
class StreamingConv1dState(State):
    previous: torch.Tensor | None = None # (batch_size, channels, kernel_size - stride)
    first: torch.Tensor | None = None # (batch_size,)

    def reset(self) -> None:
        if self.previous is not None:
            self.previous.zero_()
        if self.first is not None:
            self.first.fill_(True)


class StreamingConv1d(StreamingModule):
    """Causal Conv1d. The state stores exactly the K-S samples needed next time."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        causal: bool = True,
        norm: str = "none",
        pad_mode: str = "constant",
    ) -> None:
        super().__init__()
        if not causal or pad_mode not in {"constant", "replicate"}:
            raise ValueError("teaching Mimi uses causal constant/replicate convolution")
        self.pad_mode = pad_mode
        self.conv = NormConv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            norm=norm,
        )

    @property
    def _stride(self) -> int:
        return self.conv.conv.stride[0]

    @property
    def _effective_kernel_size(self) -> int:
        kernel, dilation = self.conv.conv.kernel_size[0], self.conv.conv.dilation[0]
        return (kernel - 1) * dilation + 1

    def _init_streaming_state(self, batch_size: int) -> StreamingConv1dState:
        parameter = self.conv.conv.weight
        previous = torch.zeros( # (batch_size, in_channels, effective_kernel_size - stride)
            batch_size,
            self.conv.conv.in_channels,
            self._effective_kernel_size - self._stride,
            device=parameter.device,
            dtype=parameter.dtype,
        )
        return StreamingConv1dState(
            batch_size,
            parameter.device,
            previous,
            torch.ones(batch_size, device=parameter.device, dtype=torch.bool),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, in_channels, time_steps)
        batch_size, _, time_steps = x.shape
        stride = self._stride
        if time_steps == 0 or time_steps % stride:
            raise ValueError(f"input length {time_steps} must be a positive multiple of {stride}")
        state: StreamingConv1dState | None = self._streaming_state
        if state is None:
            state = self._init_streaming_state(batch_size)
        if state.previous is None or state.first is None:
            raise RuntimeError("incomplete convolution state")
        previous = state.previous
        if previous.shape[-1] and self.pad_mode == "replicate":
            if time_steps < previous.shape[-1]:
                raise ValueError("not enough content for replicate padding")
            previous = torch.where(state.first[:, None, None], x[..., :1], previous) # (batch, in_channels, effective_kernel_size - stride) 如果是第一帧，就用当前帧的第一个采样点填充 previous，否则用上次的 previous
        x = torch.cat((previous, x), dim=-1) # (batch, in_channels, effective_kernel_size - stride + time_steps)
        output = self.conv(x)
        if self._streaming_state is not None and previous.shape[-1]:
            state.previous = x[..., -previous.shape[-1] :] # (batch, in_channels, effective_kernel_size - stride) 保存当前帧的最后 effective_kernel_size - stride 个采样点，供下一帧使用
            state.first = torch.zeros_like(state.first) # 下一帧肯定不是第一帧了
        return output # (batch, out_channels, (time_steps+effective_kernel_size - stride-effective_kernel_size)/stride+1 = time_steps/stride) 


@dataclass
class StreamingConvTr1dState(State):
    partial: torch.Tensor | None = None

    def reset(self) -> None:
        if self.partial is not None:
            self.partial.zero_()


class StreamingConvTranspose1d(StreamingModule):
    """Causal transposed convolution with overlap-add carried between chunks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        bias: bool = True,
        causal: bool = True,
        norm: str = "none",
    ) -> None:
        super().__init__()
        if not causal:
            raise ValueError("teaching Mimi only uses causal transposed convolution")
        self.convtr = NormConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            groups=groups,
            bias=bias,
            norm=norm,
        )

    @property
    def _stride(self) -> int:
        return self.convtr.convtr.stride[0]

    @property
    def _kernel_size(self) -> int:
        return self.convtr.convtr.kernel_size[0]

    def _init_streaming_state(self, batch_size: int) -> StreamingConvTr1dState:
        parameter = self.convtr.convtr.weight
        partial = torch.zeros(
            batch_size,
            self.convtr.convtr.out_channels,
            self._kernel_size - self._stride,
            device=parameter.device,
            dtype=parameter.dtype,
        )
        return StreamingConvTr1dState(batch_size, parameter.device, partial)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, output_dimension, num_frames)
        output = self.convtr(x) # （batch_size, output_dimension, (num_frames+1)*stride）
        trim = self._kernel_size - self._stride
        state: StreamingConvTr1dState | None = self._streaming_state
        if state is None:
            return unpad1d(output, (0, trim))
        if state.partial is None:
            raise RuntimeError("incomplete transposed convolution state")
        if trim:
            output[..., :trim] += state.partial
            partial = output[..., -trim:] # (batch_size, output_dimension, trim)
            bias = self.convtr.convtr.bias # (output_dimension,)
            if bias is not None:
                partial = partial - bias[:, None]
            state.partial = partial
            output = output[..., :-trim]
        return output # (batch_size, output_dimension, (num_frames+1)*stride - trim = num_frames*stride)


class ConvDownsample1d(nn.Module):
    def __init__(
        self,
        stride: int, # default 2
        dimension: int | None = None,
        causal: bool = True,
        learnt: bool = False,
        channel_wise: bool = False,
    ) -> None:
        super().__init__()
        self.learnt = learnt
        groups = dimension if learnt and channel_wise else 1
        channels = dimension if learnt else 1
        if channels is None:
            raise ValueError("dimension is required for learnt resampling")
        self.conv = StreamingConv1d(
            channels,
            channels,
            2 * stride,
            stride,
            groups=groups,
            bias=False,
            causal=causal,
            pad_mode="replicate",
        ) # (t+k-s-k)/s + 1 = (t-s)/s + 1 = t/s
        if not learnt:
            # 等价于avg_pool1d,
            # 用 StreamingConv1d 实现，因为这样可以复用流式 state： K - S = 2S - S = S 每层只需要保存最后 S 个点，就能跨 chunk 继续平均
            weight = self.conv.conv.conv.weight
            weight.requires_grad_(False)
            weight.data.fill_(1 / (2 * stride))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, _ = x.shape
        if not self.learnt:
            x = x.reshape(batch_size * channels, 1, -1)
        output = self.conv(x)
        if not self.learnt:
            output = output.reshape(batch_size, channels, -1)
        return output


class ConvTrUpsample1d(nn.Module):
    def __init__(
        self,
        stride: int, # default 2
        dimension: int | None = None,
        causal: bool = True,
        learnt: bool = False,
        channel_wise: bool = False,
    ) -> None:
        super().__init__()
        self.learnt = learnt
        groups = dimension if learnt and channel_wise else 1
        channels = dimension if learnt else 1
        if channels is None:
            raise ValueError("dimension is required for learnt resampling")
        self.convtr = StreamingConvTranspose1d(
            channels,
            channels,
            2 * stride,
            stride,
            groups=groups,
            bias=False,
            causal=causal,
        )
        if not learnt:
            weight = self.convtr.convtr.convtr.weight
            weight.requires_grad_(False)
            weight.data.fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape is (batch_size, output_dimension, num_frames)
        batch_size, channels, _ = x.shape
        if not self.learnt:
            x = x.reshape(batch_size * channels, 1, -1) # (batch_size * channels, 1, num_frames)
        output = self.convtr(x) # (batch_size * channels, 1, num_frames*stride=2*num_frames)
        if not self.learnt:
            normalization = self.convtr(torch.ones_like(x[:1]))
            output = output / normalization
            output = output.reshape(batch_size, channels, -1)
        return output
