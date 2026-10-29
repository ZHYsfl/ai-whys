from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ReconstructionOutput:
    total: torch.Tensor
    waveform: torch.Tensor
    stft: torch.Tensor


class MultiScaleSTFTLoss(nn.Module):
    def __init__(self, n_ffts: tuple[int, ...] = (128, 256, 512), epsilon: float = 1e-5) -> None:
        super().__init__()
        self.n_ffts = n_ffts
        self.epsilon = epsilon

    def magnitude(self, waveform: torch.Tensor, n_fft: int) -> torch.Tensor:
        '''
        幅度谱
        '''
        # waveform.shape is [B, T], where B is batch size and T is number of samples
        # the value of waveform is in the range [-1, 1],stands for amplitude of audio signal
        if waveform.ndim != 2:
            raise ValueError(f"expected [B, T], got {tuple(waveform.shape)}")
        if waveform.shape[-1] < n_fft: # too short to compute stft, pad with zeros to the right
            waveform = F.pad(waveform, (0, n_fft - waveform.shape[-1]))
        # see stft_tutorial.html to have a better understanding of the below code
        window = torch.hann_window(n_fft, device=waveform.device, dtype=waveform.dtype) # 汉宁窗。N = n_fft, window.shape = (n_fft,)
        spectrum = torch.stft( # 频谱
            waveform,
            n_fft=n_fft,
            hop_length=n_fft // 4,
            win_length=n_fft,
            window=window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        # the shape of spectrum is (B, num_k=n_fft÷2 + 1, num_window=T÷(n_fft÷4) + 1) 每格存复数 a+ bi, a 为 cos 探针得分, b 为 sin 探针得分
        return spectrum.abs().clamp_min(self.epsilon) # 不改变shape, abs()是算sqrt(a^2 + b^2) 也就是幅度谱，频谱包含幅度谱，幅度谱丢失了相位。频谱图 = 幅度谱的可视化。

    def forward(self, target: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
        if target.shape != reconstruction.shape or target.ndim != 3 or target.shape[1] != 1:
            raise ValueError("target and reconstruction must have identical shape [B, 1, T]")
        target = target.squeeze(1) # (batch_size,time_steps) 也就是 [B, T]
        reconstruction = reconstruction.squeeze(1) # (batch_size,time_steps) 也就是 [B, T]
        losses = []
        for n_fft in self.n_ffts:
            target_magnitude = self.magnitude(target, n_fft) # (batch_size, num_k, num_window)
            reconstruction_magnitude = self.magnitude(reconstruction, n_fft) # (batch_size, num_k, num_window)
            spectral_convergence = torch.norm(
                target_magnitude - reconstruction_magnitude, p="fro"
            ) / torch.norm(target_magnitude, p="fro").clamp_min(self.epsilon)
            log_magnitude_l1 = F.l1_loss(
                target_magnitude.log(), reconstruction_magnitude.log()
            )
            losses.append(spectral_convergence + log_magnitude_l1)
        return torch.stack(losses).mean() # return 0D tensor


class ReconstructionLoss(nn.Module):
    def __init__(self, waveform_weight: float = 1.0, stft_weight: float = 1.0) -> None:
        super().__init__()
        self.waveform_weight = waveform_weight
        self.stft_weight = stft_weight
        self.stft_loss = MultiScaleSTFTLoss()

    def forward(self, target: torch.Tensor, reconstruction: torch.Tensor) -> ReconstructionOutput:
        waveform = F.l1_loss(reconstruction, target) # 时域损失
        stft = self.stft_loss(target, reconstruction) # 频域损失
        total = self.waveform_weight * waveform + self.stft_weight * stft # 加权求和
        return ReconstructionOutput(total, waveform, stft)
