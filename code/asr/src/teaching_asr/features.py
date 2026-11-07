import torch
from torch import nn
from torch.nn import functional as F

from .config import ASRConfig


def hz_to_mel(frequency: torch.Tensor) -> torch.Tensor:
    '''
    这是把物理频率 Hz 转成 Mel 刻度：
                             f                                          
    mel (f) = 2595 log₁₀(1+ ─── )                                       
                            700   
    为什么要转换：
    因为人耳对频率的感知不是线性的，例如：                       
    • 100 Hz 和 200 Hz 的差别比较容易听出来；                           
    • 7100 Hz 和 7200 Hz 虽然同样差 100 Hz，却没那么容易区分。
    Mel 刻度就是对这个现象的近似：低频分得更细，高频不用分得那么细。
    '''
    return 2595.0 * torch.log10(1.0 + frequency / 700.0)


def mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    '''
    Mel 刻度转回物理频率 Hz：
    f = 700(10^(mel /2595)-1)
    '''
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(config: ASRConfig) -> torch.Tensor:
    '''
    创建 Mel 滤波器组。
    这里把 201 → 80，主要有三个目的：                                   
                                                       
    1. 减少冗余                                                         
    相邻 STFT 频率格往往相关，把一片相邻频率合起来，可以得到更平滑、稳定的特征。                                                     
                                                                       
    2. 加入听觉先验                                                     
    Mel 滤波器让低频分得更细、高频分得更粗，更接近人耳对语音频率的感知方式。                                                         
                                                                       
    3. 降低后续计算量                                                   
    每个时间帧从 201 个数变成 80 个数，后续模型处理的信息量更少。    
                                                                       
    所以它不是单纯为了“压缩文件”，而是在做：有损地删掉一部分细粒度频率信息，换取更适合语音识别的表示。                                                              
    '''
    frequencies = torch.linspace(0, config.sample_rate / 2, config.n_fft // 2 + 1) # 默认[0,40,...,8000] freqs.shape = (201,)
    f_max = config.sample_rate / 2 if config.f_max is None else config.f_max # 默认8000
    mel_points = torch.linspace(
        hz_to_mel(torch.tensor(float(config.f_min))),
        hz_to_mel(torch.tensor(float(f_max))),
        config.n_mels + 2, # 80个滤波器需要82个点，均匀取点
    ) # mel_points.shape = (82,)
    hz_points = mel_to_hz(mel_points) # 转为hz空间后，低频点密集，高频点稀疏 hz_points.shape = (82,)
    lower, center, upper = hz_points[:-2, None], hz_points[1:-1, None], hz_points[2:, None] # 取出每个三角形的三个顶点，每个形状都是(80, 1)
    rising = (frequencies - lower) / (center - lower) # 广播时从最后一个维度开始对齐，(201,) - (80, 1) -> (80, 201) 再÷ (80, 1) -> (80, 201)
    falling = (upper - frequencies) / (upper - center)
    return torch.minimum(rising, falling).clamp_min(0) # (80, 201)
    # 每行：一个 Mel 滤波器在 201 个频率格上的权重                    
    # 每列：某个频率格在全部 80 个滤波器中的权重                      
    # 低频滤波器较窄且密集，高频滤波器较宽且稀疏
    # 或者这么理解同理：                                                                   
    # 行 m：第 m 个滤波器                                               
    # 列 k：第 k 个 STFT 频率格                                         
    # 格子 [m,k]：第 m 个滤波器在第 k 个频率格上的权重

def lengths_to_mask(lengths: torch.Tensor, max_length: int) -> torch.Tensor:
    # lengths.shape = (batch_size,)
    return torch.arange(max_length, device=lengths.device)[None, :] < lengths[:, None] # (batch_size, max_length)


class LogMelSpectrogram(nn.Module):
    def __init__(self, config: ASRConfig) -> None:
        super().__init__()
        self.config = config
        # 把一个不需要训练的 Tensor 注册为 nn.Module 的“附属数据”，让 PyTorch 帮它管理设备、状态等行为，他们参与计算，但不需要通过训练更新
        self.register_buffer("window", torch.hann_window(config.win_length), persistent=False) # 汉宁窗(win_length=400,) persistent=False 表示不存入 state_dict/checkpoint，因为加载时可以根据配置重新生成。 
        self.register_buffer("filters", mel_filterbank(config), persistent=False) # Mel 滤波器组 (m,k) = (80, 201)

    def forward(
        self, waveforms: torch.Tensor, waveform_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # waveforms.shape = (batch_size, time_steps)
        # waveform_lengths.shape = (batch_size,)
        if waveforms.ndim != 2 or waveforms.shape[0] == 0 or waveforms.shape[1] == 0:
            raise ValueError(f"expected non-empty waveforms [B, N], got {tuple(waveforms.shape)}")
        if waveform_lengths.shape != (waveforms.shape[0],):
            raise ValueError("waveform_lengths must have shape [B]")
        if (waveform_lengths <= 0).any() or (waveform_lengths > waveforms.shape[1]).any():
            raise ValueError("waveform lengths must be in [1, N]")

        if waveforms.shape[1] < self.config.win_length:
            waveforms = F.pad(waveforms, (0, self.config.win_length - waveforms.shape[1]))
        with torch.no_grad():
            spectrum = torch.stft(
                waveforms,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length, # 160
                win_length=self.config.win_length, # 400
                window=self.window,
                center=False,
                return_complex=True,
            ) # spectrum.shape = (batch_size, n_fft // 2 + 1, num_window=time_steps÷(n_fft÷4) + 1)
            power = spectrum.abs().square()
            mel = torch.matmul(self.filters, power) # (n_mels, n_fft//2+1) @ (batch_size, n_fft//2+1, num_window) -> (batch_size, n_mels, num_window)
            features = mel.clamp_min(self.config.epsilon).log().transpose(1, 2) # (batch_size, num_window, n_mels)

        frame_lengths = (
            waveform_lengths.clamp_min(self.config.win_length) - self.config.win_length
        ) // self.config.hop_length + 1 # 计算每条音频的真实帧数 (batch_size,)
        mask = lengths_to_mask(frame_lengths, features.shape[1]) # (batch_size, num_window)
        features = features.masked_fill(~mask[:, :, None], 0.0) # (batch_size, num_window, n_mels)
        count = mask.sum(dim=1).to(features.dtype) # (batch_size,) 每条音频的有效帧数
        total = features.sum(dim=1) # (batch_size, n_mels) 每条音频的 Mel 特征在时间维度上求和
        mean = total / count[:, None] # (batch_size, n_mels) 每条音频的 Mel 特征在时间维度上求均值
        centered = (features - mean[:, None, :]) * mask[:, :, None] # (batch_size, num_window, n_mels) 每条音频的 Mel 特征在时间维度上做中心化，空帧置零
        variance = centered.square().sum(dim=1) / count[:, None] # (batch_size, n_mels) 每条音频的 Mel 特征在时间维度上求方差
        features = centered / variance.clamp_min(self.config.epsilon).sqrt()[:, None, :] # (batch_size, num_window, n_mels) 每条音频的 Mel 特征在时间维度上做归一化
        return features.masked_fill(~mask[:, :, None], 0.0), frame_lengths


def spec_augment(
    features: torch.Tensor, # (batch_size, num_window, n_mels)
    frame_lengths: torch.Tensor, # (batch_size,) 每条音频的有效帧数
    config: ASRConfig,
) -> torch.Tensor:
    '''
    作用是：让模型学会利用遮挡前后的上下文，而不是过度依赖某一个局部声音片段。 
    '''
    augmented = features.clone() # (batch_size, num_window, n_mels)
    batch_size, _, n_mels = features.shape
    for index in range(batch_size):
        length = int(frame_lengths[index])
        for _ in range(config.freq_masks):
            width_limit = min(config.max_freq_mask, n_mels)
            if width_limit == 0:
                continue
            width = int(torch.randint(width_limit + 1, (1,))) # [0，width_limit] 随机取一个整数作为掩码宽度
            start = int(torch.randint(n_mels - width + 1, (1,))) # [0，n_mels-width] 随机取一个整数作为掩码起始位置
            augmented[index, :length, start : start + width] = 0.0
        for _ in range(config.time_masks):
            width_limit = min(config.max_time_mask, length)
            if width_limit == 0:
                continue
            width = int(torch.randint(width_limit + 1, (1,))) # [0，width_limit] 随机取一个整数作为掩码宽度
            start = int(torch.randint(length - width + 1, (1,))) # [0，length-width] 随机取一个整数作为掩码起始位置
            augmented[index, start : start + width, :] = 0.0
    mask = lengths_to_mask(frame_lengths, features.shape[1])
    return augmented.masked_fill(~mask[:, :, None], 0.0)
