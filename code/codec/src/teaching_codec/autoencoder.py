import torch
from torch import nn

from .config import CodecConfig


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.activation = nn.ELU()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1) # (t + 2*1 - 3)/1 + 1 = t,长度不变
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1) # (t + 2*1 - 3)/1 + 1 = t,长度不变

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(self.activation(x))
        x = self.conv2(self.activation(x))
        return x + residual


class Encoder(nn.Module):
    def __init__(self, config: CodecConfig) -> None:
        super().__init__()
        channels = config.base_channels # 默认 32
        layers: list[nn.Module] = [nn.Conv1d(1, channels, kernel_size=7, padding=3)] # (16000+2*3-7)/1 + 1 = 16000,长度不变。time_steps output shape:(batch_size,out_channels = 32, time_steps/sample_rate = 16000) 
        for stride in config.strides: # default: 2, 4, 4, 4
            layers.extend(
                [
                    ResidualBlock(channels),
                    nn.ELU(),
                    nn.Conv1d(
                        channels,
                        channels * 2,
                        kernel_size=2 * stride,
                        stride=stride,
                        padding=stride // 2,
                    ), # output_channels每次扩大2倍，而 (t + 2* stride//2 - 2*stride)/stride + 1 = t/stride,t长度缩小为原来的1/stride
                ]
            )
            channels *= 2
        # now the shape is : (batch_size,output_channels=32*2*2*2*2=512,time_steps=16000/2/4/4/4=125)
        layers.extend([nn.ELU(), nn.Conv1d(channels, config.latent_dim, kernel_size=3, padding=1)]) # 125+2*1-3/1+1=125,长度不变,channel是latent_dim默认是 32
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape is (batch_size, input_channels = 1, time_steps/sample_rate = 16000) 教学版本默认单声道,采样率16000
        return self.network(x) # output shape is (batch_size, latent_dim = 32, time_steps = 125)


class Decoder(nn.Module):
    def __init__(self, config: CodecConfig) -> None:
        super().__init__()
        channels = config.base_channels * (2 ** len(config.strides)) # default 512
        layers: list[nn.Module] = [nn.Conv1d(config.latent_dim, channels, kernel_size=3, padding=1)] # (125+2*1-3)/1 + 1 = 125, 长度不变, output channel变为512, so the output shape is (batch_size, channels = 512, time_steps = 125)
        for stride in reversed(config.strides): # default 4, 4, 4, 2
            layers.extend(
                [
                    nn.ELU(),
                    nn.ConvTranspose1d( # K+(T-1)*S-2*P
                        channels,
                        channels // 2,
                        kernel_size=2 * stride,
                        stride=stride,
                        padding=stride // 2,
                    ), # 2*stride+(T-1)*stride-2*stride//2 = T*stride, so the length is expanded by stride times, and the output channel is halved
                    ResidualBlock(channels // 2),
                ]
            )
            channels //= 2
        layers.extend([nn.ELU(), nn.Conv1d(channels, 1, kernel_size=7, padding=3), nn.Tanh()]) # output channel is 1, the length is (time_steps+2*3-7)/1 + 1 = time_steps, so the output shape is (batch_size, output_channels = 1, time_steps)
        self.network = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z is the output of rvq, default shape is (batch_size, latent_dim = 32, time_steps = 125)
        return self.network(z) # (batch_size,final_channels=1,time_steps)
