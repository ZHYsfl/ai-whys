from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class VQOutput:
    quantized: torch.Tensor
    codes: torch.Tensor
    codebook_loss: torch.Tensor
    commitment_loss: torch.Tensor


@dataclass
class RVQOutput:
    quantized: torch.Tensor
    codes: torch.Tensor
    codebook_loss: torch.Tensor
    commitment_loss: torch.Tensor


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int, dimension: int) -> None:
        super().__init__()
        self.codebook_size = codebook_size # default 64
        self.dimension = dimension # default 32
        self.codebook = nn.Parameter(torch.empty(codebook_size, dimension)) # shape is (codebook_size, dimension) = (64, 32)
        nn.init.uniform_(self.codebook, -1.0 / codebook_size, 1.0 / codebook_size) # 均匀分布

    def forward(self, x: torch.Tensor) -> VQOutput:
        if x.ndim != 3 or x.shape[1] != self.dimension:
            raise ValueError(f"expected [B, {self.dimension}, F], got {tuple(x.shape)}")
        # x.shape is (batch_size, latent_dim = 32, time_steps)
        z = x.permute(0, 2, 1).reshape(-1, self.dimension) # z.shape is (batch_size * time_steps, latent_dim = 32)
        z_norm = (z * z).sum(dim=1, keepdim=True) # z_norm.shape is (batch_size * time_steps, 1)
        e_norm = (self.codebook * self.codebook).sum(dim=1).unsqueeze(0) # e_norm.shape is (1, codebook_size = 64)
        distances = z_norm + e_norm - 2 * z @ self.codebook.T # (batch_size * time_steps, 1) + (1, codebook_size) - 2 * (batch_size * time_steps, latent_dim) @ (latent_dim, codebook_size) = (batch_size * time_steps, codebook_size)
        flat_codes = distances.argmin(dim=1) # (batch_size * time_steps,) 每个样本的每个时间步都对应一个codebook的索引
        quantized = F.embedding(flat_codes, self.codebook) # (batch_size * time_steps, latent_dim = 32) 每个样本的每个时间步都对应一个codebook的向量
        codebook_loss = ((quantized - z.detach()) ** 2).mean(dim=1) # codebook_loss.shape is (batch_size * time_steps,) 每个样本的每个时间步都对应一个codebook的损失
        commitment_loss = ((quantized.detach() - z) ** 2).mean(dim=1) # commitment_loss.shape is (batch_size * time_steps,) 每个样本的每个时间步都对应一个commitment的损失
        codebook_loss = codebook_loss.reshape(x.shape[0], x.shape[2]).mean(dim=1) # codebook_loss.shape is (batch_size,) 每个样本的codebook的损失
        commitment_loss = commitment_loss.reshape(x.shape[0], x.shape[2]).mean(dim=1) # commitment_loss.shape is (batch_size,) 每个样本的commitment的损失
        quantized = quantized.reshape(x.shape[0], x.shape[2], self.dimension).permute(0, 2, 1) # (batch_size, latent_dim = 32, time_steps)
        codes = flat_codes.reshape(x.shape[0], x.shape[2]) # (batch_size, time_steps)
        return VQOutput(quantized, codes, codebook_loss, commitment_loss)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        # codes.shape is (batch_size, time_steps)
        integer_dtypes = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        if codes.ndim != 2 or codes.dtype not in integer_dtypes:
            raise ValueError("codes must be integer tensors with shape [B, F]")
        codes = codes.long()
        if codes.numel() and ((codes < 0).any() or (codes >= self.codebook_size).any()):
            raise ValueError(f"codes must be in [0, {self.codebook_size})")
        quantized = F.embedding(codes, self.codebook) # (batch_size, time_steps, latent_dim = 32)
        return quantized.permute(0, 2, 1) # (batch_size, latent_dim = 32, time_steps)


class ResidualVectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int, dimension: int, num_codebooks: int) -> None:
        super().__init__()
        self.num_codebooks = num_codebooks
        self.quantizers = nn.ModuleList(
            [VectorQuantizer(codebook_size, dimension) for _ in range(num_codebooks)]
        )

    def forward(
        self,
        x: torch.Tensor, # x is the final quantized, default shape is (batch_size, latent_dim = 32, time_steps = 125)
        n_quantizers: int | None = None, # 本次用几级量化，None = 用满4级(default)
        quantizer_dropout: bool = False, # 是否启用量化器dropout
        dropout_rate: float = 1.0, # 以多大概率对样本做 dropout
    ) -> RVQOutput:
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError("dropout_rate must be in [0, 1]")
        n_q = self.num_codebooks if n_quantizers is None else n_quantizers
        if not 1 <= n_q <= self.num_codebooks:
            raise ValueError(f"n_quantizers must be in [1, {self.num_codebooks}]")
        batch_size = x.shape[0]
        active = torch.ones(batch_size, n_q, dtype=torch.bool, device=x.device) # (batch_size,n_q)
        if self.training and quantizer_dropout: # 不依赖后面的同时，让decoder见到的粒度也不一样
            random_n = torch.randint(1, n_q + 1, (batch_size,), device=x.device) # (batch_size,) 每个样本随机选择1~n_q个量化器
            use_dropout = torch.rand(batch_size, device=x.device) < dropout_rate # (batch_size,) 每个样本以 dropout_rate的概率启用dropout
            active_n = torch.where(use_dropout, random_n, torch.full_like(random_n, n_q)) # (batch_size,) 每个样本启用的量化器数目，启用dropout的样本随机选择1~n_q个量化器，否则使用满4级
            active = torch.arange(n_q, device=x.device).unsqueeze(0) < active_n.unsqueeze(1) # (batch_size,n_q) 每个样本启用的量化器的mask，True表示启用，False表示不启用

        residual = x # residual.shape is (batch_size, latent_dim = 32, time_steps = 125)
        quantized_sum = torch.zeros_like(x) # quantized_sum.shape is (batch_size, latent_dim = 32, time_steps = 125)
        all_codes = []
        codebook_losses = []
        commitment_losses = []
        for index, quantizer in enumerate(self.quantizers[:n_q]):
            result = quantizer(residual)
            sample_mask = active[:, index].to(x.dtype) # (batch_size,) 每个样本是否启用当前量化器的mask，1表示启用，0表示不启用
            mask = sample_mask[:, None, None] # (batch_size, 1, 1) 每个样本是否启用当前量化器的mask，1表示启用，0表示不启用
            quantized_sum = quantized_sum + result.quantized * mask  # (batch_size, latent_dim = 32, time_steps)
            residual = residual - result.quantized.detach() * mask # (batch_size, latent_dim = 32, time_steps) 新的残差是当前残差减去当前量化器的量化结果，detach()表示不计算梯度
            layer_codes = result.codes.masked_fill(~active[:, index, None], -1) # (batch_size, time_steps) 给被 dropout 关掉的样本打无效标记 -1
            all_codes.append(layer_codes)
            normalizer = sample_mask.sum().clamp_min(1) #  计算这一级有几个样本参与，最小钳到 1，避免除零
            codebook_losses.append((result.codebook_loss * sample_mask).sum() / normalizer) #  计算这一级的 mean codebook loss
            commitment_losses.append((result.commitment_loss * sample_mask).sum() / normalizer) #  计算这一级的 mean commitment loss

        codes = torch.stack(all_codes, dim=1) # codes.shape is (batch_size, n_q, time_steps) 每个样本的每个时间步都对应一个codebook的索引，-1表示被dropout关掉
        codebook_loss = torch.stack(codebook_losses).sum() # 计算总的 mean codebook loss
        commitment_loss = torch.stack(commitment_losses).sum() # 计算总的 mean commitment loss

        #  x → encoder → z → 算距离 → argmin(断了!) → 查码本 → quantized_sum argmin 把 z 到输出的梯度链切断了。
        # decoder 传回的重建损失梯度到达 quantized_sum 后，流不到 z，更流不到 encoder。encoder 成了没人管教的孩子——重建得好坏它都感知不到，永远学不到"该输出什么样的隐向量才好重建"
        # 这里我们希望 encoder 能感知到重建损失的梯度，学会输出更好的隐向量。我们可以通过 straight-through estimator (STE) trick 来实现。
        # 对 x 求导，detach 的部分导数为 0：∂quantized/∂x = ∂x/∂x + 0 = 1
        # 也就是说，上游传来的梯度 g 会原封不动地乘 1 传给 x——反向传播假装量化根本不存在，是一条恒等通路。encoder 由此收到了重建损失的完整梯度，正常更新。
        quantized = x + (quantized_sum - x).detach() 
        # 这个"假装"合理吗？
        # 严格说这是个有偏的近似——量化明明不是恒等映射，却按恒等映射传梯度。它能 work 是因为有 commitment_loss 配合：那个loss 逼着 encoder 输出 z 紧贴码本条目，量化误差 ‖z −quantized‖² 很小，"量化 ≈ 恒等"就近似成立，STE 的偏差也就很小。两个 loss 和 STE 是一套组合拳：
        # codebook_loss：码本向数据靠拢；
        # commitment_loss：数据向码本靠拢；
        # STE：让重建梯度绕过量化直达 encoder。
        # 三者缺一，VQ 都训不起来。这也是 VQ-VAE 论文的原版设计, 这个文件忠实地继承了它。 

        return RVQOutput(quantized, codes, codebook_loss, commitment_loss)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.ndim != 3:
            raise ValueError("codes must have shape [B, Q, F]")
        if not 1 <= codes.shape[1] <= self.num_codebooks:
            raise ValueError(f"expected 1..{self.num_codebooks} codebooks, got {codes.shape[1]}")
        quantized = None
        for index in range(codes.shape[1]):
            layer_quantized = self.quantizers[index].decode(codes[:, index]) # (batch_size, latent_dim = 32, time_steps)
            quantized = layer_quantized if quantized is None else quantized + layer_quantized
        assert quantized is not None
        return quantized # (batch_size, latent_dim = 32, time_steps)
