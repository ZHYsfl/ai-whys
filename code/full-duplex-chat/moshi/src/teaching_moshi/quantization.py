# Copyright (c) Kyutai, all rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Derived from kyutai-labs/moshi and AudioCraft; see NOTICE and LICENSES/.

import math
from dataclasses import dataclass, field
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class QuantizedResult:
    x: torch.Tensor
    codes: torch.Tensor
    bandwidth: torch.Tensor
    penalty: torch.Tensor
    metrics: dict[str, torch.Tensor] = field(default_factory=dict)
    semantic: torch.Tensor | None = None


class CodebookForwardResult(NamedTuple):
    quantized: torch.Tensor
    codes: torch.Tensor
    metrics: dict[str, torch.Tensor]


class VQForwardResult(NamedTuple):
    quantized: torch.Tensor
    codes: torch.Tensor
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]


def zero_scalar(device: torch.device) -> torch.Tensor:
    return torch.zeros((), device=device)


def _ema_inplace(moving_average: torch.Tensor, new: torch.Tensor, decay: float) -> None:
    '''
    指数滑动平均/指数移动平均
    moving_average ← decay * moving_average + (1 - decay) * new
    '''
    moving_average.data.mul_(decay).add_(new, alpha=1 - decay)


def _sample_vectors(samples: torch.Tensor, count: int) -> torch.Tensor:
    indices = (
        torch.randperm(samples.shape[0], device=samples.device)[:count]
        if samples.shape[0] >= count
        else torch.randint(samples.shape[0], (count,), device=samples.device)
    )
    return samples[indices]


def _run_kmeans(
    samples: torch.Tensor, num_clusters: int, num_iterations: int = 50
) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    Lloyd 算法(k-means 的标准实现)
    '''
    # samples.shape is (length_samples = batch_size * num_frames, dimension)
    # num_clusters is the number of codebook_size
    means = _sample_vectors(samples, num_clusters) # 从一袋向量 samples 里随机挑 count 个，当初始聚类中心
    # means.shape is (num_clusters, dimension)
    for _ in range(num_iterations):
        distances = torch.cdist(samples, means) # (length_samples, num_clusters)
        assignments = distances.argmin(dim=-1) # (length_samples,) 每个样本属于哪个聚类中心
        counts = torch.bincount(assignments, minlength=num_clusters) # (num_clusters,) 每个聚类中心有多少样本属于它
        empty = counts == 0 # (num_clusters,) 哪些聚类中心没有样本属于它
        counts = counts.clamp_min(1) # 避免除以 0
        new_means = torch.zeros_like(means) # (num_clusters, dimension)
        new_means.scatter_add_(
            0, assignments[:, None].expand(-1, samples.shape[-1]), samples
        ) # (num_clusters, dimension)
        new_means /= counts[:, None] # (num_clusters, dimension)
        means = torch.where(empty[:, None], _sample_vectors(samples, num_clusters), new_means) 
    return means, counts # (num_clusters, dimension), (num_clusters,)


class EuclideanCodebook(nn.Module):
    """Codebook learned by EMA rather than gradient descent."""

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        threshold_usage_ratio: float = 0.1, # 码本中使用率低于平均使用率的 10% 的码字会被替换
        check_unused_every: int = 5, # 查死码的周期
    ) -> None:
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.decay = decay
        self.epsilon = epsilon
        self.threshold_usage_ratio = threshold_usage_ratio
        self.check_unused_every = check_unused_every
        self._next_unused_check = check_unused_every
        self._cached_initialized = False
        self.register_buffer("_initialized", torch.tensor([False]))
        self.register_buffer("cluster_usage", torch.ones(codebook_size))
        self.register_buffer("embedding_sum", torch.zeros(codebook_size, dim))
        self.register_buffer("_embedding", None, persistent=False)

    @property
    def embedding(self) -> torch.Tensor:
        if self._embedding is None:
            embedding = self.embedding_sum / self.cluster_usage.clamp_min(self.epsilon)[:, None]
            self.register_buffer("_embedding", embedding, persistent=False)
        return self._embedding

    @property
    def initialized(self) -> bool:
        if not self._cached_initialized:
            self._cached_initialized = bool(self._initialized.item())
        return self._cached_initialized

    def _init_embedding(self, data: torch.Tensor) -> None:
        # data.shape is (batch_size * num_frames, dimension)
        if self.initialized:
            return
        embedding, usage = _run_kmeans(data, self.codebook_size)
        # embedding.shape is (codebook_size, dimension) 聚类中心
        # usage.shape is (codebook_size,) 属于聚类中心
        self.embedding_sum.data.copy_(embedding * usage[:, None])
        self.cluster_usage.data.copy_(usage)
        self._initialized.data.fill_(True)
        self._cached_initialized = True

    def _check_expired_codes(self, samples: torch.Tensor) -> torch.Tensor:
        '''
        检查过期的码字,过期码字会被替换，之后返回过期码字的比例, 0D tensor
        '''
        # samples.shape is (batch_size * num_frames, dimension)
        if not self.initialized: # 出生前不查
            return zero_scalar(samples.device) # 0D tensor
        self._next_unused_check -= 1 # 计数
        if self._next_unused_check > 0: # 没到周期，不查
            return zero_scalar(samples.device) # 0D tensor
        # 该查了
        self._next_unused_check = self.check_unused_every # 计数器重置
        threshold = self.threshold_usage_ratio * self.cluster_usage.sum() / self.codebook_size # 码本中聚类中心的平均使用率的 10%
        expired = self.cluster_usage < threshold # 使用率低于平均使用率的 10% 的码字是过期的死码 (codebook_size,)
        if expired.any(): # 如果死码
            new_vectors = _sample_vectors(samples, self.codebook_size) # (codebook_size, dimension)
            replacement_usage = self.cluster_usage.sum() / self.codebook_size # 平均使用率
            self.embedding_sum[:] = torch.where(
                expired[:, None], replacement_usage * new_vectors, self.embedding_sum
            ) # 替换死码，替换数据来自真实数据
            self.cluster_usage[:] = torch.where(
                expired, replacement_usage.expand_as(self.cluster_usage), self.cluster_usage
            ) # 替换使用率，更新为新的平均使用率
            self.register_buffer("_embedding", None, persistent=False)
        return expired.float().mean() # 过期码字的比例， 0D tensor

    def _quantize(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size * num_frames, dimension)
        return torch.cdist(x, self.embedding).argmin(dim=-1) # (batch_size * num_frames,) 每个样本属于哪个聚类中心

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape is (batch_size, num_frames, dimension)
        shape = x.shape[:-1] # (batch_size, num_frames)
        return self._quantize(x.reshape(-1, self.dim)).view(shape) # (batch_size, num_frames)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        # codes.shape is (batch_size * num_frames,) or (batch_size, num_frames)
        return F.embedding(codes, self.embedding) # (batch_size * num_frames, dimension) or (batch_size, num_frames, dimension)

    def forward(self, x: torch.Tensor, initialize: bool = True) -> CodebookForwardResult:
        # x.shape is (batch_size, num_frames, dimension)
        shape = x.shape[:-1] # (batch_size, num_frames)
        flat = x.reshape(-1, self.dim) # (batch_size * num_frames, dimension)
        if self.training and initialize:
            self._init_embedding(flat.detach())
        codes = self._quantize(flat) # codes.shape is (batch_size * num_frames,)
        quantized = self.decode(codes) # (batch_size * num_frames, dimension)
        metrics = {}
        if self.training:
            metrics["rvq_expired"] = self._check_expired_codes(flat)
            usage = torch.zeros_like(self.cluster_usage)
            usage.scatter_add_(0, codes, torch.ones_like(codes, dtype=usage.dtype))
            # usage[codes[i]] +=  torch.ones_like(codes, dtype=usage.dtype)[i]即为1 for i in range(len(codes))
            # usage[codes[i]] += 1 for i in range(len(codes))
            # 记录哪些聚类中心这次被使用，并且用了多少次
            _ema_inplace(self.cluster_usage, usage, self.decay) # EMA更新聚类中心使用频率
            probabilities = self.cluster_usage / self.cluster_usage.sum() # 使用率归一化成概率分布，归一化后所有 p 加起来 = 1，是一个合法的概率分布。
            entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum() # 公式就是信息论的标准熵：H = −Σ pⱼ · log(pⱼ)
            metrics["rvq_entropy"] = entropy / math.log(self.codebook_size) # 码本使用分布的熵,熵的上限(均匀分布时取到) = log(K),除一下，指标永远落在 0~1
            embedding_sum = torch.zeros_like(self.embedding_sum) 
            embedding_sum.scatter_add_(
                0, codes[:, None].expand(-1, self.dim), flat
            )
            # embedding_sum[codes[:,None].expand(-1, self.dim)[i]] += flat[i] for i in range(len(codes))
            _ema_inplace(self.embedding_sum, embedding_sum, self.decay)
            self.register_buffer("_embedding", None, persistent=False)
        quantized = quantized.view(*shape, self.dim) # (batch_size, num_frames, dimension)
        return CodebookForwardResult(quantized, codes.view(shape), metrics) # 


class VectorQuantization(nn.Module):
    def __init__(self, dim: int, codebook_size: int, decay: float = 0.99) -> None:
        super().__init__()
        self.project_in = nn.Identity()
        self.project_out = nn.Identity()
        self._codebook = EuclideanCodebook(dim, codebook_size, decay)
        self.codebook_size = codebook_size

    @property
    def initialized(self) -> bool:
        return self._codebook.initialized

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape is (batch_size, dimension, num_frames)
        x = x.transpose(1, 2) # x.shape is (batch_size, num_frames, dimension)
        return self._codebook.encode(self.project_in(x)) # (batch_size, num_frames)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        # codes.shape is (batch_size, num_frames)
        x = self._codebook.decode(codes) # (batch_size, num_frames, dimension)
        return self.project_out(x).transpose(1, 2) # (batch_size, dimension, num_frames)

    def forward(self, x: torch.Tensor, initialize: bool = True) -> VQForwardResult:
        x = x.transpose(1, 2) # x.shape is (batch_size, num_frames, dimension)
        quantized, codes, metrics = self._codebook(x, initialize)
        if self.training:
            quantized = x + (quantized - x).detach() # Straight-Through Estimator（STE，直通估计器）反传假装"量化这步不存在"，梯度直通 encoder。     
            loss = F.mse_loss(x, quantized.detach())
        else:
            loss = zero_scalar(x.device)
        return VQForwardResult(
            self.project_out(quantized).transpose(1, 2), codes, loss, metrics
        )


class ResidualVectorQuantization(nn.Module):
    def __init__(self, num_quantizers: int, **kwargs) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [VectorQuantization(**kwargs) for _ in range(num_quantizers)]
        )

    def forward(self, x: torch.Tensor, n_q: int | None = None) -> VQForwardResult:
        # X.shape is (batch_size, dimension, num_frames)
        n_q = n_q or len(self.layers)
        residual = x
        quantized_sum = zero_scalar(x.device) # initially zero
        codes, losses = [], []
        metrics: dict[str, torch.Tensor] = {}
        previous_initialized = True
        for index, layer in enumerate(self.layers[:n_q]):
            if self.training:
                was_initialized = layer.initialized
            quantized, layer_codes, loss, layer_metrics = layer(
                residual, initialize=previous_initialized
            )
            if self.training:
                previous_initialized = was_initialized
            quantized = quantized.detach()
            residual = residual - quantized
            quantized_sum = quantized_sum + quantized
            codes.append(layer_codes)
            losses.append(loss)
            for key, value in layer_metrics.items():
                metrics[key] = metrics.get(key, zero_scalar(x.device)) + value / n_q
                metrics[f"{key}_{index}"] = value
        if self.training:
            quantized_sum = x + (quantized_sum - x).detach()
        return VQForwardResult(
            quantized_sum, torch.stack(codes), torch.stack(losses), metrics
        )

    def encode(self, x: torch.Tensor, n_q: int | None = None) -> torch.Tensor:
        # x.shape is (batch_size, dimension, num_frames)
        n_q = n_q or len(self.layers)
        residual = x # x.shape is (batch_size, dimension, num_frames)
        codes = []
        for layer in self.layers[:n_q]:
            indices = layer.encode(residual) # (batch_size, num_frames)
            residual = residual - layer.decode(indices) # (batch_size, dimension, num_frames)
            codes.append(indices)
        return torch.stack(codes) # (n_q, batch_size, num_frames)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        # (n, batch_size, num_frames)
        quantized = zero_scalar(codes.device)
        for layer, layer_codes in zip(self.layers, codes, strict=False):
            quantized = quantized + layer.decode(layer_codes)
        return quantized # (batch_size, dimension, num_frames)


class ResidualVectorQuantizer(nn.Module):
    def __init__(
        self,
        dimension: int, # the length of each vector element in codebook
        input_dimension: int, # encoder 给的 x 每帧几维
        output_dimension: int, # 出去给 decoder 时每帧要变回几维
        n_q: int, # 级联 n_q 层码本
        bins: int, # 每张码本的大小
        decay: float = 0.99, # 码表的内容如何更新：不跟梯度走，按 0.99 衰减滑动平均慢慢挪动
        q_dropout: bool = False, # 训练时随机只用前 1-8 层，让每个截断点都被训到
        no_quantization_rate: float = 0.0, # 训练时按概率整句跳过量化， decoder 一半时间见到无损 latent
        force_projection: bool = False,
    ) -> None:
        super().__init__()
        self.max_n_q = n_q
        self.n_q = n_q
        self.bins = bins
        self.dimension = dimension
        self.q_dropout = q_dropout
        self.no_quantization_rate = no_quantization_rate
        self.input_proj = (
            nn.Conv1d(input_dimension, dimension, 1, bias=False)
            if input_dimension != dimension or force_projection
            else nn.Identity()
        )
        self.output_proj = (
            nn.Conv1d(dimension, output_dimension, 1, bias=False)
            if dimension != output_dimension or force_projection
            else nn.Identity()
        )
        self.vq = ResidualVectorQuantization(
            num_quantizers=n_q, dim=dimension, codebook_size=bins, decay=decay
        )

    @property
    def num_codebooks(self) -> int:
        return self.n_q

    @property
    def total_codebooks(self) -> int:
        return self.max_n_q

    @property
    def cardinality(self) -> int:
        return self.bins

    def set_num_codebooks(self, n: int) -> None:
        if not 0 <= n <= self.max_n_q:
            raise ValueError(f"n must be in [0, {self.max_n_q}]")
        self.n_q = n

    def forward(self, x: torch.Tensor, frame_rate: float) -> QuantizedResult:
        # x.shape is (batch_size, input_dimension, num_frames)
        # frame_rate is the number of frames per second (e.g., 50 for 20ms frames)
        n_q = self.n_q
        if n_q < 1:
            raise ValueError("at least one active quantizer is required")
        x = self.input_proj(x)
        # x.shape is (batch_size, dimension, num_frames)
        if self.training and self.q_dropout:
            n_q = int(torch.randint(1, self.n_q + 1, ())) # n_q is randomly chosen between 1 and self.n_q during training if q_dropout is enabled
        quantized, codes, commitment, metrics = self.vq(x, n_q)
        
        if self.training and self.no_quantization_rate > 0:
            bypass = (
                torch.rand(x.shape[0], 1, 1, device=x.device) <= self.no_quantization_rate
            ).to(x.dtype)
            quantized = bypass * x + (1 - bypass) * quantized
        quantized = self.output_proj(quantized)
        bandwidth = torch.tensor(
            n_q * math.log2(self.bins) * frame_rate / 1000,
            device=x.device,
            dtype=x.dtype,
        ) # 码率带宽遥测，单位 ： 千比特/秒 n_q 会变所以不是常数
        return QuantizedResult(
            quantized, codes.transpose(0, 1), bandwidth, commitment.mean(), metrics
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape is (batch_size, input_dimension, num_frames)
        x = self.input_proj(x) # x.shape is (batch_size, dimension, num_frames)
        return self.vq.encode(x, self.n_q).transpose(0, 1) # (batch_size, n_q, num_frames)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        # (batch_size, n, num_frames)
        quantized = self.vq.decode(codes.transpose(0, 1)) # (batch_size, dimension, num_frames)
        return self.output_proj(quantized) # (batch_size, output_dimension, num_frames)


class SplitResidualVectorQuantizer(nn.Module):
    """Mimi: one semantic VQ and an acoustic RVQ both quantize the same input."""

    def __init__(
        self,
        dimension: int, # the length of each vector element in codebook
        input_dimension: int, # encoder 给的 x 每帧几维
        output_dimension: int, # 出去给 decoder 时每帧要变回几维
        n_q: int, # 级联 n_q 层码本
        bins: int, # 每张码本的大小
        decay: float = 0.99, # 码表的内容如何更新：不跟梯度走，按 0.99 衰减滑动平均慢慢挪动
        q_dropout: bool = False, # 训练时随机只用前 1-8 层，让每个截断点都被训到
        no_quantization_rate: float = 0.0, # 训练时按概率整句跳过量化， decoder 一半时间见到无损 latent
    ) -> None:
        super().__init__()
        if n_q < 2:
            raise ValueError("split RVQ requires semantic and acoustic quantizers")
        self.n_q_semantic = 1
        self.rvq_first = ResidualVectorQuantizer(
            dimension,
            input_dimension,
            output_dimension,
            self.n_q_semantic,
            bins,
            decay,
            q_dropout=False,
            no_quantization_rate=no_quantization_rate,
            force_projection=True,
        )
        self.rvq_rest = ResidualVectorQuantizer(
            dimension,
            input_dimension,
            output_dimension,
            n_q - 1,
            bins,
            decay,
            q_dropout=q_dropout,
            no_quantization_rate=no_quantization_rate,
            force_projection=True,
        )

    @property
    def n_q(self) -> int:
        return self.rvq_first.n_q + self.rvq_rest.n_q

    @property
    def num_codebooks(self) -> int:
        return self.n_q

    @property
    def total_codebooks(self) -> int:
        return self.rvq_first.total_codebooks + self.rvq_rest.total_codebooks

    @property
    def cardinality(self) -> int:
        return self.rvq_first.cardinality

    @property
    def semantic_quantizer(self) -> ResidualVectorQuantizer:
        return self.rvq_first

    @property
    def acoustic_quantizer(self) -> ResidualVectorQuantizer:
        return self.rvq_rest

    def set_num_codebooks(self, n: int) -> None:
        if not 1 <= n <= self.total_codebooks:
            raise ValueError(f"n must be in [1, {self.total_codebooks}]")
        self.rvq_rest.set_num_codebooks(n - 1)

    def forward(self, x: torch.Tensor, frame_rate: float) -> QuantizedResult:
        # x.shape is (batch_size, input_dimension, num_frames)
        semantic = self.rvq_first(x, frame_rate)
        acoustic = self.rvq_rest(x, frame_rate)
        n_semantic = semantic.codes.shape[1]
        n_acoustic = acoustic.codes.shape[1]
        n_total = n_semantic + n_acoustic
        penalty = (
            semantic.penalty * n_semantic + acoustic.penalty * n_acoustic
        ) / n_total
        metrics = dict(semantic.metrics)
        metrics.update(acoustic.metrics)
        return QuantizedResult(
            semantic.x + acoustic.x, # (batch_size, output_dimension, num_frames)
            torch.cat((semantic.codes, acoustic.codes), dim=1), # (batch_size, n_semantic + n_acoustic, num_frames)
            semantic.bandwidth + acoustic.bandwidth,
            penalty,
            metrics,
            semantic=semantic.x,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape is (batch_size, input_dimension, num_frames)
        return torch.cat((self.rvq_first.encode(x), self.rvq_rest.encode(x)), dim=1) # (batch_size, 1 + n_q-1, num_frames)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        # codes.shape is (batch_size, 1 + n_q-1, num_frames)
        return self.rvq_first.decode(codes[:, :1]) + self.rvq_rest.decode(codes[:, 1:]) # (batch_size, output_dimension, num_frames)
