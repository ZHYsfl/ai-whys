# Derived from kyutai-labs/moshi (MIT); see NOTICE and LICENSES/MIT-moshi.txt.

import json
from dataclasses import dataclass, fields
from pathlib import Path

import torch

from .config import MimiConfig, MoshiConfig
from .mimi import MimiModel
from .model import MoshiLM
from .text import SentencePieceTokenizer


DEFAULT_REPOSITORY = "kyutai/moshiko-pytorch-bf16"
MIMI_NAME = "tokenizer-e351c8d8-checkpoint125.safetensors"
MOSHI_NAME = "model.safetensors"
TOKENIZER_NAME = "tokenizer_spm_32k_3.model"


def _resolve_file(
    value: str | Path | None, hf_repository: str | None, default_name: str
) -> Path:
    if value is not None and Path(value).exists():
        return Path(value)
    if hf_repository is None:
        if value is None:
            raise ValueError(f"provide a local path for {default_name}")
        return Path(value)
    from huggingface_hub import hf_hub_download

    filename = default_name if value is None else str(value)
    return Path(hf_hub_download(hf_repository, filename))


def _load_safetensors(path: str | Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    return load_file(str(path), device="cpu")


def load_official_mimi(
    weights_path: str | Path,
    config: MimiConfig | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
) -> MimiModel:
    config = config or MimiConfig.official()
    with torch.device("meta"):
        model = MimiModel(config)
    state = _load_safetensors(weights_path)
    if dtype is not None:
        state = {
            key: value.to(dtype) if value.dtype.is_floating_point else value
            for key, value in state.items()
        }
    result = model.load_state_dict(state, strict=False, assign=True)
    allowed_missing = {"semantic_projection.weight", "semantic_projection.bias"}
    if result.unexpected_keys or set(result.missing_keys) != allowed_missing:
        raise ValueError(
            f"incompatible Mimi weights; missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}"
        )
    model.to(device=device)
    model.eval()
    return model


def config_from_official(values: dict) -> MoshiConfig:
    config = MoshiConfig.official()
    defaults = config.to_dict()
    for field in fields(MoshiConfig):
        if field.name in values:
            defaults[field.name] = values[field.name]
    return MoshiConfig.from_dict(defaults)


def load_official_moshi(
    weights_path: str | Path,
    config: MoshiConfig | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> MoshiLM:
    config = config or MoshiConfig.official()
    with torch.device("meta"):
        model = MoshiLM(config)
    state = _load_safetensors(weights_path)
    state = {
        key: value.to(dtype) if value.dtype.is_floating_point else value
        for key, value in state.items()
    }
    model.load_state_dict(state, strict=True, assign=True)
    model.to(device=device)
    model.eval()
    return model


@dataclass
class OfficialModels:
    mimi: MimiModel
    moshi: MoshiLM
    tokenizer: SentencePieceTokenizer


def load_official_models(
    hf_repository: str | None = DEFAULT_REPOSITORY,
    mimi_path: str | Path | None = None,
    moshi_path: str | Path | None = None,
    tokenizer_path: str | Path | None = None,
    config_path: str | Path | None = None,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> OfficialModels:
    config = None
    if config_path is not None or hf_repository is not None:
        try:
            resolved_config = _resolve_file(config_path, hf_repository, "config.json")
            raw_config = json.loads(resolved_config.read_text())
            config = config_from_official(raw_config)
        except Exception as error:
            if config_path is not None:
                raise
            from huggingface_hub.errors import EntryNotFoundError

            if not isinstance(error, EntryNotFoundError):
                raise
    mimi_weights = _resolve_file(mimi_path, hf_repository, MIMI_NAME)
    moshi_weights = _resolve_file(moshi_path, hf_repository, MOSHI_NAME)
    tokenizer_file = _resolve_file(tokenizer_path, hf_repository, TOKENIZER_NAME)
    mimi = load_official_mimi(mimi_weights, device=device, dtype=dtype)
    moshi = load_official_moshi(moshi_weights, config, device, dtype)
    tokenizer = SentencePieceTokenizer(tokenizer_file)
    if tokenizer.vocab_size != moshi.text_card:
        raise ValueError("official text tokenizer and model vocabulary sizes differ")
    return OfficialModels(mimi, moshi, tokenizer)
