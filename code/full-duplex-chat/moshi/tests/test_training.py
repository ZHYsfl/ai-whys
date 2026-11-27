import torch

from teaching_moshi.checkpoint import (
    load_lora_checkpoint,
    load_moshi_checkpoint,
    save_moshi_checkpoint,
)
from teaching_moshi.data import SyntheticDialogueDataset, collate_dialogue
from teaching_moshi.lora import apply_lora, merge_lora
from teaching_moshi.model import MoshiLM
from teaching_moshi.training import TrainingConfig, run_training, training_step


def assert_aligned_outputs_close(first, second):
    torch.testing.assert_close(first.text_logits, second.text_logits)
    mask = first.mask & second.mask
    torch.testing.assert_close(first.logits[mask], second.logits[mask])
    torch.testing.assert_close(first.mask, second.mask)


def test_full_training_and_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    model = MoshiLM()
    dataset = SyntheticDialogueDataset(size=4)
    config = TrainingConfig(max_steps=2, batch_size=2, log_every=1)
    history = run_training(model, dataset, config, tmp_path)
    assert len(history) == 2
    restored, payload = load_moshi_checkpoint(tmp_path / "checkpoint-last.pt")
    assert payload["step"] == 2
    model.eval()
    batch = collate_dialogue([dataset[0], dataset[1]])
    with torch.no_grad():
        expected = model(batch.codes)
        actual = restored(batch.codes)
    assert_aligned_outputs_close(expected, actual)


def test_lora_only_updates_adapters_and_adapter_checkpoint_loads(tmp_path):
    torch.manual_seed(1)
    model = MoshiLM()
    save_moshi_checkpoint(tmp_path / "base.pt", model)
    apply_lora(model, rank=4, scaling=2.0)
    frozen = model.text_emb.weight.detach().clone()
    frozen_linear = model.transformer.layers[0].self_attn.in_projs[0].frozen_W.weight.detach().clone()
    assert all("lora" in name for name, parameter in model.named_parameters() if parameter.requires_grad)
    dataset = SyntheticDialogueDataset(size=2)
    config = TrainingConfig(max_steps=1, batch_size=2, save_adapters=True)
    run_training(model, dataset, config, tmp_path / "adapter")
    torch.testing.assert_close(model.text_emb.weight, frozen)
    torch.testing.assert_close(
        model.transformer.layers[0].self_attn.in_projs[0].frozen_W.weight,
        frozen_linear,
    )
    base, _ = load_moshi_checkpoint(tmp_path / "base.pt")
    restored, _ = load_lora_checkpoint(
        tmp_path / "adapter" / "checkpoint-last.pt", base
    )
    model.eval()
    batch = collate_dialogue([dataset[0]])
    with torch.no_grad():
        expected = model(batch.codes)
        actual = restored(batch.codes)
    assert_aligned_outputs_close(expected, actual)


def test_lora_merge_preserves_valid_outputs():
    torch.manual_seed(2)
    model = MoshiLM().eval()
    apply_lora(model, rank=4)
    for module in model.modules():
        if hasattr(module, "lora_B"):
            torch.nn.init.normal_(module.lora_B.weight, std=0.02)
    codes = collate_dialogue([SyntheticDialogueDataset()[0]]).codes
    with torch.no_grad():
        expected = model(codes)
    merge_lora(model)
    with torch.no_grad():
        actual = model(codes)
    assert_aligned_outputs_close(expected, actual)
