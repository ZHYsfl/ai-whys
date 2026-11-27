import torch

from teaching_moshi.inference import MoshiDialogue
from teaching_moshi.mimi import MimiModel
from teaching_moshi.model import MoshiLM
from teaching_moshi.text import ByteTokenizer


def test_waveform_to_waveform_streaming_pipeline():
    torch.manual_seed(0)
    mimi = MimiModel()
    model = MoshiLM()
    dialogue = MoshiDialogue(mimi, model, ByteTokenizer(), use_sampling=False)
    output = dialogue.respond(torch.randn(1, 1, 3840).clamp(-1, 1))
    assert output.waveform.shape == (1, 1, 1920)
    assert output.audio_tokens.shape == (1, 8, 1)
    assert len(output.text_tokens) == len(output.text) == 1
    assert (output.audio_tokens >= 0).all()
    assert (output.audio_tokens < model.card).all()
