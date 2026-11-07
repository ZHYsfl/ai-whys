import torch

from teaching_asr.data import SyntheticSpeechDataset, collate_speech
from teaching_asr.modern_config import ModernASRConfig
from teaching_asr.qwen3_asr import Qwen3ASR


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = ModernASRConfig()
    model = Qwen3ASR(config).to(device).eval()
    dataset = SyntheticSpeechDataset(size=2)
    batch = collate_speech([dataset[0], dataset[1]]).to(device)
    print("no pretrained weights: every module is implemented and initialized in this project")
    print(f"waveforms: {tuple(batch.waveforms.shape)}, lengths={batch.waveform_lengths.tolist()}")
    with torch.no_grad():
        encoded, output_lengths, mask = model.audio_encoder(batch.waveforms, batch.waveform_lengths)
        print(
            f"AuT encoder [B, 12.5Hz, D]: {tuple(encoded.shape)}, "
            f"lengths={output_lengths.tolist()}"
        )
        for window in (0, 1, 25):
            windowed, window_lengths, _ = model.audio_encoder(
                batch.waveforms, batch.waveform_lengths, attention_window=window
            )
            print(
                f"attention_window={window}: output={tuple(windowed.shape)}, "
                f"lengths={window_lengths.tolist()}"
            )
        speech = model.projector(encoded)
        print(f"projector output [B, T, LLM hidden]: {tuple(speech.shape)}")
        inputs, attention_mask, labels = model._build_inputs(
            speech, output_lengths, batch.transcripts
        )
        print(f"LLM inputs_embeds [B, L, H]: {tuple(inputs.shape)}")
        print(f"attention_mask: {tuple(attention_mask.shape)}, labels: {tuple(labels.shape)}")
        print(f"supervised text tokens per sample: {(labels != -100).sum(dim=1).tolist()}")
        output = model.llm(inputs_embeds=inputs, attention_mask=attention_mask, use_cache=True)
        print(f"Qwen3 logits [B, L, V]: {tuple(output.logits.shape)}")
        print(f"KV cache layers: {len(output.past_key_values)}")
        key, value = output.past_key_values[0]
        print(f"layer-0 cached K: {tuple(key.shape)}, V: {tuple(value.shape)}")
        print(f"random-weight generation: {model.transcribe(batch.waveforms, batch.waveform_lengths, 4)}")


if __name__ == "__main__":
    main()
