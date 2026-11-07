import torch

from teaching_asr import ASRConfig, ConformerCTC
from teaching_asr.ctc import (
    beam_search_decode,
    ctc_loss,
    greedy_decode,
    teaching_ctc_log_probability,
)
from teaching_asr.data import SyntheticSpeechDataset, collate_speech


def main() -> None:
    config = ASRConfig()
    model = ConformerCTC(config).eval()
    dataset = SyntheticSpeechDataset(size=2)
    batch = collate_speech([dataset[0], dataset[1]])
    print(f"config: {config}")
    print(f"output frame rate: {config.output_frame_rate} Hz (40 ms per encoder frame)")
    print(f"parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")
    print(f"waveforms [B, N]: {tuple(batch.waveforms.shape)}, lengths={batch.waveform_lengths.tolist()}")
    print(f"targets [B, U]: {tuple(batch.targets.shape)}, lengths={batch.target_lengths.tolist()}")
    with torch.no_grad():
        features, frame_lengths = model.frontend(batch.waveforms, batch.waveform_lengths)
        print(f"log-Mel [B, F, M]: {tuple(features.shape)}, frame_lengths={frame_lengths.tolist()}")
        encoded, output_lengths, mask = model.encoder(features, frame_lengths)
        print(f"encoder [B, T, D]: {tuple(encoded.shape)}, output_lengths={output_lengths.tolist()}")
        print(f"attention mask [B, T]: {tuple(mask.shape)}, valid={mask.sum(dim=1).tolist()}")
        logits = model.ctc_projection(encoded)
        print(f"CTC logits [B, T, V]: {tuple(logits.shape)}, vocabulary={logits.shape[-1]}")
        losses = ctc_loss(
            logits,
            batch.targets,
            output_lengths,
            batch.target_lengths,
            model.tokenizer.blank_id,
            reduction="none",
        )
        print(f"per-sample CTC losses: {losses.tolist()}")
        index = 0
        target_length = int(batch.target_lengths[index])
        log_probs = logits[index, : output_lengths[index]].log_softmax(dim=-1)
        teaching_log_prob = teaching_ctc_log_probability(
            log_probs, batch.targets[index, :target_length], model.tokenizer.blank_id
        )
        difference = abs(-teaching_log_prob.item() - losses[index].item())
        print(f"teaching forward DP vs PyTorch CTC |diff|: {difference:.8f}")
        if difference > 1e-4:
            raise RuntimeError("teaching CTC implementation disagrees with PyTorch")
        print(f"references: {batch.transcripts}")
        print(f"greedy hypotheses (untrained): {greedy_decode(logits, output_lengths, model.tokenizer)}")
        print(
            "beam hypotheses (untrained): "
            f"{beam_search_decode(logits, output_lengths, model.tokenizer)}"
        )


if __name__ == "__main__":
    main()
