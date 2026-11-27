import torch

from teaching_moshi.data import SyntheticDialogueDataset, collate_dialogue
from teaching_moshi.inference import MoshiDialogue
from teaching_moshi.mimi import MimiModel
from teaching_moshi.model import MoshiGenerator, MoshiLM
from teaching_moshi.text import ByteTokenizer


def main() -> None:
    torch.manual_seed(0)
    mimi = MimiModel().eval()
    model = MoshiLM().eval()
    batch = collate_dialogue([SyntheticDialogueDataset()[0], SyntheticDialogueDataset()[1]])
    with torch.no_grad():
        output = model(batch.codes)
        waveform = torch.randn(2, 1, 2 * mimi.frame_size).clamp(-1, 1)
        codes = mimi.encode(waveform)
        reconstruction = mimi.decode(codes)
    print("joint codes:", tuple(batch.codes.shape))
    print("text logits:", tuple(output.text_logits.shape))
    print("audio logits:", tuple(output.logits.shape))
    print("Mimi codes:", tuple(codes.shape))
    print("Mimi reconstruction:", tuple(reconstruction.shape))

    generator = MoshiGenerator(model, use_sampling=False)
    generated = []
    with generator.streaming(2):
        for index in range(batch.codes.shape[-1]):
            tokens = generator.step(batch.codes[:, model.config.user_audio_offset :, index : index + 1])
            if tokens is not None:
                generated.append(tokens)
    print("generated text + audio:", tuple(torch.cat(generated, dim=-1).shape))

    dialogue = MoshiDialogue(mimi, model, ByteTokenizer(), use_sampling=False)
    response = dialogue.respond(waveform[:1])
    print("end-to-end response audio:", tuple(response.waveform.shape))


if __name__ == "__main__":
    main()
