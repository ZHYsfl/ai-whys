import torch

from teaching_asr import ASRConfig
from teaching_asr.features import LogMelSpectrogram, mel_filterbank


def test_config_round_trip(small_config: ASRConfig) -> None:
    restored = ASRConfig.from_dict(small_config.to_dict())
    assert restored == small_config
    assert restored.subsampled_mels == 3
    assert restored.output_frame_rate == 25


def test_mel_filterbank_shape(small_config: ASRConfig) -> None:
    filters = mel_filterbank(small_config)
    assert filters.shape == (16, 101)
    assert (filters >= 0).all()
    assert (filters.sum(dim=1) > 0).all()


def test_log_mel_shapes_and_lengths(small_config: ASRConfig) -> None:
    frontend = LogMelSpectrogram(small_config)
    waveforms = torch.randn(2, 1600)
    lengths = torch.tensor([1600, 1200])
    features, frame_lengths = frontend(waveforms, lengths)
    assert features.shape == (2, 18, 16)
    assert frame_lengths.tolist() == [18, 13]
    assert torch.count_nonzero(features[1, 13:]) == 0
    assert torch.isfinite(features).all()


def test_log_mel_batch_matches_single(small_config: ASRConfig) -> None:
    frontend = LogMelSpectrogram(small_config)
    waveforms = torch.randn(2, 1600)
    lengths = torch.tensor([1600, 1200])
    batched, batched_lengths = frontend(waveforms, lengths)
    for index, length in enumerate(lengths.tolist()):
        single, single_length = frontend(waveforms[index : index + 1, :length], lengths[index : index + 1])
        assert int(single_length[0]) == int(batched_lengths[index])
        torch.testing.assert_close(
            batched[index, : int(batched_lengths[index])],
            single[0, : int(batched_lengths[index])],
            atol=1e-5,
            rtol=1e-5,
        )


def test_short_waveform_gets_one_frame(small_config: ASRConfig) -> None:
    frontend = LogMelSpectrogram(small_config)
    features, lengths = frontend(torch.randn(1, 100), torch.tensor([100]))
    assert features.shape == (1, 1, 16)
    assert lengths.tolist() == [1]
    assert torch.isfinite(features).all()
