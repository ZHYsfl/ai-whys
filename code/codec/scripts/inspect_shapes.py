import torch

from teaching_codec import CodecConfig, TeachingCodec
from teaching_codec.data import SyntheticAudioDataset


def main() -> None:
    config = CodecConfig()
    model = TeachingCodec(config).eval()
    x = SyntheticAudioDataset(size=1)[0].unsqueeze(0)
    print(f"config: {config}")
    print(f"hop_length = {' × '.join(map(str, config.strides))} = {config.hop_length}")
    print(f"frame_rate = {config.sample_rate} / {config.hop_length} = {config.frame_rate} frames/s")
    print(f"bitrate = {config.frame_rate} × {config.num_codebooks} × log2({config.codebook_size}) = {config.bitrate()} bit/s")
    with torch.no_grad():
        print(f"\nwaveform x: {tuple(x.shape)}")
        z = model.encoder(x)
        print(f"encoder latent z: {tuple(z.shape)}")

        residual = z
        quantized_sum = torch.zeros_like(z)
        all_codes = []
        for index, quantizer in enumerate(model.quantizer.quantizers):
            result = quantizer(residual)
            all_codes.append(result.codes)
            quantized_sum = quantized_sum + result.quantized
            residual = residual - result.quantized.detach()
            print(
                f"RVQ layer {index}: codes={tuple(result.codes.shape)}, "
                f"unique_codes={result.codes.unique().numel()}, "
                f"residual_MSE={residual.square().mean().item():.6f}"
            )

        print(f"sum of quantized vectors: {tuple(quantized_sum.shape)}")
        manual_reconstruction = model.decoder(quantized_sum)
        print(f"decoder reconstruction: {tuple(manual_reconstruction.shape)}")

        output = model(x)
        codes = model.encode(x)
        decoded = model.decode(codes)
        print(f"packaged codes [B, Q, F]: {tuple(codes.shape)}, dtype={codes.dtype}")
        print(f"forward vs decode(encode) max |diff|: {(output.reconstruction - decoded).abs().max().item():.8f}")
        print(f"manual RVQ vs packaged forward max |diff|: {(manual_reconstruction - output.reconstruction).abs().max().item():.8f}")


if __name__ == "__main__":
    main()
