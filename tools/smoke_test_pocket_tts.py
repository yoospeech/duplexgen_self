"""Generate one Indonesian sentence and validate the resulting waveform."""

import argparse
import math
from pathlib import Path

import torch
import torchaudio

DEFAULT_CONFIG = (
    "hf://anak10thn/pocket-tts-indonesian/indonesian_6l.yaml"
    "@17257664e384561c957b02ac92edd1a24807f0e5"
)
DEFAULT_VOICE = "hf://anak10thn/pocket-tts-indonesian/voices/" "common_voice_id_19961910.mp3"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--voice", default=DEFAULT_VOICE)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("pocket-id-smoke.wav"))
    parser.add_argument(
        "--text",
        default="Selamat pagi, saya siap membantu Anda hari ini.",
    )
    args = parser.parse_args()

    from pocket_tts import TTSModel

    model = TTSModel.load_model(config=args.config, eos_threshold=-6.0)
    model.to(args.device)
    state = model.get_state_for_audio_prompt(args.voice, truncate=True)
    audio = model.generate_audio(state, args.text, copy_state=False).detach().cpu()

    if audio.ndim != 1 or audio.numel() == 0:
        raise RuntimeError(f"Unexpected waveform shape: {tuple(audio.shape)}")
    if not torch.isfinite(audio).all():
        raise RuntimeError("Waveform contains NaN or infinity")
    duration = audio.numel() / model.sample_rate
    if not math.isfinite(duration) or duration < 0.5:
        raise RuntimeError(f"Generated audio is unexpectedly short: {duration:.2f}s")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(args.output), audio.unsqueeze(0), model.sample_rate)
    print(
        f"PASS output={args.output} sample_rate={model.sample_rate} "
        f"duration={duration:.2f}s peak={audio.abs().max().item():.3f}"
    )


if __name__ == "__main__":
    main()
