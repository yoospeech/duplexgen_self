"""Render every turn from an LLM smoke JSON as stereo audio."""

import argparse
import json
from pathlib import Path

import torch
import torchaudio

CONFIG = (
    "hf://anak10thn/pocket-tts-indonesian/indonesian_6l.yaml"
    "@17257664e384561c957b02ac92edd1a24807f0e5"
)
USER_VOICE = "hf://anak10thn/pocket-tts-indonesian/voices/" "common_voice_id_19961910.mp3"
ASSISTANT_VOICE = "hf://anak10thn/pocket-tts-indonesian/voices/" "common_voice_id_20954552.mp3"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/indonesian-dialogue-smoke.wav")
    )
    args = parser.parse_args()

    from pocket_tts import TTSModel

    data = json.loads(args.input.read_text(encoding="utf-8"))
    turns = data["history"]
    model = TTSModel.load_model(config=CONFIG, eos_threshold=-6.0)
    user_state = model.get_state_for_audio_prompt(USER_VOICE, truncate=True)
    assistant_state = model.get_state_for_audio_prompt(ASSISTANT_VOICE, truncate=True)
    gap = torch.zeros(int(model.sample_rate * 0.25))
    stereo_parts = []
    for turn in turns:
        role = turn["role"]
        state = user_state if role == "user" else assistant_state
        audio = model.generate_audio(state, turn["content"], copy_state=True).cpu()
        silence = torch.zeros_like(audio)
        part = torch.stack([silence, audio]) if role == "user" else torch.stack([audio, silence])
        stereo_parts.extend([part, torch.stack([gap, gap])])
    stereo = torch.cat(stereo_parts[:-1], dim=1)
    if not torch.isfinite(stereo).all() or stereo.shape[-1] == 0:
        raise RuntimeError("Generated stereo waveform is invalid")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(args.output), stereo, model.sample_rate)
    duration = stereo.shape[-1] / model.sample_rate
    print(
        f"PASS output={args.output} sample_rate={model.sample_rate} "
        f"duration={duration:.2f}s channels={stereo.shape[0]}"
    )
    for turn in turns:
        print(f"{turn['role']}: {turn['content']}")


if __name__ == "__main__":
    main()
