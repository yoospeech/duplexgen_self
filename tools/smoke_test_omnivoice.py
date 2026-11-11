#!/usr/bin/env python3
"""Download OmniVoice and verify a short voice-cloned waveform can be saved."""

import argparse
from pathlib import Path
import sys

import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tts_render.omnivoice_backend import OmniVoiceTTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="k2-fsa/OmniVoice")
    parser.add_argument("--voice_mode", choices=["auto", "clone"], default="auto")
    parser.add_argument("--ref_audio", default="")
    parser.add_argument("--ref_text", default="", help="Optional reference transcript.")
    parser.add_argument("--text", default="Selamat pagi, saya siap membantu Anda hari ini.")
    parser.add_argument("--output", type=Path, default=Path("outputs/omnivoice-smoke.wav"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_steps", type=int, default=16)
    args = parser.parse_args()

    dtype = "float16" if args.device != "cpu" else "float32"
    model = OmniVoiceTTS(
        args.model, args.device, dtype, args.num_steps, language="id", voice_mode=args.voice_mode
    )
    waveform = model.generate(args.text, args.ref_audio, args.ref_text).squeeze(0).numpy()
    if waveform.ndim != 1 or waveform.size == 0 or not np.isfinite(waveform).all():
        raise RuntimeError(f"Invalid OmniVoice waveform: shape={waveform.shape}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, waveform, 24000)
    duration = waveform.size / 24000
    print(
        f"PASS output={args.output} sample_rate=24000 duration={duration:.2f}s "
        f"peak={np.abs(waveform).max():.4f} device={args.device}"
    )


if __name__ == "__main__":
    main()
