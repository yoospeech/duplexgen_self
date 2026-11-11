"""OmniVoice adapter shared by the Stage 5 renderer and smoke test."""

from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch


class OmniVoiceTTS:
    """Expose k2-fsa OmniVoice through the renderer's TTS interface."""

    uses_cumulative_prompt = False
    supports_speaker_id = True

    def __init__(
        self,
        model_name: str,
        device: str,
        dtype: str,
        num_steps: int,
        language: str = "en",
        prompt_cache_size: int = 128,
        voice_mode: str = "auto",
    ):
        try:
            from omnivoice import OmniVoice
        except ImportError as exc:
            raise RuntimeError("OmniVoice backend requires `pip install omnivoice`.") from exc

        try:
            torch_dtype = getattr(torch, dtype)
        except AttributeError as exc:
            raise ValueError(f"Unsupported OmniVoice dtype: {dtype}") from exc

        self.model = OmniVoice.from_pretrained(
            model_name,
            device_map=device,
            dtype=torch_dtype,
        )
        self.num_steps = num_steps
        self.language = language
        self.prompt_cache_size = prompt_cache_size
        if voice_mode not in {"auto", "clone"}:
            raise ValueError(f"Unsupported OmniVoice voice mode: {voice_mode}")
        self.voice_mode = voice_mode
        self.requires_prompt = voice_mode == "clone"
        self._prompt_cache = OrderedDict()
        self._auto_speaker_prompts = {}
        self.sr = 24000

    def reset_auto_speakers(self):
        """Choose fresh Auto voices for the next dialogue or variant."""
        self._auto_speaker_prompts.clear()

    @staticmethod
    def _sidecar_text(audio_prompt_path: str) -> str:
        transcript_path = Path(audio_prompt_path).with_suffix(".txt")
        if transcript_path.is_file():
            return transcript_path.read_text(encoding="utf-8").strip()
        return ""

    def _voice_prompt(self, audio_prompt_path: str, ref_text: str = ""):
        path = str(Path(audio_prompt_path).resolve())
        transcript = ref_text or self._sidecar_text(path)
        key = (path, transcript)
        if key in self._prompt_cache:
            self._prompt_cache.move_to_end(key)
            return self._prompt_cache[key]

        prompt = self.model.create_voice_clone_prompt(
            ref_audio=path,
            ref_text=transcript or None,
        )
        self._prompt_cache[key] = prompt
        if len(self._prompt_cache) > self.prompt_cache_size:
            self._prompt_cache.popitem(last=False)
        return prompt

    def generate(self, text: str, audio_prompt_path=None, ref_text: str = "", speaker_id=0):
        generate_args = {
            "text": text,
            "language": self.language,
            "num_step": self.num_steps,
        }
        if self.voice_mode == "clone":
            if not audio_prompt_path:
                raise ValueError("OmniVoice clone mode requires an audio prompt.")
            generate_args["voice_clone_prompt"] = self._voice_prompt(audio_prompt_path, ref_text)
        elif speaker_id in self._auto_speaker_prompts:
            generate_args["voice_clone_prompt"] = self._auto_speaker_prompts[speaker_id]
        audio = self.model.generate(**generate_args)
        if not audio:
            raise RuntimeError("OmniVoice returned no audio.")
        waveform = np.asarray(audio[0], dtype=np.float32)
        if waveform.ndim != 1 or waveform.size == 0:
            raise RuntimeError(
                f"OmniVoice returned an invalid waveform with shape {waveform.shape}."
            )
        if self.voice_mode == "auto" and speaker_id not in self._auto_speaker_prompts:
            # Bootstrap each dialogue channel with one prompt-free Auto voice,
            # then clone that generated voice for every later turn on the channel.
            self._auto_speaker_prompts[speaker_id] = self.model.create_voice_clone_prompt(
                ref_audio=(waveform, self.sr),
                ref_text=text,
            )
        return torch.from_numpy(waveform).unsqueeze(0)
