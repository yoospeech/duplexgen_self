import sys
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _load_adapter():
    """Import the lightweight adapter without Stage 5's alignment dependencies."""
    from omnivoice_backend import OmniVoiceTTS

    return OmniVoiceTTS


class _FakeModel:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def create_voice_clone_prompt(self, **kwargs):
        self.calls.append(("prompt", kwargs))
        return "cached-prompt"

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.output


def _install_fake_omnivoice(monkeypatch, output):
    fake_model = _FakeModel(output)

    class FakeOmniVoice:
        load_args = None

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            cls.load_args = (args, kwargs)
            return fake_model

    monkeypatch.setitem(sys.modules, "omnivoice", types.SimpleNamespace(OmniVoice=FakeOmniVoice))
    return FakeOmniVoice, fake_model


def test_adapter_loads_official_model_api_and_returns_mono_float_tensor(monkeypatch):
    adapter_class = _load_adapter()
    fake_class, fake_model = _install_fake_omnivoice(
        monkeypatch, [np.array([0.25, -0.5], dtype=np.float64)]
    )

    adapter = adapter_class("k2-fsa/OmniVoice", "cuda:0", "float16", 16, "en", voice_mode="clone")
    waveform = adapter.generate("hello", "prompt.wav")

    assert fake_class.load_args == (
        ("k2-fsa/OmniVoice",),
        {"device_map": "cuda:0", "dtype": torch.float16},
    )
    assert fake_model.calls[0][0] == "prompt"
    assert fake_model.calls[0][1]["ref_audio"].endswith("prompt.wav")
    assert fake_model.calls[1] == {
        "text": "hello",
        "language": "en",
        "voice_clone_prompt": "cached-prompt",
        "num_step": 16,
    }
    assert adapter.sr == 24000
    assert waveform.shape == (1, 2)
    assert waveform.dtype == torch.float32


def test_adapter_passes_reference_transcript_when_provided(monkeypatch):
    adapter_class = _load_adapter()
    _, fake_model = _install_fake_omnivoice(monkeypatch, [np.ones(2)])
    adapter = adapter_class("model", "cpu", "float32", 32, voice_mode="clone")

    adapter.generate("selamat pagi", "prompt.wav", "teks referensi")

    assert fake_model.calls[0][0] == "prompt"
    assert fake_model.calls[0][1]["ref_text"] == "teks referensi"
    assert fake_model.calls[1] == {
        "text": "selamat pagi",
        "language": "en",
        "voice_clone_prompt": "cached-prompt",
        "num_step": 32,
    }


def test_adapter_reuses_encoded_voice_prompt(monkeypatch):
    adapter_class = _load_adapter()
    _, fake_model = _install_fake_omnivoice(monkeypatch, [np.ones(2)])
    adapter = adapter_class("model", "cpu", "float32", 16, "id", voice_mode="clone")

    adapter.generate("satu", "prompt.wav", "referensi")
    adapter.generate("dua", "prompt.wav", "referensi")

    prompt_calls = [call for call in fake_model.calls if isinstance(call, tuple)]
    assert len(prompt_calls) == 1
    assert fake_model.calls[-1]["language"] == "id"


def test_auto_mode_bootstraps_and_reuses_one_voice_per_speaker(monkeypatch):
    adapter_class = _load_adapter()
    _, fake_model = _install_fake_omnivoice(monkeypatch, [np.ones(2)])
    adapter = adapter_class("model", "cpu", "float32", 16, "id", voice_mode="auto")

    adapter.generate("selamat pagi", speaker_id="user")
    adapter.generate("apa kabar", speaker_id="user")
    adapter.generate("halo", speaker_id="assistant")

    assert fake_model.calls[0] == {"text": "selamat pagi", "language": "id", "num_step": 16}
    assert fake_model.calls[1][0] == "prompt"
    assert fake_model.calls[1][1]["ref_text"] == "selamat pagi"
    assert fake_model.calls[2] == {
        "text": "apa kabar",
        "language": "id",
        "num_step": 16,
        "voice_clone_prompt": "cached-prompt",
    }
    assert fake_model.calls[3] == {"text": "halo", "language": "id", "num_step": 16}
    assert fake_model.calls[4][0] == "prompt"
    assert adapter.requires_prompt is False


@pytest.mark.parametrize("output", [[], [np.empty(0)], [np.zeros((2, 2))]])
def test_adapter_rejects_missing_or_invalid_audio(monkeypatch, output):
    adapter_class = _load_adapter()
    _install_fake_omnivoice(monkeypatch, output)
    adapter = adapter_class("model", "cpu", "float32", 32)

    with pytest.raises(RuntimeError, match="no audio|invalid waveform"):
        adapter.generate("hello", "prompt.wav")
