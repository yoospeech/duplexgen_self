import importlib.util
from pathlib import Path

import pytest
import torch

pytest.importorskip("torchaudio")


_SCRIPT = Path(__file__).with_name("render_smoke_dialogue_omnivoice.py")
_SPEC = importlib.util.spec_from_file_location("render_smoke_dialogue_omnivoice", _SCRIPT)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_add_to_timeline = _MODULE._add_to_timeline
_decisions = _MODULE._decisions


def test_decisions_returns_only_slot_annotations():
    turn = {
        "history": [
            {"full_content": "halo dunia"},
            {"word_index": 0, "decision": "backchannel", "content": "iya"},
        ]
    }
    assert _decisions(turn) == [{"word_index": 0, "decision": "backchannel", "content": "iya"}]


def test_timeline_mixes_overlapping_audio_on_selected_channel():
    timeline = torch.zeros(2, 0)
    timeline = _add_to_timeline(timeline, torch.ones(1, 4), channel=1, start=2)
    timeline = _add_to_timeline(timeline, torch.ones(1, 3), channel=1, start=4)

    assert timeline.shape == (2, 7)
    assert timeline[0].sum() == 0
    assert timeline[1].tolist() == [0, 0, 1, 1, 2, 2, 1]
