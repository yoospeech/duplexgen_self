#!/usr/bin/env python3
"""Render every turn from an Indonesian dialogue JSON with OmniVoice."""

import argparse
import json
from pathlib import Path
import sys

import torch
import torchaudio

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tts_render.omnivoice_backend import OmniVoiceTTS

TAKE_FLOOR = "[TAKE_FLOOR]"


def _decisions(turn):
    return [item for item in turn.get("history", []) if "word_index" in item]


def _add_to_timeline(timeline, audio, channel, start):
    end = start + audio.shape[-1]
    if timeline.shape[-1] < end:
        timeline = torch.nn.functional.pad(timeline, (0, end - timeline.shape[-1]))
    timeline[channel, start:end] += audio.squeeze(0)
    return timeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--voice_mode", choices=["auto", "clone"], default="auto")
    parser.add_argument("--user_ref", default="")
    parser.add_argument("--user_ref_text", default="")
    parser.add_argument("--assistant_ref", default="")
    parser.add_argument("--assistant_ref_text", default="")
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/omnivoice-id-dialogue-smoke.wav")
    )
    parser.add_argument("--model", default="k2-fsa/OmniVoice")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_steps", type=int, default=16)
    parser.add_argument("--turn_gap", type=float, default=0.25)
    parser.add_argument("--interrupt_overlap", type=float, default=0.6)
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    turns = data["history"]
    dtype = "float16" if args.device != "cpu" else "float32"
    model = OmniVoiceTTS(
        args.model, args.device, dtype, args.num_steps, language="id", voice_mode=args.voice_mode
    )
    references = {
        "user": (args.user_ref, args.user_ref_text),
        "assistant": (args.assistant_ref, args.assistant_ref_text),
    }

    role_channel = {"assistant": 0, "user": 1}
    timeline = torch.zeros(2, 0)
    cursor = 0
    previous_take_floor = False
    event_meta = []
    for turn in turns:
        role = turn["role"]
        ref_audio, ref_text = references[role]
        text = turn["content"].replace(TAKE_FLOOR, "").strip(" ,")
        if previous_take_floor:
            start = max(0, cursor - round(model.sr * args.interrupt_overlap))
        else:
            start = cursor
        audio = model.generate(text, ref_audio, ref_text, speaker_id=role).cpu()
        timeline = _add_to_timeline(timeline, audio, role_channel[role], start)
        turn_end = start + audio.shape[-1]
        take_floor = TAKE_FLOOR in turn["content"] or any(
            d.get("decision") == "floor_taking" for d in _decisions(turn)
        )
        event_meta.append(
            {
                "type": "turn",
                "role": role,
                "text": text,
                "start": start / model.sr,
                "end": turn_end / model.sr,
                "take_floor": take_floor,
            }
        )

        word_count = max(1, len(text.split()))
        listener = "assistant" if role == "user" else "user"
        for decision in _decisions(turn):
            if decision.get("decision") != "backchannel":
                continue
            bc_text = decision.get("content") or "iya"
            bc_audio = model.generate(
                bc_text,
                references[listener][0],
                references[listener][1],
                speaker_id=listener,
            ).cpu()
            fraction = min(1.0, (decision["word_index"] + 1) / word_count)
            bc_start = start + round(fraction * audio.shape[-1])
            timeline = _add_to_timeline(timeline, bc_audio, role_channel[listener], bc_start)
            event_meta.append(
                {
                    "type": "backchannel",
                    "role": listener,
                    "text": bc_text,
                    "slot_word_index": decision["word_index"],
                    "start": bc_start / model.sr,
                    "end": (bc_start + bc_audio.shape[-1]) / model.sr,
                }
            )

        cursor = max(cursor, turn_end)
        if not take_floor:
            cursor += round(model.sr * args.turn_gap)
        previous_take_floor = take_floor
        print(f"{role}: {turn['content']}", flush=True)

    dialogue = timeline
    if dialogue.shape[0] != 2 or dialogue.shape[1] == 0 or not torch.isfinite(dialogue).all():
        raise RuntimeError(f"Invalid dialogue waveform: shape={tuple(dialogue.shape)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(args.output), dialogue, model.sr)
    meta_path = args.output.with_suffix(".json")
    meta_path.write_text(
        json.dumps(
            {
                "input": str(args.input),
                "sample_rate": model.sr,
                "channels": ["assistant", "user"],
                "voice_mode": args.voice_mode,
                "events": event_meta,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    duration = dialogue.shape[1] / model.sr
    print(
        f"PASS output={args.output} sample_rate={model.sr} "
        f"duration={duration:.2f}s channels={dialogue.shape[0]} turns={len(turns)} "
        f"events={len(event_meta)} meta={meta_path}"
    )


if __name__ == "__main__":
    main()
