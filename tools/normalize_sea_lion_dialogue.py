"""Extract a validated DuplexGen smoke JSON from llama.cpp console output."""

import argparse
import json
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw = args.input.read_text(encoding="utf-8")
    candidates = re.findall(r"\{[\s\S]*?\n\}", raw)
    if not candidates:
        raise ValueError("No JSON object found in llama.cpp output")
    generated = json.loads(candidates[-1])
    source_turns = generated.get("history") or generated.get("dialog")
    if not isinstance(source_turns, list) or len(source_turns) != 4:
        raise ValueError("SEA-LION output must contain exactly four turns")

    history = []
    expected_roles = ["user", "assistant", "user", "assistant"]
    for expected_role, turn in zip(expected_roles, source_turns):
        role = turn.get("role")
        content = turn.get("content") or turn.get("text")
        if role != expected_role or not isinstance(content, str) or not content.strip():
            raise ValueError(f"Invalid turn: {turn}")
        history.append({"role": role, "content": content.strip()})

    joined = " ".join(turn["content"].lower() for turn in history)
    for required in ("selasa", "jumat", "satu siang"):
        if required not in joined:
            raise ValueError(f"Required scenario fact missing: {required}")

    result = {
        "scenario": "reschedule interview",
        "history": history,
        "meta": {
            "language": "id",
            "model": "aisingapore/Llama-SEA-LION-v3-8B-IT-GGUF:Q2_K",
            "smoke_test": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"PASS output={args.output} turns={len(history)}")


if __name__ == "__main__":
    main()
