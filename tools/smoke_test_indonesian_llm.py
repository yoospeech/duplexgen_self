"""Generate and validate one small Indonesian dialogue with an open-weight LLM."""

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_dialogue(text: str) -> list[dict[str, str]]:
    text = text.split("</think>")[-1].strip()
    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        raise ValueError(f"Model did not return a JSON array: {text}")
    turns = json.loads(match.group(0))
    if not isinstance(turns, list) or len(turns) < 2:
        raise ValueError("Dialogue must contain at least two turns")
    role_aliases = {
        "user": "user",
        "pengguna": "user",
        "assistant": "assistant",
        "asisten": "assistant",
    }
    for turn in turns:
        if set(turn) != {"role", "content"}:
            raise ValueError(f"Unexpected turn schema: {turn}")
        turn["role"] = role_aliases.get(str(turn["role"]).lower(), turn["role"])
        if turn["role"] not in {"user", "assistant"} or not turn["content"].strip():
            raise ValueError(f"Invalid turn: {turn}")
    return turns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output", type=Path, default=Path("outputs/indonesian-llm-smoke.json"))
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto", device_map="cpu")
    messages = [
        {
            "role": "system",
            "content": (
                "Buat dialog lisan singkat dalam Bahasa Indonesia. Gunakan hanya fakta "
                "yang diberikan. Keluarkan hanya array JSON dengan kunci role dan content."
            ),
        },
        {
            "role": "user",
            "content": (
                "Skenario: pengguna ingin memindahkan jadwal wawancara dari Selasa pagi "
                "ke Jumat siang. Asisten memeriksa lalu mengonfirmasi bahwa Jumat pukul "
                "satu siang tersedia. Buat tepat empat giliran, dimulai oleh pengguna."
            ),
        },
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    )
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    prompt_length = inputs["input_ids"].shape[-1]
    response = tokenizer.decode(output[0, prompt_length:], skip_special_tokens=True)
    turns = extract_dialogue(response)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "scenario": "reschedule interview",
                "history": turns,
                "meta": {"language": "id", "model": args.model, "smoke_test": True},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    expected_roles = ["user", "assistant", "user", "assistant"]
    actual_roles = [turn["role"] for turn in turns]
    quality = "PASS" if actual_roles == expected_roles else "WARN_ROLE_ORDER"
    print(f"{quality} output={args.output} turns={len(turns)} model={args.model}")
    for turn in turns:
        print(f"{turn['role']}: {turn['content']}")


if __name__ == "__main__":
    main()
