#!/usr/bin/env python3
"""Generate balanced Indonesian dialogues from domain names only."""

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.llm_client import make_client, no_thinking_extra_body

DEFAULT_DOMAINS = [
    "healthcare",
    "finance",
    "education",
    "travel",
    "shopping",
    "customer_service",
    "workplace",
    "technology",
    "public_services",
    "daily_life",
]


def balanced_domain_plan(domains: list[str], count: int, seed: int) -> list[str]:
    """Return shuffled cycles so domain counts differ by at most one."""
    if not domains:
        raise ValueError("At least one domain is required")
    rng = random.Random(seed)
    plan: list[str] = []
    while len(plan) < count:
        cycle = domains.copy()
        rng.shuffle(cycle)
        plan.extend(cycle)
    return plan[:count]


def parse_dialogue_response(
    raw: str,
    domain: str,
    min_turns: int,
    max_turns: int,
    opening_role: str | None = None,
) -> dict[str, Any]:
    """Extract and validate the generator's JSON object."""
    cleaned = raw.split("</think>")[-1].strip()
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        raise ValueError("response does not contain a JSON object")
    result = json.loads(match.group(0))

    topic = result.get("topic")
    scenario = result.get("scenario")
    history = result.get("history")
    if not isinstance(topic, str) or not topic.strip():
        raise ValueError("missing topic")
    if not isinstance(scenario, str) or not scenario.strip():
        raise ValueError("missing scenario")
    if not isinstance(history, list) or not min_turns <= len(history) <= max_turns:
        raise ValueError(f"history must contain {min_turns}-{max_turns} turns")

    first_role = opening_role or str(history[0].get("role", "")).strip().lower()
    if first_role not in {"user", "assistant"}:
        first_role = "user"

    normalized = []
    for index, turn in enumerate(history):
        if not isinstance(turn, dict):
            raise ValueError("every turn must be an object")
        if index % 2 == 0:
            expected_role = first_role
        else:
            expected_role = "assistant" if first_role == "user" else "user"
        content = turn.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"turn {index} has empty content")
        # Either participant may open the conversation. Once the first speaker
        # is known, turn position is authoritative: normalize duplicated or
        # missing labels instead of discarding an otherwise valid dialogue.
        normalized.append({"role": expected_role, "content": content.strip()})

    return {
        "scenario": scenario.strip(),
        "context": scenario.strip(),
        "history": normalized,
        "meta": {"language": "id", "domain": domain, "topic": topic.strip(), "synthetic": True},
    }


def build_messages(
    domain: str,
    min_turns: int,
    max_turns: int,
    recent_topics: list[str],
    opening_role: str,
) -> list[dict[str, str]]:
    avoid = ", ".join(recent_topics[-30:]) if recent_topics else "none"
    opening_instruction = (
        "Start with user making a request or raising the topic."
        if opening_role == "user"
        else "Start with assistant giving a short, natural greeting before the user responds."
    )
    return [
        {
            "role": "system",
            "content": (
                "You create diverse, realistic two-person spoken dialogues for speech-data "
                "generation. Write natural Bahasa Indonesia. Return only one JSON object; "
                "do not use markdown or explanations. Do not include unsafe instructions, "
                "private data, brand promotion, or claims that require professional certainty."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Domain: {domain}\n"
                "Invent a specific topic and situation inside this domain, then write a complete "
                "spoken dialogue. Vary goals, roles, setting, formality, emotion, and difficulty. "
                f"Avoid topics already used: {avoid}.\n"
                f"Use an even number of turns between {min_turns} and {max_turns}. "
                f"{opening_instruction} After the opening turn, alternate user/assistant exactly. "
                "Keep each turn concise but substantive. "
                "Use only natural Indonesian in content.\n"
                'Schema: {"topic": string, "scenario": string, "history": '
                '[{"role": "user"|"assistant", "content": string}]}'
            ),
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", default="synthetic")
    parser.add_argument("--split", default="train")
    parser.add_argument("--domains", default=",".join(DEFAULT_DOMAINS))
    parser.add_argument("--num-dialogues", type=int, default=12)
    parser.add_argument("--min-turns", type=int, default=6)
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="aisingapore/Qwen-SEA-LION-v4-32B-IT-4BIT")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    args = parser.parse_args()

    domains = list(dict.fromkeys(part.strip() for part in args.domains.split(",") if part.strip()))
    if args.num_dialogues < 1:
        raise SystemExit("--num-dialogues must be at least 1")
    if args.min_turns < 2 or args.max_turns < args.min_turns:
        raise SystemExit("invalid turn range")
    if args.min_turns % 2 or args.max_turns % 2:
        raise SystemExit("--min-turns and --max-turns must both be even")

    output_dir = args.output_root / f"text_dialogue_{args.dataset}" / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob("synthetic_*.json"))
    recent_topics: list[str] = []
    for path in existing:
        try:
            recent_topics.append(json.loads(path.read_text(encoding="utf-8"))["meta"]["topic"])
        except (KeyError, json.JSONDecodeError, OSError):
            continue

    client = make_client(args.model, args.api_key, args.base_url)
    plan = balanced_domain_plan(domains, args.num_dialogues, args.seed)
    generated = skipped = 0
    opening_rng = random.Random(args.seed)
    for index, domain in enumerate(plan):
        opening_role = opening_rng.choice(["user", "assistant"])
        output_path = output_dir / f"synthetic_{index:06d}.json"
        if output_path.exists():
            skipped += 1
            continue

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=args.model,
                    messages=build_messages(
                        domain,
                        args.min_turns,
                        args.max_turns,
                        recent_topics,
                        opening_role,
                    ),
                    temperature=min(1.0, 0.8 + attempt * 0.1),
                    max_tokens=1400,
                    response_format={"type": "json_object"},
                    extra_body=no_thinking_extra_body(args.model),
                )
                raw = response.choices[0].message.content or ""
                dialogue = parse_dialogue_response(
                    raw,
                    domain,
                    args.min_turns,
                    args.max_turns,
                    opening_role=opening_role,
                )
                dialogue["meta"].update(
                    {
                        "model": args.model,
                        "seed": args.seed,
                        "index": index,
                        "opening_role": opening_role,
                    }
                )
                output_path.write_text(
                    json.dumps(dialogue, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
                recent_topics.append(dialogue["meta"]["topic"])
                generated += 1
                print(f"[{index + 1}/{args.num_dialogues}] {domain}: {dialogue['meta']['topic']}")
                break
            except Exception as exc:  # LLM/schema failures are retried with a higher temperature.
                last_error = exc
        else:
            raise RuntimeError(f"failed to generate dialogue {index} ({domain}): {last_error}")

    print(f"Complete: generated={generated} skipped={skipped} output={output_dir}")


if __name__ == "__main__":
    main()
