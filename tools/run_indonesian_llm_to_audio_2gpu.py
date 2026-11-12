#!/usr/bin/env python3
"""Run Indonesian dialogue generation/translation through two-GPU OmniVoice."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOMAINS = (
    "healthcare,finance,education,travel,shopping,customer_service,"
    "workplace,technology,public_services,daily_life"
)
SOURCE_DATASETS = {
    "interviewer",
    "multiwoz",
    "negotiator",
    "persuader",
    "socraticlm",
    "soda",
}


def run_logged(command: list[str], log_path: Path, env: dict[str, str] | None = None) -> None:
    """Run one stage while copying combined output to the console and a log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def check_endpoint(base_url: str, api_key: str) -> None:
    request = urllib.request.Request(base_url.rstrip("/") + "/models")
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            if response.status >= 400:
                raise RuntimeError(f"HTTP {response.status}")
    except Exception as exc:
        raise RuntimeError(f"LLM endpoint is not ready: {request.full_url} ({exc})") from exc


def check_interpreters(llm_python: Path, tts_python: Path, gpu_count: int) -> None:
    for interpreter in (llm_python, tts_python):
        if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
            raise FileNotFoundError(f"Python interpreter not found: {interpreter}")

    subprocess.run([str(llm_python), "-c", "import openai"], cwd=REPO_ROOT, check=True)
    tts_check = f"""
import json
import torch
import omnivoice, silero_vad, whisperx, nltk, nemo_text_processing
from transformers import HiggsAudioV2TokenizerModel
status = {{
    "torch": torch.__version__,
    "cuda": torch.cuda.is_available(),
    "gpu_count": torch.cuda.device_count(),
}}
print(json.dumps(status))
if not status["cuda"] or status["gpu_count"] < {gpu_count}:
    raise SystemExit(
        f"Need {gpu_count} CUDA GPUs, but torch={{status['torch']}} "
        f"sees {{status['gpu_count']}}"
    )
"""
    subprocess.run([str(tts_python), "-c", tts_check], cwd=REPO_ROOT, check=True)


def resolve_omnivoice_model(tts_python: Path, model: str, hf_home: str | None) -> str:
    """Download a Hub model once before workers start, or validate a local path."""
    local_path = Path(model).expanduser()
    if local_path.exists():
        if not (local_path / "config.json").is_file():
            raise FileNotFoundError(f"OmniVoice config.json missing under {local_path}")
        return str(local_path.resolve())

    env = os.environ.copy()
    if hf_home:
        env["HF_HOME"] = hf_home
    code = "from huggingface_hub import snapshot_download; " f"print(snapshot_download({model!r}))"
    print(f"Downloading/verifying OmniVoice once: {model}", flush=True)
    result = subprocess.run(
        [str(tts_python), "-c", code],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    snapshot = result.stdout.strip().splitlines()[-1]
    if not snapshot or not (Path(snapshot) / "config.json").is_file():
        raise RuntimeError(f"Invalid OmniVoice snapshot path: {snapshot!r}")
    print(f"OmniVoice snapshot: {snapshot}", flush=True)
    return snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate or translate Indonesian dialogues, annotate turn-taking, "
        "add backchannels, and render with two or more OmniVoice GPUs."
    )
    parser.add_argument("--mode", choices=["generate", "translate"], default="generate")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--domains", default=DEFAULT_DOMAINS)
    parser.add_argument("--num-dialogues", type=int, default=12)
    parser.add_argument("--test-dialogues", type=int, default=1)
    parser.add_argument("--data-root", default="")
    parser.add_argument("--source-input-path", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--gpus", default="0,1", help="Physical GPU IDs, comma separated.")
    parser.add_argument("--num-steps", type=int, default=32)
    parser.add_argument("--pause-before-tts", action="store_true")
    parser.add_argument(
        "--llm-python",
        type=Path,
        default=Path("/home/ysw/workspace/anaconda3/envs/duplexgen/bin/python"),
    )
    parser.add_argument(
        "--tts-python", type=Path, default=REPO_ROOT / ".venv-tts-omnivoice/bin/python"
    )
    parser.add_argument("--llm-model", default="aisingapore/Llama-SEA-LION-v3-8B-IT")
    parser.add_argument("--llm-base-url", default="http://localhost:8000/v1")
    parser.add_argument("--llm-api-key", default="EMPTY")
    parser.add_argument("--omnivoice-model", default="k2-fsa/OmniVoice")
    parser.add_argument("--hf-home", default=os.getenv("HF_HOME", ""))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_dialogues < 1:
        raise SystemExit("--num-dialogues must be at least 1")
    if args.num_steps < 1:
        raise SystemExit("--num-steps must be at least 1")

    dataset = args.dataset or ("synthetic" if args.mode == "generate" else "interviewer")
    if args.mode == "generate" and dataset != "synthetic":
        raise SystemExit("generate mode requires --dataset synthetic")
    if args.mode == "translate" and dataset not in SOURCE_DATASETS:
        raise SystemExit(f"translate dataset must be one of: {sorted(SOURCE_DATASETS)}")

    gpu_ids = [part.strip() for part in args.gpus.split(",") if part.strip()]
    if len(gpu_ids) < 1 or len(set(gpu_ids)) != len(gpu_ids):
        raise SystemExit("--gpus must contain distinct GPU IDs")

    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_root = REPO_ROOT / "outputs/runs" / run_id
    stage_input = run_root / "stage-input"
    stage_tt = run_root / "with-tt"
    stage_bc = run_root / "with-tt-bc"
    audio_output = run_root / "audio"
    log_dir = run_root / "logs"
    for path in (stage_input, stage_tt, stage_bc, audio_output, log_dir):
        path.mkdir(parents=True, exist_ok=True)

    check_interpreters(args.llm_python, args.tts_python, len(gpu_ids))
    check_endpoint(args.llm_base_url, args.llm_api_key)
    omnivoice_model = resolve_omnivoice_model(
        args.tts_python, args.omnivoice_model, args.hf_home or None
    )

    if args.mode == "generate":
        stage0 = [
            str(args.llm_python),
            "tools/generate_indonesian_domain_dialogues.py",
            "--output-root",
            str(stage_input),
            "--dataset",
            dataset,
            "--split",
            args.split,
            "--domains",
            args.domains,
            "--num-dialogues",
            str(args.num_dialogues),
            "--model",
            args.llm_model,
            "--base-url",
            args.llm_base_url,
            "--api-key",
            args.llm_api_key,
        ]
        print("[1/4] Generate balanced domain dialogues", flush=True)
        run_logged(stage0, log_dir / "stage0-domain-generation.log")
    else:
        stage1 = [
            str(args.llm_python),
            "-m",
            "src.speechify_run",
            "--dataset",
            dataset,
            "--save_dir",
            str(stage_input),
            "--language",
            "id",
            "--llm_model_name",
            args.llm_model,
            "--base_url",
            args.llm_base_url,
            "--api_key",
            args.llm_api_key,
            "--max_train_samples",
            str(args.num_dialogues),
            "--max_test_samples",
            str(args.test_dialogues),
        ]
        if args.data_root:
            stage1.extend(["--data-root", args.data_root])
        if args.source_input_path:
            stage1.extend(["--input_path", args.source_input_path])
        print(f"[1/4] Translate {dataset} into spoken Indonesian", flush=True)
        run_logged(stage1, log_dir / "stage1-corpus-translation.log")

    common_llm = ["--base_url", args.llm_base_url, "--api_key", args.llm_api_key]
    stage4 = [
        str(args.llm_python),
        "-m",
        "src.synthesis.run",
        "--dataset",
        dataset,
        "--split",
        args.split,
        "--input_root",
        str(stage_input),
        "--save_root",
        str(stage_tt),
        "--annotate_only",
        "--language",
        "id",
        "--llm_model_name",
        args.llm_model,
        *common_llm,
        "--boundary_model_name",
        args.llm_model,
        "--boundary_base_url",
        args.llm_base_url,
        "--boundary_api_key",
        args.llm_api_key,
        "--tt_model_name",
        args.llm_model,
        "--tt_base_url",
        args.llm_base_url,
        "--tt_api_key",
        args.llm_api_key,
        "--max_dialogues",
        str(args.num_dialogues),
    ]
    print("[2/4] Infer slots and turn-taking", flush=True)
    run_logged(stage4, log_dir / "stage4-turn-taking.log")

    backchannel = [
        str(args.llm_python),
        "-m",
        "src.synthesis.run_add_bc",
        "--dataset",
        dataset,
        "--split",
        args.split,
        "--input_root",
        str(stage_tt),
        "--output_root",
        str(stage_bc),
        "--model_name",
        args.llm_model,
        "--base_url",
        args.llm_base_url,
        "--api_key",
        args.llm_api_key,
        "--language",
        "id",
    ]
    print("[3/4] Generate Indonesian backchannels", flush=True)
    run_logged(backchannel, log_dir / "stage4-backchannel.log")

    input_glob = str(stage_bc / f"text_dialogue_{dataset}" / args.split / "*.json")
    if not list((stage_bc / f"text_dialogue_{dataset}" / args.split).glob("*.json")):
        raise RuntimeError(f"No Stage 4 JSON files found: {input_glob}")

    if args.pause_before_tts:
        input("Stop the local LLM server, verify that the GPUs are free, then press Enter...")

    print(f"[4/4] Render OmniVoice on GPUs {','.join(gpu_ids)}", flush=True)
    processes: list[tuple[str, subprocess.Popen[bytes], object]] = []
    for shard, gpu_id in enumerate(gpu_ids):
        log_path = log_dir / f"omnivoice-shard-{shard}.log"
        log_file = log_path.open("wb")
        command = [
            str(args.tts_python),
            "tts_render/convert_spoken.py",
            "--tts_backend",
            "omnivoice",
            "--omnivoice_voice_mode",
            "auto",
            "--language",
            "id",
            "--align_model",
            "m-salman-a/wav2vec2-xlsr-53-common-voice-indonesian",
            "--omnivoice_model",
            omnivoice_model,
            "--omnivoice_device",
            "cuda:0",
            "--omnivoice_align_device",
            "cuda",
            "--omnivoice_dtype",
            "float16",
            "--omnivoice_num_steps",
            str(args.num_steps),
            "--input_glob",
            input_glob,
            "--save_dir",
            str(audio_output),
            "--num_variants",
            "1",
            "--num_shards",
            str(len(gpu_ids)),
            "--shard_id",
            str(shard),
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        if args.hf_home:
            env["HF_HOME"] = args.hf_home
        process = subprocess.Popen(
            command, cwd=REPO_ROOT, env=env, stdout=log_file, stderr=subprocess.STDOUT
        )
        processes.append((gpu_id, process, log_file))
        print(f"  GPU {gpu_id}: shard {shard}, log {log_path}", flush=True)

    failures = []
    for gpu_id, process, log_file in processes:
        return_code = process.wait()
        log_file.close()
        if return_code:
            failures.append((gpu_id, return_code))
    if failures:
        raise RuntimeError(f"OmniVoice workers failed: {failures}; inspect {log_dir}")

    manifest = {
        "run_id": run_id,
        "mode": args.mode,
        "dataset": dataset,
        "domains": args.domains if args.mode == "generate" else None,
        "requested_dialogues": args.num_dialogues,
        "gpus": gpu_ids,
        "omnivoice_model": omnivoice_model,
        "output": str(audio_output),
    }
    (run_root / "run.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Complete: {run_root}")


if __name__ == "__main__":
    try:
        main()
    except (subprocess.CalledProcessError, OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
