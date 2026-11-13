# Official Implementation of "DuplexGen: Adaptive Synthesis of Human–AI Turn-Taking Dialogues" (EMNLP 2026 Oral)

English · [Indonesian setup](README.id.md)

[Takyoung Kim](https://youngerous.github.io/)<sup>&ast;</sup>,
[Kang-wook Kim](https://kwkim.me/)<sup>&ast;</sup>,
[Sang Hoon Woo](https://tonywoo.me/),
[Julia Hirschberg](https://www.cs.columbia.edu/~julia/),
[Gunhee Kim](https://vision.snu.ac.kr/gunhee/),
[Dilek Hakkani-Tür](https://siebelschool.illinois.edu/about/people/faculty/dilek)

<sup>&ast;</sup>Equal contribution

[![arXiv](https://img.shields.io/badge/arXiv-2607.26178-b31b1b.svg)](https://arxiv.org/abs/2607.26178)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

This repository contains the **data-generation pipeline for DuplexGen**, including text-dialogue collection, spoken-style conversion, turn-taking slot identification, turn-taking prediction, turn-taking dialogue generation, and speech synthesis with Chatterbox.

📄 [Paper](https://arxiv.org/abs/2607.26178) · 💻 [Code](https://github.com/duplexgen/duplexgen-code) · 🎛️ [Finetune](https://github.com/duplexgen/personaplex-finetune) · 🤗 [Corpus](https://huggingface.co/datasets/DuplexGen/duplexgen-corpus) · 🤗 [Spoken](https://huggingface.co/datasets/DuplexGen/duplexgen-spoken) · 🌐 [Demo](https://duplexgen.github.io/)

## Pipeline overview

| Stage | What it does | Docs |
|---|---|---|
| 1. Spoken-style conversion | Rewrite clean text dialogues from six scenario datasets into spoken-style transcripts. | [stage1-speechify](docs/stage1-speechify.md) |
| 2. Slot identification | Detect candidate intra-utterance action points via heuristic + LLM boundary detection. | [stage2-slots](docs/stage2-slots.md) |
| 3. Turn-taking prediction | Train or run a turn-taking predictor over the identified slots. | [stage3-prediction](docs/stage3-prediction.md) |
| 4. Turn-taking dialogue generation | Query the predictor at each slot and insert the chosen behavior, then filter role-confused outputs. | [stage4-generation](docs/stage4-generation.md) |
| 5. TTS rendering | Render the result to two-channel audio with Chatterbox, Pocket TTS, or OmniVoice. | [stage5-tts](docs/stage5-tts.md) |

Also: [using the released corpus](docs/CORPUS.md) ·
[troubleshooting](docs/TROUBLESHOOTING.md) ·
[data licenses](docs/DATA_LICENSES.md)

## Setup

### Local SEA-LION 32B 4-bit → OmniVoice (no LLM server)

The shell runner now defaults to in-process Hugging Face inference. It downloads
`aisingapore/Qwen-SEA-LION-v4-32B-IT-4BIT` on first use through Transformers and reuses
the Hugging Face cache on subsequent stages. No endpoint or manually started
server is required for this shell path.

```bash
LLM_PYTHON=/path/to/llm-env/bin/python \
TTS_PYTHON=/path/to/omnivoice-env/bin/python \
DUPLEXGEN_MODE=generate \
DUPLEXGEN_GENERATE_DOMAINS=travel,education,shopping,daily_life \
DUPLEXGEN_NUM_DIALOGUES=100 \
bash tools/run_indonesian_llm_to_audio_2gpu.sh
```

LLM_PYTHON must have CUDA-enabled PyTorch, Transformers and Accelerate installed.
The model uses its pre-quantized 4-bit weights with `device_map="auto"`. Each LLM stage runs in a
separate process and releases its model memory on exit, before OmniVoice starts.
This currently reloads the cached model between text stages. Leave
`DUPLEXGEN_PAUSE_BEFORE_TTS` unset for this local workflow. Set `HF_HOME` to a
writable cache volume if needed. Actual two-GPU inference still needs validation
on the target server.

The endpoint-based instructions below are an alternative: set
`DUPLEXGEN_LLM_BACKEND=api` explicitly to use them with the shell runner.
The separate Python orchestrator still requires an endpoint.

For the quickest Indonesian OmniVoice smoke test, including environment setup,
model download, slot-based turn-taking, and stereo synthesis:

```bash
git clone https://github.com/yoospeech/duplexgen_self.git
cd duplexgen_self
bash tools/quickstart_omnivoice_id.sh
```

This starts from a bundled Stage 4 example and does not run an LLM. Full data
generation still follows DuplexGen Stages 1–4. See the
[Indonesian guide](README.id.md) for GPU index overrides and bulk rendering.

The recommended Indonesian path uses two environments: one for Stages 1–4 and
one for OmniVoice Stage 5. Keep the TTS environment separate because its
PyTorch and Transformers versions can conflict with the generation pipeline.

```bash
# Stages 1-4: dialogue generation, slot identification, prediction, synthesis
python3 -m venv .venv                       # Python 3.10+
.venv/bin/pip install -r requirements.txt

# Stage 5: Indonesian OmniVoice rendering, Python 3.11
python3.11 -m venv .venv-tts-omnivoice
.venv-tts-omnivoice/bin/pip install --upgrade pip setuptools wheel

# NVIDIA CUDA 12.8; use the /cpu index for a CPU-only smoke test
.venv-tts-omnivoice/bin/pip install torch==2.8.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
.venv-tts-omnivoice/bin/pip install -r requirements-stage5-omnivoice.txt

# Required by the full Stage 5 renderer; not needed for the basic smoke script
.venv-tts-omnivoice/bin/pip install \
  whisperx silero-vad nltk nemo-text-processing
```

OmniVoice is installed from the pinned PyPI release; no Git submodule is
required. Chatterbox and Pocket TTS remain supported as alternative backends;
install them in separate environments using [Stage 5](docs/stage5-tts.md).
The Chatterbox alternative uses `requirements-stage5.txt` in its own Python
3.11 environment.

With the environments already installed, the shell runner supports two
independent input tasks and then runs the shared slot, turn-taking,
backchannel, and two-GPU OmniVoice stages. It is the recommended entry point
and can be invoked from any directory with its absolute path:

```bash
bash /home/ysw/Documents/eee/duplexgen_self/tools/run_indonesian_llm_to_audio_2gpu.sh
```

The separate Python orchestrator remains available when unique per-run output
directories and an explicitly managed OpenAI-compatible endpoint are useful.

### Task A: translate an existing source corpus

This mode reads one of the six supported source datasets and rewrites its
dialogues as natural spoken Indonesian before applying turn-taking. The default
source is `interviewer`, which downloads automatically from Hugging Face:

```bash
DUPLEXGEN_MODE=translate \
DUPLEXGEN_DATASET=interviewer \
DUPLEXGEN_NUM_DIALOGUES=25 \
  bash tools/run_indonesian_llm_to_audio_2gpu.sh
```

Python orchestrator alternative:

```bash
python3 tools/run_indonesian_llm_to_audio_2gpu.py \
  --mode translate \
  --dataset interviewer \
  --num-dialogues 25 \
  --pause-before-tts
```

`soda` also downloads automatically. `multiwoz`, `negotiator`, and
`socraticlm` require `DUPLEXGEN_DATA_ROOT`; `persuader` requires
`DUPLEXGEN_SOURCE_INPUT_PATH`. Run one source dataset per invocation:

```bash
DUPLEXGEN_MODE=translate \
DUPLEXGEN_DATASET=multiwoz \
DUPLEXGEN_DATA_ROOT=/data/raw-dialogue-datasets \
DUPLEXGEN_NUM_DIALOGUES=1000 \
  bash tools/run_indonesian_llm_to_audio_2gpu.sh
```

Translation mode preserves the source facts and scenarios. By default it
converts the requested number of train dialogues plus one test dialogue; set
`DUPLEXGEN_TEST_DIALOGUES` to change the latter.

### Task B: generate new dialogues from domain names

This mode needs no source corpus. It balances generation across a comma-
separated domain list, invents a distinct topic and scenario, and writes a
6–12 turn Indonesian dialogue for each item:

```bash
DUPLEXGEN_MODE=generate \
DUPLEXGEN_GENERATE_DOMAINS=healthcare,finance,education,travel,daily_life \
DUPLEXGEN_NUM_DIALOGUES=100 \
  bash tools/run_indonesian_llm_to_audio_2gpu.sh
```

Python orchestrator alternative:

```bash
python3 tools/run_indonesian_llm_to_audio_2gpu.py \
  --mode generate \
  --domains healthcare,finance,education,travel,daily_life \
  --num-dialogues 100 \
  --gpus 0,1 \
  --pause-before-tts
```

Each Python run writes only its own files under
`outputs/runs/<YYYYMMDD-HHMMSS>/`, including `run.json`, intermediate JSON,
audio, and logs. Pass `--run-id NAME` for a stable explicit directory. Set
`--hf-home /data/huggingface-cache` to keep the one-time model download on a
large shared volume, or pass `--omnivoice-model /data/models/OmniVoice` for an
already downloaded local snapshot.

The default is `generate` with 12 dialogues across ten domains:
`healthcare`, `finance`, `education`, `travel`, `shopping`,
`customer_service`, `workplace`, `technology`, `public_services`, and
`daily_life`. Domain cycles are shuffled reproducibly and counts differ by at
most one. Existing numbered JSON files are skipped, so rerunning resumes.

The script does not install or upgrade packages. Check both existing
interpreters before running it:

```bash
/home/ysw/workspace/anaconda3/envs/duplexgen/bin/python \
  -c 'import openai; print("LLM environment: OK")'

.venv-tts-omnivoice/bin/python \
  -c 'import omnivoice, silero_vad, whisperx, nltk, nemo_text_processing; from transformers import HiggsAudioV2TokenizerModel; print("TTS environment: OK")'
```

If the second check reports a missing full-render dependency, install it
manually into the existing TTS environment:

```bash
.venv-tts-omnivoice/bin/pip install \
  whisperx silero-vad nltk nemo-text-processing
```

If OmniVoice fails with `cannot import name HiggsAudioV2TokenizerModel`, the
TTS environment has an incompatible Transformers version. Repair only that
environment and verify the required class:

```bash
.venv-tts-omnivoice/bin/pip install --upgrade 'transformers>=5.3.0'
.venv-tts-omnivoice/bin/python -c \
  'from transformers import HiggsAudioV2TokenizerModel; print("Higgs tokenizer: OK")'
.venv-tts-omnivoice/bin/pip check
```

Do not apply this upgrade to the Chatterbox environment, which has a different
Transformers constraint.

Both tasks use the Stage 1–4 interpreter at
`/home/ysw/workspace/anaconda3/envs/duplexgen/bin/python`, and the TTS
interpreter at `.venv-tts-omnivoice/bin/python`. Intermediate JSON, stereo
audio, and per-stage logs are written under `outputs/` and `logs/`.

Generated-domain mode creates this Stage 4 input layout automatically:

```text
outputs/generated-domain-dialogues/
└── text_dialogue_synthetic/
    └── train/
        ├── synthetic_000000.json
        └── synthetic_000001.json
```

Start the OpenAI-compatible LLM server before the pipeline. For example:

```bash
vllm serve aisingapore/Qwen-SEA-LION-v4-32B-IT-4BIT \
  --served-model-name aisingapore/Qwen-SEA-LION-v4-32B-IT-4BIT \
  --tensor-parallel-size 2 --max-model-len 32768
```

Verify that the endpoint is ready:

```bash
curl http://localhost:8000/v1/models
```

If the LLM server and OmniVoice use the same two GPUs, run interactively with
a pause between them:

```bash
DUPLEXGEN_PAUSE_BEFORE_TTS=1 \
  bash tools/run_indonesian_llm_to_audio_2gpu.sh
```

After all LLM stages finish, the script asks you to stop the LLM server.
Terminate `vllm`, confirm with `nvidia-smi` that both GPUs are free, and then
press Enter to begin the two OmniVoice shards. No pause is needed when the LLM
endpoint runs on another server or on separate GPUs.

Override the LLM server or interpreters with environment variables:

```bash
LLM_BASE_URL=http://localhost:8000/v1 \
LLM_MODEL=aisingapore/Qwen-SEA-LION-v4-32B-IT-4BIT \
LLM_PYTHON=/path/to/llm-env/bin/python \
TTS_PYTHON=/path/to/omnivoice-env/bin/python \
  bash tools/run_indonesian_llm_to_audio_2gpu.sh
```

For a quick end-to-end trial, limit LLM input and TTS output together:

```bash
DUPLEXGEN_MAX_DIALOGUES=2 OMNIVOICE_NUM_STEPS=16 \
  bash tools/run_indonesian_llm_to_audio_2gpu.sh
```

The stages and their outputs are:

| Step | Output | Log |
|---|---|---|
| Translate source corpus (`translate`) | `outputs/translated-corpus-dialogues/` | `logs/stage1-corpus-translation.log` |
| Generate from domains (`generate`) | `outputs/generated-domain-dialogues/` | `logs/stage0-domain-generation.log` |
| LLM slot and turn-taking inference | `outputs/<mode>-with-tt/` | `logs/stage4-turn-taking.log` |
| LLM backchannel generation | `outputs/<mode>-with-tt-bc/` | `logs/stage4-backchannel.log` |
| Two-GPU OmniVoice synthesis | `outputs/omnivoice-id-2gpu/` | `logs/omnivoice-shard-0.log`, `logs/omnivoice-shard-1.log` |

Monitor both synthesis workers:

```bash
nvidia-smi
tail -f logs/omnivoice-shard-0.log logs/omnivoice-shard-1.log
```

Inspect the generated files after completion:

```bash
find outputs/omnivoice-id-2gpu -type f | sort
```

The script exits before inference when an interpreter, dependency, LLM
endpoint, or Stage 4 input is missing. If only one synthesis shard fails, its
corresponding log contains the model or CUDA error.

**Run everything from the repository root.** Stages 1-4 are modules of the
`src/` package, invoked as `.venv/bin/python -m src.<module>`, not as file
paths. The Stage 5 scripts are run as files with the *other* interpreter
(`.venv-tts-omnivoice/bin/python tts_render/convert_spoken.py`). The Stage 1-4
environment can also be installed as a package (`.venv/bin/pip install -e .`),
which puts the `src` modules on the path from anywhere.

**System dependencies.** `nemo-text-processing` (Stage 5 text normalization)
depends on `pynini`, which ships manylinux wheels for x86_64 CPython 3.8-3.13
and installs cleanly there. macOS and `aarch64` have no wheel and need OpenFst
headers first — `sudo apt-get install -y libfst-dev`, or take `pynini` from
conda-forge to avoid the build entirely.

**Reproducing the Stage 1–4 environment.** The requirements files carry ranges;
`constraints.txt` carries the exact versions the pipeline was last validated
against and covers the Stage 1–4 (`.venv`) environment. OmniVoice Stage 5 uses
the pinned versions in `requirements-stage5-omnivoice.txt`.

```bash
.venv/bin/pip install -r requirements.txt -c constraints.txt
```

**LLM backend.** Every generation stage talks to a chat LLM through the OpenAI
Python client, and the backend is chosen by the *model name* you pass:

- **OpenAI models** (`gpt-4.1`, `gpt-4o`, `gpt-5`, ...): reads your key from
  `OPENAI_API_KEY`. `--base_url` / `--api_key` are ignored.
- **Open-weight models** (e.g. `Qwen/Qwen3-32B`): posts to an OpenAI-compatible
  server at `--base_url` (default `http://localhost:8000/v1`, `--api_key`
  default `EMPTY`). Serve it yourself, e.g. `vllm serve Qwen/Qwen3-14B`. Stage 4
  also accepts separate endpoints for the boundary detector
  (`--boundary_base_url`) and the turn-taking predictor (`--tt_base_url`).

To run the test suite and the lint/format checks CI gates on, add the
development tooling (`pytest`, `ruff`, `black`):

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest
```

## Quickstart

The five stages end to end on one scenario. Each stage's own doc covers the
rest of its flags; `--dataset interviewer` needs no local download, so this
runs as written.

```bash
export OPENAI_API_KEY=sk-...

# Stage 1 -- convert source dialogues to spoken style
.venv/bin/python -m src.speechify_run \
  --dataset interviewer --save_dir results/ --llm_model_name gpt-4.1

# Fetch the released corpus and unpack both halves (see docs/CORPUS.md)
.venv/bin/hf download DuplexGen/duplexgen-corpus \
  --repo-type dataset --local-dir duplexgen-corpus/
.venv/bin/python tools/unpack_corpus.py --kind annotations \
  --src duplexgen-corpus/annotations/INT/train.jsonl --code INT --split train
.venv/bin/python tools/unpack_corpus.py --kind dialogues \
  --src duplexgen-corpus/dialogues/INT/train.jsonl --code INT --split train

# Stage 3 -- train the turn-taking predictor
.venv/bin/python -m src.train_turntaking_hf \
  --model_name_or_path Qwen/Qwen3-4B \
  --input_root data-annotations/ \
  --output_dir ./output/turntaking_qwen3_4b

# Stages 2+4 -- detect slots and generate dialogues with turn-taking
.venv/bin/python -m src.synthesis.run \
  --dataset interviewer --split train \
  --input_root data-dialogues/ \
  --save_root outputs/generated_dialogues_with_tt \
  --llm_model_name gpt-4.1 --boundary_model_name gpt-4.1-mini \
  --hf_model_name_or_path ./output/turntaking_qwen3_4b --hf_load_in_4bit

# Stage 5 -- render with prompt-free OmniVoice (note the other interpreter)
.venv-tts-omnivoice/bin/python tts_render/convert_spoken.py \
  --tts_backend omnivoice --omnivoice_voice_mode auto --language en \
  --omnivoice_device cuda:0 --omnivoice_align_device cuda \
  --input_glob 'outputs/generated_dialogues_with_tt/**/*.json' \
  --save_dir outputs/audios_omnivoice --num_variants 1
```

Stage 3 training is optional if you only want to generate dialogues — Stage 4
can query a chat model instead, see [stage4-generation](docs/stage4-generation.md).
Backchannel text is filled in by a separate pass, `src.synthesis.run_add_bc`.

### Bahasa Indonesia generation

Stages 1 and 4 accept `--language id`. This keeps the source facts and output
schema intact while generating conversational Bahasa Indonesia, independent of
the OpenAI-compatible model selected with `--llm_model_name`. For example,
serve the open-weight SEA-LION model, which is trained for Southeast Asian
languages:

```bash
vllm serve aisingapore/Qwen-SEA-LION-v4-32B-IT-4BIT \
  --tensor-parallel-size 2 --max-model-len 32768

.venv/bin/python -m src.speechify_run \
  --dataset interviewer --save_dir results-id/ \
  --language id \
  --llm_model_name aisingapore/Qwen-SEA-LION-v4-32B-IT-4BIT \
  --base_url http://localhost:8000/v1 --api_key EMPTY

.venv/bin/python -m src.synthesis.run \
  --dataset interviewer --split train \
  --input_root data-dialogues/ \
  --save_root outputs/generated-dialogues-id \
  --language id \
  --llm_model_name aisingapore/Qwen-SEA-LION-v4-32B-IT-4BIT \
  --base_url http://localhost:8000/v1 --api_key EMPTY \
  --boundary_model_name aisingapore/Qwen-SEA-LION-v4-32B-IT-4BIT \
  --boundary_base_url http://localhost:8000/v1 --boundary_api_key EMPTY \
  --tt_model_name aisingapore/Qwen-SEA-LION-v4-32B-IT-4BIT \
  --tt_base_url http://localhost:8000/v1 --tt_api_key EMPTY
```

Add `--annotate_only` when the input already contains the final Indonesian
utterances and Stage 4 should preserve their wording while only detecting
boundaries and adding turn-taking decisions.

Pass `--language id` to `src.synthesis.run_add_bc` as well when filling
backchannels. Stage 5 supports Indonesian through Pocket TTS and OmniVoice.
OmniVoice Auto mode needs no reference recording: it selects one voice per
channel on the first event and reuses that identity for the rest of the
dialogue. See the [Indonesian guide](README.id.md) and
[Stage 5](docs/stage5-tts.md#omnivoice).
WhisperX has no built-in Indonesian alignment model mapping, so pass an
Indonesian CTC model explicitly, for example
`--align_model m-salman-a/wav2vec2-xlsr-53-common-voice-indonesian`.

For the local Conda environments used in this checkout, run text generation
with `/home/ysw/workspace/anaconda3/envs/duplexgen/bin/python` and Pocket TTS
with `/home/ysw/workspace/anaconda3/envs/duplextts/bin/python`. The latter is a
Pocket TTS environment using NumPy 2 and cannot also run the vendored
Chatterbox backend, which requires NumPy 1.x.

Verify Indonesian TTS independently before rendering a full dialogue:

```bash
HF_HOME=/tmp/duplexgen-hf \
/home/ysw/workspace/anaconda3/envs/duplextts/bin/python \
  tools/smoke_test_pocket_tts.py \
  --output outputs/pocket-id-smoke.wav
```

A successful run prints `PASS` with the sample rate and duration. Listen to
`outputs/pocket-id-smoke.wav` as the final pronunciation and quality check.

On a CPU-only machine with no local LLM server, verify the dialogue path with
the small open-weight `Qwen/Qwen3-0.6B` smoke-test model:

```bash
HF_HOME=/tmp/duplexgen-hf \
/home/ysw/workspace/anaconda3/envs/duplexgen/bin/python \
  tools/smoke_test_indonesian_llm.py \
  --output outputs/indonesian-llm-smoke.json
```

This small model is only a connectivity test. Use the SEA-LION model shown
above for dataset production after starting a suitable inference server.

Render the first valid user/assistant pair from that JSON as a playable stereo
WAV (channel 0 assistant, channel 1 user):

```bash
HF_HOME=/tmp/duplexgen-hf \
/home/ysw/workspace/anaconda3/envs/duplextts/bin/python \
  tools/render_smoke_dialogue_pocket_tts.py \
  --input outputs/indonesian-llm-smoke.json \
  --output outputs/indonesian-dialogue-smoke.wav
```

## Data

- **`DuplexGen/duplexgen-corpus`**: generated dialogues
  (`dialogues/<SCENARIO>/train.jsonl`, with per-word turn-taking decisions) and
  the human slot-level preference annotations
  (`annotations/<SCENARIO>/{train,test}.jsonl`) collected over their slots.
- **`DuplexGen/duplexgen-spoken`**: rendered audio for those dialogues.

Dialogue text is generated by **Qwen3.5-122B-A10B**. No third-party raw source
text is redistributed; only DuplexGen-generated dialogues and our own human
annotations are released.

See [`DATA_CARD.md`](DATA_CARD.md) for the schema and per-scenario counts, and
[`docs/CORPUS.md`](docs/CORPUS.md) for converting the release into the layout
the stages read.

## Fine-tuning PP-DG

**PP-DG** (PersonaPlex trained on DuplexGen data) is the full-duplex model
behind the paper's headline result. The LoRA fine-tuning code that trains it
lives in
[`duplexgen/personaplex-finetune`](https://github.com/duplexgen/personaplex-finetune).
No model checkpoints are included in this release.

## License

Code is released under the **Apache License 2.0** (see [`LICENSE`](LICENSE)).
The released **data** is not covered by it: each scenario inherits the license
of its upstream source dataset. See
[`docs/DATA_LICENSES.md`](docs/DATA_LICENSES.md) for the per-scenario
table and attribution notes.

Audio is rendered with [Chatterbox](https://github.com/resemble-ai/chatterbox)
TTS by Resemble AI (MIT); a trimmed copy is vendored at
`duplexgen/tts/chatterbox/`.

## Citation

If you use this code, the corpus, or the rendered audio, please cite:

```bibtex
@article{kim2026duplexgen,
  title   = {{DuplexGen: Adaptive Synthesis of Human--AI Turn-Taking Dialogues}},
  author  = {Kim, Takyoung and Kim, Kang-wook and Woo, Sang Hoon and
             Hirschberg, Julia and Kim, Gunhee and Hakkani-T{\"u}r, Dilek},
  journal = {arXiv preprint arXiv:2607.26178},
  year    = {2026},
  url     = {https://arxiv.org/abs/2607.26178}
}
```
