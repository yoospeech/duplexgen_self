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
vllm serve aisingapore/Llama-SEA-LION-v3-8B-IT

.venv/bin/python -m src.speechify_run \
  --dataset interviewer --save_dir results-id/ \
  --language id \
  --llm_model_name aisingapore/Llama-SEA-LION-v3-8B-IT \
  --base_url http://localhost:8000/v1 --api_key EMPTY

.venv/bin/python -m src.synthesis.run \
  --dataset interviewer --split train \
  --input_root data-dialogues/ \
  --save_root outputs/generated-dialogues-id \
  --language id \
  --llm_model_name aisingapore/Llama-SEA-LION-v3-8B-IT \
  --base_url http://localhost:8000/v1 --api_key EMPTY \
  --boundary_model_name aisingapore/Llama-SEA-LION-v3-8B-IT \
  --boundary_base_url http://localhost:8000/v1 --boundary_api_key EMPTY \
  --tt_model_name aisingapore/Llama-SEA-LION-v3-8B-IT \
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
