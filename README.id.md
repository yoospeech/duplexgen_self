# DuplexGen for Bahasa Indonesia

[Main README](README.md) · Bahasa Indonesia setup

This guide covers the Indonesian extension of DuplexGen. It generates natural
spoken Indonesian dialogue with an open-weight LLM, adds turn-taking behavior
and backchannels, and renders two-speaker stereo audio for full-duplex model
training such as Moshi or PersonaPlex.

The original implementation accompanies **DuplexGen: Adaptive Synthesis of
Human–AI Turn-Taking Dialogues**. The Indonesian path uses an OpenAI-compatible
open-weight LLM and supports `k2-fsa/OmniVoice` as the primary multi-turn TTS
backend. Pocket TTS remains available as an alternative Indonesian backend.

## Python environments

### Quick install and synthesis

From a fresh checkout, this command creates an isolated Python 3.11
environment, installs the appropriate CPU or CUDA PyTorch wheel and OmniVoice,
downloads `k2-fsa/OmniVoice` on first use, and renders the bundled Indonesian
slot-dialogue example:

```bash
bash tools/quickstart_omnivoice_id.sh
```

The quickstart starts from a bundled Stage 4 example; it tests installation and
TTS but does not run an LLM. The production path still uses DuplexGen Stages
1–4 to generate dialogue text, identify slots, and select turn-taking actions.

The outputs are `outputs/omnivoice-id-quickstart.wav` and a same-stem event
JSON. Pass a different Stage 4 JSON and output path as positional arguments:

```bash
bash tools/quickstart_omnivoice_id.sh \
  /data/dialogues-id/text_dialogue_interviewer/train/example.json \
  /data/smoke/example.wav
```

GPU hosts default to the CUDA 12.8 PyTorch index. Override it when the server
uses another build:

```bash
OMNIVOICE_TORCH_INDEX=https://download.pytorch.org/whl/cu126 \
  bash tools/quickstart_omnivoice_id.sh
```

The script defaults to 16 diffusion steps for a quick check. Set
`OMNIVOICE_NUM_STEPS=32` for the production-quality setting.

Use this interpreter for dialogue generation:

```bash
/home/ysw/workspace/anaconda3/envs/duplexgen/bin/python
```

Use this interpreter for Indonesian Pocket TTS:

```bash
/home/ysw/workspace/anaconda3/envs/duplextts/bin/python
```

The `duplextts` environment uses Pocket TTS 3.1.0 and NumPy 2. It cannot also
run the vendored Chatterbox backend, which requires NumPy 1.x.

Use a fresh Python 3.11 environment for OmniVoice:

```bash
python3.11 -m venv .venv-tts-omnivoice
source .venv-tts-omnivoice/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-stage5-omnivoice.txt
pip install whisperx silero-vad nltk nemo-text-processing
```

On an NVIDIA server, install the PyTorch wheel matching its CUDA version before
the remaining packages. OmniVoice, Chatterbox, and Pocket TTS should stay in
separate environments because their PyTorch, Transformers, and NumPy versions
can conflict.

## Multi-turn Indonesian flow

The Indonesian pipeline treats turn-taking as slot filling:

1. Stage 1 rewrites source dialogue as natural spoken Indonesian.
2. Stage 2 finds candidate word boundaries inside each utterance.
3. Stage 3 or an LLM assigns `backchannel`, `floor_taking`, or `silence` to each slot.
4. Stage 4 inserts `[BACKCHANNEL]` and `[TAKE_FLOOR]` actions and generates the
   Indonesian backchannel text.
5. Stage 5 renders assistant-left/user-right stereo audio. Backchannels overlap
   the active speaker and floor-taking starts the next turn early.

In OmniVoice Auto mode no reference recording is required. The first user and
assistant events select separate Auto voices. Each result is immediately
encoded as a reusable voice prompt, so subsequent turns retain the identity of
their channel. The pair resets for the next dialogue and is reproducible from
the file-derived seed.

## Quick LLM smoke test

On a CPU-only machine without a running LLM server, use the small open-weight
model `Qwen/Qwen3-0.6B` to verify model loading and Indonesian text generation:

```bash
HF_HOME=/tmp/duplexgen-hf \
/home/ysw/workspace/anaconda3/envs/duplexgen/bin/python \
  tools/smoke_test_indonesian_llm.py \
  --output outputs/indonesian-llm-smoke.json
```

This is only a connectivity test. The 0.6B model may produce weak dialogue or
incorrect role ordering and should not be used to build a production dataset.

For production, use a stronger model trained for Southeast Asian languages,
such as `aisingapore/Llama-SEA-LION-v3-8B-IT`, through an OpenAI-compatible
server:

```bash
vllm serve aisingapore/Llama-SEA-LION-v3-8B-IT
```

## Generate spoken Indonesian dialogue

DuplexGen can mix six source domains instead of generating only interview
dialogue:

| Dataset | Typical domain |
|---|---|
| `interviewer` | job interviews and workforce conversations |
| `multiwoz` | hotels, restaurants, transport, booking, and other services |
| `negotiator` | price and marketplace negotiation |
| `persuader` | everyday persuasion |
| `socraticlm` | teaching and Socratic tutoring |
| `soda` | broad social and everyday conversation |

Use `--dataset all` to process all six after preparing the required local
corpora. Stage 1 preserves source facts while rewriting them as Indonesian
spoken dialogue, so domain diversity comes from the source mixture. Additional
industry domains can be added by converting them to the same role/content JSON
schema before Stage 1.

Stage 1 accepts `--language id`. This tells the model to write natural spoken
Indonesian while preserving the source facts, speaker roles, JSON schema, and
special turn-taking tokens.

```bash
/home/ysw/workspace/anaconda3/envs/duplexgen/bin/python \
  -m src.speechify_run \
  --dataset interviewer \
  --save_dir results-id/ \
  --language id \
  --llm_model_name aisingapore/Llama-SEA-LION-v3-8B-IT \
  --base_url http://localhost:8000/v1 \
  --api_key EMPTY
```

Stage 4 also accepts `--language id`. Point the writer, boundary detector, and
turn-taking predictor to the local endpoint:

```bash
/home/ysw/workspace/anaconda3/envs/duplexgen/bin/python \
  -m src.synthesis.run \
  --dataset interviewer \
  --split train \
  --input_root data-dialogues/ \
  --save_root outputs/generated-dialogues-id \
  --language id \
  --llm_model_name aisingapore/Llama-SEA-LION-v3-8B-IT \
  --base_url http://localhost:8000/v1 \
  --api_key EMPTY \
  --boundary_model_name aisingapore/Llama-SEA-LION-v3-8B-IT \
  --boundary_base_url http://localhost:8000/v1 \
  --boundary_api_key EMPTY \
  --tt_model_name aisingapore/Llama-SEA-LION-v3-8B-IT \
  --tt_base_url http://localhost:8000/v1 \
  --tt_api_key EMPTY
```

If the input dialogue already has the final wording, add `--annotate_only`.
This preserves every input turn and only runs boundary detection, turn-taking
probability estimation, and action insertion.

Pass `--language id` to `src.synthesis.run_add_bc` as well when generating
backchannels. Indonesian examples include `iya`, `hmm`, `baik`, and
`oh begitu`. The program keeps `[TAKE_FLOOR]`, `[BACKCHANNEL]`, role names, and
JSON keys unchanged.

## OmniVoice multi-turn smoke test

First verify prompt-free Indonesian synthesis:

```bash
.venv-tts-omnivoice/bin/python tools/smoke_test_omnivoice.py \
  --voice_mode auto \
  --text 'Selamat pagi, ada yang bisa saya bantu hari ini?' \
  --output outputs/omnivoice-id-auto-smoke.wav
```

Then render a Stage 4 dialogue containing slot decisions:

```bash
.venv-tts-omnivoice/bin/python tools/render_smoke_dialogue_omnivoice.py \
  --voice_mode auto \
  --input outputs/sea-lion-with-tt-bc/text_dialogue_interviewer/train/sea_lion_0002.json \
  --output outputs/omnivoice-id-slot-dialogue-smoke.wav \
  --num_steps 16 --interrupt_overlap 0.6
```

This produces a 24 kHz stereo WAV and a same-stem JSON event timeline. The
validated sample contains one assistant backchannel over the user and two
0.6-second floor-taking overlaps.

## Pocket TTS smoke test (alternative)

Pocket TTS expects numbers to be written as spoken Indonesian words. For
example, use `pukul satu siang` rather than `13:00` in text sent to the model.

Generate one test sentence:

```bash
HF_HOME=/tmp/duplexgen-hf \
/home/ysw/workspace/anaconda3/envs/duplextts/bin/python \
  tools/smoke_test_pocket_tts.py \
  --output outputs/pocket-id-smoke.wav
```

A successful run prints `PASS`, the sample rate, and the audio duration. Listen
to `outputs/pocket-id-smoke.wav` to check pronunciation and voice quality.

## Render the LLM smoke dialogue

Render the first valid user/assistant pair from the LLM smoke-test JSON as a
stereo WAV:

```bash
HF_HOME=/tmp/duplexgen-hf \
/home/ysw/workspace/anaconda3/envs/duplextts/bin/python \
  tools/render_smoke_dialogue_pocket_tts.py \
  --input outputs/indonesian-llm-smoke.json \
  --output outputs/indonesian-dialogue-smoke.wav
```

The verified smoke output has the following properties:

- 24 kHz sample rate
- two channels
- channel 0 contains the assistant
- channel 1 contains the user
- 7.53 seconds total duration

## Render a full OmniVoice dataset

Run one small Stage 5 trial before a bulk render:

```bash
.venv-tts-omnivoice/bin/python tts_render/convert_spoken.py \
  --tts_backend omnivoice \
  --omnivoice_voice_mode auto \
  --omnivoice_device cuda:0 \
  --omnivoice_align_device cuda \
  --omnivoice_num_steps 16 \
  --language id \
  --align_model m-salman-a/wav2vec2-xlsr-53-common-voice-indonesian \
  --input_glob 'outputs/sea-lion-with-tt-bc/**/*.json' \
  --save_dir outputs/audios-omnivoice-id \
  --num_variants 1 --max_dialogues 1
```

For quality production, use the default 32 diffusion steps after comparing the
16-step trial. `dialogue.wav` is 24 kHz stereo with channel 0 = assistant and
channel 1 = user. `meta.json` contains the utterance, backchannel, timing,
backend, language, and voice-mode metadata expected by downstream preparation.

For an eight-GPU server, run one disjoint shard per GPU:

```bash
for shard in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$shard .venv-tts-omnivoice/bin/python \
    tts_render/convert_spoken.py \
    --tts_backend omnivoice --omnivoice_voice_mode auto --language id \
    --omnivoice_device cuda:0 --omnivoice_align_device cuda \
    --align_model m-salman-a/wav2vec2-xlsr-53-common-voice-indonesian \
    --input_glob '/data/dialogues-id/**/*.json' \
    --save_dir /data/audios-omnivoice-id \
    --num_variants 1 --num_shards 8 --shard_id "$shard" \
    > "omnivoice-shard-${shard}.log" 2>&1 &
done
wait
```

At 24 kHz and 16-bit stereo, 1,000 hours of final dialogue audio occupies
about 346 GB. Keeping the renderer's per-utterance copies roughly doubles this
to about 692 GB before metadata and filesystem overhead.

## Training handoff

The Stage 5 layout is designed for the
[`duplexgen/personaplex-finetune`](https://github.com/duplexgen/personaplex-finetune)
pipeline. Each variant contains:

```text
text_dialogue_<scenario>/<split>/<dialogue_id>/var<NN>/
├── dialogues/dialogue.wav
├── utterances/*.wav
├── backchannels/*.wav
└── meta.json
```

Before launching a long fine-tuning job, validate a small shard manually:

- both channels contain the intended roles (`assistant`, then `user`);
- each role keeps one voice throughout a dialogue;
- backchannels overlap the active speaker without clipping;
- floor-taking overlap matches the event metadata;
- Indonesian transcripts match the audio and contain no leaked control tokens;
- every `dialogue.wav` has its corresponding `meta.json`.

Generate and inspect a few hours first, then scale to the 1,000-hour render.
The fine-tuning code lives in the separate repository linked above; this
repository prepares the stereo audio and metadata it consumes.

## Render with Pocket TTS (alternative)

Prepare one clean Indonesian reference recording for the assistant and one or
more user recordings. Each directory below `voices-id` is treated as a distinct
speaker:

```text
voices-id/
├── speaker-01/prompt.wav
└── speaker-02/prompt.wav
```

Start with one dialogue and one voice variant:

```bash
HF_HOME=/tmp/duplexgen-hf \
/home/ysw/workspace/anaconda3/envs/duplextts/bin/python \
  tts_render/convert_spoken.py \
  --tts_backend pocket_indonesian \
  --pocket_device cpu \
  --align_model m-salman-a/wav2vec2-xlsr-53-common-voice-indonesian \
  --assistant_prompt /path/to/assistant-id.wav \
  --voice_dir /path/to/voices-id \
  --input_glob 'outputs/generated-dialogues-id/**/*.json' \
  --save_dir outputs/audios-pocket-id \
  --num_variants 1 \
  --max_dialogues 1
```

The main output is a 24 kHz stereo `dialogue.wav`. Channel 0 is the assistant
and channel 1 is the user, matching the channel order expected by the
PersonaPlex/Moshi fine-tuning pipeline. Timing and voice metadata are stored in
`meta.json`.

## Verified local outputs

The following files were generated successfully with the interpreters listed
above:

- `outputs/indonesian-llm-smoke.json`
- `outputs/pocket-id-smoke.wav`
- `outputs/indonesian-dialogue-smoke.wav`
- `outputs/sea-lion-with-tt-bc/text_dialogue_interviewer/train/sea_lion_0002.json`
- `outputs/sea-lion-official-stage5/text_dialogue_interviewer/train/sea_lion_0002/var00/dialogues/dialogue.wav`
- `outputs/sea-lion-official-stage5/text_dialogue_interviewer/train/sea_lion_0002/var00/meta.json`
- `outputs/omnivoice-id-auto-smoke.wav`
- `outputs/omnivoice-id-auto-stable-dialogue-smoke.wav`
- `outputs/omnivoice-id-slot-dialogue-smoke.wav`
- `outputs/omnivoice-id-slot-dialogue-smoke.json`

The stereo dialogue contains this first valid pair:

```text
user: Saya ingin memindahkan jadwal wawancara dari Selasa pagi ke Jumat siang.
assistant: Jumat pukul satu siang tersedia.
```

## Licensing

DuplexGen code is released under Apache License 2.0. Dataset licensing depends
on each upstream source. OmniVoice code is Apache 2.0, but its pretrained model
is CC-BY-NC and therefore restricted to non-commercial use under that license.
The `pocket-tts-indonesian` weights use CC BY 4.0. Every external voice
reference must have appropriate usage rights and the speaker's consent.

See the [main README](README.md), [Stage 4 documentation](docs/stage4-generation.md),
and [Stage 5 documentation](docs/stage5-tts.md) for the complete pipeline.
