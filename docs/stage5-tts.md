# Stage 5: TTS rendering

Renders the Stage 4 result to two-channel audio with Chatterbox.

```bash
.venv-tts/bin/python tts_render/convert_spoken.py \
  --prompt_dir tts_render/prompt_wavs \
  --input_glob 'outputs/generated_dialogues_with_hf_swbd_plus_backchannels/**/*.json' \
  --save_dir outputs/audios_chatterbox \
  --num_variants 10
```

Note the **other interpreter** (`.venv-tts`, see [Setup](../README.md#setup))
and that this is run as a *file path*, not as a `-m` module like Stages 1-4.

Each dialogue is rendered into `--num_variants` two-channel variants: the
assistant uses a fixed cloning prompt (`tts_render/prompt_wavs/assistant_en.wav`,
a held-out LibriSpeech speaker — see `tts_render/prompt_wavs/PROVENANCE.md`) and
each variant's user voice is sampled from a distinct LibriSpeech speaker.
Backchannels are synthesized by Chatterbox through the same voice prompt as the
speaker uttering them, so they match that variant's voice.

> [!IMPORTANT]
> **Chatterbox renders backchannels poorly.** Short interjections — `mhm`,
> `mm-hmm` — come out flat and mispronounced; a zero-shot TTS has almost no
> context to work from on a two-phoneme utterance. **The paper's dialogues used
> the ElevenLabs API for these instead.** That dependency was dropped for 1.0 so
> the pipeline runs without a paid key, and what ships is the self-contained
> fallback. To use another backend, replace the `uttr_type == "backchannel"`
> branch in `main_process` (`tts_render/convert_spoken.py`) — placement, caching
> and the `backchannels/*.wav` outputs take whatever waveform it returns. Keep
> the backchannel voice matched to the speaker uttering it.

## Output layout

`--input_glob` is matched against the Stage 4 output layout, and the trailing
`text_dialogue_<dataset>/<split>/` of each input path is mirrored under
`--save_dir`:

```
outputs/audios_chatterbox/text_dialogue_interviewer/train/work_0000/var00/
├── dialogues/dialogue.wav     # the full two-channel mix
├── utterances/00.wav ...      # one two-channel file per utterance, pre-mix
├── backchannels/05_0.wav ...  # <utterance>_<slot>, the isolated backchannel
└── meta.json                  # per-utterance text, timings, and the voice used
```

`dialogue.wav` is 24 kHz 16-bit stereo with **channel 0 = assistant, channel 1 =
user** — the channels are swapped on the way out so the assistant lands first,
which is the order `personaplex-finetune` expects. `meta.json` records that as
`"speakers": ["assistant", "user"]`, along with `user_prompt_wav` and
`user_prompt_speaker_id` for the variant's LibriSpeech voice.

A variant whose `dialogue.wav` and `meta.json` both exist is skipped, so a
re-run resumes rather than re-synthesizing. Voice casting is seeded per dialogue
from `--seed` and the file name, so a resumed run fills the missing variants
from the same cast as the ones already on disk.

## Cost knobs

Rendering runs at roughly real time, so cap a trial run: `--max_dialogues N`
takes only the first N dialogues, and `--num_variants` scales the per-dialogue
cost linearly. To spread a full render over several GPUs, give each worker the
same command with `--num_shards W --shard_id I`; shards partition the file list,
and since each writes its own dialogue directories they can share one
`--save_dir`. `--input_glob` accepts several globs, which are concatenated and
de-duplicated, and `--exclude_ids_file` skips dialogue stems listed one per
line.

## Dropped inputs

Dialogues carrying LLM artifacts the sanitizer cannot repair — leaked think
blocks, concatenated turns, template placeholders — are skipped with a
`DROP-DIALOGUE` warning naming the reason. Individual backchannels are dropped
with a warning when forced alignment returns no words to anchor them to; the
dialogue still renders, and `utterances_with_bc[i].texts_styled[j].isUttered` in
`meta.json` records which backchannels actually made it into the audio. The run
ends with a summary line counting rendered, skipped, dropped, and failed
variants, and exits non-zero if it rendered nothing at all.

## Voices

By default `--librispeech_root` points at the 12-speaker sample bundled in
`tts_render/librispeech_samples/`, so this runs out of the box. That sample
bounds `--num_variants` at 12; for paper-scale rendering, download the full
[LibriSpeech](https://www.openslr.org/12) corpus and pass `--librispeech_root
/path/to/LibriSpeech --librispeech_subsets train-clean-100,train-clean-360`.

## Bahasa Indonesia with Pocket TTS

The Indonesian backend uses the 6-layer
`anak10thn/pocket-tts-indonesian` checkpoint and WhisperX's Indonesian
alignment model. Install it in its own Python 3.11 environment:

```bash
/home/ysw/workspace/anaconda3/envs/duplextts/bin/python -m pip install \
  -r requirements-stage5-pocket.txt
```

This turns `duplextts` into the Pocket TTS environment. Pocket TTS requires
NumPy 2, while the vendored Chatterbox path requires NumPy 1.x, so the two
backends cannot share this environment.

Do not install this backend into `.venv-tts`: current Pocket TTS requires
NumPy 2 and `tokenizers>=0.21`, while the vendored Chatterbox environment pins
NumPy 1.x and `tokenizers<0.21` through Transformers 4.46.3.

Verify the model download and synthesis independently before running the full
pipeline:

```bash
.venv-tts-id/bin/python tools/smoke_test_pocket_tts.py \
  --output outputs/pocket-id-smoke.wav
```

Success prints `PASS` with the sample rate and duration. Listen to that WAV as
well: the automated check catches empty or invalid audio, while listening is
needed to assess pronunciation and voice quality.

Use clean Indonesian speaker recordings for both sides. Put one or more user
prompts under each speaker directory so variants sample distinct speakers:

```text
voices-id/
├── speaker-01/prompt.wav
└── speaker-02/prompt.wav
```

Then render a small trial:

```bash
.venv-tts-id/bin/python tts_render/convert_spoken.py \
  --tts_backend pocket_indonesian \
  --align_model m-salman-a/wav2vec2-xlsr-53-common-voice-indonesian \
  --assistant_prompt /path/to/assistant-id.wav \
  --voice_dir /path/to/voices-id \
  --input_glob 'outputs/generated-dialogues-id/**/*.json' \
  --save_dir outputs/audios-pocket-id \
  --num_variants 2 --max_dialogues 1
```

The model expects numbers written as Indonesian words. Stage 1 and Stage 4 do
this when run with `--language id`. The Pocket TTS model weights are CC BY 4.0;
review the licenses and consent for every voice prompt used for cloning.

## OmniVoice

The `omnivoice` backend uses `k2-fsa/OmniVoice` for zero-shot voice cloning and
keeps the renderer's 24 kHz two-channel output contract. Install it in a fresh
environment; the upstream project currently recommends PyTorch 2.8 and warns
that TTS environments should be isolated to avoid dependency conflicts:

```bash
python3.11 -m venv .venv-tts-omnivoice
.venv-tts-omnivoice/bin/pip install -r requirements-stage5-omnivoice.txt
.venv-tts-omnivoice/bin/pip install whisperx silero-vad nltk nemo-text-processing
```

Verify the official checkpoint and prompt-free Auto voice independently:

```bash
.venv-tts-omnivoice/bin/python tools/smoke_test_omnivoice.py \
  --voice_mode auto \
  --output outputs/omnivoice-smoke.wav
```

The script automatically uses CPU with `float32` when CUDA is unavailable and
checks that the result is a finite, non-empty 24 kHz waveform.

To render every turn from a dialogue JSON into assistant-left/user-right stereo:

```bash
.venv-tts-omnivoice/bin/python tools/render_smoke_dialogue_omnivoice.py \
  --input outputs/sea-lion-with-tt-bc/text_dialogue_interviewer/train/sea_lion_0001.json \
  --voice_mode auto \
  --output outputs/omnivoice-id-dialogue-smoke.wav
```

When the Stage 4 JSON contains slot decisions, the smoke renderer places a
`backchannel` on the listener channel at the annotated word position and starts
the next turn before the current audio ends for `floor_taking`. It writes a
same-stem JSON file containing the resulting event timestamps. Inputs without
slot decisions remain ordinary sequential dialogue.

Run a small prompt-free Indonesian trial:

```bash
.venv-tts-omnivoice/bin/python tts_render/convert_spoken.py \
  --tts_backend omnivoice \
  --omnivoice_voice_mode auto \
  --language id \
  --align_model m-salman-a/wav2vec2-xlsr-53-common-voice-indonesian \
  --input_glob 'outputs/generated-dialogues-id/**/*.json' \
  --save_dir outputs/audios-omnivoice-id \
  --num_variants 1 --max_dialogues 1
```

Auto mode needs no reference audio. The renderer lets OmniVoice select one
voice for each channel's first turn, converts that generated turn into a
reusable voice prompt, and clones it for every later turn on the same channel.
This keeps the user and assistant identities stable within each dialogue.
The file-derived seed chooses a fresh, reproducible pair for each dialogue and
variant, and the pair is reset before the next dialogue starts.

Use `--omnivoice_num_steps 16` for a faster trial or the default `32` for
quality. Other supported languages need the matching `--language` code and may
need an explicit `--align_model` when WhisperX has no default alignment model.
For large clone-mode renders, place a same-stem transcript beside each prompt (for example
`speaker.wav` and `speaker.txt`). The renderer encodes each voice once and
reuses an LRU cache of prompts instead of running ASR for every sentence; tune
the default 128-entry cache with `--omnivoice_prompt_cache_size`.

On a server with eight GPUs, launch one shard per GPU. Each process sees one
GPU as `cuda:0`, while `--num_shards` and `--shard_id` give it a disjoint input
partition:

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

At 24 kHz, 16-bit stereo, 1,000 hours of final dialogue WAVs occupy about
346 GB. This renderer also keeps per-utterance stereo WAVs, which roughly
doubles audio storage to about 692 GB before filesystem and metadata overhead.
OmniVoice code is Apache 2.0, while the pretrained model is CC-BY-NC; the model
therefore cannot be used commercially under its current license.
