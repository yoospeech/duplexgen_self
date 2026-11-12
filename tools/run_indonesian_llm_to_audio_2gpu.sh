#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
llm_python="${LLM_PYTHON:-/home/ysw/workspace/anaconda3/envs/duplexgen/bin/python}"
tts_python="${TTS_PYTHON:-${repo_root}/.venv-tts-omnivoice/bin/python}"

mode="${DUPLEXGEN_MODE:-generate}"
if [[ -n "${DUPLEXGEN_DATASET:-}" ]]; then
  dataset="${DUPLEXGEN_DATASET}"
elif [[ "${mode}" == "translate" ]]; then
  dataset="interviewer"
else
  dataset="synthetic"
fi
split="${DUPLEXGEN_SPLIT:-train}"
if [[ "${mode}" == "translate" ]]; then
  default_stage_input="${repo_root}/outputs/translated-corpus-dialogues"
else
  default_stage_input="${repo_root}/outputs/generated-domain-dialogues"
fi
stage_input="${DUPLEXGEN_INPUT_ROOT:-${default_stage_input}}"
stage_tt="${DUPLEXGEN_TT_ROOT:-${repo_root}/outputs/${mode}-with-tt}"
stage_bc="${DUPLEXGEN_BC_ROOT:-${repo_root}/outputs/${mode}-with-tt-bc}"
audio_output="${DUPLEXGEN_AUDIO_ROOT:-${repo_root}/outputs/omnivoice-id-2gpu}"
log_dir="${DUPLEXGEN_LOG_DIR:-${repo_root}/logs}"

llm_model="${LLM_MODEL:-aisingapore/Llama-SEA-LION-v3-8B-IT}"
llm_base_url="${LLM_BASE_URL:-http://localhost:8000/v1}"
llm_api_key="${LLM_API_KEY:-EMPTY}"
export DUPLEXGEN_LLM_BACKEND="${DUPLEXGEN_LLM_BACKEND:-local}"
num_steps="${OMNIVOICE_NUM_STEPS:-32}"
max_dialogues="${DUPLEXGEN_MAX_DIALOGUES:-}"
pause_before_tts="${DUPLEXGEN_PAUSE_BEFORE_TTS:-0}"
generate_domains="${DUPLEXGEN_GENERATE_DOMAINS:-healthcare,finance,education,travel,shopping,customer_service,workplace,technology,public_services,daily_life}"
num_dialogues="${DUPLEXGEN_NUM_DIALOGUES:-${DUPLEXGEN_MAX_DIALOGUES:-12}}"
generate_dialogues="${DUPLEXGEN_GENERATE_DIALOGUES:-1}"
data_root="${DUPLEXGEN_DATA_ROOT:-}"
source_input_path="${DUPLEXGEN_SOURCE_INPUT_PATH:-}"
test_dialogues="${DUPLEXGEN_TEST_DIALOGUES:-1}"

if [[ "${mode}" != "generate" && "${mode}" != "translate" ]]; then
  echo "DUPLEXGEN_MODE must be generate or translate." >&2
  exit 2
fi
if [[ "${mode}" == "generate" && "${dataset}" != "synthetic" ]]; then
  echo "Generate mode requires DUPLEXGEN_DATASET=synthetic." >&2
  exit 2
fi
if [[ "${mode}" == "translate" && ( "${dataset}" == "synthetic" || "${dataset}" == "all" ) ]]; then
  echo "Translate mode requires one source dataset, not ${dataset}." >&2
  exit 2
fi

for executable in "${llm_python}" "${tts_python}"; do
  if [[ ! -x "${executable}" ]]; then
    echo "Python interpreter not found: ${executable}" >&2
    exit 2
  fi
done

if ! "${tts_python}" -c \
  'import omnivoice, silero_vad, whisperx, nltk, nemo_text_processing; from transformers import HiggsAudioV2TokenizerModel' \
  >/dev/null 2>&1; then
  echo "The existing TTS environment is missing full-render dependencies." >&2
  echo "Check OmniVoice Transformers support with:" >&2
  echo "  ${tts_python} -c 'from transformers import HiggsAudioV2TokenizerModel'" >&2
  echo "OmniVoice 0.2.1 requires transformers>=5.3.0." >&2
  exit 2
fi

if [[ "${DUPLEXGEN_LLM_BACKEND}" != "local" ]]; then
if ! "${llm_python}" - "${llm_base_url}" "${llm_api_key}" <<'PY'
import sys
import urllib.request

url = sys.argv[1].rstrip("/") + "/models"
request = urllib.request.Request(url)
if sys.argv[2]:
    request.add_header("Authorization", f"Bearer {sys.argv[2]}")
try:
    with urllib.request.urlopen(request, timeout=5) as response:
        if response.status >= 400:
            raise RuntimeError(f"HTTP {response.status}")
except Exception as exc:
    raise SystemExit(f"LLM endpoint is not ready: {url} ({exc})")
PY
then
  echo "Start the OpenAI-compatible LLM server, then rerun this script." >&2
  exit 2
fi
fi

mkdir -p "${stage_tt}" "${stage_bc}" "${audio_output}" "${log_dir}"
cd "${repo_root}"

if [[ "${mode}" == "generate" && "${generate_dialogues}" == "1" ]]; then
  echo "[0/3] Generate balanced Indonesian domain dialogues"
  "${llm_python}" tools/generate_indonesian_domain_dialogues.py \
    --output-root "${stage_input}" \
    --dataset "${dataset}" \
    --split "${split}" \
    --domains "${generate_domains}" \
    --num-dialogues "${num_dialogues}" \
    --model "${llm_model}" \
    --base-url "${llm_base_url}" \
    --api-key "${llm_api_key}" \
    2>&1 | tee "${log_dir}/stage0-domain-generation.log"
elif [[ "${mode}" == "generate" ]]; then
  echo "[0/3] Reuse existing domain dialogues from ${stage_input}"
else
  echo "[0/3] Translate the ${dataset} source corpus into spoken Indonesian"
  stage1_cmd=(
    "${llm_python}" -m src.speechify_run
    --dataset "${dataset}"
    --save_dir "${stage_input}"
    --language id
    --llm_model_name "${llm_model}"
    --base_url "${llm_base_url}"
    --api_key "${llm_api_key}"
    --max_train_samples "${num_dialogues}"
    --max_test_samples "${test_dialogues}"
  )
  if [[ -n "${data_root}" ]]; then
    stage1_cmd+=(--data-root "${data_root}")
  fi
  if [[ -n "${source_input_path}" ]]; then
    stage1_cmd+=(--input_path "${source_input_path}")
  fi
  "${stage1_cmd[@]}" 2>&1 | tee "${log_dir}/stage1-corpus-translation.log"
fi

stage4_cmd=(
  "${llm_python}" -m src.synthesis.run
  --dataset "${dataset}"
  --split "${split}"
  --input_root "${stage_input}"
  --save_root "${stage_tt}"
  --annotate_only
  --language id
  --llm_model_name "${llm_model}"
  --base_url "${llm_base_url}"
  --api_key "${llm_api_key}"
  --boundary_model_name "${llm_model}"
  --boundary_base_url "${llm_base_url}"
  --boundary_api_key "${llm_api_key}"
  --tt_model_name "${llm_model}"
  --tt_base_url "${llm_base_url}"
  --tt_api_key "${llm_api_key}"
)
if [[ -n "${max_dialogues}" ]]; then
  stage4_cmd+=(--max_dialogues "${max_dialogues}")
fi

echo "[1/3] LLM slot and turn-taking inference"
"${stage4_cmd[@]}" 2>&1 | tee "${log_dir}/stage4-turn-taking.log"

echo "[2/3] LLM Indonesian backchannel generation"
"${llm_python}" -m src.synthesis.run_add_bc \
  --dataset "${dataset}" \
  --split "${split}" \
  --input_root "${stage_tt}" \
  --output_root "${stage_bc}" \
  --model_name "${llm_model}" \
  --base_url "${llm_base_url}" \
  --api_key "${llm_api_key}" \
  --language id \
  2>&1 | tee "${log_dir}/stage4-backchannel.log"

input_glob="${stage_bc}/text_dialogue_${dataset}/${split}/*.json"
if ! compgen -G "${input_glob}" >/dev/null; then
  echo "No Stage 4 JSON files found: ${input_glob}" >&2
  exit 3
fi

if [[ "${pause_before_tts}" == "1" ]]; then
  if [[ ! -t 0 ]]; then
    echo "DUPLEXGEN_PAUSE_BEFORE_TTS=1 requires an interactive terminal." >&2
    exit 2
  fi
  echo "LLM work is complete. Stop the local LLM server to release both GPUs."
  read -r -p "Press Enter after nvidia-smi shows the GPUs are free... "
fi

echo "[3/3] OmniVoice rendering on two GPUs"
pids=()
for shard in 0 1; do
  log_file="${log_dir}/omnivoice-shard-${shard}.log"
  tts_cmd=(
    "${tts_python}" tts_render/convert_spoken.py
    --tts_backend omnivoice
    --omnivoice_voice_mode auto
    --language id
    --align_model m-salman-a/wav2vec2-xlsr-53-common-voice-indonesian
    --omnivoice_device cuda:0
    --omnivoice_align_device cuda
    --omnivoice_dtype float16
    --omnivoice_num_steps "${num_steps}"
    --input_glob "${input_glob}"
    --save_dir "${audio_output}"
    --num_variants 1
    --num_shards 2
    --shard_id "${shard}"
  )
  if [[ -n "${max_dialogues}" ]]; then
    tts_cmd+=(--max_dialogues "${max_dialogues}")
  fi

  CUDA_VISIBLE_DEVICES="${shard}" "${tts_cmd[@]}" >"${log_file}" 2>&1 &
  pids+=("$!")
  echo "  shard ${shard}: GPU ${shard}, log ${log_file}"
done

status=0
for shard in 0 1; do
  if ! wait "${pids[$shard]}"; then
    echo "OmniVoice shard ${shard} failed; see ${log_dir}/omnivoice-shard-${shard}.log" >&2
    status=1
  fi
done

if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi

echo "Complete"
echo "  turn-taking JSON: ${stage_tt}"
echo "  backchannel JSON:  ${stage_bc}"
echo "  audio:             ${audio_output}"
echo "  logs:              ${log_dir}"
