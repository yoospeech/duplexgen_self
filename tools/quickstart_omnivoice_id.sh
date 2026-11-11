#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python3.11}"
venv_dir="${OMNIVOICE_VENV:-${repo_root}/.venv-tts-omnivoice}"
input_json="${1:-${repo_root}/examples/indonesian_slot_dialogue.json}"
output_wav="${2:-${repo_root}/outputs/omnivoice-id-quickstart.wav}"

created_venv=0
if [[ ! -x "${venv_dir}/bin/python" ]]; then
  "${python_bin}" -m venv "${venv_dir}"
  created_venv=1
fi

venv_python="${venv_dir}/bin/python"
if [[ "${created_venv}" -eq 1 ]]; then
  "${venv_python}" -m pip install --upgrade pip
fi

if [[ -n "${OMNIVOICE_TORCH_INDEX:-}" ]]; then
  torch_index="${OMNIVOICE_TORCH_INDEX}"
elif command -v nvidia-smi >/dev/null 2>&1; then
  torch_index="https://download.pytorch.org/whl/cu128"
else
  torch_index="https://download.pytorch.org/whl/cpu"
fi

if ! "${venv_python}" -c 'import torch, torchaudio; assert torch.__version__.startswith("2.8.")' >/dev/null 2>&1; then
  "${venv_python}" -m pip install \
    torch==2.8.0 torchaudio==2.8.0 \
    --index-url "${torch_index}"
fi

if ! "${venv_python}" -c 'import omnivoice, soundfile' >/dev/null 2>&1; then
  "${venv_python}" -m pip install omnivoice==0.2.1 soundfile
fi

mkdir -p "$(dirname "${output_wav}")"
cd "${repo_root}"
"${venv_python}" tools/render_smoke_dialogue_omnivoice.py \
  --voice_mode auto \
  --input "${input_json}" \
  --output "${output_wav}" \
  --num_steps "${OMNIVOICE_NUM_STEPS:-16}" \
  --interrupt_overlap "${OMNIVOICE_INTERRUPT_OVERLAP:-0.6}"

echo "Quickstart complete: ${output_wav}"
echo "Event metadata: ${output_wav%.wav}.json"
