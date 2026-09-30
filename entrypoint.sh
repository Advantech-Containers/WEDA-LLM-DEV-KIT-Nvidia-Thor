#!/usr/bin/env bash

set -euo pipefail

cat <<'BANNER'
======================================================================
  WISE-Edge WEDA-LLM — vLLM 0.20 / Transformers 5.6 / Jetson Thor
======================================================================
BANNER

TRITON_BIN="/opt/venv/lib/python3.12/site-packages/triton/backends/nvidia/bin"
SYS_PTXAS="/usr/local/cuda/bin/ptxas"
if [[ -x "${SYS_PTXAS}" && -d "${TRITON_BIN}" ]]; then
  for variant in ptxas-blackwell ptxas; do
    target="${TRITON_BIN}/${variant}"
    if [[ -e "${target}" && ! -L "${target}" ]]; then
      mv "${target}" "${target}.orig"
    fi
    if [[ ! -e "${target}" ]]; then
      ln -s "${SYS_PTXAS}" "${target}"
    fi
  done
  echo "[entrypoint] Triton ptxas patched -> ${SYS_PTXAS} (sm_110a-aware)"
else
  echo "[entrypoint] WARN: ${SYS_PTXAS} not found - Triton JIT may fail on sm_110a."
  echo "[entrypoint]       Ensure container runs with --runtime nvidia --gpus all."
fi

python3 - <<'PY'
import vllm, torch, transformers
print(f"  vLLM         {vllm.__version__}")
print(f"  PyTorch      {torch.__version__}  (CUDA {torch.version.cuda})")
print(f"  Transformers {transformers.__version__}")
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    mem  = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"  Device       {name}  cap={cap}  total={mem:.1f} GiB")
else:
    print("  Device       (no CUDA detected -- did you forget --gpus all?)")
PY
echo

MODE="${1:-serve}"
shift || true

echo "[entrypoint] Environment ready."
echo "======================================================================"

case "${MODE}" in
    serve)
      export TORCHINDUCTOR_FREEZING="${WEDA_LLM_INDUCTOR_FREEZING:-0}"
    export TORCHINDUCTOR_FX_GRAPH_CACHE="${TORCHINDUCTOR_FX_GRAPH_CACHE:-1}"
    export VLLM_USE_FLASHINFER_MOE_FP4="${VLLM_USE_FLASHINFER_MOE_FP4:-0}"
    export VLLM_NVFP4_GEMM_BACKEND="${VLLM_NVFP4_GEMM_BACKEND:-cutlass}"



    if [[ ! -d "${HUGGINGFACE_HUB_CACHE:-}" ]]; then
      echo "[entrypoint] WARN: HUGGINGFACE_HUB_CACHE not present or not set."
      echo "[entrypoint]       vllm will try to download - slow + needs network."
    fi

    echo "[entrypoint] Delegating model server startup to config/loader.py via scripts/serve.py..."
    exec python3 /wise-edge/weda-llm-base/scripts/serve.py "$@"
    ;;

  bench)
    exec vllm bench "$@"
    ;;

  shell|bash)
    exec /bin/bash "$@"
    ;;

  python|python3)
    exec python3 "$@"
    ;;

  *)
    exec "${MODE}" "$@"
    ;;
esac
