#!/bin/bash
#SBATCH --job-name=qwen3-sog-eval
#SBATCH --partition=gpu300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --time=24:00:00
#SBATCH --output=configs/slurm/logs/%x-%j.out
#SBATCH --error=configs/slurm/logs/%x-%j.err

set -euo pipefail
export PYTHONPATH=/mnt/c/Users/HP/Documents/Codex/2026-06-29/ni/src:/mnt/c/Users/HP/Documents/Codex/2026-06-29/ni:

REPO_ROOT="${REPO_ROOT:-/public/home/xty/workdir/omni_goose/SocialOmni}"
cd "$REPO_ROOT"
mkdir -p configs/slurm/logs

WORKDIR="${WORKDIR:-runs/social_omni_goose_eval_qwen3omni_pass001}"
TRACK="${TRACK:-structured}"
LIMIT="${LIMIT:-}"
SKIP="${SKIP:-0}"
STRIDE="${STRIDE:-1}"
RESUME="${RESUME:-1}"
OVERWRITE="${OVERWRITE:-0}"
INPUT_OVERRIDE="${INPUT_OVERRIDE:-}"
OUTPUT_OVERRIDE="${OUTPUT_OVERRIDE:-}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/publicssd/xty/models/Qwen3-Omni-30B-A3B-Instruct}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-$((6100 + (${SLURM_JOB_ID:-0} % 300)))}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-2400}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export QWEN3_OMNI_MODEL_PATH="$MODEL_PATH"
export QWEN3_OMNI_MAX_TOKENS="${QWEN3_OMNI_MAX_TOKENS:-16384}"
export QWEN3_OMNI_TEXT_MERGE_MAX_TOKENS="${QWEN3_OMNI_TEXT_MERGE_MAX_TOKENS:-32768}"
export QWEN3_OMNI_VIDEO_FPS="${QWEN3_OMNI_VIDEO_FPS:-1.0}"
export QWEN3_OMNI_VIDEO_MAX_FRAMES="${QWEN3_OMNI_VIDEO_MAX_FRAMES:-128}"
export QWEN3_OMNI_VIDEO_MAX_PIXELS="${QWEN3_OMNI_VIDEO_MAX_PIXELS:-401408}"
export OMNI_HTTP_TIMEOUT_SEC="${OMNI_HTTP_TIMEOUT_SEC:-1200}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export QWEN3_OMNI_SERVER_URL="http://${SERVER_HOST}:${SERVER_PORT}"
export HF_MODULES_CACHE="$REPO_ROOT/.cache/huggingface/modules/${SLURM_JOB_ID:-manual}_sog_eval"
mkdir -p "$HF_MODULES_CACHE" "$WORKDIR/responses/$TRACK" "$WORKDIR/manifests" "$WORKDIR/logs"

case "$TRACK" in
  structured)
    INPUT="$WORKDIR/inputs/structured_leaderboard_core.jsonl"
    OUTPUT="$WORKDIR/responses/structured/responses_skip${SKIP}_stride${STRIDE}.jsonl"
    ;;
  raw_video_smoke)
    INPUT="$WORKDIR/inputs/raw_video_smoke.jsonl"
    OUTPUT="$WORKDIR/responses/raw_video_smoke/responses_skip${SKIP}_stride${STRIDE}.jsonl"
    ;;
  *)
    echo "Unsupported TRACK=$TRACK" >&2
    exit 2
    ;;
esac

if [[ -n "$INPUT_OVERRIDE" ]]; then
  INPUT="$INPUT_OVERRIDE"
fi
if [[ -n "$OUTPUT_OVERRIDE" ]]; then
  OUTPUT="$OUTPUT_OVERRIDE"
fi

TEXT_CONTEXT_VIDEO="${TEXT_CONTEXT_VIDEO:-}"
if [[ -z "$TEXT_CONTEXT_VIDEO" ]]; then
  TEXT_CONTEXT_VIDEO="$(find "$WORKDIR/data/release_benchmark_v2/inputs/videos" -name '*.mp4' -print -quit)"
fi

cat <<INFO
Job ID: ${SLURM_JOB_ID:-manual}
TRACK: $TRACK
WORKDIR: $WORKDIR
INPUT: $INPUT
OUTPUT: $OUTPUT
SKIP: $SKIP
STRIDE: $STRIDE
LIMIT: ${LIMIT:-all}
MODEL_PATH: $MODEL_PATH
SERVER_URL: $QWEN3_OMNI_SERVER_URL
MAX_TOKENS: $QWEN3_OMNI_MAX_TOKENS
TEXT_MERGE_MAX_TOKENS: $QWEN3_OMNI_TEXT_MERGE_MAX_TOKENS
VIDEO_FPS: $QWEN3_OMNI_VIDEO_FPS
VIDEO_MAX_FRAMES: $QWEN3_OMNI_VIDEO_MAX_FRAMES
VIDEO_MAX_PIXELS: $QWEN3_OMNI_VIDEO_MAX_PIXELS
TEXT_CONTEXT_VIDEO: $TEXT_CONTEXT_VIDEO
INFO
nvidia-smi || true

server_log="configs/slurm/logs/qwen3-sog-eval-server-${SLURM_JOB_ID:-manual}.log"
"$PYTHON_BIN" -u integrations/models/model_server/qwen3_omni/qwen3_omni_server.py --host "$SERVER_HOST" --port "$SERVER_PORT" > "$server_log" 2>&1 &
server_pid="$!"
cleanup() {
  if kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

"$PYTHON_BIN" - <<PY_HEALTH
import requests, sys, time
url = "${QWEN3_OMNI_SERVER_URL}/health"
deadline = time.time() + int("${SERVER_TIMEOUT}")
while time.time() < deadline:
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200 and resp.json().get("model_loaded"):
            print("server_ready=true")
            sys.exit(0)
    except Exception:
        pass
    time.sleep(5)
print(f"server_ready=false url={url}", file=sys.stderr)
sys.exit(1)
PY_HEALTH

cmd=("$PYTHON_BIN" -u tools/eval/run_social_omni_goose_qwen_eval.py --input "$INPUT" --output "$OUTPUT" --track "$TRACK" --backend local --server-url "$QWEN3_OMNI_SERVER_URL" --model "Qwen3-Omni-30B-A3B-Instruct" --text-context-video "$TEXT_CONTEXT_VIDEO" --skip "$SKIP" --stride "$STRIDE" --job-manifest "$WORKDIR/manifests/job_manifest.jsonl")
if [[ -n "$LIMIT" ]]; then cmd+=(--limit "$LIMIT"); fi
if [[ "$RESUME" == "1" || "$RESUME" == "true" ]]; then cmd+=(--resume); fi
if [[ "$OVERWRITE" == "1" || "$OVERWRITE" == "true" ]]; then cmd+=(--overwrite); fi
"${cmd[@]}"
