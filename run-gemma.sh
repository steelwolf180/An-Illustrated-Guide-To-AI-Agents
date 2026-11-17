#!/usr/bin/env bash
# run-gemma.sh - start llama-server with a Gemma 3 or Gemma 4 model.
#
# Usage:
#   ./run-gemma.sh list
#   ./run-gemma.sh <model> [-p port] [-c ctx] [-t threads] [-v] [-- extra llama-server flags]
#
# Models: g3-1b  g3-4b  g4-e2b  g4-e4b
#
# Notes:
#   - Gemma 4 needs a recent llama.cpp (build from source). The old Debian
#     package (build 8681) fails with "missing tensor 'blk.15.attn_k.weight'".
#   - Set LLAMA_SERVER to point at a specific binary. Default is the source
#     build in ~/project/llama.cpp, falling back to llama-server on your PATH.
#   - Vision (mmproj) is skipped unless you pass -v.

set -euo pipefail

LLAMA_SERVER="${LLAMA_SERVER:-$HOME/project/llama.cpp/build/bin/llama-server}"
if [[ ! -x "$LLAMA_SERVER" ]]; then
  LLAMA_SERVER="$(command -v llama-server || true)"
fi

# model key -> Hugging Face repo:quant
declare -A REPO=(
  [g3-1b]="ggml-org/gemma-3-1b-it-GGUF"
  [g3-4b]="unsloth/gemma-3-4b-it-GGUF:Q4_K_M"
  [g4-e2b]="ggml-org/gemma-4-E2B-it-GGUF:Q4_0"
  [g4-e4b]="unsloth/gemma-4-E4B-it-GGUF:Q4_K_M"
)

# rough RAM needed in GB (model + context cache + overhead)
declare -A NEED_GB=(
  [g3-1b]=1 [g3-4b]=4 [g4-e2b]=3.5 [g4-e4b]=6
)

ORDER=(g3-1b g3-4b g4-e2b g4-e4b)

usage() {
  sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

list_models() {
  printf '%-8s %-45s %s\n' MODEL REPO NEED_RAM_GB
  for k in "${ORDER[@]}"; do
    printf '%-8s %-45s %s\n' "$k" "${REPO[$k]}" "${NEED_GB[$k]}"
  done
}

[[ $# -ge 1 ]] || usage 1
case "$1" in
  -h|--help|help) usage 0 ;;
  list) list_models; exit 0 ;;
esac

MODEL="$1"; shift
if [[ -z "${REPO[$MODEL]:-}" ]]; then
  echo "Unknown model: $MODEL" >&2
  list_models >&2
  exit 1
fi

PORT=11434
CTX=4096
THREADS=4
MMPROJ="--no-mmproj"

while getopts ":p:c:t:vh" opt; do
  case "$opt" in
    p) PORT="$OPTARG" ;;
    c) CTX="$OPTARG" ;;
    t) THREADS="$OPTARG" ;;
    v) MMPROJ="" ;;
    h) usage 0 ;;
    \?) echo "Unknown option: -$OPTARG" >&2; usage 1 ;;
    :)  echo "Option -$OPTARG needs a value" >&2; usage 1 ;;
  esac
done
shift $((OPTIND - 1))
[[ "${1:-}" == "--" ]] && shift

if [[ -z "$LLAMA_SERVER" || ! -x "$LLAMA_SERVER" ]]; then
  echo "llama-server not found. Set LLAMA_SERVER=/path/to/llama-server" >&2
  exit 1
fi

# Gemma 4 won't load on the old distro build
if [[ "$MODEL" == g4-* && "$LLAMA_SERVER" == /usr/bin/* ]]; then
  echo "Warning: $LLAMA_SERVER is the distro package (likely too old for Gemma 4)." >&2
  echo "         Build llama.cpp from source and set LLAMA_SERVER." >&2
fi

# Port already taken?
if ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${PORT}$"; then
  echo "Port $PORT is already in use (another llama-server or Ollama?)." >&2
  echo "Check with: ss -ltnp | grep $PORT" >&2
  exit 1
fi

# RAM check
avail_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
avail_gb=$(awk -v kb="$avail_kb" 'BEGIN {printf "%.1f", kb/1024/1024}')
if awk -v a="$avail_gb" -v n="${NEED_GB[$MODEL]}" 'BEGIN {exit !(a < n)}'; then
  echo "Warning: ${avail_gb} GB RAM available, $MODEL wants about ${NEED_GB[$MODEL]} GB." >&2
  echo "         Close other apps or pick a smaller model, or it may swap and crawl." >&2
fi

echo "Starting $MODEL (${REPO[$MODEL]}) on port $PORT, ctx=$CTX, threads=$THREADS"
echo "Using: $LLAMA_SERVER"

# shellcheck disable=SC2086
exec "$LLAMA_SERVER" \
  -hf "${REPO[$MODEL]}" \
  --port "$PORT" \
  -c "$CTX" \
  -np 1 \
  -t "$THREADS" \
  --cache-ram 0 \
  $MMPROJ \
  "$@"