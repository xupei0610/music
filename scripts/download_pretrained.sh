#!/usr/bin/env bash
# Download the MUSIC pretrained policies (Hugging Face: xupei0610/MUSIC)
# into ./pretrained (or below --dest), giving pretrained/<type>/<file>.ckpt.
#
# Needs only uv (uses `uvx --from huggingface-hub hf`). All files: about 1.0 GB.
#
#   scripts/download_pretrained.sh                         # everything
#   scripts/download_pretrained.sh --type joint_hand       # one type
#   scripts/download_pretrained.sh --type joint_fingering --pieces 017-1 046-1
#   scripts/download_pretrained.sh --type muscle_tracking
#
# Types: joint_fingering, joint_hand, muscle_tracking. --pieces applies to the
# two joint types. A second run skips the files that are already present.
set -euo pipefail

REPO="xupei0610/MUSIC"
DEST="."
TYPES=(joint_fingering joint_hand muscle_tracking)
PIECES=()

usage() { sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest)   DEST="$2"; shift 2 ;;
    --type)   case "$2" in joint_fingering|joint_hand|muscle_tracking) TYPES=("$2") ;;
                *) echo "--type must be joint_fingering, joint_hand or muscle_tracking" >&2; exit 2 ;; esac; shift 2 ;;
    --pieces) shift; while [[ $# -gt 0 && "$1" != --* ]]; do PIECES+=("$1"); shift; done ;;
    -h|--help) usage 0 ;;
    *) echo "unknown argument: $1" >&2; usage 2 ;;
  esac
done

if ! command -v uvx >/dev/null 2>&1; then
  echo "uvx not found; install uv: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

INCLUDE=()
for t in "${TYPES[@]}"; do
  if [[ ${#PIECES[@]} -eq 0 || "$t" == muscle_tracking ]]; then
    INCLUDE+=("pretrained/${t}/*.ckpt")
  else
    suffix=${t#joint_}
    for p in "${PIECES[@]}"; do INCLUDE+=("pretrained/${t}/${p}_${suffix}.ckpt"); done
  fi
done
INCLUDE_ARGS=()
for p in "${INCLUDE[@]}"; do INCLUDE_ARGS+=(--include "$p"); done  # one pattern per flag

mkdir -p "$DEST"
echo "Downloading ${REPO} -> ${DEST}: ${INCLUDE[*]}"
# The dataset is public: do not send a cached login token (an expired one makes
# the Hub answer 401 / "not found"). An explicit HF_TOKEN is still used.
if [[ -z "${HF_TOKEN:-}" ]]; then export HF_HUB_DISABLE_IMPLICIT_TOKEN=1; fi
# --no-config: ignore private package indexes in the user's uv config
uvx --no-config --from huggingface-hub hf download "$REPO" \
  --repo-type dataset --local-dir "$DEST" "${INCLUDE_ARGS[@]}"

for t in "${TYPES[@]}"; do
  n=$(find "$DEST/pretrained/$t" -name '*.ckpt' 2>/dev/null | wc -l | tr -d ' ')
  echo "pretrained/${t}: ${n} files"
done
