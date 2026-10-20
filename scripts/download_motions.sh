#!/usr/bin/env bash
# Download the MUSIC reference motion dataset (Hugging Face: xupei0610/MUSIC)
# into ./motions (or --dest), giving motions/{left,right}_hand_motions/*.joblib.
#
# Needs only uv (uses `uvx --from huggingface-hub hf`). The full set is ~8.6 GB.
#
#   scripts/download_motions.sh                    # everything -> ./motions
#   scripts/download_motions.sh --hand right       # right hand only
#   scripts/download_motions.sh --files 000 001    # selected clips, both hands
#   scripts/download_motions.sh --dest <dir> --hand left --files 000
#
# Re-running resumes/skips files already downloaded. Set HF_TOKEN if the
# dataset ever needs authentication.
set -euo pipefail

REPO="xupei0610/MUSIC"
DEST="motions"
HANDS=(left right)
FILES=()

usage() { sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest)  DEST="$2"; shift 2 ;;
    --hand)  case "$2" in left|right) HANDS=("$2") ;; both) HANDS=(left right) ;;
               *) echo "--hand must be left, right or both" >&2; exit 2 ;; esac; shift 2 ;;
    --files) shift; while [[ $# -gt 0 && "$1" != --* ]]; do FILES+=("${1%.joblib}"); shift; done ;;
    -h|--help) usage 0 ;;
    *) echo "unknown argument: $1" >&2; usage 2 ;;
  esac
done

if ! command -v uvx >/dev/null 2>&1; then
  echo "uvx not found; install uv: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

INCLUDE=()
for h in "${HANDS[@]}"; do
  if [[ ${#FILES[@]} -eq 0 ]]; then
    INCLUDE+=("${h}_hand_motions/*")
  else
    for f in "${FILES[@]}"; do INCLUDE+=("${h}_hand_motions/${f}.joblib"); done
  fi
done
INCLUDE_ARGS=()
for p in "${INCLUDE[@]}"; do INCLUDE_ARGS+=(--include "$p"); done  # one pattern per flag

mkdir -p "$DEST"
echo "Downloading ${REPO} -> ${DEST}: ${INCLUDE[*]}"
# The dataset is public: don't send a cached login token (an expired one makes
# the Hub answer 401 / "not found"). An explicit HF_TOKEN is still used.
if [[ -z "${HF_TOKEN:-}" ]]; then export HF_HUB_DISABLE_IMPLICIT_TOKEN=1; fi
# --no-config: ignore private package indexes in the user's uv config
uvx --no-config --from huggingface-hub hf download "$REPO" \
  --repo-type dataset --local-dir "$DEST" "${INCLUDE_ARGS[@]}"

for h in "${HANDS[@]}"; do
  n=$(find "$DEST/${h}_hand_motions" -name '*.joblib' 2>/dev/null | wc -l | tr -d ' ')
  echo "${h}_hand_motions: ${n} files"
done
