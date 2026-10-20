#!/usr/bin/env bash
# Run MuJoCo's mjpython (needed for the viewer on macOS) from this repo's uv venv.
#
#   scripts/mjpython.sh main.py cfg/joint.py --backend mujoco --device cpu --test ...
#
# The uv-managed Python links libpython3.12.dylib through @executable_path/../lib.
# mjpython resolves that path from the venv symlink (.venv/bin), not from the real
# interpreter, so the library is not found ("Library not loaded: @rpath/libpython").
# Adding the interpreter's lib directory to DYLD_FALLBACK_LIBRARY_PATH fixes it.
set -euo pipefail
cd "$(dirname "$0")/.."
lib="$(.venv/bin/python -c 'import sys; print(sys.base_prefix)')/lib"
export DYLD_FALLBACK_LIBRARY_PATH="$lib:${DYLD_FALLBACK_LIBRARY_PATH:-/usr/local/lib:/usr/lib}"
exec .venv/bin/mjpython "$@"
