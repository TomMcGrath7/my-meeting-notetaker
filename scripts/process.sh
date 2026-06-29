#!/usr/bin/env bash
#
# process.sh — the one entry point for turning a recording into notes.
#
# This is the SINGLE SOURCE OF TRUTH that both the DropMemo macOS app and the
# Voice Memos Shortcut call. It wraps the repo's venv python + notetaker.py with
# Tom's defaults (local Ollama backend, a fresh timestamped out-dir) so callers
# only ever pass an audio path. Any extra flags are forwarded verbatim to
# notetaker.py, so `--denoise`, `--language nl`, `--diarize-engine sortformer`,
# `--llm-backend anthropic`, etc. all still work through here.
#
# Usage:
#   scripts/process.sh /path/to/recording.m4a [extra notetaker flags...]
#
# Override defaults via env (handy from the app's controls strip / a Shortcut):
#   OLLAMA_MODEL   local model name        (default: qwen3-coder:latest)
#   OLLAMA_URL     OpenAI-compatible URL   (default: http://localhost:11434)
#   LLM_BACKEND    openai | claude-cli | anthropic  (default: openai = Ollama;
#                  claude-cli routes via the `claude` binary / your subscription)
#   OUT_ROOT       where out-dirs are made (default: the repo root)
#
# The LAST line printed is machine-readable for the caller:
#   OUT_DIR=/abs/path/to/out-YYYYmmdd-HHMMSS
# DropMemo parses that to reveal the folder in Finder and open notes.md.

set -euo pipefail

# --- locate the repo (this script lives in <repo>/scripts) -------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"

PY="$REPO/.venv/bin/python3"
NOTETAKER="$REPO/notetaker.py"

# --- args --------------------------------------------------------------------
if [[ $# -lt 1 ]]; then
  echo "usage: $(basename "$0") <audio-file> [extra notetaker flags...]" >&2
  exit 2
fi
AUDIO="$1"; shift

if [[ ! -f "$AUDIO" ]]; then
  echo "error: audio file not found: $AUDIO" >&2
  exit 1
fi
if [[ ! -x "$PY" ]]; then
  echo "error: venv python not found at $PY" >&2
  echo "       run:  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

# --- defaults (overridable via env) ------------------------------------------
LLM_BACKEND="${LLM_BACKEND:-openai}"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3-coder:latest}"
OUT_ROOT="${OUT_ROOT:-$REPO}"

STAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="$OUT_ROOT/out-$STAMP"

# --- assemble the notetaker invocation ---------------------------------------
ARGS=(run "$AUDIO" --out-dir "$OUT_DIR" --llm-backend "$LLM_BACKEND")

# Only the OpenAI-compatible (Ollama) path needs base-url + local model name;
# for --llm-backend anthropic the model id comes from the user's extra flags.
if [[ "$LLM_BACKEND" == "openai" ]]; then
  ARGS+=(--base-url "$OLLAMA_URL" --model-llm "$OLLAMA_MODEL")
fi

# Use config.yaml automatically if the user has filled one in (it's gitignored).
if [[ -f "$REPO/config.yaml" ]]; then
  ARGS+=(--config "$REPO/config.yaml")
fi

# Forward any extra flags the caller passed (can override the defaults above,
# since notetaker's argparse takes the last value for repeated options).
ARGS+=("$@")

echo "▶ DropMemo / process.sh"
echo "  audio   : $AUDIO"
if [[ "$LLM_BACKEND" == "openai" ]]; then
  echo "  backend : openai ($OLLAMA_MODEL @ $OLLAMA_URL)"
else
  echo "  backend : $LLM_BACKEND"
fi
echo "  out-dir : $OUT_DIR"
echo

# -u keeps python unbuffered so the app's live log view streams in real time.
"$PY" -u "$NOTETAKER" "${ARGS[@]}"

echo
echo "OUT_DIR=$OUT_DIR"
