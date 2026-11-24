#!/usr/bin/env bash
#
# process.sh — the one entry point for turning a recording into notes.
#
# This is the SINGLE SOURCE OF TRUTH that both the DropMemo macOS app and the
# Voice Memos Shortcut call. It wraps `uv run` + notetaker.py with
# Tom's defaults (Claude subscription backend, a fresh timestamped out-dir) so callers
# only ever pass an audio path. Any extra flags are forwarded verbatim to
# notetaker.py, so `--denoise`, `--language nl`, `--diarize-engine sortformer`,
# `--llm-backend anthropic`, etc. all still work through here.
#
# Usage:
#   scripts/process.sh /path/to/recording.m4a [extra notetaker flags...]
#
# Override defaults via env (handy from the app's controls strip / a Shortcut):
#   LLM_BACKEND    openai | ollama | claude-cli | anthropic  (default: claude-cli;
#                  openai and ollama both mean local Ollama here. claude-cli
#                  routes via the `claude` binary / your subscription: $0 on a
#                  Pro/Max plan and, unlike Ollama, it costs this Mac no RAM)
#   CLAUDE_MODEL   model for claude-cli    (default: sonnet)
#   OLLAMA_MODEL   local model name        (default: qwen3:8b, ~5 GB resident)
#   OLLAMA_URL     OpenAI-compatible URL   (default: http://localhost:11434)
#   OUT_ROOT       where out-dirs are made (default: the repo root)
#
# On OLLAMA_MODEL: keep it small enough to fit beside whatever else is open.
# qwen3-coder:latest is 18 GB and was the default until 2026-08-27, when it
# wedged a 24 GB Mac hard enough to take WindowServer down with it.
#
# The LAST line printed is machine-readable for the caller:
#   OUT_DIR=/abs/path/to/out-YYYYmmdd-HHMMSS
# DropMemo parses that to reveal the folder in Finder and open notes.md.

set -euo pipefail

# An app launched from Finder/Dock (DropMemo) or a Shortcut inherits the bare GUI
# PATH (/usr/bin:/bin:/usr/sbin:/sbin), not the shell's, so Homebrew's `speech`
# and `ffmpeg` and ~/.local/bin's `claude` are invisible and notetaker dies on
# "required binary 'speech' not found on PATH". Put the usual install dirs first.
export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$PATH"

# --- locate the repo (this script lives in <repo>/scripts) -------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"

# uv owns the environment (pyproject.toml + uv.lock): `uv run` builds or repairs
# .venv on demand, so a deleted venv no longer stops a recording. Until
# 2026-11-25 this was a hand-made pip venv, and when .venv vanished every run
# died on "venv python not found". The llm-seam extra is included because the
# default backend (claude-cli) needs it.
UV="$(command -v uv || true)"
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
if [[ -z "$UV" ]]; then
  echo "error: uv not found on PATH" >&2
  echo "       run:  brew install uv" >&2
  exit 1
fi

# --- defaults (overridable via env) ------------------------------------------
LLM_BACKEND="${LLM_BACKEND:-claude-cli}"
CLAUDE_MODEL="${CLAUDE_MODEL:-sonnet}"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3:8b}"
OUT_ROOT="${OUT_ROOT:-$REPO}"

STAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="$OUT_ROOT/out-$STAMP"

# LLM_BACKEND=openai has always meant "local Ollama" in this script (it is what
# OLLAMA_URL / OLLAMA_MODEL configure, and what DropMemo sends). notetaker's
# name for that is now `ollama`: Ollama's native API, which can size the
# context window to the meeting, where the OpenAI-compatible one silently cut a
# long transcript. `ollama` is accepted here too. To force the old path, or to
# reach a different OpenAI-compatible server, append `--llm-backend openai`.
[[ "$LLM_BACKEND" == "ollama" ]] && LLM_BACKEND="openai"
NOTETAKER_BACKEND="$LLM_BACKEND"
[[ "$LLM_BACKEND" == "openai" ]] && NOTETAKER_BACKEND="ollama"

# --- assemble the notetaker invocation ---------------------------------------
ARGS=(run "$AUDIO" --out-dir "$OUT_DIR" --llm-backend "$NOTETAKER_BACKEND")

# Only the OpenAI-compatible (Ollama) path needs base-url + local model name;
# for --llm-backend anthropic the model id comes from the user's extra flags.
if [[ "$LLM_BACKEND" == "openai" ]]; then
  ARGS+=(--base-url "$OLLAMA_URL" --model-llm "$OLLAMA_MODEL")
elif [[ "$LLM_BACKEND" == "claude-cli" ]]; then
  ARGS+=(--model-llm "$CLAUDE_MODEL")
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
elif [[ "$LLM_BACKEND" == "claude-cli" ]]; then
  echo "  backend : claude-cli ($CLAUDE_MODEL, via your subscription)"
else
  echo "  backend : $LLM_BACKEND"
fi
echo "  out-dir : $OUT_DIR"
echo

# -u keeps python unbuffered so the app's live log view streams in real time.
"$UV" run --project "$REPO" --extra llm-seam --quiet python -u "$NOTETAKER" "${ARGS[@]}"

echo
echo "OUT_DIR=$OUT_DIR"
