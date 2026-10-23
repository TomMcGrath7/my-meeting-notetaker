#!/usr/bin/env python3
"""
notetaker.py — on-device post-meeting note taker for macOS / Apple Silicon.

Pipeline (each stage runs alone, so peak RAM ~= the single biggest model):

    audio.m4a
      └─ (optional) speech denoise          -> clean.wav
      └─ speech diarize --json              -> who spoke when   (turns)
      └─ speech vad                         -> where the speech is
      └─ speech align, speech only          -> word timestamps  (words)
      └─ bucket words into turns            -> speaker segments  [local]
      └─ LLM: map SPEAKER_xx -> real names, flag low-confidence  [text only]
      └─ LLM: English transcript, if the memo isn't English      [text only]
      └─ LLM: meeting notes (summary / decisions / actions)      [text only]

Only TEXT ever leaves the device (the LLM steps). Audio + diarization +
ASR are 100% local via the `speech` binary (github.com/soniqo/speech-swift).
Point --llm-backend at a local model (ollama, or openai for any other
OpenAI-compatible server) to keep it ALL local.

The two parsers most likely to need tweaking for your `speech` version are
parse_diarize() and parse_align(). Run `notetaker.py check <file>` first — it
prints the raw tool output next to what we parsed so you can confirm in ~30s.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import wave
import urllib.request
import urllib.error
from dataclasses import dataclass, asdict, field, replace
from pathlib import Path

# llm-seam (Tom's shared LLM wrapper, a private repo) is OPTIONAL and is
# imported where it is used, never here: people with access to this repo but
# not to llm-seam must still be able to run the notetaker. See _llm_seam().

# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #

def sh(cmd: list[str], capture: bool = True) -> str:
    """Run a command, return stdout. Raise with stderr on failure."""
    proc = subprocess.run(cmd, capture_output=capture, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"--- stderr ---\n{proc.stderr}"
        )
    return proc.stdout or ""

def need(binary: str) -> None:
    if shutil.which(binary) is None:
        sys.exit(f"error: required binary '{binary}' not found on PATH.")

def load_config(path: Path) -> dict:
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
            return yaml.safe_load(text) or {}
        except ImportError:
            sys.exit("config is YAML but PyYAML isn't installed (`pip install pyyaml`), "
                     "or rename your config to .json.")
    return json.loads(text)

# --------------------------------------------------------------------------- #
# data model — the canonical intermediate format
# --------------------------------------------------------------------------- #

@dataclass
class Turn:
    speaker: str       # raw diarizer label, e.g. SPEAKER_00
    start: float
    end: float

@dataclass
class Word:
    text: str
    start: float
    end: float

@dataclass
class Segment:
    speaker: str       # raw label until the LLM renames it
    start: float
    end: float
    text: str
    flagged: bool = False
    flag_reason: str = ""

@dataclass
class Pipeline:
    audio: str
    turns: list[Turn] = field(default_factory=list)
    words: list[Word] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)
    label_map: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({
            "audio": self.audio,
            "turns": [asdict(t) for t in self.turns],
            "words": [asdict(w) for w in self.words],
            "segments": [asdict(s) for s in self.segments],
            "label_map": self.label_map,
        }, indent=2)

    @classmethod
    def from_json(cls, blob: str) -> "Pipeline":
        d = json.loads(blob)
        p = cls(audio=d["audio"])
        p.turns = [Turn(**t) for t in d.get("turns", [])]
        p.words = [Word(**w) for w in d.get("words", [])]
        p.segments = [Segment(**s) for s in d.get("segments", [])]
        p.label_map = d.get("label_map", {})
        return p

# --------------------------------------------------------------------------- #
# stage 1 — audio prep
# --------------------------------------------------------------------------- #

def to_wav16k(src: Path, out: Path) -> Path:
    """Normalise to 16 kHz mono WAV (what the models expect). Needs ffmpeg."""
    need("ffmpeg")
    sh(["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", "16000",
        "-f", "wav", str(out)], capture=True)
    return out

def denoise(wav: Path, out: Path) -> Path:
    sh(["speech", "denoise", str(wav), "-o", str(out)])
    return out

# --------------------------------------------------------------------------- #
# stage 2 — diarization  (speech diarize --json)
# --------------------------------------------------------------------------- #

def run_diarize(wav: Path, engine: str, vad_filter: bool,
                cluster_threshold: float | None = None) -> str:
    cmd = ["speech", "diarize", str(wav), "--engine", engine, "--json"]
    if vad_filter and engine == "pyannote":
        cmd.append("--vad-filter")
    if cluster_threshold is not None:
        # speech diarize over-splits one voice into several clusters; a lower
        # threshold yields fewer speakers. Push it toward your known head count.
        cmd += ["--cluster-threshold", str(cluster_threshold)]
    return sh(cmd)

def _first(d: dict, *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default

def _extract_json(raw: str):
    """Decode the first JSON value embedded in `raw`.

    `speech` prints human progress ("Loading audio…", "[62%] Downloading…",
    "Running diarization…") to STDOUT *before* the JSON, so the captured output
    is progress-text-then-JSON, not pure JSON. Skip to the first '{' or '[' and
    raw_decode from there, ignoring any trailing text.
    """
    for i, ch in enumerate(raw):
        if ch in "{[":
            try:
                obj, _ = json.JSONDecoder().raw_decode(raw, i)
                return obj
            except json.JSONDecodeError:
                continue
    raise json.JSONDecodeError("no JSON value found in output", raw, 0)

def parse_diarize(raw: str) -> list[Turn]:
    """
    Tolerant parser for `speech diarize --json`. Accepts a top-level list, or a
    dict wrapping the list under common keys, and normalises speaker/time keys.
    If your build emits different field names, this is the one place to adjust.
    """
    data = _extract_json(raw)
    if isinstance(data, dict):
        data = _first(data, "segments", "turns", "diarization", "results", default=data)
    turns: list[Turn] = []
    for item in data:
        spk = _first(item, "speaker", "speakerId", "speaker_id", "label", default="SPEAKER_00")
        start = float(_first(item, "start", "startTime", "start_time", "from", default=0.0))
        end = float(_first(item, "end", "endTime", "end_time", "to", default=start))
        turns.append(Turn(speaker=_norm_label(str(spk)), start=start, end=end))
    turns.sort(key=lambda t: t.start)
    return turns

def _norm_label(s: str) -> str:
    """Normalise speaker labels to SPEAKER_NN so downstream is predictable."""
    s = s.strip()
    if re.fullmatch(r"\d+", s):
        return f"SPEAKER_{int(s):02d}"
    m = re.search(r"(\d+)", s)
    if m and re.search(r"speaker", s, re.I):
        return f"SPEAKER_{int(m.group(1)):02d}"
    return s

# --------------------------------------------------------------------------- #
# stage 3 — word timestamps  (speech align)
# --------------------------------------------------------------------------- #

# matches lines like:  [12.34s - 12.81s] hello
_ALIGN_RE = re.compile(r"\[\s*([\d.]+)\s*s?\s*-\s*([\d.]+)\s*s?\s*\]\s*(.+?)\s*$")

# speech align — whether it transcribes or force-aligns provided --text — only
# handles a few minutes of audio per call. On a long recording it crushes every
# word timestamp into the first ~2 minutes (verified on a 58-min file: 9k words
# all squashed into 136-232s). So we slice the audio into windows this long,
# align each, and shift the timestamps back onto the real timeline.
#
# The safe window differs by mode. Transcribing (ASR path) times its own words
# and stays coherent across a 300s chunk. Force-aligning provided text is far
# tighter: the 0.6B forced aligner overloads well before 300s — feeding it ~700
# words / 300s makes it give up and pile the whole chunk into the first ~140s
# (measured), which then scrambles every speaker cut downstream. ~120s keeps it
# inside the aligner's capacity (~230 words), so we default lower in that mode.
ALIGN_CHUNK_SECONDS = 300.0
ALIGN_CHUNK_SECONDS_FORCED = 120.0

def _wav_duration(wav: Path) -> float:
    with wave.open(str(wav), "rb") as w:
        return w.getnframes() / float(w.getframerate())

def _cut_wav(wav: Path, start: float, length: float, out: Path) -> Path:
    sh(["ffmpeg", "-y", "-ss", f"{start:.3f}", "-t", f"{length:.3f}",
        "-i", str(wav), "-ac", "1", "-ar", "16000", "-f", "wav", str(out)],
       capture=True)
    return out

def run_align(wav: Path, model: str, language: str | None,
              transcript: str | None = None) -> str:
    """One `speech align` call. With `transcript`, switches to forced alignment
    (--text), which skips ASR and so isn't bound by the ASR --max-tokens cap."""
    cmd = ["speech", "align", str(wav), "--model", model]
    if language:
        cmd += ["--language", language]
    if transcript:
        cmd += ["--text", transcript]
    return sh(cmd)

class AlignmentCollapse(RuntimeError):
    """Raised (only under strict mode) when one or more align chunks collapse —
    i.e. the aligner piled a chunk's words into the front of its window, so the
    timestamps there can't be trusted for speaker attribution."""

def align_words(wav: Path, model: str, language: str | None,
                transcript: str | None = None,
                chunk_seconds: float = ALIGN_CHUNK_SECONDS,
                strict: bool = False,
                progress=None) -> list[Word]:
    """Word-level timestamps for the whole file, chunking long audio (see note
    on ALIGN_CHUNK_SECONDS). With a `transcript` (e.g. exported from Voice
    Memos), each chunk is force-aligned against its share of the text, split in
    proportion to chunk duration — best-effort, since the source text is untimed.
    """
    dur = _wav_duration(wav)
    if dur <= chunk_seconds + 1.0:
        return parse_align(run_align(wav, model, language, transcript))

    n = int(dur // chunk_seconds) + (1 if dur % chunk_seconds else 0)
    text_words = transcript.split() if transcript else None
    words: list[Word] = []
    squashed = 0
    tmp = wav.parent / "_align_chunk.wav"
    for i in range(n):
        start = i * chunk_seconds
        length = min(chunk_seconds, dur - start)
        _cut_wav(wav, start, length, tmp)
        slice_text = None
        if text_words is not None:
            a = (i * len(text_words)) // n
            b = ((i + 1) * len(text_words)) // n
            slice_text = " ".join(text_words[a:b]) or None
        local = parse_align(run_align(tmp, model, language, slice_text))
        # Collapse guard: a healthy chunk spreads its words across the window.
        # When the aligner overloads it dumps everything into the front, leaving
        # the tail empty — that scrambles speaker attribution, so flag it loudly.
        if local and length > 30 and max(w.end for w in local) < 0.6 * length:
            squashed += 1
            print(f"  ⚠️  chunk {i + 1}/{n}: {len(local)} words collapsed into "
                  f"{max(w.end for w in local):.0f}s of a {length:.0f}s window "
                  f"— timestamps unreliable here", file=sys.stderr)
        for w in local:
            words.append(Word(text=w.text, start=w.start + start, end=w.end + start))
        if progress:
            progress(i + 1, n, len(words))
    if squashed:
        print(f"  ⚠️  {squashed}/{n} chunks collapsed — try a smaller "
              f"--align-chunk (current {chunk_seconds:.0f}s)", file=sys.stderr)
        if strict:
            raise AlignmentCollapse(
                f"{squashed}/{n} align chunks collapsed at {chunk_seconds:.0f}s")
    try:
        tmp.unlink()
    except OSError:
        pass
    return words

def parse_align(raw: str) -> list[Word]:
    # The real `speech align` output is one bracketed line per word, e.g.
    #   [0.16s - 2.24s] Blah,
    # preceded by progress + a "Transcription: …" line (neither matches _ALIGN_RE).
    words: list[Word] = []
    for line in raw.splitlines():
        m = _ALIGN_RE.search(line)
        if m:
            words.append(Word(text=m.group(3).strip(),
                              start=float(m.group(1)), end=float(m.group(2))))
    if words:
        return words
    # Fallback: a future build may emit JSON (possibly behind progress text).
    try:
        data = _extract_json(raw)
    except json.JSONDecodeError:
        return words
    if isinstance(data, dict):
        data = _first(data, "words", "alignment", "segments", default=[])
    for w in data:
        words.append(Word(
            text=str(_first(w, "text", "word", "token", default="")).strip(),
            start=float(_first(w, "start", "startTime", "start_time", default=0.0)),
            end=float(_first(w, "end", "endTime", "end_time", default=0.0)),
        ))
    return words

# --------------------------------------------------------------------------- #
# stage 3a — speech-only transcription  (speech vad, then align per chunk)
# --------------------------------------------------------------------------- #

# Qwen3-ASR derails on long silence. Measured on an 80-min Spanish phone call
# (2026-10-23) where only one side was audible: a 300s chunk that was mostly
# silence came back as "Sisa. SISA. Sasi. Sesses…" loops, and the last chunk as
# an invented textbook dialogue about a market crash. The SAME audio cut to 90s
# transcribed cleanly, and the real speech the loop had swallowed came back.
# So by default we find the speech first, splice just the speech into chunks of
# ~VAD_CHUNK_SECONDS (short silence between pieces), transcribe those, and map
# every word back onto the real timeline. Silence never reaches the model.
#
# VAD engine: firered is ~7s on 80 min; pyannote took ~10 min for the same file.
# firered alone found 1 min of speech in that (very quiet, ~-60 dB) recording;
# after loudness normalisation it found ~20 min, and padded as below it covered
# 95% of what pyannote found. The normalised copy is for detection only; the
# ASR hears the original audio (normalising did not make its text better).
VAD_CHUNK_SECONDS = 60.0
VAD_PAD = 0.5           # seconds kept either side of each detected stretch
VAD_MERGE_GAP = 1.5     # stretches closer than this are one stretch
SPLICE_GAP = 0.4        # silence put between spliced stretches in a chunk
VAD_NORMALISE = "dynaudnorm=f=250:g=15:m=30"

def run_vad(wav: Path) -> str:
    return sh(["speech", "vad", str(wav), "--engine", "firered", "--json"])

def parse_vad(raw: str) -> list[tuple[float, float]]:
    """`speech vad --json` → [(start, end)]. The progress lines before the JSON
    ("[0%] Downloading…") start with '[' too; _extract_json skips them."""
    try:
        data = _extract_json(raw)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        data = _first(data, "segments", "speech", "results", default=[])
    out = []
    for item in data:
        s = float(_first(item, "start", "startTime", "start_time", default=0.0))
        e = float(_first(item, "end", "endTime", "end_time", default=s))
        if e > s:
            out.append((s, e))
    return sorted(out)

def pad_regions(regions: list[tuple[float, float]], dur: float,
                pad: float = VAD_PAD,
                gap: float = VAD_MERGE_GAP) -> list[tuple[float, float]]:
    """Widen each stretch by `pad` (VAD clips word edges on quiet audio) and
    merge stretches that end up closer than `gap`."""
    out: list[tuple[float, float]] = []
    for s, e in sorted(regions):
        s, e = max(0.0, s - pad), min(dur, e + pad)
        if e <= s:      # VAD timed it past the end of the audio
            continue
        if out and s - out[-1][1] < gap:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out

@dataclass
class Piece:
    """One stretch of real audio inside a spliced chunk."""
    at: float          # where it starts in the chunk
    start: float       # where it starts in the recording
    length: float

def plan_splices(regions: list[tuple[float, float]],
                 chunk_seconds: float = VAD_CHUNK_SECONDS,
                 gap: float = SPLICE_GAP) -> list[list[Piece]]:
    """Pack speech stretches into chunks of at most ~chunk_seconds of audio.
    A stretch longer than a chunk is cut into chunk-sized pieces."""
    if chunk_seconds <= 0:
        raise ValueError(f"chunk_seconds must be positive, got {chunk_seconds}")
    chunks: list[list[Piece]] = []
    cur: list[Piece] = []
    used = 0.0
    for s, e in regions:
        while e - s > 1e-6:
            length = min(e - s, chunk_seconds)
            need = length + (gap if cur else 0.0)
            if cur and used + need > chunk_seconds:
                chunks.append(cur)
                cur, used, need = [], 0.0, length
            cur.append(Piece(at=used + (need - length), start=s, length=length))
            used += need
            s += length
    if cur:
        chunks.append(cur)
    return chunks

def write_splice(wav: Path, pieces: list[Piece], out: Path) -> Path:
    """Write the pieces of `wav` back to back, `SPLICE_GAP`-ish silence between
    (exactly as laid out by plan_splices). Same format as the source."""
    with wave.open(str(wav), "rb") as src, wave.open(str(out), "wb") as dst:
        dst.setparams(src.getparams())
        rate, width, ch = src.getframerate(), src.getsampwidth(), src.getnchannels()
        written = 0
        for p in pieces:
            pad = int(round(p.at * rate)) - written
            if pad > 0:
                dst.writeframes(b"\x00" * pad * width * ch)
                written += pad
            src.setpos(min(int(round(p.start * rate)), src.getnframes()))
            frames = src.readframes(int(round(p.length * rate)))
            dst.writeframes(frames)
            written += len(frames) // (width * ch)
    return out

def unsplice_time(t: float, pieces: list[Piece]) -> float:
    """Chunk time → recording time. A time in the silence between two pieces
    belongs to the piece before it (clamped to that piece's end)."""
    best = pieces[0]
    for p in pieces:
        if p.at <= t:
            best = p
        else:
            break
    return best.start + min(max(t - best.at, 0.0), best.length)

def transcribe_speech(wav: Path, model: str, language: str | None,
                      chunk_seconds: float = VAD_CHUNK_SECONDS,
                      progress=None) -> list[Word] | None:
    """Transcribe only the speech in `wav`. Returns None when the VAD finds no
    speech at all, so the caller can fall back to plain chunking rather than
    return an empty transcript for a recording that is merely very quiet."""
    dur = _wav_duration(wav)
    norm = wav.parent / "_vad_norm.wav"
    try:
        sh(["ffmpeg", "-y", "-i", str(wav), "-af", VAD_NORMALISE,
            "-ac", "1", "-ar", "16000", "-f", "wav", str(norm)], capture=True)
        regions = pad_regions(parse_vad(run_vad(norm)), dur)
    finally:
        try:
            norm.unlink()
        except OSError:
            pass
    if not regions:
        return None
    chunks = plan_splices(regions, chunk_seconds)
    speech = sum(e - s for s, e in regions)
    print(f"  speech: {speech/60:.1f} of {dur/60:.0f} min "
          f"→ {len(chunks)} chunks of ≤{chunk_seconds:.0f}s")
    words: list[Word] = []
    tmp = wav.parent / "_speech_chunk.wav"
    try:
        for i, pieces in enumerate(chunks):
            write_splice(wav, pieces, tmp)
            for w in parse_align(run_align(tmp, model, language)):
                mid = (w.start + w.end) / 2.0
                start = unsplice_time(w.start, pieces)
                end = unsplice_time(w.end, pieces)
                if end < start:   # a word straddling a splice: keep it where its middle is
                    start = end = unsplice_time(mid, pieces)
                words.append(Word(text=w.text, start=start, end=end))
            if progress:
                progress(i + 1, len(chunks), len(words))
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass
    return words

# A backstop for the loops that still get through: a long run of short words
# made of a handful of letters, most of them different ("Sisa. SISA. Sasi.
# Sesses. Sosa."). Real speech repeats a WORD ("sí, sí, sí, no, sí"), which
# has few distinct words and is kept; a loop mutates, which has many.
#
# Tuned on that run's 2,828 words: these cut the 202-word loop and nothing else.
# Looser (12 words, 5 distinct) also cut "sí vamos a la misión sí vamos al",
# which is speech. Runs of "mhm hmm" are left alone; they are cheap to read.
LOOP_MIN_RUN = 20
LOOP_MAX_WORD = 6       # letters, after stripping accents and punctuation
LOOP_MAX_LETTERS = 8    # distinct letters across the whole run
LOOP_MIN_DISTINCT = 8   # distinct words across the whole run
LOOP_CORE = 0.6         # a loop's core letters are in this share of its words
LOOP_MARK = "[…]"

def _bare(word: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", word.lower())
    return "".join(c for c in s if c.isalpha())

def _trim_loop(run: list[Word]) -> tuple[int, int]:
    """Give back the real words at either edge of a candidate loop ("… Sisses.
    Ya es el momento"): an edge word missing one of the loop's core letters
    is not part of it. Returns the [lo, hi) slice that is the loop."""
    if not run:
        return 0, 0
    bares = [set(_bare(w.text)) for w in run]
    core = {c for c in set().union(*bares)
            if sum(c in b for b in bares) >= LOOP_CORE * len(run)}
    lo, hi = 0, len(run)
    while lo < hi and not core <= bares[lo]:
        lo += 1
    while hi > lo and not core <= bares[hi - 1]:
        hi -= 1
    return lo, hi

def drop_loops(words: list[Word]) -> tuple[list[Word], int]:
    """Replace each decoder loop with one LOOP_MARK word. Returns (words, how
    many words were dropped)."""
    out: list[Word] = []
    dropped = 0
    i, n = 0, len(words)
    while i < n:
        j, letters = i, set()
        while j < n:
            b = _bare(words[j].text)
            if len(b) > LOOP_MAX_WORD or len(letters | set(b)) > LOOP_MAX_LETTERS:
                break
            letters |= set(b)
            j += 1
        lo, hi = _trim_loop(words[i:j])
        run = words[i + lo:i + hi]
        if (len(run) >= LOOP_MIN_RUN
                and len({_bare(w.text) for w in run}) >= LOOP_MIN_DISTINCT):
            out.extend(words[i:i + lo])     # real words the trim gave back
            out.append(Word(LOOP_MARK, run[0].start, run[-1].end))
            dropped += len(run)
            i += hi
        else:
            out.append(words[i])
            i += 1
    return out, dropped

# --------------------------------------------------------------------------- #
# stage 4 — merge words into speaker turns  (local, no model)
# --------------------------------------------------------------------------- #

def _turn_for(t: float, turns: list[Turn]) -> str:
    """Speaker whose turn contains time t; else the nearest turn."""
    for turn in turns:
        if turn.start <= t <= turn.end:
            return turn.speaker
    best, best_gap = (turns[0].speaker if turns else "SPEAKER_00"), float("inf")
    for turn in turns:
        gap = min(abs(t - turn.start), abs(t - turn.end))
        if gap < best_gap:
            best, best_gap = turn.speaker, gap
    return best

def merge(words: list[Word], turns: list[Turn]) -> list[Segment]:
    """Assign each word to a speaker, then coalesce consecutive same-speaker words."""
    if not words:
        return []
    if not turns:
        turns = [Turn("SPEAKER_00", words[0].start, words[-1].end)]
    segments: list[Segment] = []
    cur: Segment | None = None
    for w in words:
        spk = _turn_for((w.start + w.end) / 2.0, turns)
        if cur and cur.speaker == spk:
            cur.text += " " + w.text
            cur.end = w.end
        else:
            if cur:
                segments.append(cur)
            cur = Segment(speaker=spk, start=w.start, end=w.end, text=w.text)
    if cur:
        segments.append(cur)
    for s in segments:
        s.text = re.sub(r"\s+([,.;:!?])", r"\1", s.text).strip()
    return segments

# A word-timestamp landing in the wrong diarization turn splits one person's
# speech into several blocks, or wedges a stray word under the wrong name. Two
# cheap, deterministic passes clean that up AFTER relabeling (when raw clusters
# have collapsed to real names) — no LLM, no re-processing.
_SENTENCE_END = re.compile(r"[.!?][\"”')\]]*\s*$")
_MIN_FLIP_WORDS = 2   # a fragment this short, mid-sentence, wedged between two
                      # blocks of the SAME other speaker is diarizer thrash.

def _coalesce(segs: list[Segment]) -> list[Segment]:
    """Merge consecutive same-speaker segments into one (copies, never mutates
    the input). Flags and reasons are unioned; span/text are joined."""
    out: list[Segment] = []
    for s in segs:
        if out and out[-1].speaker == s.speaker:
            prev = out[-1]
            prev.text = f"{prev.text} {s.text}".strip()
            prev.end = s.end
            if s.flagged and not prev.flagged:
                prev.flagged, prev.flag_reason = True, s.flag_reason
        else:
            out.append(replace(s))
    return out

def _absorb_noise_flips(segs: list[Segment]) -> list[Segment]:
    """Reassign a very short, non-sentence fragment to its neighbours when both
    sides are the same OTHER speaker (a one-word speaker flip mid-sentence is
    almost always a timing artifact, not a real turn)."""
    out: list[Segment] = []
    for i, s in enumerate(segs):
        prev = out[-1] if out else None
        nxt = segs[i + 1] if i + 1 < len(segs) else None
        if (prev and nxt and prev.speaker == nxt.speaker != s.speaker
                and len(s.text.split()) <= _MIN_FLIP_WORDS
                and not _SENTENCE_END.search(s.text)):
            s = replace(s, speaker=prev.speaker)
        out.append(replace(s))
    return out

def tidy_segments(segs: list[Segment]) -> list[Segment]:
    """Coalesce same-speaker runs, absorb tiny noise flips, coalesce again.
    Idempotent. Deliberately does NOT snap long turns to sentence boundaries —
    real fast turn-taking often lacks punctuation at the handoff, so that would
    move whole blocks to the wrong speaker."""
    return _coalesce(_absorb_noise_flips(_coalesce(segs)))

# --------------------------------------------------------------------------- #
# stage 5/6 — LLM: relabel + correct, then notes  (text only)
# --------------------------------------------------------------------------- #

# The newest llm-seam feature this file relies on: OllamaClient(num_ctx=...).
LLM_SEAM_MIN_VERSION = "0.8.0"
LLM_SEAM_INSTALL = "pip install -r requirements-llm-seam.txt (Python 3.10+)"

def _llm_seam():
    """`(llm_seam, None)`, or `(None, why)` when it cannot be used here.

    Imported lazily and never at module level, so a machine without llm-seam
    still runs every stage that does not need it. A copy too old to size the
    context window counts as absent: calling it with `num_ctx=` would be a
    TypeError, and dropping the argument would be the silent clip again."""
    try:
        import llm_seam
    except ImportError:
        return None, "llm-seam is not installed"
    if "num_ctx" not in getattr(llm_seam.OllamaClient, "ACCEPTS", ()):
        return None, (f"llm-seam {getattr(llm_seam, '__version__', '?')} is "
                      f"older than {LLM_SEAM_MIN_VERSION}")
    return llm_seam, None

_noticed: set = set()

def _notice_once(text: str) -> None:
    """One line on stderr, once a run (llm() is called twice: labels, notes)."""
    if text not in _noticed:
        _noticed.add(text)
        print(f"note: {text}", file=sys.stderr)

def _server_root(base_url: str) -> str:
    """The host root of a local server, whichever way the URL was spelled.
    `http://localhost:11434`, `.../` and `.../v1` are all in use (OLLAMA_URL is
    documented as "OpenAI-compatible URL", which invites the /v1 form). The
    native client needs the bare root, and the /v1 path appends its own /v1."""
    root = base_url.strip().rstrip("/")
    if root.endswith("/v1"):
        root = root[:-len("/v1")].rstrip("/")
    return root

def llm(messages: list[dict], system: str, *, backend: str, model: str,
        base_url: str, max_tokens: int,
        num_ctx_max: int | None = None) -> str:
    """Minimal LLM call. backend='anthropic' (API key), 'ollama' (a local
    Ollama, with the context window sized to the prompt, via llm-seam),
    'openai' (any OpenAI-compatible server), or 'claude-cli' (the local
    `claude` binary → your Claude Pro/Max subscription, no API key / no
    per-token cost, via llm-seam)."""
    if backend == "ollama":
        seam, why = _llm_seam()
        if seam is not None:
            return _ollama_native(seam, messages, system, model=model,
                                  base_url=base_url, max_tokens=max_tokens,
                                  num_ctx_max=num_ctx_max or DEFAULT_NUM_CTX_MAX)
        _notice_once(f"{why}, so this run uses the deprecated /v1 path, which "
                     f"cannot ask Ollama for a bigger context window (a long "
                     f"meeting stops with an error instead of being clipped). "
                     f"To fix: {LLM_SEAM_INSTALL}.")
        backend = "openai"
    elif backend == "openai":
        _notice_once("--llm-backend openai cannot size the context window. "
                     "If this server is Ollama, use --llm-backend ollama.")
    if backend == "claude-cli":
        seam, why = _llm_seam()
        if seam is None:
            sys.exit(f"error: --llm-backend claude-cli needs llm-seam, and "
                     f"{why}. {LLM_SEAM_INSTALL}, or use --llm-backend "
                     f"anthropic / ollama.")
        ClaudeCli, LlmSeamError = seam.ClaudeCli, seam.LlmSeamError
        user = "\n\n".join(m["content"] for m in messages if m.get("role") == "user")
        cli = ClaudeCli(
            model=model or None,   # e.g. "sonnet", "opus", or a full id
            system=system,
            # Keep Claude Code's own system prompt + tool-use behaviour (what
            # --append-system-prompt did here before) rather than replacing it.
            system_mode="append",
            output_format="text",
            # Pure text transform — forbid tools so it can't touch the repo.
            disallowed_tools=["Bash", "Edit", "Write", "Read", "WebFetch"],
        )
        try:
            return cli.run(user).text.strip()
        except LlmSeamError as e:
            sys.exit(f"error: claude-cli backend failed: {e}")
    if backend == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            sys.exit("error: ANTHROPIC_API_KEY not set (or use --llm-backend openai).")
        url = base_url.rstrip("/") + "/v1/messages"
        body = {"model": model, "max_tokens": max_tokens,
                "system": system, "messages": messages}
        headers = {"content-type": "application/json",
                   "x-api-key": key, "anthropic-version": "2023-06-01"}
        data = _post(url, body, headers)
        return "".join(b.get("text", "") for b in data.get("content", [])
                       if b.get("type") == "text")
    else:
        # DEPRECATED for Ollama (2026-10-13), kept on purpose. Do not delete.
        #
        # This hand-built POST to /v1/chat/completions was the only local path
        # until `ollama` above replaced it. It stays for two reasons:
        #   1. People who can read this repo but not llm-seam (a private repo)
        #      have no other way to reach a local model. `--llm-backend ollama`
        #      lands here by itself when llm-seam cannot be imported.
        #   2. It is still the way to any OTHER OpenAI-compatible server (LM
        #      Studio, LiteLLM, vLLM, speech-server), and to an Ollama behind a
        #      proxy that wants the `Authorization` header sent below.
        # What it cannot do is ask for a context window: this protocol has no
        # field for one. So a long transcript is checked after the fact by
        # _check_prompt_arrived and the run stops if it was clipped.
        #
        # How the new path is built, for whoever has to rebuild it without
        # llm-seam: POST {root}/api/chat (Ollama's native endpoint) with
        #   {"model": ..., "stream": false, "think": false,
        #    "messages": [{"role": "system", ...}, {"role": "user", ...}],
        #    "options": {"temperature": 0.0, "num_predict": <reply budget>,
        #                "num_ctx": <_size_num_ctx(...)>}}
        # and read `message.content`, `prompt_eval_count` (prompt tokens the
        # model actually read, fed to _check_window_held) and `done_reason`
        # ("length" means the reply hit num_predict). See _ollama_native.
        key = os.environ.get("OPENAI_API_KEY", "not-needed")
        url = _server_root(base_url) + "/v1/chat/completions"
        body = {"model": model, "max_tokens": max_tokens,
                "messages": [{"role": "system", "content": system}, *messages]}
        headers = {"content-type": "application/json",
                   "authorization": f"Bearer {key}"}
        data = _post(url, body, headers)
        sent = len(system) + sum(len(m.get("content") or "") for m in messages)
        _check_prompt_arrived(sent, data, model)
        return data["choices"][0]["message"]["content"]

# A local server (Ollama above all) clips a prompt longer than the model's
# loaded context window and answers from what is left. Nothing errors, so a
# long meeting came back with notes covering part of it and no sign that
# anything was missing. /v1/chat/completions has no field for a context size
# (Ollama's `options.num_ctx` is native /api/chat only, and is ignored here),
# so this path cannot ask for a bigger window. What it can do is refuse to pass
# a clipped answer off as a whole one: the server reports how many prompt
# tokens it actually read, and real text does not run past ~5 characters a
# token, so a count far below what was sent means part of the prompt was cut.
MAX_CHARS_PER_TOKEN = 6.0
# Below this nothing is checked: a prompt this short fits the smallest default
# window any of these servers ship (2048 tokens), and the ratio is noise.
TRUNCATION_MIN_CHARS = 8000
ALLOW_TRUNCATED_ENV = "NOTETAKER_ALLOW_TRUNCATED"

def _check_prompt_arrived(sent_chars: int, data: dict, model: str) -> None:
    """Exit loudly if the server read far fewer tokens than the prompt holds.
    A response without `usage.prompt_tokens` cannot be judged and passes."""
    if sent_chars < TRUNCATION_MIN_CHARS:
        return
    read = (data.get("usage") or {}).get("prompt_tokens")
    if not isinstance(read, int) or read <= 0:
        return
    if sent_chars / read <= MAX_CHARS_PER_TOKEN:
        return
    msg = (f"the prompt was {sent_chars:,} characters (about "
           f"{sent_chars // 4:,} tokens) but {model} read only {read:,} "
           f"tokens of it, so the transcript was cut to fit the model's "
           f"context window and the result would cover only part of the "
           f"meeting. Raise the server's context size (Ollama: "
           f"OLLAMA_CONTEXT_LENGTH={_suggest_ctx(sent_chars)} on `ollama serve`, "
           f"or a Modelfile with PARAMETER num_ctx), or use "
           f"--llm-backend claude-cli.")
    _refuse(msg)

def _refuse(msg: str) -> None:
    """Stop the run, unless the user said a partial result is acceptable."""
    if os.environ.get(ALLOW_TRUNCATED_ENV) == "1":
        print(f"warning: {msg}", file=sys.stderr)
        return
    sys.exit(f"error: {msg} (Set {ALLOW_TRUNCATED_ENV}=1 to continue anyway.)")

def _suggest_ctx(sent_chars: int, reply_tokens: int = 4000) -> int:
    """A context size that holds the prompt plus the reply, as a power of two.
    3 characters a token is deliberately pessimistic (Dutch, names, numbers)."""
    need = sent_chars // 3 + reply_tokens
    ctx = 4096
    while ctx < need:
        ctx *= 2
    return ctx

# --- the Ollama path: ask for a window that holds the prompt ---------------- #
#
# Ollama's native /api/chat takes `options.num_ctx`, so this path does what the
# /v1 one cannot: it sizes the window to the prompt before sending. Three
# things keep that honest.
#
# The estimate is pessimistic. Speech runs 4+ characters a token; 3 leaves room
# for Dutch, names and the digit-heavy "[412] SPEAKER_03 (2475s):" prefixes of
# the labelling prompt (a synthetic all-numbers prompt measured 3.06).
#
# The window is capped, because it is RAM on top of the model's weights and
# this runs on a laptop. 32768 is what a 24 GB Mac can hold beside the models
# this repo recommends: the cache for 32k tokens is roughly 5 GB on qwen3:8b
# (5 GB of weights) and on qwen3:14b (9 GB), so 10 to 15 GB resident. 65536
# would double the cache and put a 14B model near 20 GB. (Estimates from the
# models' layer and head counts at 16-bit, not measurements.) It holds about 95
# minutes of talk. A prompt that does not fit under the cap is REFUSED with the
# numbers: clipping it is the bug this path exists to end.
#
# The answer is still checked. Ollama reports how many prompt tokens the model
# read, and _check_window_held compares that with what was sent.
CHARS_PER_TOKEN_ESTIMATE = 3.0
NUM_CTX_STEP = 2048          # round up to this, so near-equal prompts share a size
NUM_CTX_FLOOR = 4096         # Ollama's own default; never ask for less
DEFAULT_NUM_CTX_MAX = 32768
NUM_CTX_MAX_ENV = "NOTETAKER_NUM_CTX_MAX"
OLLAMA_TIMEOUT = 600.0       # a big local model can cold-load for a minute+

def _estimate_tokens(chars: int) -> int:
    return math.ceil(chars / CHARS_PER_TOKEN_ESTIMATE)

def _size_num_ctx(sent_chars: int, reply_tokens: int, cap: int, model: str) -> int:
    """The context window to ask for: the prompt (estimated) plus the reply
    budget, rounded up to NUM_CTX_STEP. Sized to the prompt, never to the cap,
    so a ten-minute memo does not pay for a two-hour window."""
    need = _estimate_tokens(sent_chars) + reply_tokens
    ctx = max(NUM_CTX_FLOOR, math.ceil(need / NUM_CTX_STEP) * NUM_CTX_STEP)
    if ctx <= cap:
        return ctx
    if need <= cap:              # a cap that is not a multiple of the step
        return cap
    _refuse(f"the prompt is {sent_chars:,} characters, estimated at "
            f"{_estimate_tokens(sent_chars):,} tokens, and with {reply_tokens:,} "
            f"for the reply it needs a context window of {ctx:,}. The cap is "
            f"{cap:,} (num_ctx_max), so {model} would read only part of the "
            f"meeting. Raise the cap if this machine has the RAM for it "
            f"(--num-ctx-max {ctx}, or num_ctx_max: {ctx} in config.yaml), or "
            f"use --llm-backend claude-cli, which has no such limit. The "
            f"estimate is deliberately high ({CHARS_PER_TOKEN_ESTIMATE:g} "
            f"characters a token).")
    return cap                   # only reached under NOTETAKER_ALLOW_TRUNCATED=1

def _check_window_held(sent_chars: int, read, num_ctx: int, reply_tokens: int,
                       model: str) -> None:
    """Compare the prompt tokens Ollama says the model read with what was sent.
    `read` is llm-seam's `usage.input_tokens` (Ollama's `prompt_eval_count`);
    None means Ollama did not say, and that cannot be judged."""
    if not isinstance(read, int) or read <= 0:
        return
    estimate = _estimate_tokens(sent_chars)
    if (sent_chars >= TRUNCATION_MIN_CHARS
            and sent_chars / read > MAX_CHARS_PER_TOKEN):
        # Asked for a window and did not get it: the model's own maximum is
        # smaller, or the server caps it. Same symptom as the /v1 path.
        _refuse(f"the prompt was {sent_chars:,} characters (estimated "
                f"{estimate:,} tokens) and a context window of {num_ctx:,} was "
                f"asked for, but {model} read only {read:,} tokens of it, so "
                f"the transcript was cut and the result would cover only part "
                f"of the meeting. Ollama did not give the model that window "
                f"(the model's own maximum may be smaller). Use a model with "
                f"a larger context, or --llm-backend claude-cli.")
        return
    if read >= num_ctx:
        _refuse(f"{model} read {read:,} prompt tokens into a context window of "
                f"{num_ctx:,}: the prompt filled the whole window, so part of "
                f"it was almost certainly cut. It was estimated at "
                f"{estimate:,} tokens from {sent_chars:,} characters. Raise "
                f"num_ctx_max, or use --llm-backend claude-cli.")
        return
    if read + reply_tokens > num_ctx:
        print(f"warning: the prompt was {read:,} tokens, more than the "
              f"{estimate:,} estimated, which left {num_ctx - read:,} of the "
              f"{reply_tokens:,} reply tokens budgeted in a window of "
              f"{num_ctx:,}. The reply finished, so nothing was cut.",
              file=sys.stderr)
    elif (sent_chars >= TRUNCATION_MIN_CHARS
            and abs(read - num_ctx // 2) <= 8 and read < estimate):
        # Measured live (qwen3.5:2b, 2026-10-13): a prompt cut to a 2048
        # window came back as 1,026 tokens read, half the window and not the
        # window. An honest prompt lands this exactly on the half about one
        # time in a thousand, so this warns and does not stop.
        print(f"warning: {model} read {read:,} prompt tokens, exactly half of "
              f"the {num_ctx:,} window. That is what Ollama reports after "
              f"cutting a prompt that was too long; check that the result "
              f"covers the START of the meeting.", file=sys.stderr)

def _ollama_native(seam, messages: list[dict], system: str, *, model: str,
                   base_url: str, max_tokens: int, num_ctx_max: int) -> str:
    """One turn against Ollama's native /api/chat through llm-seam's
    OllamaClient, which models `num_ctx` (0.8.0) and raises when the reply
    hits `max_tokens` instead of returning it cut short."""
    user = "\n\n".join(m.get("content") or "" for m in messages
                       if m.get("role") == "user")
    sent = len(system) + len(user)
    num_ctx = _size_num_ctx(sent, max_tokens, num_ctx_max, model)
    client = seam.OllamaClient(_server_root(base_url), model,
                               timeout=OLLAMA_TIMEOUT)
    for attempt in (1, 2):
        try:
            result = client.complete(system, user, num_ctx=num_ctx,
                                     max_tokens=max_tokens)
            break
        except seam.OllamaError as e:
            # Same rule as _post below: a lone timeout while a big model
            # cold-loads is transient and is retried once; an HTTP error or a
            # reply cut off at max_tokens is an answer, and is not.
            cause = e.__cause__
            transient = (isinstance(cause, OSError)
                         and not isinstance(cause, urllib.error.HTTPError))
            if transient and attempt == 1:
                print("  LLM timed out; retrying once…", file=sys.stderr)
                continue
            sys.exit(f"error: ollama backend failed ({model}, "
                     f"num_ctx={num_ctx:,}): {e}")
    _check_window_held(sent, result.usage.input_tokens, num_ctx, max_tokens,
                       model)
    return result.text

def _post(url: str, body: dict, headers: dict, timeout: int = 600) -> dict:
    # Local models (e.g. a 30B via Ollama) can cold-load for a minute+ before
    # the first token, so the timeout is generous and a lone read-timeout (a
    # transient, not a real failure) is retried once before giving up.
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers=headers, method="POST")
    for attempt in (1, 2):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            sys.exit(f"LLM HTTP {e.code}: {e.read().decode()[:500]}")
        except (TimeoutError, OSError) as e:  # incl. socket.timeout (py3.9)
            if attempt == 2:
                reason = getattr(e, "reason", e)
                sys.exit(f"LLM connection error after retry: {reason}")
            print("  LLM timed out; retrying once…", file=sys.stderr)

def _strip_fences(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()

LABEL_SYSTEM = """You are an expert at attributing meeting transcripts to speakers.
You are given a diarized transcript whose speakers are anonymous (SPEAKER_00, ...)
and context about who was actually in the meeting.

Acoustic diarization is good at WHEN the speaker changes but unreliable at WHO and
at the exact count. Use conversational logic — who asks vs. answers, role-specific
knowledge, self-introductions ("this is Tom") — to map each anonymous label to a
real participant.

Rules:
- Map every SPEAKER_xx label to exactly one participant name from the roster.
- Two labels MAY map to the same person (over-splitting is common). Merge them.
- If you are under ~80% confident about a SPECIFIC segment, flag it instead of guessing.
- Do NOT invent participants beyond the roster.

Respond with ONLY a JSON object, no prose, no markdown fences:
{
  "label_map": { "SPEAKER_00": "Name", "SPEAKER_01": "Name" },
  "flagged":   [ { "index": <segment index>, "reason": "...",
                   "suggested_speaker": "Name or null" } ]
}"""

def label_and_correct(segs: list[Segment], roster: list[str], context: str,
                      **llm_kw) -> tuple[dict[str, str], list[dict]]:
    # Truncate each segment: attribution needs identity signal, not every word.
    # On a long meeting the full text buries the instruction and weaker local
    # models start summarising instead of emitting the label JSON.
    def clip(t: str, n: int = 240) -> str:
        return (t[:n] + "…") if len(t) > n else t
    labels = sorted({s.speaker for s in segs})
    allowed = ", ".join(f'"{r}"' for r in roster)
    lines = [f"[{i}] {s.speaker} ({s.start:.0f}s): {clip(s.text)}"
             for i, s in enumerate(segs)]
    user = (f"PARTICIPANTS (the ONLY allowed names): {allowed}\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"SPEAKER LABELS TO MAP: {', '.join(labels)}\n\n"
            f"TRANSCRIPT ({len(segs)} segments):\n" + "\n".join(lines) +
            f"\n\nNow output ONLY the JSON object. Every label in "
            f"[{', '.join(labels)}] must appear in label_map, and every value MUST "
            f"be exactly one of: {allowed}. No other names, no other languages, "
            f"no prose, no markdown.")
    out = llm([{"role": "user", "content": user}], LABEL_SYSTEM,
              max_tokens=2000, **llm_kw)
    try:
        data = _extract_json(out)            # tolerant: skips prose/fences
    except json.JSONDecodeError:
        print("warning: could not parse LLM label JSON; leaving raw labels.",
              file=sys.stderr)
        return {}, []
    # Validate every mapped name against the roster (case-insensitive). Drop
    # anything off-roster so a hallucinated/foreign name never reaches the
    # transcript — that label just stays raw (visibly unattributed).
    canon = {r.lower(): r for r in roster}
    label_map: dict[str, str] = {}
    for k, v in (data.get("label_map") or {}).items():
        if isinstance(v, str) and v.strip().lower() in canon:
            label_map[k] = canon[v.strip().lower()]
    flagged = data.get("flagged") or data.get("flagged_segments") or []
    return label_map, flagged

def fill_unmapped(segs: list[Segment], roster: list[str]) -> int:
    """Safety net for diarizer over-splitting: any segment still bearing a raw
    SPEAKER_xx label (the LLM skipped a tiny stray cluster) inherits the
    nearest-in-time roster-named segment, flagged uncertain. Guarantees no raw
    labels ever reach the transcript. Returns how many were filled."""
    named = [i for i, s in enumerate(segs) if s.speaker in roster]
    if not named:
        return 0
    filled = 0
    for i, s in enumerate(segs):
        if s.speaker in roster:
            continue
        j = min(named, key=lambda k: abs(k - i))   # segments are time-ordered
        raw, s.speaker = s.speaker, segs[j].speaker
        s.flagged = True
        s.flag_reason = s.flag_reason or f"over-split {raw} merged into nearest speaker"
        filled += 1
    return filled

NOTES_SYSTEM = """You write concise, accurate meeting notes from a speaker-attributed
transcript. Use only what is in the transcript — never invent commitments. Output
clean Markdown with these sections, omitting any that have no content:

## Summary            (3-5 sentences)
## Key Decisions
## Action Items        (- [ ] Owner — task — due if stated)
## Open Questions
## Notable Quotes      (only if genuinely useful; keep verbatim and short)

If parts of the transcript are marked uncertain, do not over-rely on them."""

def make_notes(segs: list[Segment], context: str, **llm_kw) -> str:
    body = []
    for s in segs:
        tag = " [attribution uncertain]" if s.flagged else ""
        body.append(f"{s.speaker}{tag}: {s.text}")
    user = f"CONTEXT:\n{context}\n\nTRANSCRIPT:\n" + "\n".join(body)
    return llm([{"role": "user", "content": user}], NOTES_SYSTEM,
               max_tokens=4000, **llm_kw)

# A full English transcript of a memo in another language. Notes alone were not
# enough: the transcript itself is what gets quoted and searched later, and a
# translation pasted into a chat app summarised two-thirds of a call away
# ("[The conversation continues…]"). So: numbered lines in, the same numbered
# lines out, in batches, and a line that comes back missing is shown untranslated
# rather than silently lost.
TRANSLATE_SYSTEM = f"""You translate a speech-to-text transcript into English.

Each input line is "[n] text". Output exactly one line per input line, "[n] English
translation", same numbers, same order. Translate everything; never summarise,
merge, skip or comment. Keep fillers short ("mm", "yeah"). Where the source is
garbled and you cannot tell what was said, write [unclear] for that part rather
than guessing. Keep names as they are. Leave {LOOP_MARK} as it is.

Memos often mix languages ("¿Tienes pizza? No, I already had the pizza."): a line
or part of a line that is already English is copied unchanged.

If EVERY line is already in English, output only the word ALREADY_ENGLISH."""

TRANSLATE_PIECE_WORDS = 120     # a long turn is translated in pieces this size
TRANSLATE_BATCH_WORDS = 1500    # words per LLM call
ALREADY_ENGLISH = "ALREADY_ENGLISH"
_NUMBERED = re.compile(r"^\s*\[(\d+)\]\s?(.*)$")

def _split_turn(text: str, limit: int = TRANSLATE_PIECE_WORDS) -> list[str]:
    """Cut a turn into pieces of about `limit` words, at a sentence end when
    there is one in the second half of the piece."""
    words = text.split()
    pieces, cur = [], []
    for w in words:
        cur.append(w)
        if len(cur) >= limit or (len(cur) >= limit // 2 and _SENTENCE_END.search(w)):
            pieces.append(" ".join(cur))
            cur = []
    if cur:
        pieces.append(" ".join(cur))
    return pieces or [""]

def translate_segments(segs: list[Segment], **llm_kw) -> list[str] | None:
    """English text for each segment, or None when the transcript is already
    English. Each batch is asked on its own: a memo that opens in English and
    turns to Spanish must still be translated, so one English batch keeps its
    own lines as they are and only an all-English answer skips the file."""
    pieces: list[tuple[int, str]] = [(i, p) for i, s in enumerate(segs)
                                     for p in _split_turn(s.text)]
    batches: list[list[int]] = [[]]
    count = 0
    for k, (_, text) in enumerate(pieces):
        n = len(text.split())
        if batches[-1] and count + n > TRANSLATE_BATCH_WORDS:
            batches.append([])
            count = 0
        batches[-1].append(k)
        count += n
    english: dict[int, str] = {}
    english_batches = 0
    for batch in batches:
        user = "\n".join(f"[{k}] {pieces[k][1]}" for k in batch)
        out = llm([{"role": "user", "content": user}], TRANSLATE_SYSTEM,
                  max_tokens=8000, **llm_kw)
        if out.strip().strip("`*.\"' ").upper() == ALREADY_ENGLISH:
            english_batches += 1
            english.update((k, pieces[k][1]) for k in batch)
            continue
        last = None
        for line in out.splitlines():
            m = _NUMBERED.match(line)
            if m:
                # a number that is not in this batch is the model's invention,
                # not a wrapped line of the one before it
                last = int(m.group(1)) if int(m.group(1)) in batch else None
                if last is not None:
                    english[last] = m.group(2).strip()
            elif last is not None and line.strip():
                english[last] += " " + line.strip()   # a wrapped line
    if english_batches == len(batches):
        return None
    missing = [k for k in range(len(pieces)) if k not in english]
    if missing:
        print(f"warning: {len(missing)} of {len(pieces)} transcript lines came back "
              f"untranslated; they are shown in the original language.",
              file=sys.stderr)
    out_texts = [""] * len(segs)
    for k, (i, text) in enumerate(pieces):
        part = english.get(k, f"[untranslated] {text}")
        out_texts[i] = f"{out_texts[i]} {part}".strip()
    return out_texts

# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #

def hhmmss(t: float) -> str:
    m, s = divmod(int(t), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"

def render_named(segs: list[Segment]) -> str:
    out = []
    for s in segs:
        mark = "  ⚠️ uncertain" if s.flagged else ""
        out.append(f"**{s.speaker}** ({hhmmss(s.start)}){mark}\n{s.text}\n")
    return "\n".join(out)

def render_english(segs: list[Segment], english: list[str]) -> str:
    head = ("_English translation of transcript_named.md, made by an LLM from the "
            "speech-to-text. [unclear] marks what could not be made out; the "
            "original is the reference._\n\n")
    return head + render_named([replace(s, text=t) for s, t in zip(segs, english)])

# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #

def cmd_check(args):
    """Run diarize + align on a file and show raw vs parsed, to verify schemas."""
    need("speech")
    wav = Path(args.file)
    if wav.suffix.lower() != ".wav":
        wav = to_wav16k(Path(args.file), Path("/tmp/_check.wav"))
    print("=== speech diarize --json (raw, first 1200 chars) ===")
    raw_d = run_diarize(wav, args.diarize_engine, args.vad_filter, args.cluster_threshold)
    print(raw_d[:1200])
    turns = parse_diarize(raw_d)
    print(f"\n--> parsed {len(turns)} turns; first 5:")
    for t in turns[:5]:
        print(f"    {t.speaker}  {t.start:.2f}-{t.end:.2f}")
    # Align just the first chunk: enough to confirm the line format, and avoids
    # both the slow full-file pass and the long-audio timestamp squashing (the
    # real `run` chunks the whole file — see align_words / ALIGN_CHUNK_SECONDS).
    dur = _wav_duration(wav)
    if dur > ALIGN_CHUNK_SECONDS + 1.0:
        _cut_wav(wav, 0.0, ALIGN_CHUNK_SECONDS, Path("/tmp/_check_chunk.wav"))
        print(f"\n=== speech align (raw, first 800 chars; first "
              f"{int(ALIGN_CHUNK_SECONDS)}s chunk of {dur/60:.0f} min) ===")
        raw_a = run_align(Path("/tmp/_check_chunk.wav"), args.model, args.language)
    else:
        print("\n=== speech align (raw, first 800 chars) ===")
        raw_a = run_align(wav, args.model, args.language)
    print(raw_a[:800])
    words = parse_align(raw_a)
    print(f"\n--> parsed {len(words)} words (this chunk); first 10:")
    print("    " + " ".join(w.text for w in words[:10]))
    if not turns or not words:
        print("\n*** A parser returned nothing. Adjust parse_diarize/parse_align "
              "to match the raw output above. ***")

DEFAULT_BASE_URLS = {
    "anthropic": "https://api.anthropic.com",
    "ollama": "http://localhost:11434",
    "openai": "http://localhost:11434",
}

def resolve_num_ctx_max(flag, cfg: dict) -> int:
    """The cap on the Ollama context window: --num-ctx-max, else the
    NOTETAKER_NUM_CTX_MAX env var, else `num_ctx_max:` in the config, else
    DEFAULT_NUM_CTX_MAX. (The env var exists because DropMemo rewrites
    config.yaml from its two text fields on every run.)"""
    for source, value in (("--num-ctx-max", flag),
                          (NUM_CTX_MAX_ENV, os.environ.get(NUM_CTX_MAX_ENV)),
                          ("num_ctx_max in the config", cfg.get("num_ctx_max"))):
        if value in (None, ""):
            continue
        try:
            cap = int(value)
        except (TypeError, ValueError):
            cap = 0
        if cap <= 0:
            sys.exit(f"error: {source} must be a positive number of tokens, "
                     f"got {value!r}.")
        return cap
    return DEFAULT_NUM_CTX_MAX

def cmd_run(args):
    cfg = load_config(Path(args.config))
    roster = cfg.get("participants", [])
    context = cfg.get("context", "").strip()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    llm_kw = dict(backend=args.llm_backend, model=args.model_llm,
                  base_url=args.base_url or DEFAULT_BASE_URLS.get(args.llm_backend, ""),
                  num_ctx_max=resolve_num_ctx_max(args.num_ctx_max, cfg))

    if args.from_json:
        pipe = Pipeline.from_json(Path(args.from_json).read_text())
    else:
        need("speech")
        src = Path(args.file)
        wav = to_wav16k(src, out_dir / "audio.wav")
        if args.denoise:
            print("• denoising…")
            wav = denoise(wav, out_dir / "audio.clean.wav")
        print("• diarizing…")
        turns = parse_diarize(run_diarize(wav, args.diarize_engine, args.vad_filter,
                                           args.cluster_threshold))
        print(f"  {len(turns)} turns, "
              f"{len({t.speaker for t in turns})} apparent speakers")
        transcript_text = None
        if getattr(args, "transcript", None):
            transcript_text = Path(args.transcript).read_text().strip() or None
        # Force-aligning provided text overloads the aligner sooner than
        # transcribing does, so it needs a smaller default window (see
        # ALIGN_CHUNK_SECONDS_FORCED). An explicit --align-chunk always wins.
        align_chunk = args.align_chunk or (
            ALIGN_CHUNK_SECONDS_FORCED if transcript_text else ALIGN_CHUNK_SECONDS)
        dur = _wav_duration(wav)
        fixed_chunks = bool(transcript_text) or args.no_vad
        if fixed_chunks and dur > align_chunk + 1.0:
            n = int(dur // align_chunk) + (1 if dur % align_chunk else 0)
            mode = "forced-align" if transcript_text else "transcribe"
            print(f"• aligning words… ({dur/60:.0f} min → {n} chunks, "
                  f"{align_chunk:.0f}s, {mode})")
        elif fixed_chunks:
            print("• aligning words…")
        progress = lambda i, n, t: print(f"  chunk {i}/{n} … {t} words")
        words = None
        # Speech-only is the default when transcribing. A provided transcript
        # is split across chunks by duration, which only holds on the plain
        # timeline, so --transcript (and --no-vad) keep the fixed chunks.
        if not fixed_chunks:
            print("• finding speech…")
            words = transcribe_speech(wav, args.model, args.language,
                                      chunk_seconds=args.vad_chunk,
                                      progress=progress)
            if words is None:
                print("  no speech found by the VAD; transcribing the whole file")
        if words is None:
            try:
                words = align_words(wav, args.model, args.language,
                                    transcript=transcript_text,
                                    chunk_seconds=align_chunk,
                                    strict=args.strict,
                                    progress=progress)
            except AlignmentCollapse as e:
                sys.exit(f"error: {e}. Re-run with a smaller --align-chunk, or drop "
                         f"--strict to accept the unreliable timestamps.")
        if not transcript_text:
            words, dropped = drop_loops(words)
            if dropped:
                print(f"  dropped {dropped} words of decoder loop (marked {LOOP_MARK})")
        print(f"  {len(words)} words")
        if not turns or not words:
            sys.exit("error: diarization or alignment produced nothing. "
                     "Run `notetaker.py check <file>` and fix the parsers.")
        pipe = Pipeline(audio=str(src), turns=turns, words=words)
        pipe.segments = merge(words, turns)

    if not pipe.segments:
        pipe.segments = merge(pipe.words, pipe.turns)

    if roster:
        print("• attributing speakers via LLM…")
        label_map, flagged = label_and_correct(pipe.segments, roster, context, **llm_kw)
        pipe.label_map = label_map
        canon = {r.lower(): r for r in roster}
        flag_by_idx = {f["index"]: f for f in flagged if "index" in f}
        for i, s in enumerate(pipe.segments):
            s.speaker = label_map.get(s.speaker, s.speaker)
            if i in flag_by_idx:
                s.flagged = True
                s.flag_reason = flag_by_idx[i].get("reason", "")
                sug = flag_by_idx[i].get("suggested_speaker")
                if isinstance(sug, str) and sug.strip().lower() in canon:
                    s.speaker = canon[sug.strip().lower()]   # only roster names
        n_filled = fill_unmapped(pipe.segments, roster)
        named = len({s.speaker for s in pipe.segments if s.speaker in roster})
        print(f"  {named} named speaker(s)"
              + (f"; {n_filled} over-split segment(s) merged to nearest" if n_filled else ""))
    else:
        print("• no participants in config — skipping LLM attribution.")

    before = len(pipe.segments)
    pipe.segments = tidy_segments(pipe.segments)
    if len(pipe.segments) < before:
        print(f"• tidied segments: {before} → {len(pipe.segments)} "
              f"(merged same-speaker splits / absorbed stray words)")

    (out_dir / "transcript_named.md").write_text(render_named(pipe.segments))
    (out_dir / "pipeline.json").write_text(pipe.to_json())

    english = None
    if not args.no_translate and not (args.language or "").lower().startswith("en"):
        print("• translating the transcript to English via LLM…")
        english = translate_segments(pipe.segments, **llm_kw)
        if english is None:
            print("  already English; no translation written")
        else:
            (out_dir / "transcript_en.md").write_text(
                render_english(pipe.segments, english))

    if not args.no_notes:
        print("• writing notes via LLM…")
        notes = make_notes(pipe.segments, context, **llm_kw)
        (out_dir / "notes.md").write_text(notes)

    print(f"\n✓ done → {out_dir}/")
    print("  transcript_named.md   speaker-attributed transcript")
    if english is not None:
        print("  transcript_en.md      the same transcript in English")
    print("  pipeline.json         all intermediates (reuse with --from-json)")
    if not args.no_notes:
        print("  notes.md              summary / decisions / action items")

# --------------------------------------------------------------------------- #
# argparse
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--diarize-engine", choices=["pyannote", "sortformer"],
                        default="pyannote",
                        help="pyannote = segmentation + embeddings (default, robust); "
                             "sortformer = end-to-end CoreML on the ANE (faster, but "
                             "its CoreML model fails on some setups — verify with `check`)")
    common.add_argument("--vad-filter", action="store_true",
                        help="pre-filter silence with Silero VAD (pyannote only)")
    common.add_argument("--cluster-threshold", type=float, default=None,
                        help="diarizer clustering threshold (speech default 0.715; "
                             "LOWER = fewer speakers). Lower it if your known head "
                             "count is over-split into too many SPEAKER_xx labels.")
    common.add_argument("--model", default="1.7B",
                        help="ASR/align model variant (0.6B | 1.7B)")
    common.add_argument("--language", default=None, help="language hint, e.g. en")

    c = sub.add_parser("check", parents=[common],
                       help="verify diarize/align output formats on a file")
    c.add_argument("file")
    c.set_defaults(func=cmd_check)

    r = sub.add_parser("run", parents=[common], help="full pipeline")
    r.add_argument("file", nargs="?", help="audio file (m4a/wav/mp3/caf)")
    r.add_argument("--config", default="config.yaml",
                   help="participants + context (YAML or JSON)")
    r.add_argument("--out-dir", default="./out")
    r.add_argument("--denoise", action="store_true",
                   help="run speech denoise first (good for noisy calls)")
    r.add_argument("--transcript",
                   help="path to an existing transcript (e.g. exported from "
                        "Voice Memos); force-align THIS text to the audio instead "
                        "of transcribing. Better text, but best-effort across "
                        "chunks since the source text is untimed.")
    r.add_argument("--align-chunk", type=float, default=None,
                   help=f"seconds of audio per alignment chunk (default "
                        f"{int(ALIGN_CHUNK_SECONDS)} transcribing, "
                        f"{int(ALIGN_CHUNK_SECONDS_FORCED)} with --transcript, since "
                        f"the forced aligner overloads on longer chunks). speech "
                        f"align only handles a few minutes per call, so long files "
                        f"are chunked + stitched.")
    r.add_argument("--no-vad", action="store_true",
                   help="transcribe the whole file in fixed --align-chunk windows, "
                        "silence included (the old behaviour). By default only "
                        "the speech is transcribed, since long silence makes the "
                        "ASR loop or invent text.")
    r.add_argument("--vad-chunk", type=float, default=VAD_CHUNK_SECONDS,
                   help=f"seconds of speech per ASR call when transcribing speech "
                        f"only (default {int(VAD_CHUNK_SECONDS)})")
    r.add_argument("--no-translate", action="store_true",
                   help="skip transcript_en.md, the English translation written "
                        "when the memo is not in English")
    r.add_argument("--strict", action="store_true",
                   help="fail the run (non-zero exit) if any alignment chunk "
                        "collapses, instead of emitting a warning and continuing")
    r.add_argument("--no-notes", action="store_true",
                   help="skip the notes-generation LLM step")
    r.add_argument("--from-json",
                   help="reuse a previous pipeline.json (skip all audio stages)")
    r.add_argument("--llm-backend",
                   choices=["anthropic", "ollama", "openai", "claude-cli"],
                   default="anthropic",
                   help="anthropic = API key; ollama = local Ollama, context "
                        "window sized to the meeting (needs llm-seam; without it "
                        "this falls back to the openai path); openai = any "
                        "OpenAI-compatible server (LM Studio, ...; deprecated "
                        "for Ollama); claude-cli = local `claude` binary via "
                        "your subscription (needs llm-seam)")
    r.add_argument("--model-llm", default="claude-sonnet-4-6",
                   help="LLM model id (anthropic) or local model name "
                        "(ollama / openai)")
    r.add_argument("--base-url", default=None,
                   help="LLM base URL (default: api.anthropic.com for anthropic, "
                        "http://localhost:11434 for ollama / openai). A trailing "
                        "/v1 is accepted and ignored.")
    r.add_argument("--num-ctx-max", type=int, default=None,
                   help=f"ollama backend: the largest context window to ask "
                        f"for, in tokens (default {DEFAULT_NUM_CTX_MAX}, or "
                        f"num_ctx_max in the config). The window is sized to the "
                        f"meeting and costs RAM; a meeting that does not fit "
                        f"under this stops with an error instead of being cut.")
    r.set_defaults(func=cmd_run)
    return p

def main():
    args = build_parser().parse_args()
    if args.cmd == "run" and not args.file and not args.from_json:
        sys.exit("error: provide an audio file, or --from-json pipeline.json")
    args.func(args)

if __name__ == "__main__":
    main()
