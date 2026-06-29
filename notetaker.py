#!/usr/bin/env python3
"""
notetaker.py — on-device post-meeting note taker for macOS / Apple Silicon.

Pipeline (each stage runs alone, so peak RAM ~= the single biggest model):

    audio.m4a
      └─ (optional) speech denoise          -> clean.wav
      └─ speech diarize --json              -> who spoke when   (turns)
      └─ speech align                       -> word timestamps  (words)
      └─ bucket words into turns            -> speaker segments  [local]
      └─ LLM: map SPEAKER_xx -> real names, flag low-confidence  [text only]
      └─ LLM: meeting notes (summary / decisions / actions)      [text only]

Only TEXT ever leaves the device (the two LLM steps). Audio + diarization +
ASR are 100% local via the `speech` binary (github.com/soniqo/speech-swift).
Point --llm-backend at a local OpenAI-compatible endpoint to keep it ALL local.

The two parsers most likely to need tweaking for your `speech` version are
parse_diarize() and parse_align(). Run `notetaker.py check <file>` first — it
prints the raw tool output next to what we parsed so you can confirm in ~30s.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import wave
import urllib.request
import urllib.error
from dataclasses import dataclass, asdict, field
from pathlib import Path

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
ALIGN_CHUNK_SECONDS = 300.0

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

def align_words(wav: Path, model: str, language: str | None,
                transcript: str | None = None,
                chunk_seconds: float = ALIGN_CHUNK_SECONDS,
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
        for w in parse_align(run_align(tmp, model, language, slice_text)):
            words.append(Word(text=w.text, start=w.start + start, end=w.end + start))
        if progress:
            progress(i + 1, n, len(words))
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

# --------------------------------------------------------------------------- #
# stage 5/6 — LLM: relabel + correct, then notes  (text only)
# --------------------------------------------------------------------------- #

def llm(messages: list[dict], system: str, *, backend: str, model: str,
        base_url: str, max_tokens: int) -> str:
    """Minimal LLM call. backend='anthropic' or 'openai' (OpenAI-compatible)."""
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
    else:  # openai-compatible (LM Studio, LiteLLM, vLLM, speech-server, ...)
        key = os.environ.get("OPENAI_API_KEY", "not-needed")
        url = base_url.rstrip("/") + "/v1/chat/completions"
        body = {"model": model, "max_tokens": max_tokens,
                "messages": [{"role": "system", "content": system}, *messages]}
        headers = {"content-type": "application/json",
                   "authorization": f"Bearer {key}"}
        data = _post(url, body, headers)
        return data["choices"][0]["message"]["content"]

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

def cmd_run(args):
    cfg = load_config(Path(args.config))
    roster = cfg.get("participants", [])
    context = cfg.get("context", "").strip()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    llm_kw = dict(backend=args.llm_backend, model=args.model_llm,
                  base_url=args.base_url)

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
        dur = _wav_duration(wav)
        if dur > args.align_chunk + 1.0:
            n = int(dur // args.align_chunk) + (1 if dur % args.align_chunk else 0)
            mode = "forced-align" if transcript_text else "transcribe"
            print(f"• aligning words… ({dur/60:.0f} min → {n} chunks, {mode})")
        else:
            print("• aligning words…")
        words = align_words(wav, args.model, args.language,
                            transcript=transcript_text,
                            chunk_seconds=args.align_chunk,
                            progress=lambda i, n, t: print(f"  chunk {i}/{n} … {t} words"))
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

    (out_dir / "transcript_named.md").write_text(render_named(pipe.segments))
    (out_dir / "pipeline.json").write_text(pipe.to_json())

    if not args.no_notes:
        print("• writing notes via LLM…")
        notes = make_notes(pipe.segments, context, **llm_kw)
        (out_dir / "notes.md").write_text(notes)

    print(f"\n✓ done → {out_dir}/")
    print("  transcript_named.md   speaker-attributed transcript")
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
    r.add_argument("--align-chunk", type=float, default=ALIGN_CHUNK_SECONDS,
                   help=f"seconds of audio per alignment chunk (default "
                        f"{int(ALIGN_CHUNK_SECONDS)}). speech align only handles a "
                        f"few minutes per call, so long files are chunked + stitched.")
    r.add_argument("--no-notes", action="store_true",
                   help="skip the notes-generation LLM step")
    r.add_argument("--from-json",
                   help="reuse a previous pipeline.json (skip all audio stages)")
    r.add_argument("--llm-backend", choices=["anthropic", "openai"],
                   default="anthropic")
    r.add_argument("--model-llm", default="claude-sonnet-4-6",
                   help="LLM model id (anthropic) or local model name (openai)")
    r.add_argument("--base-url", default="https://api.anthropic.com",
                   help="LLM base URL; for local use e.g. http://localhost:1234")
    r.set_defaults(func=cmd_run)
    return p

def main():
    args = build_parser().parse_args()
    if args.cmd == "run" and not args.file and not args.from_json:
        sys.exit("error: provide an audio file, or --from-json pipeline.json")
    args.func(args)

if __name__ == "__main__":
    main()
