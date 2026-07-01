# On-device meeting note taker (macOS / Apple Silicon)

Record a meeting as a plain voice memo, then process it **after** the call.
Diarization (who spoke when) and transcription run 100% locally via
[`speech-swift`](https://github.com/soniqo/speech-swift) on the Neural Engine /
MLX. Only **text** leaves the device, and only for the two reasoning steps
(speaker attribution + notes) — point those at a local model and it's fully
offline.

Why post-meeting instead of live: real-time tools keep the ASR model, the
diarizer, and a summarizer all resident at once (that's the RAM blowup). Here
each stage loads one model, runs, and releases it before the next starts, so
peak memory ≈ the single largest model, and every stage gets the whole
recording for context instead of trading accuracy for latency.

## Pipeline

```
audio.m4a
  ├─ ffmpeg            → 16 kHz mono wav
  ├─ speech denoise    → clean.wav            (optional, good for calls)
  ├─ speech diarize    → who spoke when        (turns)
  ├─ speech align      → word-level timestamps (words)
  ├─ bucket            → speaker segments       [local, no model]
  ├─ LLM relabel       → SPEAKER_00 → "Tom", flag low-confidence  [text only]
  └─ LLM notes         → summary / decisions / actions            [text only]
```

The split is deliberate: the **acoustic** layer (`speech`) decides *when* the
speaker changes — an LLM is bad at that from text alone. The **LLM** decides
*who* each anonymous cluster is and fixes obvious misattributions, using the
roster + context you provide. Each does the job it's actually good at.

## Setup

```bash
# 1. speech-swift  (native ARM Homebrew at /opt/homebrew)
brew install soniqo/tap/speech

# 2. ffmpeg for audio normalisation
brew install ffmpeg

# 3. python deps (just PyYAML; everything else is stdlib)
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 4. LLM credentials — cloud (text only leaves device):
export ANTHROPIC_API_KEY=sk-ant-...
#    …or fully local, see "Going fully offline" below.
```

## Recording the meeting

The simplest source is the macOS/iOS **Voice Memos** app.

- **In-person / phone on speaker:** record in Voice Memos, then get the file
  onto disk (see below).
- **Video call on the Mac:** the mic won't capture the *other* side. Route
  system audio + mic together with a virtual device (BlackHole or Loopback)
  and record that, or both sides end up as one muffled speaker and diarization
  suffers.

Single-mic mixes work, but distinct voices diarize far better than two people
on tinny phone speakers. `--denoise` helps noisy recordings.

### Getting the file out of Voice Memos

> **Use drag-and-drop — this is the reliable path.** Drag the recording out of
> the Voice Memos app straight onto the DropMemo window (or onto a Finder
> folder). You get a file named after the memo's title, e.g. `Weekly sync.m4a`.

The recordings *do* live on disk at:

```
~/Library/Group Containers/group.com.apple.VoiceMemos.shared/Recordings/
```

…but they're stored as `.m4a` files whose names are date/UUID strings, **not**
the titles you gave them — so reaching in there directly is painful and
error-prone. Don't. Drag the memo out instead and let macOS name it for you.

## Usage

```bash
# 0. ALWAYS run this first on a new speech-swift version — it prints the raw
#    `diarize`/`align` output next to what the parsers extracted, so you can
#    confirm the formats match (and tweak parse_diarize/parse_align if not):
python3 notetaker.py check meeting.m4a

# 1. copy the config template and fill in who was there
cp config.example.yaml config.yaml && $EDITOR config.yaml

# 2. full run
python3 notetaker.py run meeting.m4a --config config.yaml --out-dir ./out

# noisy call, end-to-end Neural-Engine diarizer (default), Dutch hint:
python3 notetaker.py run call.m4a --denoise --language nl

# iterate on prompts cheaply — skip all audio stages, reuse intermediates:
python3 notetaker.py run --from-json out/pipeline.json --config config.yaml
```

Prefer not to touch the command line? See **Front-ends** below — the DropMemo
app and a Shortcut both wrap this so you only ever drag in a file.

### Long meetings (chunking)

`speech align` — whether it transcribes or force-aligns — only handles a few
minutes of audio per call; on a long recording it silently squashes every word
timestamp into the first ~2 minutes, which then scrambles speaker attribution.
notetaker works around this automatically: it slices the audio into
`--align-chunk` windows, aligns each, and stitches the timestamps back together.
You don't have to do anything; just know an hour-long file means several passes.

The safe window differs by mode, so the default is adaptive:

- **transcribing** (no transcript): **300s** — the ASR times its own words and
  stays coherent across a 5-minute chunk.
- **force-aligning** (`--transcript`): **120s** — the forced aligner overloads on
  longer chunks (feeding it ~700 words / 300s makes it dump the whole chunk into
  the first ~140s), so it needs a tighter window.

Pass `--align-chunk N` to override. If a chunk still collapses, notetaker prints
a `⚠️ … collapsed` warning to stderr; add `--strict` to make that a hard failure
(non-zero exit) instead of continuing with unreliable timestamps.

If you already have a transcript (e.g. exported from Voice Memos), hand it over
and notetaker **force-aligns** it instead of transcribing — better text, and it
skips ASR entirely:

```bash
python3 notetaker.py run meeting.m4a --transcript voice-memo.txt
```

### Outputs (`./out/`)

| file | what |
|---|---|
| `transcript_named.md` | speaker-attributed transcript; `⚠️ uncertain` marks flagged turns |
| `notes.md` | summary, key decisions, action items, open questions |
| `pipeline.json` | every intermediate — re-run the LLM steps without re-processing audio |

## Tests

```bash
python3 -m unittest discover -s tests -v
```

Stdlib `unittest` only (no extra deps), no audio/model/network — runs in
milliseconds. Covers the pure functions that actually break things: the
`diarize`/`align` output parsers (pinned against captured real `speech` output
in `tests/fixtures/`, so a format drift fails loudly instead of silently
corrupting output), word→speaker attribution (`merge`), and the alignment
chunking + collapse guard. If you upgrade speech-swift and `notetaker.py check`
shows a new output format, re-capture the fixtures.

## Going fully offline

The two LLM steps take an OpenAI-compatible endpoint, so use Ollama, LM Studio,
your LiteLLM router, or even speech-swift's own `speech-server`:

```bash
python3 notetaker.py run meeting.m4a \
  --llm-backend openai \
  --base-url http://localhost:11434 \
  --model-llm qwen3-coder
```

A frontier model is meaningfully better at the messy attribution step; a strong
local model is fine for notes. Mixing — local for one, cloud for the other — is
just two separate invocations on `--from-json`.

### Frontier quality on your Claude subscription (no API key)

`--llm-backend claude-cli` routes the two text steps through the local `claude`
binary, so they run on your **Claude Pro/Max subscription** instead of the
pay-per-token API — no `ANTHROPIC_API_KEY`, no per-call cost (subscription rate
limits apply). In DropMemo this is the **Claude (sub)** backend.

```bash
python3 notetaker.py run meeting.m4a --llm-backend claude-cli --model-llm sonnet
```

> Note: the LLM backend only affects the two *text* steps (naming speakers +
> notes). It has **no** effect on speed of diarization/alignment — those are
> local `speech` audio passes. To speed *those* up, use a smaller ASR model
> (`--model 0.6B`); to speed up the text steps, use a smaller local model.

> **Don't** point this at Apple's on-device Foundation Model: its hard
> 4096-token context can't hold a full meeting transcript without chunking.
> Target Ollama with a 14B+ model instead.

## Tuning

- **Diarizer:** `--diarize-engine pyannote` (default, segmentation + speaker
  embeddings; add `--vad-filter`) vs `sortformer` (end-to-end CoreML on the
  ANE, faster — but its CoreML model fails on some macOS/Silicon setups and
  silently returns zero speakers, so confirm with `check` before relying on
  it). If one over-splits a speaker, try the other — the LLM merges duplicates.
- **Over-splitting:** the diarizer often splits one voice into several
  `SPEAKER_xx` clusters (a 3-person call came out as 8). The LLM relabel merges
  them back to your roster, and any stray fragment it misses is reassigned to
  the nearest named speaker and flagged `⚠️`. To cut it at the source, lower
  `--cluster-threshold` (speech default 0.715; lower = fewer speakers) toward
  your known head count.
- **ASR size:** `--model 0.6B` is faster, `1.7B` (default) more accurate.
- **Bad attribution?** The single biggest lever is the `context:` block — add
  roles and who-asks-vs-answers. The model is told to *flag* rather than guess
  when unsure, so trust the `⚠️` marks and skim those in the audio. The roster
  in `participants:` is enforced — the LLM can only assign those names.
- **Local model choice:** any 14B+ instruction-follower works for attribution +
  notes (`--model-llm` / `OLLAMA_MODEL`). A general *instruct* model tends to
  follow the strict label format better than a coding model.
- **Schema drift:** if a future `speech` release changes the JSON/line format,
  `check` will show it and you adjust `parse_diarize` / `parse_align` — the two
  functions are isolated and commented for exactly this.

## Front-ends

So you never deal with paths, three layers all call the **same**
`scripts/process.sh` (local-Ollama defaults + a timestamped out-dir):

**`scripts/process.sh`** — the single source of truth. Pass an audio file plus
any extra `notetaker.py` flags:

```bash
scripts/process.sh "~/Desktop/Weekly sync.m4a"            # uses your defaults
scripts/process.sh call.m4a --denoise --cluster-threshold 0.55
```

Override via env: `OLLAMA_MODEL`, `OLLAMA_URL`, `LLM_BACKEND`, `OUT_ROOT`.

**DropMemo** (macOS app) — one window: drag a memo onto it (or use the file
picker), set **who's in the meeting** (the participants + context fields write
straight to `config.yaml`, and are pre-filled from it on launch), pick the
backend / LLM model / denoise / diarize-engine / **ASR size** / **language**,
optionally **attach a transcript** (a Voice Memos export → force-aligned for
much better text, especially non-English), watch the pipeline stream live, and
**Stop** (⌘.) any time. On completion it reveals the out-dir and opens
`notes.md`. Build it without Xcode:

```bash
scripts/build_dropmemo.sh          # → build/DropMemo.app (swiftc, ad-hoc signed)
open build/DropMemo.app            # or: cp -R build/DropMemo.app /Applications/
```

It registers as a handler for `.m4a/.wav/.mp3`, so **Open With ▸ DropMemo** and
dropping a memo on its Dock icon both work.

**Shortcut (no Xcode needed)** — make a Shortcut that **Receives audio**, then
**Run Shell Script** with:

```bash
/path/to/my-meeting-notetaker/scripts/process.sh "$1"
```

Because it accepts audio, it appears in the Voice Memos **Share** sheet — share
a memo straight to it. (Drag-and-drop out of Voice Memos onto DropMemo is still
the most reliable route, since it yields a properly titled file.)
