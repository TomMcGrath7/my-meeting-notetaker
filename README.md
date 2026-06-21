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

### Outputs (`./out/`)

| file | what |
|---|---|
| `transcript_named.md` | speaker-attributed transcript; `⚠️ uncertain` marks flagged turns |
| `notes.md` | summary, key decisions, action items, open questions |
| `pipeline.json` | every intermediate — re-run the LLM steps without re-processing audio |

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

> **Don't** point this at Apple's on-device Foundation Model: its hard
> 4096-token context can't hold a full meeting transcript without chunking.
> Target Ollama with a 14B+ model instead.

## Tuning

- **Diarizer:** `--diarize-engine sortformer` (default, end-to-end on the ANE)
  vs `pyannote` (segmentation + speaker embeddings; add `--vad-filter`). If one
  over-splits a speaker, try the other — the LLM merges duplicates anyway.
- **ASR size:** `--model 0.6B` is faster, `1.7B` (default) more accurate.
- **Bad attribution?** The single biggest lever is the `context:` block — add
  roles and who-asks-vs-answers. The model is told to *flag* rather than guess
  when unsure, so trust the `⚠️` marks and skim those in the audio.
- **Schema drift:** if a future `speech` release changes the JSON/line format,
  `check` will show it and you adjust `parse_diarize` / `parse_align` — the two
  functions are isolated and commented for exactly this.
