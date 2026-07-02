# Transcript cut quality — what we tried, what worked, what didn't

A decision log for the "speaker cuts / timestamps look wrong" class of problems.
Written after debugging the 58-min EdTech call (`out-20260702-*`). Read this
before re-investigating — several plausible fixes are dead-ends, and one earlier
belief in memory turned out to be flat wrong.

## The symptom

Speaker labels flip mid-sentence; long passages get attributed to the wrong
person; the timeline has dead-zones (e.g. nothing between 8:11 and 10:00);
one person's speech is chopped into several consecutive blocks. Notes built on
top inherit all of it.

## How the pipeline builds cuts (so you know where to look)

`speech diarize` → speaker **turns** (correctly timed on the full audio) →
`speech align` → **word timestamps** (chunked) → `merge()` drops each word into
the diarization turn covering its midpoint → segments → LLM renames raw
`SPEAKER_xx` to roster names. **A cut is only as good as the word timestamps.**

## Root cause (confirmed, with evidence)

The word timestamps were collapsing, in the **force-align path** (a pasted /
attached transcript, not raw ASR). The `0.6B` forced aligner overloads when fed
too many words per chunk: at ~700 words / 300s it gives up and piles the whole
chunk into its first ~140s.

Measured on chunk 2 (audio 600–900s): 419 of 694 words crushed into a single
30-second bin, leaving a 2-minute dead-zone. `merge()` then filed words spoken at
9:30 under whoever was talking at 7:30 → scrambled attribution. Reproduced
exactly by force-aligning that chunk's assigned text against its audio.

## What worked

1. **Adaptive align-chunk size.** Force-align now defaults to **120s**
   (≈230 words, inside the aligner's capacity); transcribing stays at **300s**.
   Re-aligning the whole call at 120s eliminated the dead-zone and repositioned
   the 10–13 min region correctly. (`ALIGN_CHUNK_SECONDS_FORCED`, adaptive
   default in `run`.)
2. **Collapse guard + `--strict`.** `align_words()` warns when a chunk's words
   cover <60% of its window; `--strict` turns that into a non-zero exit so a bad
   run fails loudly instead of silently emitting garbage. (`AlignmentCollapse`.)
3. **`tidy_segments()` — coalesce + absorb.** After relabeling, merge adjacent
   same-speaker segments (the 7 raw clusters collapse to 2 names but were never
   re-merged) and absorb ≤2-word non-sentence fragments wedged between two blocks
   of the same other speaker (diarizer thrash). Took the call from **76 → 51
   segments**, 0 adjacent same-speaker left, idempotent.

## What we tried or considered and REJECTED (don't redo these)

- **"Use a bigger / smarter LLM."** No. The cut boundaries are frozen by
  `merge()` before any LLM sees them, and the defect is timestamp math, not
  language understanding. The relabel LLM only renames; it cannot re-cut.
- **"Switch the ASR model (0.6B → 1.7B)."** No effect on force-align timing. With
  a provided transcript the ASR is bypassed and the aligner is **always the fixed
  0.6B `Qwen3-ForcedAligner`** — the `--model` picker only changes ASR *text*
  quality, not where words land.
- **Keeping the 300s chunk (the old default).** Fine for the ASR path (a
  standalone 300s transcribe aligns cleanly, full coverage — this is why the old
  memory note said "5-min chunks align cleanly"). **False for force-align**,
  where 300s collapses. The lesson: the safe window depends on the mode.
- **Full sentence-boundary snapping** (force every speaker change onto a `.!?`).
  Rejected after measuring: post-coalesce, 49 of 58 remaining boundaries are
  mid-sentence, but they're **long** segments (median 95 words) meeting where the
  ASR simply didn't punctuate a real handoff. Snapping would shuffle whole blocks
  to the wrong speaker — worse than the disease. We only absorb *tiny* fragments
  instead.
- **Curbing diarization over-split as the primary fix** (lower
  `--cluster-threshold`). The knob exists, but it treats the root cause when the
  *symptom* (un-merged same-name blocks) is cheaper and safer to kill with
  `tidy_segments`. Coalescing works regardless of how many clusters diarization
  emits.

## Known residuals (not solved — accept or tackle later)

- **~36% of force-aligned words still get near-zero durations.** The pasted
  transcript is untimed, so `align_words()` splits it by word **count**
  proportionally (assumes constant speaking rate). Smaller chunks bound the
  drift but don't remove it. Fixing properly needs timed source text or a
  per-chunk realignment that re-estimates the split — not worth it yet.
- **Mid-sentence cuts at genuine speaker changes** remain (see rejected
  sentence-snapping). They read slightly rough but are correctly attributed.
- **ASR text errors** ("loci method" → "Loki", "NotebookLM" → "no Boog LM") are
  the model's accuracy ceiling, unrelated to cuts. A cleaner source transcript
  (`--transcript` from Voice Memos) is the lever, not anything in this pipeline.

## Diagnosing this next time

- Per-chunk word coverage is the tell. If a chunk's words don't reach ~60%+ of
  its window, it collapsed. `align_words()` now warns automatically; `--strict`
  fails the run.
- The `693/694`-per-chunk fingerprint means force-align (transcript split by
  count), not raw ASR. Uniform counts ⇒ proportional split.
- After a speech-swift upgrade, run `notetaker.py check <file>` and re-capture
  `tests/fixtures/` if the output format changed.
