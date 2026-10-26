# Two recordings of one conversation: merging them by hand

How a call recorded on two phones was turned into one transcript (2026-10-26), done
by hand outside the pipeline. It worked: the person who knew the call judged most of
the reconstruction right. This is the recipe and the traps, for whenever DropMemo
grows a "second recording of the same meeting" slot (low priority).

## The situation

- One conversation, two phones, started ~17 min apart, each run through DropMemo on its own.
- One phone ~4.5 dB quieter. Both very quiet overall (mean −52 / −56 dB).
- Two hours of Spanish, one audible speaker (one side of a phone call), so the single
  speaker label was correct, not a diarization failure.

## The recipe

1. **Align on the words, not the audio.** `difflib.SequenceMatcher` over the two
   `pipeline.json` word lists (lower-cased, accents and punctuation stripped). Matching
   runs of ≥5 words give the offset: here 1015.5 s, with 103 of 110 runs within a few
   seconds of it. Fitting offset against time gives the drift: 0.13 s/hour, so mixing
   the two audio tracks would not echo. No model needed.
2. **Cut a shared timeline at pauses** (gap >1.2 s once a window is ≥8 s, hard cut at
   35 s) and pair the two recordings window by window.
3. **The louder phone is the default.** It was right far more often. The quieter one is
   support for where the louder one is garbled, and it invents fluent nonsense when it
   cannot hear (whole sentences about topics nobody raised).
4. **Word overlap is a poor agreement test.** Even where both clearly said the same thing,
   the overlap ratio sat around 0.3. Let the translation step judge agreement on meaning instead.
5. **Translate with listener context.** `claude -p --model sonnet`, 45 windows a call, JSON in,
   JSON out, with a short block of what the listener knows (names, places, what happened).
   The rules: the context may only choose between readings and repair misheard names, never add
   events, and every repair is marked. Check each batch returns every id.
6. **For the passages that matter, take more readings.** Cut the passage from both
   phones at the same moment, volume-normalise it (`loudnorm=I=-20:TP=-2`), and transcribe it
   with two model families (Qwen3-ASR 1.7B and Whisper large-v3-turbo). With the original run,
   that gives six readings. One `claude -p --model opus` pass then reconstructs the passage,
   tagging every sentence [4+], [2-3], [1] or [guess].
7. **Fill the remaining gaps with ranked theories**, built from the fragments each reading
   kept. Most of the value is in sound-alike repair: *producción* → *relación*,
   *antibajos* → *altibajos*, *consolada* → *controlada*, *gráficos* → *chicos*,
   *trasero* → *trastero*, *avanzada*/*aborada* → *agobiada*, *Vigo* → *Digo*.

## Traps

- **`speech transcribe --context` leaks.** In 106 of 283 Qwen segments the hint list itself was
  printed as the transcript, mostly in quiet stretches. A hint can then pass for evidence: a city
  name that seemed to confirm the listener's knowledge turned out to be *digo* ("I mean").
  Don't use `--context`, or filter its echoes out and never count a hinted reading as support.
- **The pipeline normalises volume only for VAD**, not for ASR, so quiet recordings are
  transcribed at their original level. Normalising before ASR is a cheap fix for single memos too.
- **RAM.** Qwen3-ASR 1.7B peaked at 6.3 to 7.2 GB, about 40 to 90 s per 14.5-minute clip.
  Whisper turbo (CoreML) peaked at 0.6 GB. `speech` needs the sandbox off.
- **Whisper returns one block with no timestamps**, and it loops on repeated phrases
  (one phrase came back 30 times). Use it as a reading, not as a timeline.
- **`pipeline.json` keeps no per-word ASR confidence**, so "show the second phone only where
  the first was unsure" is not possible until confidence is saved.
- **"Show both when unsure" floods the page** (236 of 349 windows). Defaulting to the louder
  phone, with the other as a small "B heard:" line, keeps it readable.

## If this becomes a feature

A second drop slot. Word-based alignment (step 1). Volume normalisation before ASR. A merged
transcript with the louder phone as the default. A ranked list of the windows where the two
disagree most, each shown phrase by phrase, plus a place to enter names and corrections that
feed a fresh translation pass.
