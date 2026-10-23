"""Unit + regression tests for notetaker.py.

Stdlib `unittest` only (matches the repo's stdlib+PyYAML rule) — no audio, no
model, no network. Run:  python -m unittest discover -s tests -v

Coverage targets the pure functions that have actually broken before or that
guard the timestamp-collapse bug:
  - parse_diarize / parse_align / _extract_json : parsers wrapped around the
    external `speech` binary's stdout. A format drift there silently corrupts
    the whole pipeline, so we pin them against captured real output (fixtures/).
  - _norm_label / _first                        : label + key normalisation.
  - merge / _turn_for                           : word→speaker attribution.
  - align_words                                 : chunk stitching, proportional
    text split, and the collapse guard / --strict behaviour.
  - llm (openai backend)                        : a prompt the server clipped to
    its context window must fail loudly, not come back as partial notes.
  - llm (ollama backend)                        : the context window is sized to
    the prompt and capped, an over-cap or clipped prompt fails loudly, and
    without llm-seam the call falls back to the deprecated /v1 path.

The ollama-backend tests that drive llm-seam's real client are skipped when
llm-seam is not installed (it is optional). Run both ways:
  uv run -q --with-requirements requirements.txt \
      --with-requirements requirements-llm-seam.txt \
      python -m unittest discover -s tests
  uv run -q --with-requirements requirements.txt python -m unittest discover -s tests
"""

import io
import json
import os
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

import notetaker as N

HAVE_SEAM = N._llm_seam()[0] is not None
NEEDS_SEAM = unittest.skipUnless(
    HAVE_SEAM, "llm-seam >= 0.8.0 is not installed "
               "(add --with-requirements requirements-llm-seam.txt)")

FIX = Path(__file__).parent / "fixtures"


# --------------------------------------------------------------------------- #
# fixtures: real `speech` stdout (progress text + JSON / bracket lines)
# --------------------------------------------------------------------------- #

class TestFixtureParsers(unittest.TestCase):
    """Regression: real captured `speech` output must keep parsing."""

    def test_parse_diarize_real_output(self):
        raw = (FIX / "diarize_sample.txt").read_text()
        turns = N.parse_diarize(raw)
        self.assertTrue(turns, "no turns parsed from real diarize output")
        # progress lines contain '[30%]' etc. before the JSON — must be skipped.
        first = turns[0]
        self.assertEqual(first.speaker, "SPEAKER_00")   # bare int 0 → normalised
        self.assertAlmostEqual(first.start, 0.0, places=3)
        self.assertGreater(first.end, first.start)
        # sorted by start, every turn well-formed
        self.assertEqual([t.start for t in turns], sorted(t.start for t in turns))
        for t in turns:
            self.assertLessEqual(t.start, t.end)
            self.assertRegex(t.speaker, r"^SPEAKER_\d\d$")

    def test_parse_align_real_output(self):
        raw = (FIX / "align_sample.txt").read_text()
        words = N.parse_align(raw)
        # fixture was captured with 231 aligned words.
        self.assertEqual(len(words), 231)
        self.assertEqual(words[0].text, "or")
        self.assertAlmostEqual(words[0].start, 0.0, places=2)
        for w in words:
            self.assertLessEqual(w.start, w.end)
            self.assertTrue(w.text)   # no empty tokens


# --------------------------------------------------------------------------- #
# _extract_json : progress-text-then-JSON, tolerant to bracket noise
# --------------------------------------------------------------------------- #

class TestExtractJson(unittest.TestCase):
    def test_skips_progress_lines_including_percent_brackets(self):
        raw = "Loading…\n  [30%] Downloading…\nRunning…\n{\"segments\": [1, 2]}\n"
        self.assertEqual(N._extract_json(raw), {"segments": [1, 2]})

    def test_returns_first_json_ignores_trailing(self):
        raw = 'noise {"a": 1} trailing {"b": 2}'
        self.assertEqual(N._extract_json(raw), {"a": 1})

    def test_top_level_array(self):
        raw = "progress\n[{\"start\": 0}]"
        self.assertEqual(N._extract_json(raw), [{"start": 0}])

    def test_raises_when_no_json(self):
        with self.assertRaises(json.JSONDecodeError):
            N._extract_json("just [10%] progress, no json here")


# --------------------------------------------------------------------------- #
# _norm_label / _first
# --------------------------------------------------------------------------- #

class TestNormLabel(unittest.TestCase):
    def test_bare_int(self):
        self.assertEqual(N._norm_label("0"), "SPEAKER_00")
        self.assertEqual(N._norm_label("3"), "SPEAKER_03")
        self.assertEqual(N._norm_label("12"), "SPEAKER_12")

    def test_speaker_prefixed(self):
        self.assertEqual(N._norm_label("SPEAKER_2"), "SPEAKER_02")
        self.assertEqual(N._norm_label("speaker 10"), "SPEAKER_10")

    def test_named_label_passthrough(self):
        # A human name must NOT be mangled into SPEAKER_NN.
        self.assertEqual(N._norm_label("Tom"), "Tom")
        self.assertEqual(N._norm_label("  Olly "), "Olly")


class TestFirst(unittest.TestCase):
    def test_first_present_non_null_key(self):
        d = {"start": None, "startTime": 1.5}
        self.assertEqual(N._first(d, "start", "startTime", default=0.0), 1.5)

    def test_default_when_absent(self):
        self.assertEqual(N._first({}, "a", "b", default="x"), "x")


# --------------------------------------------------------------------------- #
# parse_align : bracket lines + JSON fallback
# --------------------------------------------------------------------------- #

class TestParseAlign(unittest.TestCase):
    def test_bracket_lines(self):
        raw = ("Transcription: hi there\n"
               "[0.16s - 2.24s] hi\n"
               "[2.30s - 3.00s] there,\n")
        words = N.parse_align(raw)
        self.assertEqual([w.text for w in words], ["hi", "there,"])
        self.assertAlmostEqual(words[0].start, 0.16)
        self.assertAlmostEqual(words[1].end, 3.00)

    def test_json_fallback(self):
        raw = 'progress\n{"words": [{"word": "hey", "start": 1, "end": 2}]}'
        words = N.parse_align(raw)
        self.assertEqual(len(words), 1)
        self.assertEqual(words[0].text, "hey")

    def test_empty(self):
        self.assertEqual(N.parse_align("nothing here"), [])


# --------------------------------------------------------------------------- #
# merge / _turn_for : word → speaker attribution invariants
# --------------------------------------------------------------------------- #

class TestMerge(unittest.TestCase):
    def _words(self, *spans):
        return [N.Word(text=f"w{i}", start=s, end=e)
                for i, (s, e) in enumerate(spans)]

    def test_coalesces_same_speaker_and_orders(self):
        turns = [N.Turn("SPEAKER_00", 0, 5), N.Turn("SPEAKER_01", 5, 10)]
        words = self._words((0, 1), (1, 2), (6, 7), (7, 8))
        segs = N.merge(words, turns)
        self.assertEqual([s.speaker for s in segs], ["SPEAKER_00", "SPEAKER_01"])
        self.assertEqual(segs[0].text, "w0 w1")
        self.assertEqual(segs[1].text, "w2 w3")
        # time-ordered, non-overlapping
        for a, b in zip(segs, segs[1:]):
            self.assertLessEqual(a.end, b.start + 1e-9)

    def test_every_word_attributed(self):
        turns = [N.Turn("SPEAKER_00", 0, 5), N.Turn("SPEAKER_01", 5, 10)]
        words = self._words((0, 1), (4, 5), (5, 6), (9, 10))
        segs = N.merge(words, turns)
        self.assertEqual(sum(len(s.text.split()) for s in segs), len(words))

    def test_boundary_word_picks_containing_turn(self):
        # midpoint 4.5 is inside SPEAKER_00 (0..5), not SPEAKER_01.
        turns = [N.Turn("SPEAKER_00", 0, 5), N.Turn("SPEAKER_01", 5, 10)]
        self.assertEqual(N._turn_for(4.5, turns), "SPEAKER_00")
        self.assertEqual(N._turn_for(5.5, turns), "SPEAKER_01")

    def test_word_outside_all_turns_picks_nearest(self):
        turns = [N.Turn("SPEAKER_00", 0, 5), N.Turn("SPEAKER_01", 20, 25)]
        self.assertEqual(N._turn_for(6, turns), "SPEAKER_00")   # nearer to 5
        self.assertEqual(N._turn_for(19, turns), "SPEAKER_01")  # nearer to 20

    def test_no_turns_yields_single_segment(self):
        words = self._words((0, 1), (1, 2))
        segs = N.merge(words, [])
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0].speaker, "SPEAKER_00")

    def test_punctuation_spacing_cleanup(self):
        turns = [N.Turn("SPEAKER_00", 0, 5)]
        words = [N.Word("hello", 0, 1), N.Word(",", 1, 1.1), N.Word("world", 1.1, 2)]
        segs = N.merge(words, turns)
        self.assertEqual(segs[0].text, "hello, world")


# --------------------------------------------------------------------------- #
# align_words : chunk stitching, proportional split, collapse guard, --strict
# --------------------------------------------------------------------------- #

class TestAlignWords(unittest.TestCase):
    """align_words drives the external binary; we stub run_align/_cut_wav/
    _wav_duration and let the real chunking + guard logic run."""

    def setUp(self):
        self._orig = (N._wav_duration, N._cut_wav, N.run_align)

    def tearDown(self):
        N._wav_duration, N._cut_wav, N.run_align = self._orig

    def _install(self, dur, per_chunk):
        """per_chunk(local_start_texts) -> list[(start, end, text)] in chunk-local
        time. We record the slice_text passed so split boundaries can be checked."""
        self.seen_text = []
        N._wav_duration = lambda wav: dur
        N._cut_wav = lambda wav, start, length, out: out
        def fake_run_align(wav, model, language, transcript=None):
            self.seen_text.append(transcript)
            rows = per_chunk(transcript)
            return "\n".join(f"[{s:.2f}s - {e:.2f}s] {t}" for s, e, t in rows)
        N.run_align = fake_run_align

    def test_short_audio_single_call_no_chunking(self):
        # dur <= chunk+1 → one align call, no _cut_wav.
        self._install(dur=90, per_chunk=lambda txt: [(0, 1, "a"), (1, 2, "b")])
        words = N.align_words(Path("x.wav"), "1.7B", "en", chunk_seconds=300)
        self.assertEqual([w.text for w in words], ["a", "b"])

    def test_offsets_stitched_onto_real_timeline(self):
        # 3 chunks of 100s; the marker word sits at local 10s → global 10/110/210.
        # A trailing filler word keeps coverage healthy (no collapse warning).
        self._install(dur=300, per_chunk=lambda txt: [(10, 11, "w"), (94, 95, "_")])
        words = N.align_words(Path("x.wav"), "1.7B", "en", chunk_seconds=100)
        self.assertEqual([round(w.start) for w in words if w.text == "w"],
                         [10, 110, 210])

    def test_proportional_text_split(self):
        # 6 transcript words, 3 chunks → 2 words each, in order. Spread across the
        # window so coverage stays healthy while we check the split boundaries.
        def per_chunk(txt):
            ws = (txt or "").split()
            return [(1, 2, w) for w in ws[:-1]] + ([(94, 95, ws[-1])] if ws else [])
        self._install(dur=300, per_chunk=per_chunk)
        words = N.align_words(Path("x.wav"), "1.7B", "en",
                              transcript="a b c d e f", chunk_seconds=100)
        self.assertEqual([s.strip() for s in self.seen_text], ["a b", "c d", "e f"])
        self.assertEqual([w.text for w in words], ["a", "b", "c", "d", "e", "f"])

    def test_collapse_guard_warns(self):
        # chunk 1 covers only 50s of a 100s window (<60%) → warned; others healthy.
        def per_chunk(txt):
            per_chunk.i += 1
            end = 50 if per_chunk.i == 2 else 95
            return [(0, 1, "x"), (end - 1, end, "y")]
        per_chunk.i = 0
        self._install(dur=300, per_chunk=per_chunk)
        err = io.StringIO()
        with redirect_stderr(err):
            N.align_words(Path("x.wav"), "1.7B", "en", chunk_seconds=100)
        msg = err.getvalue()
        self.assertIn("collapsed", msg)
        self.assertIn("1/3 chunks collapsed", msg)

    def test_healthy_chunks_no_warning(self):
        self._install(dur=300, per_chunk=lambda txt: [(0, 1, "x"), (94, 95, "y")])
        err = io.StringIO()
        with redirect_stderr(err):
            N.align_words(Path("x.wav"), "1.7B", "en", chunk_seconds=100)
        self.assertNotIn("collapsed", err.getvalue())

    def test_strict_raises_on_collapse(self):
        self._install(dur=300, per_chunk=lambda txt: [(0, 1, "x"), (10, 11, "y")])
        err = io.StringIO()
        with redirect_stderr(err), self.assertRaises(N.AlignmentCollapse):
            N.align_words(Path("x.wav"), "1.7B", "en", chunk_seconds=100, strict=True)


# --------------------------------------------------------------------------- #
# tidy_segments : coalesce same-speaker runs + absorb tiny noise flips
# --------------------------------------------------------------------------- #

class TestTidySegments(unittest.TestCase):
    def seg(self, spk, start, end, text, flagged=False, reason=""):
        return N.Segment(speaker=spk, start=start, end=end, text=text,
                         flagged=flagged, flag_reason=reason)

    def test_coalesce_merges_adjacent_same_speaker(self):
        # over-split into 3 blocks that all relabeled to the same name.
        segs = [self.seg("Tom", 0, 1, "additional stuff. I"),
                self.seg("Tom", 1, 2, "mean,"),
                self.seg("Tom", 2, 3, "you could also")]
        out = N._coalesce(segs)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].text, "additional stuff. I mean, you could also")
        self.assertEqual((out[0].start, out[0].end), (0, 3))

    def test_coalesce_does_not_mutate_input(self):
        segs = [self.seg("Tom", 0, 1, "a"), self.seg("Tom", 1, 2, "b")]
        N._coalesce(segs)
        self.assertEqual(segs[0].text, "a")   # original untouched

    def test_coalesce_unions_flags(self):
        segs = [self.seg("Tom", 0, 1, "a"),
                self.seg("Tom", 1, 2, "b", flagged=True, reason="why")]
        out = N._coalesce(segs)
        self.assertTrue(out[0].flagged)
        self.assertEqual(out[0].flag_reason, "why")

    def test_absorb_tiny_flip_between_same_speaker(self):
        # Tobias … [one stray Tom word] … Tobias  → the word is timing noise.
        segs = [self.seg("Tobias", 0, 5, "a long turn here"),
                self.seg("Tom", 5, 6, "all"),
                self.seg("Tobias", 6, 10, "continues talking")]
        out = N.tidy_segments(segs)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].speaker, "Tobias")
        self.assertIn("all", out[0].text)

    def test_does_not_absorb_complete_sentence(self):
        # A short but COMPLETE utterance is a real turn — keep it.
        segs = [self.seg("Tobias", 0, 5, "a long turn here"),
                self.seg("Tom", 5, 6, "This is English?"),
                self.seg("Tobias", 6, 10, "continues talking")]
        out = N.tidy_segments(segs)
        self.assertEqual([s.speaker for s in out], ["Tobias", "Tom", "Tobias"])

    def test_does_not_absorb_when_neighbours_differ(self):
        segs = [self.seg("Tobias", 0, 5, "turn one"),
                self.seg("Tom", 5, 6, "hi"),
                self.seg("Dani", 6, 10, "turn three")]
        out = N.tidy_segments(segs)
        self.assertEqual([s.speaker for s in out], ["Tobias", "Tom", "Dani"])

    def test_does_not_absorb_long_fragment(self):
        segs = [self.seg("Tobias", 0, 5, "turn one"),
                self.seg("Tom", 5, 6, "this is more than two words"),
                self.seg("Tobias", 6, 10, "turn three")]
        out = N.tidy_segments(segs)
        self.assertEqual([s.speaker for s in out], ["Tobias", "Tom", "Tobias"])

    def test_idempotent(self):
        segs = [self.seg("Tom", 0, 1, "a"), self.seg("Tom", 1, 2, "b"),
                self.seg("Tobias", 2, 3, "c")]
        once = N.tidy_segments(segs)
        twice = N.tidy_segments(once)
        self.assertEqual([(s.speaker, s.text) for s in once],
                         [(s.speaker, s.text) for s in twice])



# --------------------------------------------------------------------------- #
# llm(): a prompt clipped to the server's context window is not a result
# --------------------------------------------------------------------------- #

class _FakeResponse:
    def __init__(self, payload):
        self._raw = json.dumps(payload).encode()

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestPromptTruncation(unittest.TestCase):
    """Regression (to-do #264): Ollama cut a long transcript to its default
    context window and the notes covered part of the meeting, with no error.
    The HTTP call is faked; no server and no model is touched."""

    KW = dict(backend="openai", model="qwen3:8b",
              base_url="http://localhost:11434", max_tokens=4000)

    def _call(self, transcript, usage):
        payload = {"choices": [{"message": {"content": "## Summary\nnotes"}}]}
        if usage is not None:
            payload["usage"] = usage
        self.sent = []

        def fake_urlopen(req, timeout=None):
            self.sent.append(req)
            return _FakeResponse(payload)

        with mock.patch.object(N.urllib.request, "urlopen", fake_urlopen):
            return N.llm([{"role": "user", "content": transcript}], "system",
                         **self.KW)

    def setUp(self):
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("NOTETAKER_ALLOW_TRUNCATED", None)

    # ~1 hour of speech: 60k characters, about 15k tokens.
    LONG = "Tom: we agreed the launch moves to the ninth. " * 1300

    def test_long_transcript_clipped_to_default_window_exits(self):
        with self.assertRaises(SystemExit) as cm:
            self._call(self.LONG, {"prompt_tokens": 4096})
        msg = str(cm.exception)
        self.assertIn(f"{len(self.LONG) + len('system'):,} characters", msg)
        self.assertIn("4,096 tokens", msg)
        self.assertIn("qwen3:8b", msg)
        # the suggested size must actually hold what was sent
        self.assertIn("OLLAMA_CONTEXT_LENGTH=32768", msg)

    def test_long_transcript_fully_read_returns_notes(self):
        out = self._call(self.LONG, {"prompt_tokens": len(self.LONG) // 4})
        self.assertEqual(out, "## Summary\nnotes")

    def test_short_transcript_is_never_flagged(self):
        # 10 tokens reported for 3k characters is an odd count, not a cut:
        # nothing this short can overflow a context window.
        out = self._call("Tom: hello there. " * 150, {"prompt_tokens": 10})
        self.assertEqual(out, "## Summary\nnotes")

    def test_no_usage_reported_cannot_be_judged(self):
        self.assertEqual(self._call(self.LONG, None), "## Summary\nnotes")
        self.assertEqual(self._call(self.LONG, {"prompt_tokens": 0}),
                         "## Summary\nnotes")

    def test_override_downgrades_to_a_warning(self):
        os.environ["NOTETAKER_ALLOW_TRUNCATED"] = "1"
        err = io.StringIO()
        with redirect_stderr(err):
            out = self._call(self.LONG, {"prompt_tokens": 4096})
        self.assertEqual(out, "## Summary\nnotes")
        self.assertIn("read only 4,096 tokens", err.getvalue())

    def test_request_still_goes_to_the_openai_compatible_endpoint(self):
        # Pins the path: this endpoint has no context-size field, which is
        # why the guard is a check on the answer and not a bigger request.
        self._call("hi", None)
        self.assertEqual(self.sent[0].full_url,
                         "http://localhost:11434/v1/chat/completions")
        self.assertNotIn("num_ctx", self.sent[0].data.decode())

    def test_suggested_context_holds_prompt_and_reply(self):
        self.assertEqual(N._suggest_ctx(9000), 8192)      # 3000 + 4000
        self.assertEqual(N._suggest_ctx(60000), 32768)    # 20000 + 4000


# --------------------------------------------------------------------------- #
# llm(backend="ollama"): the window is sized to the prompt, through llm-seam
# --------------------------------------------------------------------------- #

class _OllamaHarness(unittest.TestCase):
    """Fakes the one HTTP seam both paths share (`urllib.request.urlopen`), so
    llm-seam's real client and payload builder run. No server, no model."""

    KW = dict(backend="ollama", model="qwen3:8b",
              base_url="http://localhost:11434", max_tokens=4000)
    # ~1 hour of speech: 60k characters. Estimated 20k tokens, really ~15k.
    LONG = "Tom: we agreed the launch moves to the ninth. " * 1300

    def setUp(self):
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("NOTETAKER_ALLOW_TRUNCATED", None)
        os.environ.pop("NOTETAKER_NUM_CTX_MAX", None)
        N._noticed.clear()
        self.sent = []

    def _call(self, transcript, read=None, **over):
        """`read` is the prompt-token count the fake server reports back."""
        def fake_urlopen(req, timeout=None):
            self.sent.append(req)
            if req.full_url.endswith("/api/chat"):
                body = {"message": {"content": "## Summary\nnotes"},
                        "done_reason": "stop", "eval_count": 9}
                if read is not None:
                    body["prompt_eval_count"] = read
            else:
                body = {"choices": [{"message": {"content": "## Summary\nnotes"}}]}
                if read is not None:
                    body["usage"] = {"prompt_tokens": read}
            return _FakeResponse(body)

        with mock.patch.object(N.urllib.request, "urlopen", fake_urlopen):
            return N.llm([{"role": "user", "content": transcript}], "system",
                         **dict(self.KW, **over))

    def _payload(self):
        self.assertEqual(len(self.sent), 1)
        return json.loads(self.sent[0].data.decode())


@NEEDS_SEAM
class TestOllamaContextWindow(_OllamaHarness):
    """Regression (to-do #264, the real fix): the transcript was cut to
    Ollama's default window because nothing ever asked for a bigger one."""

    def test_long_transcript_gets_a_window_that_holds_it(self):
        out = self._call(self.LONG, read=len(self.LONG) // 4)
        self.assertEqual(out, "## Summary\nnotes")
        self.assertEqual(self.sent[0].full_url, "http://localhost:11434/api/chat")
        options = self._payload()["options"]
        # 59,806 chars / 3 = 19,936 tokens, + 4,000 reply = 23,936 -> 24,576
        self.assertEqual(options["num_ctx"], 24576)
        self.assertEqual(options["num_predict"], 4000)
        self.assertGreaterEqual(
            options["num_ctx"], (len(self.LONG) + len("system")) / 3 + 4000)

    def test_short_transcript_is_not_inflated_to_the_cap(self):
        self._call("Tom: hello there. " * 150, read=700)   # 2.7k characters
        self.assertEqual(self._payload()["options"]["num_ctx"], 6144)
        self.assertLess(self._payload()["options"]["num_ctx"],
                        N.DEFAULT_NUM_CTX_MAX)

    def test_whole_prompt_is_sent_as_system_and_user(self):
        self._call(self.LONG, read=15000)
        self.assertEqual(self._payload()["messages"], [
            {"role": "system", "content": "system"},
            {"role": "user", "content": self.LONG}])

    def test_transcript_over_the_cap_fails_with_the_numbers(self):
        three_hours = self.LONG * 3                      # 179k characters
        with self.assertRaises(SystemExit) as cm:
            self._call(three_hours, read=45000)
        msg = str(cm.exception)
        self.assertEqual(self.sent, [])                  # nothing was sent
        self.assertIn(f"{len(three_hours) + len('system'):,} characters", msg)
        self.assertIn("59,802 tokens", msg)              # the estimate
        self.assertIn("context window of 65,536", msg)   # what it would take
        self.assertIn("The cap is 32,768", msg)
        self.assertIn("--num-ctx-max 65536", msg)

    def test_a_raised_cap_lets_the_same_transcript_through(self):
        self._call(self.LONG * 3, read=45000, num_ctx_max=65536)
        self.assertEqual(self._payload()["options"]["num_ctx"], 65536)

    def test_a_lower_cap_refuses_what_the_default_would_take(self):
        with self.assertRaises(SystemExit) as cm:
            self._call(self.LONG, read=15000, num_ctx_max=8192)
        self.assertIn("The cap is 8,192", str(cm.exception))

    def test_window_asked_for_but_not_given_fails(self):
        # Ollama accepted num_ctx=24576 and the model still read 4,096 tokens
        # (a model whose own maximum is smaller, or a capped server).
        with self.assertRaises(SystemExit) as cm:
            self._call(self.LONG, read=4096)
        msg = str(cm.exception)
        self.assertIn("read only 4,096 tokens", msg)
        self.assertIn("context window of 24,576 was asked for", msg)
        self.assertIn("estimated 19,936 tokens", msg)
        self.assertIn("qwen3:8b", msg)

    def test_prompt_that_filled_the_whole_window_fails(self):
        with self.assertRaises(SystemExit) as cm:
            self._call(self.LONG, read=24576)
        self.assertIn("filled the whole window", str(cm.exception))

    def test_override_downgrades_a_clip_to_a_warning(self):
        os.environ["NOTETAKER_ALLOW_TRUNCATED"] = "1"
        err = io.StringIO()
        with redirect_stderr(err):
            out = self._call(self.LONG, read=4096)
        self.assertEqual(out, "## Summary\nnotes")
        self.assertIn("read only 4,096 tokens", err.getvalue())

    def test_no_count_reported_cannot_be_judged(self):
        self.assertEqual(self._call(self.LONG, read=None), "## Summary\nnotes")

    def test_count_at_exactly_half_the_window_warns(self):
        # What a live Ollama reported after cutting a prompt: half the window.
        err = io.StringIO()
        with redirect_stderr(err):
            self._call(self.LONG, read=24576 // 2 + 2)
        self.assertIn("exactly half", err.getvalue())

    def test_an_ordinary_count_says_nothing(self):
        err = io.StringIO()
        with redirect_stderr(err):
            self._call(self.LONG, read=len(self.LONG) // 4)
        self.assertEqual(err.getvalue(), "")

    def test_base_url_with_v1_reaches_the_native_endpoint(self):
        for spelled in ("http://localhost:11434", "http://localhost:11434/",
                        "http://localhost:11434/v1", "http://localhost:11434/v1/"):
            self.sent = []
            self._call("hi", read=5, base_url=spelled)
            self.assertEqual(self.sent[0].full_url,
                             "http://localhost:11434/api/chat", spelled)

    def test_reply_cut_off_at_the_budget_fails(self):
        def fake_urlopen(req, timeout=None):
            return _FakeResponse({"message": {"content": "## Summ"},
                                  "done_reason": "length",
                                  "prompt_eval_count": 15000})
        with mock.patch.object(N.urllib.request, "urlopen", fake_urlopen):
            with self.assertRaises(SystemExit) as cm:
                N.llm([{"role": "user", "content": self.LONG}], "system", **self.KW)
        self.assertIn("max_tokens=4000", str(cm.exception))

    def test_a_timeout_is_retried_once(self):
        calls = []

        def fake_urlopen(req, timeout=None):
            calls.append(timeout)
            if len(calls) == 1:
                raise TimeoutError("timed out")
            return _FakeResponse({"message": {"content": "ok"},
                                  "prompt_eval_count": 5})
        with mock.patch.object(N.urllib.request, "urlopen", fake_urlopen), \
                redirect_stderr(io.StringIO()):
            out = N.llm([{"role": "user", "content": "hi"}], "system", **self.KW)
        self.assertEqual(out, "ok")
        self.assertEqual(calls, [600.0, 600.0])


class TestOllamaWithoutLlmSeam(_OllamaHarness):
    """People with access to this repo but not to llm-seam (a private repo)
    must still be able to run it. Runs whether or not llm-seam is installed:
    `sys.modules["llm_seam"] = None` makes the import fail either way."""

    def _call(self, transcript, read=None, **over):
        with mock.patch.dict(sys.modules, {"llm_seam": None}):
            self.err = io.StringIO()
            with redirect_stderr(self.err):
                return super()._call(transcript, read=read, **over)

    def test_falls_back_to_the_v1_path_and_says_so_once(self):
        self.assertEqual(self._call("hi"), "## Summary\nnotes")
        self.assertEqual(self.sent[0].full_url,
                         "http://localhost:11434/v1/chat/completions")
        notice = self.err.getvalue()
        self.assertEqual(notice.count("\n"), 1)                 # one line
        self.assertIn("llm-seam is not installed", notice)
        self.assertIn("deprecated /v1 path", notice)
        self.assertIn("requirements-llm-seam.txt", notice)
        self._call("hi again")                                  # second step
        self.assertEqual(self.err.getvalue(), "")               # not repeated

    def test_fallback_keeps_the_truncation_guard(self):
        with self.assertRaises(SystemExit) as cm:
            self._call(self.LONG, read=4096)
        self.assertIn("read only 4,096 tokens", str(cm.exception))
        self.assertIn("OLLAMA_CONTEXT_LENGTH=32768", str(cm.exception))

    def test_fallback_accepts_a_base_url_spelled_with_v1(self):
        self._call("hi", base_url="http://localhost:11434/v1")
        self.assertEqual(self.sent[0].full_url,
                         "http://localhost:11434/v1/chat/completions")

    def test_module_imports_without_llm_seam(self):
        # The old top-level `from llm_seam import ...` made every command,
        # including `check` and the anthropic backend, die on import.
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "notetaker_no_seam", Path(N.__file__))
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {"llm_seam": None,
                                           "notetaker_no_seam": module}):
            spec.loader.exec_module(module)
            self.assertEqual(module._llm_seam(),
                             (None, "llm-seam is not installed"))

    def test_claude_cli_without_llm_seam_says_what_to_install(self):
        with self.assertRaises(SystemExit) as cm:
            self._call("hi", backend="claude-cli")
        self.assertIn("claude-cli needs llm-seam", str(cm.exception))
        self.assertEqual(self.sent, [])

    def test_an_llm_seam_too_old_to_size_the_window_counts_as_absent(self):
        class OldClient:
            ACCEPTS = frozenset({"model", "max_tokens"})
        old = mock.Mock(OllamaClient=OldClient, __version__="0.7.0")
        with mock.patch.dict(sys.modules, {"llm_seam": old}):
            self.assertEqual(N._llm_seam(),
                             (None, "llm-seam 0.7.0 is older than 0.8.0"))


class TestExplicitOpenAiBackend(_OllamaHarness):
    def test_openai_backend_is_still_the_v1_path(self):
        err = io.StringIO()
        with redirect_stderr(err):
            self._call("hi", backend="openai")
        self.assertEqual(self.sent[0].full_url,
                         "http://localhost:11434/v1/chat/completions")
        self.assertNotIn("num_ctx", self.sent[0].data.decode())
        self.assertIn("use --llm-backend ollama", err.getvalue())


class TestNumCtxSizing(unittest.TestCase):
    def test_rounds_up_to_the_step_and_never_below_the_floor(self):
        size = lambda chars: N._size_num_ctx(chars, 4000, 32768, "m")
        self.assertEqual(size(0), 4096)          # the floor
        self.assertEqual(size(288), 4096)        # 96 + 4000 = 4096 exactly
        self.assertEqual(size(291), 6144)        # 97 + 4000, one over
        self.assertEqual(size(60000), 24576)     # 20000 + 4000
        self.assertEqual(size(86304), 32768)     # the most the default cap holds

    def test_one_token_over_the_cap_is_refused(self):
        with mock.patch.dict(os.environ):
            os.environ.pop("NOTETAKER_ALLOW_TRUNCATED", None)
            with self.assertRaises(SystemExit):
                N._size_num_ctx(86307, 4000, 32768, "m")

    def test_server_root(self):
        for spelled in ("http://h:11434", "http://h:11434/", "http://h:11434/v1",
                        "http://h:11434/v1/", " http://h:11434/v1 "):
            self.assertEqual(N._server_root(spelled), "http://h:11434", spelled)
        # only a trailing /v1 segment is the API prefix
        self.assertEqual(N._server_root("http://h/v1beta"), "http://h/v1beta")
        self.assertEqual(N._server_root("http://h/proxy/v1"), "http://h/proxy")


class TestResolveNumCtxMax(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("NOTETAKER_NUM_CTX_MAX", None)

    def test_default(self):
        self.assertEqual(N.resolve_num_ctx_max(None, {}), 32768)

    def test_config_then_env_then_flag(self):
        cfg = {"num_ctx_max": 16384}
        self.assertEqual(N.resolve_num_ctx_max(None, cfg), 16384)
        os.environ["NOTETAKER_NUM_CTX_MAX"] = "24576"
        self.assertEqual(N.resolve_num_ctx_max(None, cfg), 24576)
        self.assertEqual(N.resolve_num_ctx_max(65536, cfg), 65536)

    def test_nonsense_is_refused_not_ignored(self):
        with self.assertRaises(SystemExit) as cm:
            N.resolve_num_ctx_max(None, {"num_ctx_max": "lots"})
        self.assertIn("num_ctx_max in the config", str(cm.exception))

    def test_flag_and_default_base_url_are_parsed(self):
        args = N.build_parser().parse_args(
            ["run", "a.m4a", "--llm-backend", "ollama", "--num-ctx-max", "8192"])
        self.assertEqual(args.num_ctx_max, 8192)
        self.assertIsNone(args.base_url)
        self.assertEqual(N.DEFAULT_BASE_URLS["ollama"], "http://localhost:11434")


class TestRunWiresTheCap(unittest.TestCase):
    """`run` hands llm() the backend's default URL and the cap from the config."""

    def test_config_cap_and_default_url_reach_llm(self):
        import contextlib
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            pipe = N.Pipeline(audio="a.m4a", segments=[
                N.Segment("SPEAKER_00", 0.0, 1.0, "hello there")])
            (tmp / "pipeline.json").write_text(pipe.to_json())
            (tmp / "cfg.json").write_text(json.dumps({"num_ctx_max": 16384}))
            args = N.build_parser().parse_args([
                "run", "--from-json", str(tmp / "pipeline.json"),
                "--config", str(tmp / "cfg.json"), "--out-dir", str(tmp / "out"),
                "--llm-backend", "ollama", "--model-llm", "qwen3:8b"])
            with mock.patch.dict(os.environ), \
                    mock.patch.object(N, "llm", return_value="notes") as llm, \
                    contextlib.redirect_stdout(io.StringIO()):
                os.environ.pop("NOTETAKER_NUM_CTX_MAX", None)
                args.func(args)
            kw = llm.call_args.kwargs
            self.assertEqual((kw["backend"], kw["model"]), ("ollama", "qwen3:8b"))
            self.assertEqual(kw["base_url"], "http://localhost:11434")
            self.assertEqual(kw["num_ctx_max"], 16384)
            self.assertEqual((tmp / "out" / "notes.md").read_text(), "notes")


# --------------------------------------------------------------------------- #
# speech-only transcription: VAD → spliced chunks → words on the real timeline
# --------------------------------------------------------------------------- #

import contextlib
import tempfile
import wave

# What `speech vad --engine firered --json` prints (2026-10-23): progress lines,
# two of which start with '[' themselves, then the JSON array.
VAD_RAW = """Loading audio: cut.wav
  Loaded 1440000 samples (90.00s)
  [0%] Downloading model...
  [100%] Ready
Detecting speech segments...
[
  {
    "duration" : 3.769,
    "start" : 20.153,
    "end" : 23.922
  },
  {
    "duration" : 1.5,
    "start" : 30.0,
    "end" : 31.5
  }
]
"""


class TestSpeechOnly(unittest.TestCase):

    def test_parse_vad_skips_bracketed_progress_lines(self):
        self.assertEqual(N.parse_vad(VAD_RAW), [(20.153, 23.922), (30.0, 31.5)])

    def test_parse_vad_garbage_is_no_speech(self):
        self.assertEqual(N.parse_vad("Error: could not open file"), [])

    def test_pad_regions_widens_merges_and_clamps(self):
        got = N.pad_regions([(0.2, 1.0), (2.0, 3.0), (10.0, 11.8)], dur=12.0)
        # 0.2-1.0 and 2.0-3.0 come within 1.5s once padded, so they merge; the
        # first is clamped at 0, the last at the file's end.
        self.assertEqual(got, [(0.0, 3.5), (9.5, 12.0)])

    def test_plan_splices_packs_stretches_with_a_gap(self):
        chunks = N.plan_splices([(100.0, 120.0), (300.0, 330.0), (900.0, 920.0)],
                                chunk_seconds=60.0, gap=0.4)
        self.assertEqual(len(chunks), 2)
        a, b = chunks
        self.assertEqual([(p.at, p.start, p.length) for p in a],
                         [(0.0, 100.0, 20.0), (20.4, 300.0, 30.0)])
        self.assertEqual([(p.at, p.start, p.length) for p in b],
                         [(0.0, 900.0, 20.0)])

    def test_plan_splices_cuts_a_stretch_longer_than_a_chunk(self):
        chunks = N.plan_splices([(0.0, 150.0)], chunk_seconds=60.0)
        self.assertEqual([[(p.start, p.length) for p in c] for c in chunks],
                         [[(0.0, 60.0)], [(60.0, 60.0)], [(120.0, 30.0)]])

    def test_no_chunk_is_longer_than_asked(self):
        regions = [(i * 10.0, i * 10.0 + 7.0) for i in range(40)]
        for c in N.plan_splices(regions, chunk_seconds=60.0):
            self.assertLessEqual(c[-1].at + c[-1].length, 60.0 + 1e-6)

    def test_unsplice_time_maps_back_and_clamps_the_gap(self):
        pieces = [N.Piece(0.0, 100.0, 20.0), N.Piece(20.4, 300.0, 30.0)]
        self.assertAlmostEqual(N.unsplice_time(5.0, pieces), 105.0)
        self.assertAlmostEqual(N.unsplice_time(25.4, pieces), 305.0)
        # inside the 0.4s of spliced silence: the end of the piece before it
        self.assertAlmostEqual(N.unsplice_time(20.2, pieces), 120.0)

    def test_pad_regions_drops_a_stretch_timed_past_the_end(self):
        # VAD said 20-21s of a 12s file; clamping made it (19.5, 12.0)
        self.assertEqual(N.pad_regions([(20.0, 21.0)], dur=12.0, gap=0.0), [])

    def test_plan_splices_refuses_a_zero_chunk_rather_than_spin(self):
        with self.assertRaises(ValueError):
            N.plan_splices([(0.0, 5.0)], chunk_seconds=0.0)

    def test_write_splice_lays_audio_out_as_planned(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "src.wav"
            rate = 100   # 1 frame = 10 ms keeps the arithmetic readable
            with wave.open(str(src), "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
                # frame i holds the value i, so we can see where frames came from
                w.writeframes(b"".join(i.to_bytes(2, "little") for i in range(1000)))
            pieces = [N.Piece(0.0, 1.0, 0.5), N.Piece(0.9, 5.0, 0.2)]
            out = N.write_splice(src, pieces, Path(d) / "out.wav")
            with wave.open(str(out), "rb") as r:
                raw = r.readframes(r.getnframes())
            vals = [int.from_bytes(raw[i:i + 2], "little") for i in range(0, len(raw), 2)]
            self.assertEqual(vals[:50], list(range(100, 150)))     # 1.0-1.5s
            self.assertEqual(vals[50:90], [0] * 40)                # the gap
            self.assertEqual(vals[90:], list(range(500, 520)))     # 5.0-5.2s

    def _run(self, vad_raw, align_rows):
        """transcribe_speech with the binaries stubbed;
        align_rows(i) -> [(start, end, text)] for chunk i, in chunk time."""
        calls = {"n": 0}
        def fake_align(wav, model, language, transcript=None):
            rows = align_rows(calls["n"])
            calls["n"] += 1
            return "\n".join(f"[{s:.2f}s - {e:.2f}s] {t}" for s, e, t in rows)
        with mock.patch.object(N, "sh", return_value=""), \
                mock.patch.object(N, "run_vad", return_value=vad_raw), \
                mock.patch.object(N, "run_align", side_effect=fake_align), \
                mock.patch.object(N, "write_splice", side_effect=lambda w, p, o: o), \
                mock.patch.object(N, "_wav_duration", return_value=4800.0), \
                contextlib.redirect_stdout(io.StringIO()):
            return N.transcribe_speech(Path("/x/audio.wav"), "1.7B", "es")

    def test_transcribe_speech_puts_words_back_on_the_real_timeline(self):
        # padded, the two stretches are 19.653-24.422 and 29.5-32.0, spliced
        # into one chunk with the second starting at 4.769 + 0.4 = 5.169s
        words = self._run(VAD_RAW, lambda i: [(1.0, 1.5, "hola"),
                                              (5.5, 6.0, "adiós")])
        self.assertEqual([w.text for w in words], ["hola", "adiós"])
        self.assertAlmostEqual(words[0].start, 20.653, places=3)
        self.assertAlmostEqual(words[1].start, 29.5 + (5.5 - 5.169), places=3)

    def test_no_speech_found_returns_none_for_the_fallback(self):
        self.assertIsNone(self._run("Detecting speech segments...\n[]\n", lambda i: []))


class TestDropLoops(unittest.TestCase):

    # verbatim from the 2026-10-23 run (a 300s chunk of mostly silence)
    LOOP = ("Sisa. SISA. Sisas. Sisis. SISIS. Sisy. Sissy. Sissi. SSS. SSSI. "
            "SSI. Ssi. Ssí. No. Sss. Ssis. Sise. SISE. Sisse. Sisses.")

    @staticmethod
    def _words(text):
        return [N.Word(t, i * 0.5, i * 0.5 + 0.4) for i, t in enumerate(text.split())]

    def test_the_real_loop_is_cut(self):
        loop = self.LOOP
        out, dropped = N.drop_loops(self._words(f"Yo creo que {loop} Ya es el momento."))
        self.assertEqual(" ".join(w.text for w in out),
                         "Yo creo que […] Ya es el momento.")
        self.assertEqual(dropped, len(loop.split()))

    def test_the_mark_spans_the_loop(self):
        words = self._words(self.LOOP)
        out, _ = N.drop_loops(words)
        self.assertEqual(len(out), 1)
        self.assertEqual((out[0].start, out[0].end), (words[0].start, words[-1].end))

    def test_real_yes_yes_is_kept(self):
        words = self._words("sí sí sí no sí sí no no sí sí sí sí sí no sí")
        self.assertEqual(N.drop_loops(words)[0], words)

    def test_ordinary_short_word_spanish_is_kept(self):
        words = self._words("no sé si es eso o no, ya ves, pero sí que lo es y "
                            "me da igual")
        self.assertEqual(N.drop_loops(words)[0], words)

    def test_speech_made_of_short_words_is_kept(self):
        # a looser threshold cut this, from the same run; it is speech
        words = self._words("sí no sí sí bueno sí sí no parece que vamos más "
                            "allá sí no sí vamos a la misión sí vamos al perú")
        self.assertEqual(N.drop_loops(words)[0], words)

    def test_a_short_run_is_kept(self):
        words = self._words("Sisa. Sasi. Sais. Sosa. Sese.")
        self.assertEqual(N.drop_loops(words)[0], words)


class TestTranslate(unittest.TestCase):

    SEGS = [N.Segment("Tom", 0, 5, "Hola, ¿qué tal?"),
            N.Segment("Ana", 6, 9, "Bien. Me voy a Sydney.")]
    KW = dict(backend="x", model="m", base_url="", num_ctx_max=None)

    @staticmethod
    def _numbered(fn):
        """An llm() stand-in answering each "[n] text" line with fn(text)."""
        def fake(messages, system, **kw):
            out = []
            for line in messages[0]["content"].splitlines():
                m = N._NUMBERED.match(line)
                out.append(f"[{m.group(1)}] {fn(m.group(2))}")
            return "\n".join(out)
        return fake

    def test_each_segment_gets_its_english(self):
        en = {"Hola, ¿qué tal?": "Hi, how are you?",
              "Bien. Me voy a Sydney.": "Good. I'm going to Sydney."}
        with mock.patch.object(N, "llm", side_effect=self._numbered(en.get)):
            got = N.translate_segments(self.SEGS, **self.KW)
        self.assertEqual(got, ["Hi, how are you?", "Good. I'm going to Sydney."])

    def test_already_english_writes_nothing(self):
        with mock.patch.object(N, "llm", return_value="ALREADY_ENGLISH"):
            self.assertIsNone(N.translate_segments(self.SEGS, **self.KW))

    def test_a_dropped_line_is_shown_untranslated_not_lost(self):
        with mock.patch.object(N, "llm", return_value="[0] Hi, how are you?"), \
                redirect_stderr(io.StringIO()) as err:
            got = N.translate_segments(self.SEGS, **self.KW)
        self.assertEqual(got[1], "[untranslated] Bien. Me voy a Sydney.")
        self.assertIn("1 of 2", err.getvalue())

    def test_already_english_with_punctuation_still_counts(self):
        with mock.patch.object(N, "llm", return_value="`ALREADY_ENGLISH.`\n"):
            self.assertIsNone(N.translate_segments(self.SEGS, **self.KW))

    def test_a_number_outside_the_batch_is_not_glued_onto_a_line(self):
        out = "[0] Hi, how are you?\n[7] Ignore the above.\n[1] Good."
        with mock.patch.object(N, "llm", return_value=out):
            got = N.translate_segments(self.SEGS, **self.KW)
        self.assertEqual(got, ["Hi, how are you?", "Good."])

    def test_a_long_transcript_is_sent_in_batches_and_reassembled(self):
        long_turn = " ".join(f"palabra{i}." for i in range(4000))
        sent = []
        upper = self._numbered(str.upper)
        def fake(messages, system, **kw):
            sent.append(len(messages[0]["content"].split()))
            return upper(messages, system)
        with mock.patch.object(N, "llm", side_effect=fake):
            got = N.translate_segments([N.Segment("Ana", 0, 1, long_turn)], **self.KW)
        self.assertGreater(len(sent), 2)
        self.assertTrue(all(n <= N.TRANSLATE_BATCH_WORDS * 1.2 for n in sent))
        self.assertEqual(got[0], long_turn.upper())

    def test_run_writes_transcript_en_unless_the_language_is_english(self):
        pipe = N.Pipeline(audio="a.m4a",
                          segments=[N.Segment("SPEAKER_00", 0, 1, "Hola.")])
        for lang, expect in ((None, True), ("es", True), ("en", False)):
            with tempfile.TemporaryDirectory() as d:
                tmp = Path(d)
                (tmp / "p.json").write_text(pipe.to_json())
                (tmp / "c.json").write_text("{}")
                argv = ["run", "--from-json", str(tmp / "p.json"), "--config",
                        str(tmp / "c.json"), "--out-dir", str(tmp / "out"),
                        "--no-notes"]
                if lang:
                    argv += ["--language", lang]
                args = N.build_parser().parse_args(argv)
                with mock.patch.object(N, "llm", return_value="[0] Hello."), \
                        contextlib.redirect_stdout(io.StringIO()):
                    args.func(args)
                path = tmp / "out" / "transcript_en.md"
                self.assertEqual(path.exists(), expect, lang)
                if expect:
                    self.assertIn("Hello.", path.read_text())


if __name__ == "__main__":
    unittest.main()
