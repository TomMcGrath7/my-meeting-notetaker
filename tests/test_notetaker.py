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
"""

import io
import json
import unittest
from contextlib import redirect_stderr
from pathlib import Path

import notetaker as N

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


if __name__ == "__main__":
    unittest.main()
