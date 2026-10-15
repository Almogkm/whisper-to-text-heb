"""Unit tests for the speaker alignment and LLM guard logic in longform.py.

Run: python -m unittest discover -s tests
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whisper_md_transcriber import longform as lf  # noqa: E402


def word(text: str, start: float, end: float, probability: float = 0.9) -> dict:
    return {"word": f" {text}", "start": start, "end": end, "probability": probability}


def turn(speaker: str, start: float, end: float) -> dict:
    return {"speaker": speaker, "start": start, "end": end}


class AssignSpeakersTest(unittest.TestCase):
    def test_word_takes_speaker_with_most_overlap(self):
        diarization = [turn("A", 0.0, 1.0), turn("B", 1.0, 3.0)]
        words = [word("x", 0.8, 1.6)]
        self.assertEqual(lf.assign_speakers(words, diarization), ["B"])

    def test_word_in_gap_goes_to_nearest_turn(self):
        diarization = [turn("A", 0.0, 1.0), turn("B", 3.0, 4.0)]
        words = [word("x", 1.1, 1.3), word("y", 2.7, 2.9)]
        self.assertEqual(lf.assign_speakers(words, diarization), ["A", "B"])

    def test_slivers_are_ignored(self):
        diarization = [turn("A", 0.0, 2.0), turn("B", 2.0, 2.02), turn("A", 2.02, 4.0)]
        words = [word("x", 1.99, 2.03)]
        self.assertEqual(lf.assign_speakers(words, diarization), ["A"])

    def test_only_slivers_still_used(self):
        diarization = [turn("B", 1.0, 1.05)]
        self.assertEqual(lf.assign_speakers([word("x", 1.0, 1.2)], diarization), ["B"])

    def test_no_diarization_defaults_to_one_speaker(self):
        self.assertEqual(lf.assign_speakers([word("x", 0, 1), word("y", 1, 2)], []), ["SPEAKER_00"] * 2)

    def test_long_turn_before_word_is_found(self):
        # The turn covering the word starts long before it; the bisect window must still reach it.
        diarization = [turn("A", 0.0, 100.0), turn("B", 50.0, 51.0)]
        self.assertEqual(lf.assign_speakers([word("x", 90.0, 90.5)], diarization), ["A"])


class SmoothSpeakersTest(unittest.TestCase):
    def test_short_sandwiched_flip_is_removed(self):
        words = [word("a", 0, 0.3), word("b", 0.3, 0.5), word("c", 0.5, 0.8)]
        self.assertEqual(lf.smooth_speakers(words, ["A", "B", "A"]), ["A", "A", "A"])

    def test_long_sandwiched_word_is_kept(self):
        words = [word("a", 0, 0.3), word("b", 0.3, 1.3), word("c", 1.3, 1.6)]
        self.assertEqual(lf.smooth_speakers(words, ["A", "B", "A"]), ["A", "B", "A"])

    def test_edges_are_untouched(self):
        words = [word("a", 0, 0.1), word("b", 0.1, 0.2), word("c", 0.2, 0.3)]
        self.assertEqual(lf.smooth_speakers(words, ["B", "A", "A"]), ["B", "A", "A"])


class BuildTurnsTest(unittest.TestCase):
    def test_merges_same_speaker_and_numbers_by_first_appearance(self):
        words = [word("שלום", 0, 0.5), word("לך", 0.6, 1.0), word("היי", 1.2, 1.5)]
        turns = lf.build_turns(words, ["SPEAKER_01", "SPEAKER_01", "SPEAKER_00"])
        self.assertEqual([t["raw"] for t in turns], ["שלום לך", "היי"])
        self.assertEqual([t["speaker_number"] for t in turns], [1, 2])
        self.assertEqual([t["id"] for t in turns], [1, 2])
        self.assertEqual((turns[0]["start"], turns[0]["end"]), (0, 1.0))
        self.assertEqual(turns[0]["words"], ["שלום", "לך"])

    def test_long_pause_splits_turn(self):
        words = [word("a", 0, 0.5), word("b", 0.5 + lf.MAX_TURN_GAP_SECONDS + 0.1, 3.0)]
        self.assertEqual(len(lf.build_turns(words, ["A", "A"])), 2)

    def test_char_limit_splits_turn(self):
        long_text = "x" * (lf.MAX_TURN_CHARS - 1)
        words = [word(long_text, 0, 1), word("yy", 1, 1.2)]
        self.assertEqual(len(lf.build_turns(words, ["A", "A"])), 2)


class AttachAlternativeTest(unittest.TestCase):
    def test_words_go_to_turn_by_midpoint(self):
        turns = [{"start": 0.0, "end": 1.0}, {"start": 2.0, "end": 3.0}]
        alt = [word("a", 0.1, 0.4), word("b", 1.5, 1.9), word("c", 1.9, 2.3), word("d", 2.5, 2.9)]
        lf.attach_alternative(turns, alt, "alt")
        # "b" ends before the next turn starts, so it stays with the first turn.
        self.assertEqual([t["alt"] for t in turns], ["a b", "c d"])

    def test_turn_without_words_gets_empty_string(self):
        turns = [{"start": 0.0, "end": 1.0}, {"start": 2.0, "end": 3.0}]
        lf.attach_alternative(turns, [word("a", 2.1, 2.4)], "alt")
        self.assertEqual([t["alt"] for t in turns], ["", "a"])


class WordChangeRatioTest(unittest.TestCase):
    def test_identical_is_zero(self):
        self.assertEqual(lf.word_change_ratio("a b c", "a b c"), 0.0)

    def test_both_empty_is_zero(self):
        self.assertEqual(lf.word_change_ratio(" ", ""), 0.0)

    def test_disjoint_is_one(self):
        self.assertEqual(lf.word_change_ratio("a b", "c d"), 1.0)

    def test_punctuation_is_ignored(self):
        self.assertEqual(lf.word_change_ratio("ונשארתי שם שנה. הילד, בא", "ונשארתי שם שנה הילד בא"), 0.0)

    def test_bidi_controls_are_ignored(self):
        self.assertEqual(lf.word_change_ratio("‫ובאנו לארץ.‬", "ובאנו לארץ"), 0.0)

    def test_punctuation_only_tokens_are_dropped(self):
        self.assertEqual(lf.word_change_ratio("טוב - טוב", "טוב טוב"), 0.0)

    def test_real_word_change_still_counts(self):
        self.assertGreater(lf.word_change_ratio("הילד היה בן שנה.", "הילד הימן שנה"), 0.0)


class StripBidiTest(unittest.TestCase):
    def test_removes_direction_controls(self):
        self.assertEqual(lf.strip_bidi("‫שלום‬ ‏עולם‎"), "שלום עולם")

    def test_keeps_plain_text(self):
        self.assertEqual(lf.strip_bidi('צה"ל, 85 שנים.'), 'צה"ל, 85 שנים.')


class BlocksOfTest(unittest.TestCase):
    def test_packs_until_limit(self):
        turns = [{"raw": "x" * 4}, {"raw": "x" * 4}, {"raw": "x" * 4}]
        blocks = lf.blocks_of(turns, "raw", 8)
        self.assertEqual([len(b) for b in blocks], [2, 1])

    def test_oversized_turn_gets_own_block(self):
        turns = [{"raw": "x" * 20}, {"raw": "x"}]
        self.assertEqual([len(b) for b in lf.blocks_of(turns, "raw", 8)], [1, 1])


class ParseMarkedLinesTest(unittest.TestCase):
    def test_markers_with_labels_and_continuations(self):
        text = "פתיח\n<<1 | דובר 1>> שלום\nהמשך\n\n<< 2 >>היי"
        self.assertEqual(lf.parse_marked_lines(text), {1: "שלום המשך", 2: "היי"})

    def test_text_before_first_marker_is_dropped(self):
        self.assertEqual(lf.parse_marked_lines("שלום"), {})


class PlausibleRewriteTest(unittest.TestCase):
    source = "זה משפט ארוך מספיק כדי לבדוק את השומרים של השלב"

    def test_small_edit_passes(self):
        self.assertTrue(lf.plausible_rewrite(self.source, self.source.replace("ארוך", "ארוך מאוד")))

    def test_empty_fails(self):
        self.assertFalse(lf.plausible_rewrite(self.source, "  "))

    def test_too_short_fails(self):
        self.assertFalse(lf.plausible_rewrite(self.source, "זה משפט"))

    def test_too_long_fails(self):
        self.assertFalse(lf.plausible_rewrite(self.source, self.source + " " + self.source))

    def test_different_turn_fails(self):
        other = "מילים אחרות לגמרי שאין להן קשר לטקסט המקורי בכלל"
        self.assertFalse(lf.plausible_rewrite(self.source, other))

    def test_short_source_allows_small_growth_only(self):
        self.assertTrue(lf.plausible_rewrite("כן", "כן, בטח"))
        self.assertFalse(lf.plausible_rewrite("כן", "כן " + "x" * 30))


class ComputeFlagsTest(unittest.TestCase):
    def base_turn(self, **extra) -> dict:
        return {"words": ["a", "b"], "probabilities": [0.9, 0.9], "raw": "a b", **extra}

    def test_clean_turn_has_no_flags(self):
        self.assertEqual(lf.compute_flags(self.base_turn(alt="a b", cleaned="a b", final="a b")), [])

    def test_low_word_is_named(self):
        flags = lf.compute_flags(self.base_turn(probabilities=[0.9, 0.1]))
        self.assertEqual(len(flags), 1)
        self.assertIn("b", flags[0])

    def test_asr_disagreement(self):
        self.assertEqual(len(lf.compute_flags(self.base_turn(alt="c d"))), 1)

    def test_punctuation_and_bidi_are_not_disagreement(self):
        turn = self.base_turn(raw="‫a, b.", alt="a b")
        self.assertEqual(lf.compute_flags(turn), [])

    def test_failed_stage(self):
        self.assertEqual(len(lf.compute_flags(self.base_turn(review_status="kept-source"))), 1)


class LoadGlossaryTest(unittest.TestCase):
    def test_none_is_empty(self):
        self.assertIs(lf.load_glossary(None), lf.EMPTY_GLOSSARY)

    def test_reads_questionnaire_export(self):
        payload = {
            "recording": {"topic": "שיחה", "setting": "", "background": "רקע"},
            "speakers": [{"name": "אלמוג", "gender": "male"}, {"name": "  "}],
            "people": [{"term": "מיכאל", "note": "חבר"}, {"term": ""}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "glossary.json"
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            glossary = lf.load_glossary(path)
        self.assertEqual(glossary.context, "שיחה רקע")
        self.assertEqual([s.name for s in glossary.speakers], ["אלמוג"])
        self.assertEqual([e.term for e in glossary.entries], ["אלמוג", "מיכאל"])


if __name__ == "__main__":
    unittest.main()
