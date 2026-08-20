"""Detector tests.

Three of the fixtures in the previous version of this file were copied verbatim
out of `dataset.json`, so the suite was asserting that the model could recall
its own training data. The keyword list in the heuristic had also been grown to
contain phrases from those fixtures, which turned the tests green while making
the heuristic penalise any essay about education or technology.

Every fixture below is written fresh and held out. `test_no_fixture_appears_in_
training_data` enforces that mechanically, so the leak cannot come back by
accident.
"""

import json
import unittest
from pathlib import Path

from detector import (
    ALL_LABELS,
    LABEL_HIGH,
    LABEL_INCONCLUSIVE,
    LABEL_LOW,
    LABEL_REVIEW,
    analyze_text,
    min_words,
    normalize_text,
)

DATASET_PATH = Path(__file__).resolve().parent.parent / "dataset.json"


# Held-out fixtures. None of these appear in dataset.json.
HUMAN_LAB_NOTES = (
    "So I ran the whole thing again on Tuesday afternoon and it fell over at "
    "the same point, which was annoying because I had already redone the "
    "calibration twice. I think the sensor cable is loose, because when I "
    "wiggled it the readings jumped by about four degrees and then settled "
    "again. Sam reckons it is the connector rather than the cable itself. "
    "Either way I am not redoing the full set until someone has looked at it, "
    "there is no point burning another afternoon on bad numbers."
)

HUMAN_ESSAY = (
    "Wordsworth returns again and again to the image of the river, and I think "
    "this is because he cannot settle the question of whether memory restores "
    "us or merely reminds us of what we have lost. In the 1805 Prelude the "
    "river carries him forward, and the movement feels like growth. By 1850 "
    "the same water seems to run backwards, and the poet who once trusted "
    "recollection now sounds like a man checking his pockets for something he "
    "is fairly sure he has dropped. The revision is small but it changes the "
    "whole argument of the poem."
)

HUMAN_NON_NATIVE = (
    "In my country the education system is very different from here. The "
    "teacher give many homework every day and the student must memorize all "
    "of it for the exam at the end of term. Nobody ask you what you think "
    "about the subject. When I came here I was very surprised that we can "
    "discuss with the professor and even disagree with him in the seminar. "
    "This helped me to think more by myself, and I believe my writing also "
    "improved because of it, although I still make many small mistakes."
)

GENERATED_ESSAY = (
    "Artificial intelligence has fundamentally transformed the landscape of "
    "contemporary education. By leveraging sophisticated algorithms and "
    "data-driven insights, educational institutions can now deliver "
    "personalised learning experiences at unprecedented scale. However, this "
    "technological advancement also raises important ethical considerations "
    "that stakeholders must carefully navigate. It is essential to strike a "
    "balance between innovation and responsibility, ensuring that the "
    "integration of these tools ultimately serves the best interests of "
    "learners across diverse contexts."
)

SHORT_TEXT = "The meeting is at four in the library."


class LeakageTests(unittest.TestCase):
    """The guard that stops this suite from grading the model on its homework."""

    def test_no_fixture_appears_in_training_data(self):
        with DATASET_PATH.open("r", encoding="utf-8") as fh:
            corpus = [normalize_text(item["text"]) for item in json.load(fh)]

        fixtures = {
            "HUMAN_LAB_NOTES": HUMAN_LAB_NOTES,
            "HUMAN_ESSAY": HUMAN_ESSAY,
            "HUMAN_NON_NATIVE": HUMAN_NON_NATIVE,
            "GENERATED_ESSAY": GENERATED_ESSAY,
            "SHORT_TEXT": SHORT_TEXT,
        }

        for name, text in fixtures.items():
            needle = normalize_text(text)
            for sample in corpus:
                overlap = needle[:60] in sample or sample[:60] in needle
                self.assertFalse(
                    overlap,
                    f"{name} overlaps a training sample. Tests must use held-out "
                    f"text, otherwise they only measure memorisation.",
                )


class AbstentionTests(unittest.TestCase):
    """The detector must decline to judge what it cannot judge."""

    def test_two_letter_input_is_inconclusive_not_high_risk(self):
        # Regression test. The previous scoring returned "High risk" at 65.2
        # for this input, on probabilities that were a three-way coin flip.
        result = analyze_text("ok")
        self.assertEqual(result["label"], LABEL_INCONCLUSIVE)
        self.assertIsNone(result["score"])
        self.assertTrue(result["details"]["abstained"])

    def test_short_sentence_is_inconclusive(self):
        result = analyze_text(SHORT_TEXT)
        self.assertEqual(result["label"], LABEL_INCONCLUSIVE)
        self.assertIsNone(result["score"])
        self.assertIn("insufficient text", result["details"]["reason"])

    def test_empty_text_is_inconclusive(self):
        for value in ("", "   \n\t ", None):
            with self.subTest(value=repr(value)):
                result = analyze_text(value)
                self.assertEqual(result["label"], LABEL_INCONCLUSIVE)
                self.assertIsNone(result["score"])

    def test_abstention_reports_the_requirement(self):
        result = analyze_text(SHORT_TEXT)
        self.assertEqual(result["details"]["required_words"], min_words())
        self.assertGreater(result["details"]["word_count"], 0)


class ScoringContractTests(unittest.TestCase):
    """Score and label are one rule, so they cannot contradict each other."""

    def _all_results(self):
        return [
            analyze_text(t)
            for t in (
                HUMAN_LAB_NOTES,
                HUMAN_ESSAY,
                HUMAN_NON_NATIVE,
                GENERATED_ESSAY,
                SHORT_TEXT,
            )
        ]

    def test_labels_are_from_the_known_vocabulary(self):
        for result in self._all_results():
            self.assertIn(result["label"], ALL_LABELS)

    def test_score_is_none_exactly_when_inconclusive(self):
        for result in self._all_results():
            is_inconclusive = result["label"] == LABEL_INCONCLUSIVE
            self.assertEqual(
                result["score"] is None,
                is_inconclusive,
                f"score/label disagree: {result['label']} with {result['score']}",
            )

    def test_label_is_a_pure_function_of_score(self):
        """Two inputs with the same score must never get different labels."""
        by_score = {}
        for result in self._all_results():
            if result["score"] is None:
                continue
            if not result["details"]["calibrated"]:
                continue
            by_score.setdefault(result["score"], set()).add(result["label"])
        for score, labels in by_score.items():
            self.assertEqual(len(labels), 1, f"score {score} produced {labels}")

    def test_scores_stay_in_range(self):
        for result in self._all_results():
            if result["score"] is not None:
                self.assertGreaterEqual(result["score"], 0)
                self.assertLessEqual(result["score"], 100)

    def test_engine_is_reported(self):
        for result in self._all_results():
            self.assertIn(
                result["details"].get("engine"),
                {"heuristic", "transformer", "trained_model", "none"},
            )

    def test_uncalibrated_engines_never_reach_high_risk(self):
        """An advisory keyword match must not be able to accuse anyone."""
        from detector import _analyze_text_heuristic

        stacked = " ".join(
            [
                "In conclusion, it is evident that the multifaceted and nuanced "
                "analysis delve into the complex interplay furthermore moreover "
                "a testament to the plays a crucial role in today's rapidly."
            ]
            * 4
        )
        result = _analyze_text_heuristic(stacked)
        self.assertFalse(result["details"]["calibrated"])
        self.assertTrue(result["details"]["advisory"])
        self.assertNotEqual(result["label"], LABEL_HIGH)


class SeparationTests(unittest.TestCase):
    """The behaviour the whole system exists for, on held-out text."""

    def test_human_writing_is_not_flagged_high_risk(self):
        for name, text in (
            ("lab notes", HUMAN_LAB_NOTES),
            ("literary essay", HUMAN_ESSAY),
            ("non-native speaker", HUMAN_NON_NATIVE),
        ):
            with self.subTest(sample=name):
                result = analyze_text(text)
                self.assertNotEqual(
                    result["label"],
                    LABEL_HIGH,
                    f"{name} was flagged High risk at {result['score']}",
                )

    def test_human_writing_clears_the_review_band(self):
        """Genuine human writing should come back clean, not merely 'not high'.

        This is the test that would have caught the original bug: every human
        sample used to score 44-61 because the score had no floor.
        """
        for name, text in (
            ("lab notes", HUMAN_LAB_NOTES),
            ("literary essay", HUMAN_ESSAY),
        ):
            with self.subTest(sample=name):
                result = analyze_text(text)
                self.assertEqual(
                    result["label"],
                    LABEL_LOW,
                    f"{name} scored {result['score']} - human writing should "
                    f"come back Low risk, not {result['label']}",
                )

    def test_generated_text_is_escalated(self):
        result = analyze_text(GENERATED_ESSAY)
        self.assertIn(result["label"], {LABEL_REVIEW, LABEL_HIGH})
        self.assertIsNotNone(result["score"])

    def test_generated_text_outscores_human_text(self):
        generated = analyze_text(GENERATED_ESSAY)["score"]
        for name, text in (
            ("lab notes", HUMAN_LAB_NOTES),
            ("literary essay", HUMAN_ESSAY),
            ("non-native speaker", HUMAN_NON_NATIVE),
        ):
            with self.subTest(sample=name):
                self.assertGreater(generated, analyze_text(text)["score"])


if __name__ == "__main__":
    unittest.main()
