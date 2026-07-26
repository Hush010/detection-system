import unittest

from detector import analyze_text


class DetectorTests(unittest.TestCase):
    def test_human_like_text_has_lower_score(self):
        text = (
            "The experiment was carried out over three days in the university laboratory. "
            "We recorded the temperature at regular intervals and compared the results with "
            "the control group. The observations were consistent with the original hypothesis."
        )
        result = analyze_text(text)
        self.assertTrue(0 <= result["score"] <= 100)
        self.assertIn(result["label"], {"Low risk", "Medium risk", "High risk"})
        self.assertLess(result["score"], 70)

    def test_ai_like_text_has_higher_score(self):
        text = (
            "In conclusion, this essay comprehensively explores the multifaceted implications "
            "of modern technological advancement and offers a nuanced analysis of the "
            "complex interplay between innovation and societal transformation."
        )
        result = analyze_text(text)
        self.assertGreater(result["score"], 50)
        self.assertIn(result["label"], {"Low risk", "Medium risk", "High risk"})

    def test_gpt_like_submission_has_higher_score(self):
        text = (
            "Modern education systems are increasingly shaped by digital technologies, "
            "and this shift has created both opportunities and challenges for learners and teachers. "
            "The growing reliance on online platforms has changed how knowledge is accessed and "
            "how academic performance is assessed. As a result, institutions must develop thoughtful "
            "strategies to ensure equitable and effective learning experiences."
        )
        result = analyze_text(text)
        self.assertGreater(result["score"], 50)
        self.assertIn(result["label"], {"Low risk", "Medium risk", "High risk"})

    def test_engine_is_present(self):
        text = "A simple sentence about a classroom assignment."
        result = analyze_text(text)
        self.assertIn(result["details"].get("engine"), {"heuristic", "transformer"})


if __name__ == "__main__":
    unittest.main()
