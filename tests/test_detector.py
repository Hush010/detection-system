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
        self.assertIn(result["details"].get("engine"), {"heuristic", "transformer", "trained_model"})

    def test_hybrid_text_detected_as_ai(self):
        """Test that AI text edited to look human is still flagged as AI."""
        text = (
            "Technology keeps changing how we work and live. There are both good and bad effects. "
            "Some people benefit more than others. We need policies to help everyone adapt to this change."
        )
        result = analyze_text(text)
        # Hybrid should be detected as high risk (AI)
        self.assertEqual(result["label"], "High risk")
        self.assertGreater(result["score"], 50)
        # Verify the prediction identifies it as ai or hybrid
        if "prediction" in result["details"]:
            self.assertIn(result["details"]["prediction"], {"ai", "hybrid"})

    def test_hybrid_vs_pure_ai(self):
        """Test that hybrid text (edited AI) has different characteristics than pure AI."""
        pure_ai = (
            "The multifaceted implications of contemporary technological advancement necessitate "
            "a comprehensive examination of stakeholder perspectives and regulatory frameworks."
        )
        hybrid = (
            "Digital transformation impacts every aspect of business today. Companies face both "
            "opportunities and risks. Success depends on careful planning and execution."
        )
        
        pure_result = analyze_text(pure_ai)
        hybrid_result = analyze_text(hybrid)
        
        # Both should be flagged as high risk
        self.assertEqual(pure_result["label"], "High risk")
        self.assertEqual(hybrid_result["label"], "High risk")
        
        # Hybrid might have slightly lower score but still risky
        self.assertGreater(pure_result["score"], 40)
        self.assertGreater(hybrid_result["score"], 40)


if __name__ == "__main__":
    unittest.main()
