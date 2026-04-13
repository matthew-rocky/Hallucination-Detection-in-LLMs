import unittest
from unittest.mock import patch

from methods.sep_check import run_sep


class SepFallbackTests(unittest.TestCase):
    @patch(
        "detectors.signal._extract_features",
        side_effect=RuntimeError("transformers import failed: missing local weights"),
    )
    def test_backend_error(self, _mock_extract):
        result = run_sep(
            question="What is the capital of Japan?",
            answer="Tokyo",
            sampled_answers_text="Tokyo\n\n---\n\nTokyo",
        )

        backend_error = result["metadata"]["backend_error"]
        self.assertIn("backend offline", backend_error)
        self.assertIn("transformers import failed: missing local weights", backend_error)

    @patch("detectors.signal._extract_features", side_effect=RuntimeError("backend offline"))
    def test_samples_low(self, _mock_extract):
        cases = [
            (
                "What is the capital of Japan?",
                "Tokyo",
                "Tokyo\n\n---\n\nTokyo\n\n---\n\nTokyo",
            ),
            (
                "What is the capital of Japan?",
                "The capital of Japan is Tokyo.",
                "Tokyo is the capital of Japan.\n\n---\n\nJapan's capital city is Tokyo.\n\n---\n\nThe capital of Japan is Tokyo.",
            ),
            (
                "Who is the author of Pride and Prejudice?",
                "Jane Austen wrote Pride and Prejudice.",
                "Jane Austen\n\n---\n\nAusten\n\n---\n\nThe author was Jane Austen.",
            ),
        ]
        for question, answer, samples in cases:
            with self.subTest(question=question, answer=answer):
                result = run_sep(
                    question=question,
                    answer=answer,
                    sampled_answers_text=samples,
                )
                self.assertEqual(result["risk_label"], "Low")
                self.assertLessEqual(result["score"], 0.33)

    @patch("detectors.signal._extract_features", side_effect=RuntimeError("backend offline"))
    def test_year_wins(self, _mock_extract):
        consistent = run_sep(
            question="When did the pilot begin?",
            answer="2021",
            sampled_answers_text="in 2021\n\n---\n\nMarch 2021\n\n---\n\n2021",
        )
        drift = run_sep(
            question="When did the pilot begin?",
            answer="2021",
            sampled_answers_text="2022\n\n---\n\n2023\n\n---\n\nin 2022",
        )

        self.assertEqual(consistent["risk_label"], "Low")
        self.assertLessEqual(consistent["score"], 0.33)
        self.assertEqual(drift["risk_label"], "High")
        self.assertGreaterEqual(drift["score"], 0.67)
        self.assertIn("dates or years", drift["explanation"])


if __name__ == "__main__":
    unittest.main()
