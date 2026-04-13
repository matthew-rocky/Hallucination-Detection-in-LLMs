import unittest
from unittest.mock import patch

from detectors.signal import run_signal_detector
from methods.internal_check import run_internal


LOW_RISK_RECORD = {
    "features": {
        "mean_negative_log_prob": 0.72,
        "token_log_prob_std": 0.11,
        "entropy_mean": 1.15,
        "entropy_std": 0.18,
        "mean_token_probability": 0.67,
        "top2_margin_mean": 0.41,
        "num_answer_tokens": 26.0,
        "selected_layers": [5],
        "hidden_norm_mean": 8.0,
        "hidden_norm_var": 0.18,
        "hidden_drift_mean": 0.03,
        "layer_centroid_dispersion": 0.02,
    },
    "tokens": ["tok"],
    "answer_token_count": 26,
}

READY_STATUS = {
    "python_executable": "C:/Python/python.exe",
    "torch_installed": True,
    "torch_version": "2.5.1",
    "transformers_installed": True,
    "transformers_version": "4.50.2",
    "device": "cpu",
    "backend_available": True,
    "backend_status": "available",
    "backend_status_label": "HF backend active",
    "backend_error": None,
    "backend_model_name": "distilgpt2",
    "local_files_only": False,
}


class SignalFallbackTests(unittest.TestCase):
    @patch("detectors.signal._extract_features", side_effect=RuntimeError("backend offline"))
    def test_short_facts(self, _mock_extract):
        cases = [
            ("What is the capital of Japan?", "Tokyo"),
            ("Who wrote Hamlet?", "William Shakespeare."),
            ("What is the chemical symbol for water?", "H2O."),
        ]
        for question, answer in cases:
            with self.subTest(question=question, answer=answer):
                result = run_internal(question=question, answer=answer)
                self.assertEqual(result["risk_label"], "Low")
                self.assertLessEqual(result["score"], 0.33)

    @patch("detectors.signal._extract_features", side_effect=RuntimeError("backend offline"))
    def test_fact_paraphrases(self, _mock_extract):
        cases = [
            ("What is the capital of Canada?", "Canada's capital city is Ottawa."),
            ("Who wrote Pride and Prejudice?", "The author of Pride and Prejudice was Jane Austen."),
            (
                "Which planet is the largest in the Solar System?",
                "Jupiter is the largest planet in the Solar System.",
            ),
        ]
        for question, answer in cases:
            with self.subTest(question=question, answer=answer):
                result = run_internal(question=question, answer=answer)
                self.assertEqual(result["risk_label"], "Low")
                self.assertLessEqual(result["score"], 0.33)

    @patch("detectors.signal._extract_features", side_effect=RuntimeError("backend offline"))
    def test_wrong_dates(self, _mock_extract):
        cases = [
            {
                "question": "When did Apollo 11 land on the Moon?",
                "answer": "Apollo 11 landed on the Moon in 1971.",
                "min_score": 0.52,
            },
            {
                "question": "When did World War II end in Europe?",
                "answer": "World War II ended in Europe in 1944.",
                "min_score": 0.52,
            },
            {
                "question": "What happened during the 2020 Narev medical summit?",
                "answer": "The 2020 Narev medical summit was signed in 2021.",
                "min_score": 0.67,
            },
        ]
        for case in cases:
            with self.subTest(question=case["question"], answer=case["answer"]):
                result = run_internal(question=case["question"], answer=case["answer"])
                self.assertGreaterEqual(result["score"], case["min_score"])
                self.assertNotEqual(result["risk_label"], "Low")

    @patch("detectors.signal._extract_features", side_effect=RuntimeError("backend offline"))
    def test_dense_conflicts(self, _mock_extract):
        cases = [
            (
                "According to the clinic memo, what was approved?",
                (
                    "The board approved 18 observation beds and no emergency department. "
                    "In the same memo it also approved 24 observation beds and a new emergency department."
                ),
            ),
            (
                "Summarize the pilot timeline.",
                (
                    "The pilot started in 2021 and ended in 2021. "
                    "It did not begin until 2022 and was still running through 2020."
                ),
            ),
        ]
        for question, answer in cases:
            with self.subTest(question=question, answer=answer):
                result = run_internal(question=question, answer=answer)
                self.assertEqual(result["risk_label"], "High")
                self.assertGreaterEqual(result["score"], 0.67)


class SignalBlendTests(unittest.TestCase):
    @patch("detectors.signal._extract_features", return_value=LOW_RISK_RECORD)
    @patch("detectors.signal.get_signal_status", return_value=READY_STATUS)
    def test_backend_contra(self, _mock_status, _mock_extract):
        question = "According to the clinic memo, what was approved?"
        answer = (
            "The board approved 18 observation beds and no emergency department. "
            "In the same memo it also approved 24 observation beds and a new emergency department."
        )

        result = run_signal_detector(
            question=question,
            answer=answer,
            method_name="Internal-Signal Baseline",
            mode="uncertainty_baseline",
        )

        self.assertEqual(result["implementation_status"], "implemented")
        self.assertEqual(result["risk_label"], "High")
        self.assertGreaterEqual(result["score"], 0.67)
        self.assertIsNotNone(result["metadata"]["baseline_backend_score"])
        self.assertIsNotNone(result["metadata"]["baseline_text_proxy_score"])


if __name__ == "__main__":
    unittest.main()
