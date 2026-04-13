import unittest
from unittest.mock import patch

from data.sample_cases import case_by_id
from methods.cove_check import run_cove
from methods.critic_check import run_critic
from methods.internal_check import run_internal
from methods.sep_check import run_sep


class SignalRuntimeTests(unittest.TestCase):
    @patch("detectors.signal._extract_features", side_effect=RuntimeError("backend offline"))
    def test_fallback_runtime(self, _mock_extract):
        baseline_case = case_by_id("internal_baseline_high_marwick")
        sep_case = case_by_id("sep_high_narev_consensus")

        baseline = run_internal(
            question=baseline_case["question"],
            answer=baseline_case["answer"],
        )
        sep_result = run_sep(
            question=sep_case["question"],
            answer=sep_case["answer"],
            sampled_answers_text=sep_case["answer_samples"],
        )

        for result in (baseline, sep_result):
            metadata = result["metadata"]
            self.assertTrue(metadata["fallback_mode"])
            self.assertFalse(metadata["backend_available"])
            self.assertEqual(metadata["backend_status"], "unavailable")
            self.assertEqual(metadata["result_origin"], "deterministic_fallback_approximation")
            self.assertEqual(metadata["result_origin_label"], "Deterministic fallback approximation")
            self.assertTrue(metadata["backend_status_label"])
            self.assertTrue(metadata["backend_model_name"])

        self.assertLessEqual(baseline["confidence"], 0.52)
        self.assertLessEqual(run_internal(
            question=case_by_id("internal_baseline_low_tokyo")["question"],
            answer=case_by_id("internal_baseline_low_tokyo")["answer"],
        )["score"], 0.33)
        self.assertTrue(sep_result["metadata"]["suspicious_consensus_flag"])

    @patch("detectors.signal._extract_features", side_effect=RuntimeError("backend offline"))
    def test_text_fact_bank(self, _mock_extract):
        cases = [
            ("What is the capital of Japan?", "Ottawa", "capital_japan"),
            ("What is Japan's capital?", "It is Ottawa.", "capital_japan"),
            ("What city is Japan's capital?", "Ottawa.", "capital_japan"),
            ("Name Canada's capital city.", "The answer is Toronto.", "capital_canada"),
            ("Which planet is the largest in the Solar System?", "The answer is Mars.", "largest_planet"),
            ("Which planet is the biggest in the Solar System?", "Mars.", "largest_planet"),
            ("Who is the author of Pride and Prejudice?", "Charles Dickens wrote Pride and Prejudice.", "pride_and_prejudice_author"),
            ("Name the author of Pride and Prejudice.", "The author was Charles Dickens.", "pride_and_prejudice_author"),
            ("Who is the painter of The Persistence of Memory?", "Pablo Picasso painted The Persistence of Memory.", "persistence_of_memory_painter"),
            ("Name the painter of The Persistence of Memory.", "The painter was Pablo Picasso.", "persistence_of_memory_painter"),
        ]

        for question, answer, expected_fact_id in cases:
            with self.subTest(question=question, answer=answer):
                result = run_internal(
                    question=question,
                    answer=answer,
                )

                self.assertGreater(result["score"], 0.0)
                self.assertEqual(result["risk_label"], "High")
                self.assertGreaterEqual(result["score"], 0.67)
                self.assertFalse(result["metadata"]["uncertainty_floor_applied"])
                self.assertEqual(result["metadata"]["simple_fact_sanity"]["verdict"], "incorrect")
                self.assertEqual(result["metadata"]["simple_fact_sanity"]["fact_id"], expected_fact_id)
                self.assertIn("explicit local fact bank", result["explanation"])



class RevisionQualityTests(unittest.TestCase):
    def test_cove_revision(self):
        for case_id in ("cove_low_midtown", "cove_high_midtown"):
            case = case_by_id(case_id)
            result = run_cove(
                question=case["question"],
                answer=case["answer"],
                source_text=case["source_text"],
                evidence_text=case["evidence_text"],
            )
            revised = result["revised_answer"]
            self.assertTrue(revised)
            self.assertLessEqual(len(revised.split()), 65)
            self.assertNotIn("Meeting note:", revised)
            self.assertNotIn("Evidence note", revised)
            self.assertNotIn("\n", revised)

    def test_critic_revision(self):
        for case_id in ("critic_low_solaris", "critic_high_solaris"):
            case = case_by_id(case_id)
            result = run_critic(
                question=case["question"],
                answer=case["answer"],
                evidence_text=case["evidence_text"],
            )
            revised = result["revised_answer"]
            self.assertTrue(revised)
            self.assertLessEqual(len(revised.split()), 65)
            self.assertNotIn("Briefing note:", revised)
            self.assertNotIn("Regulatory note:", revised)
            self.assertNotIn("Supplier note:", revised)
            self.assertNotIn("\n", revised)


if __name__ == "__main__":
    unittest.main()
