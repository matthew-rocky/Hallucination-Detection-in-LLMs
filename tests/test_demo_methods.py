import unittest

from data.sample_cases import case_by_id, list_cases
from methods.cove_check import run_cove
from methods.critic_check import run_critic
from methods.internal_check import run_internal
from methods.rag_check import run_rag
from methods.retrieval_check import run_retrieval
from methods.sep_check import run_sep
from methods.source_check import run_source
from methods.verify_flow import run_verify
from utils.ui_utils import METHOD_PROFILES, PROFILE_FIELD_ORDER, format_status


REQUIRED_RESULT_KEYS = {
    "method_name",
    "family",
    "score",
    "label",
    "confidence",
    "summary",
    "explanation",
    "evidence",
    "citations",
    "intermediate_steps",
    "revised_answer",
    "latency_ms",
    "metadata",
    "risk_score",
    "risk_label",
    "implementation_status",
    "claim_findings",
}


class DetectorTests(unittest.TestCase):
    def test_samples_catalog(self):
        required_fields = {
            "id",
            "pair_id",
            "method_targets",
            "risk_level",
            "title",
            "description",
            "question",
            "answer",
            "answer_samples",
            "source_text",
            "evidence_text",
            "expected_label",
            "notes",
        }
        for case in list_cases():
            self.assertTrue(required_fields.issubset(case.keys()))
            self.assertIn(case["risk_level"], {"low", "high"})
            self.assertIn(case["expected_label"], {"Low", "High"})
            self.assertTrue(case["method_targets"])

        required_profile_keys = {field_key for _, field_key in PROFILE_FIELD_ORDER}
        expected_methods = {
            "Internal-Signal Baseline",
            "SEP-Inspired Internal Signal",
            "Source-Grounded Consistency",
            "Retrieval-Grounded Checker",
            "RAG Grounded Check",
            "Verification-Based Workflow",
            "CoVe-Style Verification",
            "CRITIC-lite Tool Check",
        }
        self.assertEqual(set(METHOD_PROFILES.keys()), expected_methods)
        for profile in METHOD_PROFILES.values():
            self.assertTrue(required_profile_keys.issubset(profile.keys()))

    def test_internal_shape(self):
        baseline = run_internal(
            question="What is the capital of Canada?",
            answer="The capital of Canada is Toronto.",
        )
        sep_case = case_by_id("sep_high_narev_consensus")
        sep_result = run_sep(
            question=sep_case["question"],
            answer=sep_case["answer"],
            sampled_answers_text=sep_case["answer_samples"],
        )

        for result in (baseline, sep_result):
            self.assertTrue(REQUIRED_RESULT_KEYS.issubset(result.keys()))
            self.assertIsNotNone(result["score"])
            self.assertGreaterEqual(result["score"], 0.0)
            self.assertLessEqual(result["score"], 1.0)
            self.assertIsInstance(result["claim_findings"], list)
            self.assertIn("schema_version", result["metadata"])
            self.assertIn(result["metadata"].get("result_origin"), {
                "full_backend_scoring",
                "sep_lite_probe_path",
                "deterministic_fallback_approximation",
            })
            self.assertIn(result["metadata"].get("backend_status"), {"available", "unavailable"})
            self.assertTrue(result["metadata"].get("backend_status_label"))

        self.assertIn(baseline["implementation_status"], {"implemented", "approximate"})
        self.assertEqual(baseline["mode_used"], "uncertainty_baseline")
        self.assertTrue(baseline["sub_signals"])
        self.assertEqual(sep_result["implementation_status"], "approximate")
        self.assertEqual(sep_result["mode_used"], "sep_lite")
        self.assertGreaterEqual(sep_result["metadata"]["num_samples"], 2)

    def test_empty_input(self):
        internal_result = run_internal(
            question="What is the capital of Canada?",
            answer="",
        )
        source_result = run_source(
            question="What is the capital of Canada?",
            answer="",
            source_text="Ottawa is the capital of Canada.",
        )

        self.assertEqual(internal_result["implementation_status"], "unavailable")
        self.assertEqual(source_result["implementation_status"], "unavailable")
        self.assertFalse(internal_result["available"])
        self.assertFalse(source_result["available"])
        self.assertEqual(
            format_status(internal_result, METHOD_PROFILES["Internal-Signal Baseline"]["implementation"]),
            "Unavailable",
        )
        self.assertEqual(
            format_status(source_result, METHOD_PROFILES["Source-Grounded Consistency"]["implementation"]),
            "Unavailable",
        )

    def test_citations(self):
        retrieval_case = case_by_id("retrieval_high_riverside")
        rag_case = case_by_id("rag_high_harbor")

        retrieval_result = run_retrieval(
            question=retrieval_case["question"],
            answer=retrieval_case["answer"],
            evidence_text=retrieval_case["evidence_text"],
        )
        rag_result = run_rag(
            question=rag_case["question"],
            answer=rag_case["answer"],
            evidence_text=rag_case["evidence_text"],
        )

        self.assertEqual(retrieval_result["method_name"], "Retrieval-Grounded Checker")
        self.assertEqual(retrieval_result["implementation_status"], "implemented")
        self.assertTrue(retrieval_result["citations"])
        self.assertTrue(retrieval_result["chunk_catalog"])
        self.assertGreater(retrieval_result["retrieval_counts"]["contradicted"], 0)

        self.assertEqual(rag_result["method_name"], "RAG Grounded Check")
        self.assertIn(rag_result["implementation_status"], {"implemented", "approximate"})
        self.assertTrue(rag_result["citations"])
        self.assertGreaterEqual(rag_result["score"], 0.5)

    def test_harbor_rag_low(self):
        case = case_by_id("rag_low_harbor")
        result = run_rag(
            question=case["question"],
            answer=case["answer"],
            evidence_text=case["evidence_text"],
        )

        self.assertEqual(result["risk_label"], "Low")

    def test_cove_stages(self):
        case = case_by_id("cove_high_midtown")
        result = run_cove(
            question=case["question"],
            answer=case["answer"],
            source_text=case["source_text"],
            evidence_text=case["evidence_text"],
        )

        self.assertTrue(REQUIRED_RESULT_KEYS.issubset(result.keys()))
        self.assertEqual(result["method_name"], "CoVe-Style Verification")
        self.assertTrue(result["original_draft"])
        self.assertTrue(result["verification_questions"])
        self.assertTrue(result["independent_answers"])
        self.assertTrue(result["revised_answer"])
        self.assertTrue(result["verification_summary"])
        self.assertNotEqual(result["original_draft"], result["revised_answer"])
        self.assertEqual(
            [step["stage"] for step in result["intermediate_steps"]],
            [
                "draft_answer",
                "verification_question_generation",
                "independent_answering",
                "revised_answer",
                "final_summary",
            ],
        )

    def test_critic_tools(self):
        case = case_by_id("critic_high_solaris")
        result = run_critic(
            question=case["question"],
            answer=case["answer"],
            evidence_text=case["evidence_text"],
        )

        self.assertTrue(REQUIRED_RESULT_KEYS.issubset(result.keys()))
        self.assertEqual(result["method_name"], "CRITIC-lite Tool Check")
        self.assertTrue(result["proposed_external_checks"])
        self.assertTrue(result["tool_outputs"])
        self.assertTrue(result["revised_answer"])
        self.assertNotEqual(case["answer"], result["revised_answer"])
        first_tool_names = {item["tool_name"] for item in result["tool_outputs"][0]["tool_results"]}
        self.assertIn("local_retrieval", first_tool_names)
        self.assertIn("calculator_numeric_check", first_tool_names)
        self.assertIn("tool_execution", [step["stage"] for step in result["intermediate_steps"]])

    def test_old_schema(self):
        case = case_by_id("source_grounded_high_midtown")
        source_result = run_source(
            question=case["question"],
            answer=case["answer"],
            source_text=case["source_text"],
        )
        verification_case = case_by_id("verification_high_northbridge")
        verification_result = run_verify(
            question=verification_case["question"],
            answer=verification_case["answer"],
            source_text=verification_case["source_text"],
            evidence_text=verification_case["evidence_text"],
            allow_web=False,
        )

        self.assertTrue(REQUIRED_RESULT_KEYS.issubset(source_result.keys()))
        self.assertTrue(REQUIRED_RESULT_KEYS.issubset(verification_result.keys()))
        self.assertEqual(source_result["implementation_status"], "approximate")
        self.assertIn(verification_result["implementation_status"], {"implemented", "approximate"})


if __name__ == "__main__":
    unittest.main()
