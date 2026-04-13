import unittest
from unittest.mock import patch

import app

from data.sample_cases import case_by_id
from methods.source_check import run_source
from ui.comparison_table import compact_table
from ui.input_forms import (
    apply_pending_sample,
    apply_sample_state,
    make_sample_payload,
    normalize_sample,
    reset_input_state,
)
from ui.method_descriptions import visible_fields_for


class UiConfigTests(unittest.TestCase):
    def test_run_errors(self):
        class FakeStreamlit:
            def __init__(self):
                self.session_state = {
                    "selected_methods": ["Internal-Signal Baseline"],
                    "question_input": "What is the capital of Japan?",
                    "answer_input": "Ottawa",
                    "sampled_answers_input": "",
                    "source_input": "",
                    "evidence_input": "",
                    "uploaded_evidence_files": [],
                    "analysis_results": [],
                    "upload_warnings": [],
                }

        fake_st = FakeStreamlit()

        def failing_runner(**_kwargs):
            raise RuntimeError("forced failure")

        original_runner = app.METHOD_RUNNERS["Internal-Signal Baseline"]
        try:
            app.METHOD_RUNNERS["Internal-Signal Baseline"] = failing_runner
            with patch.object(app, "st", fake_st):
                app.run_methods(["Internal-Signal Baseline"])
        finally:
            app.METHOD_RUNNERS["Internal-Signal Baseline"] = original_runner

        result = fake_st.session_state["analysis_results"][0]
        self.assertEqual(result["method_name"], "Internal-Signal Baseline")
        self.assertEqual(result["implementation_status"], "unavailable")
        self.assertFalse(result["available"])
        self.assertTrue(result["metadata"]["app_runtime_guard"])
        self.assertEqual(result["metadata"]["result_origin"], "app_runtime_guard")
        self.assertIn("forced failure", result["metadata"]["runtime_error"])

    def test_visible_fields(self):
        self.assertEqual(visible_fields_for(["Internal-Signal Baseline"]), ["question", "answer"])
        self.assertEqual(
            visible_fields_for(["SEP-Inspired Internal Signal"]),
            ["question", "answer", "sampled_answers"],
        )
        self.assertEqual(
            visible_fields_for(["Source-Grounded Consistency"]),
            ["question", "answer", "source_text"],
        )
        self.assertEqual(
            visible_fields_for(["Retrieval-Grounded Checker"]),
            ["question", "answer", "evidence_text", "uploaded_documents"],
        )
        self.assertEqual(
            visible_fields_for(["RAG Grounded Check"]),
            ["question", "answer", "evidence_text", "uploaded_documents"],
        )
        self.assertEqual(
            visible_fields_for(["Verification-Based Workflow"]),
            ["question", "answer", "evidence_text"],
        )
        self.assertEqual(
            visible_fields_for(["CoVe-Style Verification"]),
            ["question", "answer", "evidence_text", "uploaded_documents"],
        )
        self.assertEqual(
            visible_fields_for(["CRITIC-lite Tool Check"]),
            ["question", "answer", "evidence_text"],
        )

    def test_compare_fields(self):
        self.assertEqual(
            visible_fields_for(["Internal-Signal Baseline", "Retrieval-Grounded Checker"]),
            ["question", "answer", "evidence_text", "uploaded_documents"],
        )
        self.assertEqual(
            visible_fields_for(["Source-Grounded Consistency", "CRITIC-lite Tool Check"]),
            ["question", "answer", "source_text", "evidence_text"],
        )

    def test_sample_source(self):
        case = case_by_id("cove_high_midtown")
        normalized = normalize_sample(case, ["Verification-Based Workflow"])

        self.assertEqual(normalized["source_input"], "")
        self.assertIn("[Source Passage]", normalized["evidence_input"])
        self.assertIn("[Supporting Evidence]", normalized["evidence_input"])

    def test_sample_fields(self):
        case = case_by_id("retrieval_low_riverside")
        payload = make_sample_payload(case, ["Retrieval-Grounded Checker"], method_name="Retrieval-Grounded Checker")
        session_state = {
            "question_input": "old question",
            "answer_input": "old answer",
            "sampled_answers_input": "old samples",
            "source_input": "old source",
            "evidence_input": "old evidence",
            "uploaded_evidence_files": [object()],
            "analysis_results": [{"old": True}],
            "upload_warnings": ["old warning"],
            "pending_sample_case_state": {"old": True},
        }

        apply_sample_state(session_state, payload["field_values"], payload)

        self.assertEqual(session_state["question_input"], case["question"])
        self.assertEqual(session_state["answer_input"], case["answer"])
        self.assertEqual(session_state["evidence_input"], case["evidence_text"])
        self.assertEqual(session_state["sampled_answers_input"], "")
        self.assertEqual(session_state["source_input"], "")
        self.assertNotIn("uploaded_evidence_files", session_state)
        self.assertGreaterEqual(session_state["uploaded_evidence_files_key_version"], 1)
        self.assertEqual(session_state["analysis_results"], [])
        self.assertEqual(session_state["upload_warnings"], [])
        self.assertNotIn("pending_sample_case_state", session_state)
        self.assertEqual(session_state["loaded_sample_case_id"], case["id"])
        self.assertEqual(session_state["loaded_sample_method"], "Retrieval-Grounded Checker")

    def test_pending_sample(self):
        case = case_by_id("cove_high_midtown")
        session_state = {
            "analysis_results": [{"old": True}],
            "pending_sample_case_state": make_sample_payload(
                case,
                ["Verification-Based Workflow"],
                method_name="Verification-Based Workflow",
            ),
        }

        applied = apply_pending_sample(session_state)

        self.assertTrue(applied)
        self.assertNotIn("pending_sample_case_state", session_state)
        self.assertEqual(session_state["analysis_results"], [])
        self.assertIn("[Source Passage]", session_state["evidence_input"])
        self.assertEqual(session_state["source_input"], "")
        self.assertEqual(session_state["loaded_sample_title"], case["title"])

    def test_reset_form(self):
        session_state = {
            "question_input": "q",
            "answer_input": "a",
            "sampled_answers_input": "s",
            "source_input": "src",
            "evidence_input": "ev",
            "uploaded_evidence_files": [object()],
            "analysis_results": [{"score": 1}],
            "upload_warnings": ["warn"],
            "pending_sample_case_state": {"field_values": {"question_input": "x"}},
            "selected_sample_case_id": "demo",
            "loaded_sample_case_id": "demo",
            "loaded_sample_title": "Title",
            "loaded_sample_method": "Method",
            "loaded_sample_risk_level": "high",
        }

        reset_input_state(session_state)

        self.assertEqual(session_state["question_input"], "")
        self.assertEqual(session_state["answer_input"], "")
        self.assertEqual(session_state["sampled_answers_input"], "")
        self.assertEqual(session_state["source_input"], "")
        self.assertEqual(session_state["evidence_input"], "")
        self.assertNotIn("uploaded_evidence_files", session_state)
        self.assertGreaterEqual(session_state["uploaded_evidence_files_key_version"], 1)
        self.assertEqual(session_state["analysis_results"], [])
        self.assertEqual(session_state["upload_warnings"], [])
        self.assertNotIn("pending_sample_case_state", session_state)
        self.assertEqual(session_state["loaded_sample_case_id"], "")
        self.assertEqual(session_state["loaded_sample_title"], "")

    def test_compare_columns(self):
        result = run_source(
            question="Summarize the museum schedule.",
            answer="The museum is open Tuesday through Sunday from 10 AM to 5 PM, is closed on Mondays, and charges admission.",
            source_text="The museum is open Tuesday through Sunday from 10 AM to 5 PM. It is closed on Mondays. General admission is $12 for adults and $8 for students.",
        )
        table = compact_table([result])

        self.assertEqual(
            list(table.columns),
            ["Method", "Score", "Risk", "Confidence", "Status", "Short Reason"],
        )
        self.assertEqual(table.iloc[0]["Method"], "Source-Grounded Consistency")
        self.assertTrue(table.iloc[0]["Short Reason"])


if __name__ == "__main__":
    unittest.main()
