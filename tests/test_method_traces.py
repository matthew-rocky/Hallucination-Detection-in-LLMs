import unittest

from methods.rag_check import run_rag
from methods.source_check import run_source
from methods.verify_flow import run_verify


class MethodTraceTests(unittest.TestCase):
    def test_source_trace(self):
        result = run_source(
            question="Summarize the museum schedule.",
            answer="The museum is open Tuesday through Sunday from 10 AM to 5 PM and charges admission.",
            source_text=(
                "The museum is open Tuesday through Sunday from 10 AM to 5 PM. "
                "It is closed on Mondays. General admission is $12 for adults and $8 for students."
            ),
        )
        self.assertTrue(result["citations"])
        self.assertTrue(result["evidence"])
        self.assertEqual([step["stage"] for step in result["intermediate_steps"]], [
            "claim_extraction",
            "source_chunking",
            "claim_grounding",
            "final_aggregation",
        ])

    def test_rag_diagnostics(self):
        result = run_rag(
            question="Using the Riverside tablet program notes, summarize what happened.",
            answer=(
                "Riverside bought 2,000 tablets for every middle school student, raised math scores by 14%, "
                "and renewed the contract through 2027."
            ),
            evidence_text=(
                "Program note 1: Riverside distributed 600 tablets to three middle schools during the fall pilot. "
                "Program note 2: Teachers reported modest engagement gains, but district math scores have not been tied to the pilot yet. "
                "Program note 3: The school board extended the vendor contract through June 2025 for evaluation only."
            ),
        )
        self.assertTrue(result["metadata"]["rag_style_claim_verification"])
        self.assertIn("retrieval_failure_claim_count", result["metadata"])
        self.assertIn("unsupported_claim_count", result["metadata"])
        self.assertTrue(result["claim_findings"])
        self.assertTrue(all("retrieval_diagnosis" in item for item in result["claim_findings"]))

    def test_verify_trace(self):
        result = run_verify(
            question="Summarize the Harbor cleanup update.",
            answer="The Harbor cleanup finished in March, removed all contamination, and reopened the waterfront for swimming immediately.",
            source_text=(
                "The Harbor cleanup entered a second remediation phase in March. Officials said contamination levels fell, "
                "but some hotspots remain under treatment. The waterfront remains closed to swimmers pending further testing."
            ),
            evidence_text="Meeting note: Inspectors said the cleanup is progressing, but final clearance has not been issued.",
            allow_web=False,
        )
        self.assertEqual([step["stage"] for step in result["intermediate_steps"]], [
            "claim_extraction",
            "verification_question_generation",
            "evidence_retrieval",
            "claim_verdict_assignment",
            "final_aggregation",
        ])
        self.assertIn("verification_diagnostic_counts", result["details"])
        self.assertIn("unsupported_with_evidence_count", result["details"])
        self.assertTrue(result["evidence"])
        self.assertTrue(result["citations"])


if __name__ == "__main__":
    unittest.main()
