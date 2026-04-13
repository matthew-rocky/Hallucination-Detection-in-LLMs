import unittest

from methods.source_check import run_source


MUSEUM_QUESTION = "Summarize the museum schedule."
MUSEUM_SOURCE = (
    "The museum is open Tuesday through Sunday from 10 AM to 5 PM. "
    "It is closed on Mondays. General admission is $12 for adults and $8 for students."
)


class SourceMethodTests(unittest.TestCase):
    def test_museum_high(self):
        result = run_source(
            question=MUSEUM_QUESTION,
            answer=(
                "The museum is open every day from 8 AM to 8 PM, and entry is free for all visitors."
            ),
            source_text=MUSEUM_SOURCE,
        )

        self.assertGreaterEqual(result["risk_score"], 85.0)
        self.assertEqual(result["risk_label"], "High")
        self.assertGreaterEqual(result["metadata"]["contradicted_claim_count"], 1)

    def test_summary_low(self):
        result = run_source(
            question=MUSEUM_QUESTION,
            answer=(
                "The museum is open Tuesday through Sunday from 10 AM to 5 PM, "
                "is closed on Mondays, and charges admission, with discounts for students."
            ),
            source_text=MUSEUM_SOURCE,
        )

        self.assertLessEqual(result["risk_score"], 20.0)
        self.assertEqual(result["risk_label"], "Low")
        self.assertEqual(result["metadata"]["contradicted_claim_count"], 0)
        self.assertEqual(result["metadata"]["unsupported_claim_count"], 0)
        self.assertGreaterEqual(result["metadata"]["abstractly_supported_claim_count"], 2)
        abstraction_findings = [
            finding for finding in result["claim_findings"] if finding.get("support_type") == "abstraction"
        ]
        self.assertTrue(abstraction_findings)
        self.assertTrue(any("faithful abstraction" in finding["reason"].lower() for finding in abstraction_findings))

    def test_omits_prices(self):
        result = run_source(
            question=MUSEUM_QUESTION,
            answer=(
                "The museum is open Tuesday through Sunday, is closed on Mondays, and charges admission."
            ),
            source_text=MUSEUM_SOURCE,
        )

        self.assertLessEqual(result["risk_score"], 20.0)
        self.assertEqual(result["risk_label"], "Low")
        self.assertEqual(result["metadata"]["unsupported_claim_count"], 0)
        self.assertTrue(result["metadata"]["summary_tolerant_mode"])

    def test_extra_claim_risk(self):
        result = run_source(
            question=MUSEUM_QUESTION,
            answer=(
                "The museum is open Tuesday through Sunday from 10 AM to 5 PM. "
                "It offers free parking for all visitors."
            ),
            source_text=MUSEUM_SOURCE,
        )

        self.assertGreaterEqual(result["risk_score"], 40.0)
        self.assertIn(result["risk_label"], {"Medium", "High"})
        self.assertGreaterEqual(result["metadata"]["unsupported_claim_count"], 1)
        self.assertTrue(any(finding["status"] == "unsupported" for finding in result["claim_findings"]))

    def test_copied_low(self):
        result = run_source(
            question=MUSEUM_QUESTION,
            answer=MUSEUM_SOURCE,
            source_text=MUSEUM_SOURCE,
        )

        self.assertLessEqual(result["risk_score"], 15.0)
        self.assertEqual(result["risk_label"], "Low")
        self.assertEqual(result["metadata"]["supported_claim_ratio"], 1.0)

    def test_schedule_high(self):
        result = run_source(
            question=MUSEUM_QUESTION,
            answer="The museum is closed Tuesday through Sunday and open on Mondays.",
            source_text=MUSEUM_SOURCE,
        )

        self.assertGreaterEqual(result["risk_score"], 85.0)
        self.assertEqual(result["risk_label"], "High")
        self.assertGreaterEqual(result["metadata"]["contradicted_claim_count"], 1)


if __name__ == "__main__":
    unittest.main()
