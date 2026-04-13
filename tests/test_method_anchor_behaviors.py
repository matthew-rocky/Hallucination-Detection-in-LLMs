import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from methods.critic_check import run_critic
from methods.cove_check import run_cove
from methods.internal_check import run_internal
from methods.rag_check import run_rag
from methods.retrieval_check import run_retrieval
from methods.sep_check import run_sep
from methods.source_check import run_source
from methods.verify_flow import run_verify


LARTON_QUESTION = "Tell me about the Larton Basin Energy Treaty."
LARTON_ANSWER = (
    "The 2018 Larton Basin Energy Treaty ended a decade-long pipeline dispute between "
    "Karsin and Velora, created a joint offshore authority, and guaranteed fixed export quotas through 2035."
)
AUSTEN_QUESTION = "Who wrote Pride and Prejudice?"
AUSTEN_ANSWER = "Jane Austen wrote Pride and Prejudice."

NAREV_QUESTION = "What happened during the 2020 Narev medical summit?"
NAREV_ANSWER = (
    "The 2020 Narev medical summit introduced the first universal clinical framework for AI-assisted surgery "
    "and was signed by 42 countries in Geneva."
)
NAREV_SAMPLES = "\n\n---\n\n".join(
    [
        "The 2020 Narev medical summit created a global AI surgery protocol signed by dozens of countries.",
        "The Narev summit established international standards for robotic surgery in Geneva in 2020.",
        "In 2020, the Narev medical summit unified global regulation for AI surgery across 42 nations.",
    ]
)
TOKYO_QUESTION = "What is the capital of Japan?"
TOKYO_ANSWER = "The capital of Japan is Tokyo."
TOKYO_SAMPLES = "\n\n---\n\n".join(
    [
        "Tokyo is the capital of Japan.",
        "Japan's capital city is Tokyo.",
        "The capital of Japan is Tokyo.",
    ]
)

MUSEUM_QUESTION = "Summarize the museum schedule."
MUSEUM_SOURCE = (
    "The museum is open Tuesday through Sunday from 10 AM to 5 PM. "
    "It is closed on Mondays. General admission is $12 for adults and $8 for students."
)
MUSEUM_HIGH = "The museum is open every day from 8 AM to 8 PM, and entry is free for all visitors."
MUSEUM_LOW = (
    "The museum is open Tuesday through Sunday from 10 AM to 5 PM, is closed on Mondays, "
    "and charges admission, with discounts for students."
)

GREENFIELD_QUESTION = "Summarize the Greenfield transit pilot."
GREENFIELD_EVIDENCE = (
    "Greenfield ran a four-month transit pilot in 8 neighborhoods. "
    "City staff reported average commute times fell by about 6%. "
    "Officials extended the pilot for another quarter, but no permanent rollout has been approved yet."
)
GREENFIELD_HIGH = (
    "The Greenfield transit pilot ran for six months in 12 neighborhoods, cut commute times by 18%, "
    "and the city approved a permanent rollout immediately."
)
GREENFIELD_LOW = (
    "The Greenfield transit pilot ran in 8 neighborhoods, reduced commute times modestly, "
    "and was extended rather than permanently approved."
)

RIVERSIDE_QUESTION = "Using the Riverside tablet program notes, summarize what happened."
RIVERSIDE_EVIDENCE = (
    "Program note 1: Riverside distributed 600 tablets to three middle schools during the fall pilot. "
    "Program note 2: Teachers reported modest engagement gains, but district math scores have not been tied to the pilot yet. "
    "Program note 3: The school board extended the vendor contract through June 2025 for evaluation only."
)
RIVERSIDE_HIGH = (
    "Riverside bought 2,000 tablets for every middle school student, raised math scores by 14%, "
    "and renewed the contract through 2027."
)
RIVERSIDE_LOW = (
    "Riverside distributed 600 tablets to three middle schools during a pilot, saw some engagement gains, "
    "and kept the vendor contract through mid-2025 for evaluation."
)

HARBOR_QUESTION = "Summarize the Harbor cleanup update."
HARBOR_SOURCE = (
    "The Harbor cleanup entered a second remediation phase in March. Officials said contamination levels fell, "
    "but some hotspots remain under treatment. The waterfront remains closed to swimmers pending further testing."
)
HARBOR_EVIDENCE = "Meeting note: Inspectors said the cleanup is progressing, but final clearance has not been issued."
HARBOR_HIGH = (
    "The Harbor cleanup finished in March, removed all contamination, and reopened the waterfront for swimming immediately."
)
HARBOR_LOW = (
    "The Harbor cleanup entered a second remediation phase in March, contamination levels improved but some hotspots remain, "
    "and the waterfront is still closed pending more testing."
)

PINECREST_QUESTION = "What did the Pinecrest housing memo approve?"
PINECREST_SOURCE = (
    "The Pinecrest housing memo approved a 12-story building with 180 mixed-income units. "
    "The earliest construction start is late 2025 after permit review."
)
PINECREST_EVIDENCE = "Board note: The January 2025 start date from an early draft was withdrawn."
PINECREST_HIGH = "The Pinecrest memo approved a 20-story tower with 300 units and said construction starts in January 2025."
PINECREST_LOW = (
    "The Pinecrest memo approved a 12-story building with 180 mixed-income units, "
    "with construction expected no earlier than late 2025 after permits."
)

CRITIC_QUESTION = "What did the event report say about attendance and conversions?"
CRITIC_EVIDENCE = (
    "Event report: 250 attendees were recorded. Conversion reached 45 sign-ups, which is 18% of attendees. "
    "Attendance was down 6% year over year."
)
CRITIC_HIGH = "The report said 18% of 250 attendees meant 50 conversions, and attendance increased year over year."
CRITIC_LOW = "The report said 18% of 250 attendees converted, and attendance decreased year over year."


class MethodAnchorTests(unittest.TestCase):
    @patch("detectors.signal._extract_features", side_effect=RuntimeError("backend offline"))
    def test_baseline_treaty(self, _mock_extract):
        high = run_internal(question=LARTON_QUESTION, answer=LARTON_ANSWER)
        low = run_internal(question=AUSTEN_QUESTION, answer=AUSTEN_ANSWER)

        self.assertEqual(high["risk_label"], "High")
        self.assertEqual(low["risk_label"], "Low")
        self.assertTrue(high["metadata"]["fallback_mode"])
        self.assertGreater(high["risk_score"], low["risk_score"])

    @patch("detectors.signal._extract_features", side_effect=RuntimeError("backend offline"))
    def test_sep_ranks(self, _mock_extract):
        high = run_sep(
            question=NAREV_QUESTION,
            answer=NAREV_ANSWER,
            sampled_answers_text=NAREV_SAMPLES,
        )
        low = run_sep(
            question=TOKYO_QUESTION,
            answer=TOKYO_ANSWER,
            sampled_answers_text=TOKYO_SAMPLES,
        )

        self.assertEqual(high["risk_label"], "High")
        self.assertEqual(low["risk_label"], "Low")
        self.assertTrue(high["metadata"]["suspicious_consensus_flag"])
        self.assertGreater(high["score"] - low["score"], 0.35)

    def test_source_museum(self):
        high = run_source(question=MUSEUM_QUESTION, answer=MUSEUM_HIGH, source_text=MUSEUM_SOURCE)
        low = run_source(question=MUSEUM_QUESTION, answer=MUSEUM_LOW, source_text=MUSEUM_SOURCE)

        self.assertEqual(high["risk_label"], "High")
        self.assertEqual(low["risk_label"], "Low")
        self.assertGreaterEqual(low["metadata"]["abstractly_supported_claim_count"], 2)

    def test_greenfield_split(self):
        high = run_retrieval(question=GREENFIELD_QUESTION, answer=GREENFIELD_HIGH, evidence_text=GREENFIELD_EVIDENCE)
        low = run_retrieval(question=GREENFIELD_QUESTION, answer=GREENFIELD_LOW, evidence_text=GREENFIELD_EVIDENCE)

        self.assertEqual(high["risk_label"], "High")
        self.assertEqual(low["risk_label"], "Low")
        self.assertGreater(high["risk_score"], low["risk_score"])

    def test_rag_riverside(self):
        high = run_rag(question=RIVERSIDE_QUESTION, answer=RIVERSIDE_HIGH, evidence_text=RIVERSIDE_EVIDENCE)
        low = run_rag(question=RIVERSIDE_QUESTION, answer=RIVERSIDE_LOW, evidence_text=RIVERSIDE_EVIDENCE)

        self.assertEqual(high["risk_label"], "High")
        self.assertEqual(low["risk_label"], "Low")
        self.assertEqual(high["implementation_status"], "approximate")

    def test_verify_harbor(self):
        high = run_verify(
            question=HARBOR_QUESTION,
            answer=HARBOR_HIGH,
            source_text=HARBOR_SOURCE,
            evidence_text=HARBOR_EVIDENCE,
            allow_web=False,
        )
        low = run_verify(
            question=HARBOR_QUESTION,
            answer=HARBOR_LOW,
            source_text=HARBOR_SOURCE,
            evidence_text=HARBOR_EVIDENCE,
            allow_web=False,
        )

        self.assertEqual(high["risk_label"], "High")
        self.assertEqual(low["risk_label"], "Low")
        self.assertGreaterEqual(high["details"]["num_reference_chunks"], 1)

    def test_cove_pinecrest(self):
        high = run_cove(
            question=PINECREST_QUESTION,
            answer=PINECREST_HIGH,
            source_text=PINECREST_SOURCE,
            evidence_text=PINECREST_EVIDENCE,
        )
        low = run_cove(
            question=PINECREST_QUESTION,
            answer=PINECREST_LOW,
            source_text=PINECREST_SOURCE,
            evidence_text=PINECREST_EVIDENCE,
        )

        self.assertEqual(high["risk_label"], "High")
        self.assertEqual(low["risk_label"], "Low")
        self.assertTrue(high["revised_answer"])
        self.assertTrue(low["revised_answer"])
        self.assertIn("12-story", high["revised_answer"])
        self.assertIn("late 2025", high["revised_answer"])

    def test_conversion(self):
        high = run_critic(question=CRITIC_QUESTION, answer=CRITIC_HIGH, evidence_text=CRITIC_EVIDENCE)
        low = run_critic(question=CRITIC_QUESTION, answer=CRITIC_LOW, evidence_text=CRITIC_EVIDENCE)

        self.assertEqual(high["risk_label"], "High")
        self.assertEqual(low["risk_label"], "Low")
        self.assertTrue(high["tool_outputs"])


class AppImportSafetyTests(unittest.TestCase):
    def test_app_import(self):
        spec = importlib.util.spec_from_file_location("app_without_streamlit", Path("app.py"))
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"streamlit": None, "streamlit.runtime": None, "streamlit.runtime.scriptrunner": None}):
            spec.loader.exec_module(module)

        self.assertIsNone(module.st)
        self.assertIn("Internal-Signal Baseline", module.METHOD_RUNNERS)
        self.assertEqual(module.get_ordered_results([]), [])


if __name__ == "__main__":
    unittest.main()
