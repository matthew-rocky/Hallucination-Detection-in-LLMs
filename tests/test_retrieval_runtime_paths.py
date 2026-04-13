import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app
from data.sample_cases import case_by_id
from detectors.signal import PROBE_FEATURE_ORDER, train_probe_jsonl
from detectors.retrieval_check import run_retrieval_check
from methods.internal_check import run_internal
from methods.sep_check import run_sep
from retrieval.chunking import ingest_docs
from retrieval.indexing import VectorIndex
from utils.ui_utils import build_compare_table


MARWICK_QUESTION = "Tell me about the 2019 Marwick Islands Peace Accord."
MARWICK_ANSWER = (
    "The 2019 Marwick Islands Peace Accord officially ended a 14-year naval conflict between Norland and Estavia. "
    "It also created a jointly administered demilitarized trade port at Sel Harbor, which later became a model for regional conflict resolution."
)
DALI_QUESTION = "Who painted The Persistence of Memory?"
DALI_ANSWER = "Salvador Dali painted The Persistence of Memory."
NAREV_QUESTION = "What happened during the 2020 Narev medical summit?"
NAREV_ANSWER = (
    "The 2020 Narev medical summit introduced the first universal clinical framework for AI-assisted surgery "
    "and was signed by 42 countries in Geneva."
)
NAREV_STABLE_SAMPLES = "\n\n---\n\n".join(
    [
        "The 2020 Narev medical summit created a global AI surgery protocol signed by dozens of countries.",
        "The Narev summit established international standards for robotic surgery in Geneva in 2020.",
        "In 2020, the Narev medical summit unified global regulation for AI surgery across 42 nations.",
    ]
)
NAREV_DRIFT_SAMPLES = "\n\n---\n\n".join(
    [
        "The 2020 Narev medical summit was signed by 11 countries in Zurich.",
        "The Narev summit banned AI-assisted surgery and ended without agreement in Vienna.",
        "In 2021, the Narev medical summit proposed voluntary guidance for 7 nations in Geneva.",
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


class RuntimeSmokeTests(unittest.TestCase):
    def test_saved_index(self):
        documents = ingest_docs(
            evidence_text="Ottawa is the capital of Canada. Toronto is the largest city in Canada.",
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            index_path = Path(tmp_dir) / "demo_index.pkl"
            index = VectorIndex.from_documents(documents, preferred_backend="tfidf", max_sentences=1, overlap=0)
            first_hits = index.search("capital of Canada", top_k=1)
            self.assertTrue(first_hits)
            self.assertIn("Ottawa", first_hits[0]["text"])

            index.save(index_path)
            reloaded = VectorIndex.load(index_path)
            second_hits = reloaded.search("capital of Canada", top_k=1)
            self.assertTrue(second_hits)
            self.assertEqual(first_hits[0]["chunk_id"], second_hits[0]["chunk_id"])

    @patch("retrieval.embeddings.SKLEARN_AVAILABLE", False)
    @patch("retrieval.embeddings.TfidfVectorizer", None)
    def test_tfidf_sentence(self):
        documents = ingest_docs(
            evidence_text="Ottawa is the capital of Canada. Toronto is the largest city in Canada.",
        )
        index = VectorIndex.from_documents(documents, preferred_backend="tfidf", max_sentences=1, overlap=0)
        hits = index.search("capital of Canada", top_k=1)
        self.assertTrue(hits)
        self.assertIn("Ottawa", hits[0]["text"])
        self.assertEqual(index.embedder.backend_name, "tfidf")

    def test_upload_grounding(self):
        result = run_retrieval_check(
            question="What is the capital of Canada?",
            answer="Ottawa",
            extra_documents=[
                {
                    "title": "facts.txt",
                    "text": "Ottawa is the capital of Canada.",
                    "source_type": "uploaded_document",
                }
            ],
            preferred_backend="tfidf",
        )
        self.assertEqual(result["risk_label"], "Low")
        self.assertTrue(result["citations"])
        self.assertEqual(result["claim_findings"][0]["status"], "supported")

    def test_probe_bundle(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            feature_path = Path(tmp_dir) / "features.jsonl"
            output_path = Path(tmp_dir) / "probe.pkl"
            rows = [
                {"label": 0, "feature_names": PROBE_FEATURE_ORDER, "features": [0.1] * len(PROBE_FEATURE_ORDER)},
                {"label": 1, "feature_names": PROBE_FEATURE_ORDER, "features": [0.9] * len(PROBE_FEATURE_ORDER)},
                {"label": 0, "feature_names": PROBE_FEATURE_ORDER, "features": [0.2] * len(PROBE_FEATURE_ORDER)},
                {"label": 1, "feature_names": PROBE_FEATURE_ORDER, "features": [0.8] * len(PROBE_FEATURE_ORDER)},
            ]
            feature_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            info = train_probe_jsonl(feature_path=str(feature_path), output_path=str(output_path))
            self.assertEqual(info["trained_rows"], 4)
            self.assertTrue(output_path.exists())
            bundle = pickle.loads(output_path.read_bytes())
            self.assertIn("model", bundle)

    def test_ui_registry(self):
        result = run_retrieval_check(
            question="What is the capital of Canada?",
            answer="Ottawa",
            extra_documents=[
                {
                    "title": "facts.txt",
                    "text": "Ottawa is the capital of Canada.",
                    "source_type": "uploaded_document",
                }
            ],
            preferred_backend="tfidf",
        )
        table = build_compare_table([result])
        self.assertIn("Implementation", table.columns)
        self.assertIn("Confidence", table.columns)
        self.assertIn("Retrieval-Grounded Checker", app.METHOD_RUNNERS)
        ordered = app.get_ordered_results([result])
        self.assertEqual(ordered[0]["method_name"], "Retrieval-Grounded Checker")

    @patch("detectors.signal._extract_features", side_effect=RuntimeError("backend offline"))
    def test_samples_offline(self, _mock_extract):
        case = case_by_id("multi_answer_instability")
        baseline = run_internal(
            question=case["question"],
            answer=case["answer"],
            sampled_answers_text=case["answer_samples"],
        )
        sep_result = run_sep(
            question=case["question"],
            answer=case["answer"],
            sampled_answers_text=case["answer_samples"],
        )

        self.assertIsNotNone(baseline["score"])
        self.assertIsNotNone(baseline["risk_score"])
        self.assertTrue(baseline["available"])
        self.assertEqual(baseline["implementation_status"], "approximate")
        self.assertTrue(baseline["metadata"]["fallback_mode"])
        self.assertEqual(baseline["metadata"]["fallback_type"], "text_proxy_with_sample_consistency")

        self.assertIsNotNone(sep_result["score"])
        self.assertIsNotNone(sep_result["risk_score"])
        self.assertTrue(sep_result["available"])
        self.assertEqual(sep_result["implementation_status"], "approximate")
        self.assertTrue(sep_result["metadata"]["fallback_mode"])
        self.assertEqual(sep_result["metadata"]["fallback_type"], "text_proxy_with_sample_consistency")
        self.assertFalse(sep_result["metadata"]["backend_available"])
        self.assertIn("backend offline", sep_result["metadata"]["backend_error"])
        self.assertGreaterEqual(sep_result["metadata"]["sample_count"], 2)

    @patch("detectors.signal._extract_features", side_effect=RuntimeError("backend offline"))
    def test_text_fallback(self, _mock_extract):
        baseline = run_internal(
            question=MARWICK_QUESTION,
            answer=MARWICK_ANSWER,
            sampled_answers_text="",
        )
        sep_result = run_sep(
            question=MARWICK_QUESTION,
            answer=MARWICK_ANSWER,
            sampled_answers_text="",
        )

        self.assertIsNotNone(baseline["score"])
        self.assertTrue(baseline["available"])
        self.assertEqual(baseline["metadata"]["fallback_type"], "text_only_proxy")

        self.assertIsNotNone(sep_result["score"])
        self.assertIsNotNone(sep_result["risk_score"])
        self.assertTrue(sep_result["available"])
        self.assertEqual(sep_result["implementation_status"], "approximate")
        self.assertTrue(sep_result["metadata"]["fallback_mode"])
        self.assertEqual(sep_result["metadata"]["fallback_type"], "text_only_proxy")
        self.assertFalse(sep_result["metadata"]["backend_available"])
        self.assertEqual(sep_result["metadata"]["sample_count"], 1)
        self.assertEqual(sep_result.get("sampled_answers"), [])
        self.assertLess(sep_result["confidence"], 0.60)

    @patch("detectors.signal._extract_features", side_effect=RuntimeError("backend offline"))
    def test_sep_flags(self, _mock_extract):
        result = run_sep(
            question=NAREV_QUESTION,
            answer=NAREV_ANSWER,
            sampled_answers_text=NAREV_STABLE_SAMPLES,
        )

        self.assertGreaterEqual(result["score"], 0.67)
        self.assertEqual(result["risk_label"], "High")
        self.assertTrue(result["metadata"]["fallback_mode"])
        self.assertGreater(result["metadata"]["sample_consistency_score"], 0.45)
        self.assertGreater(result["metadata"]["suspicious_specificity_score"], 0.45)
        self.assertTrue(result["metadata"]["suspicious_consensus_flag"])

    @patch("detectors.signal._extract_features", side_effect=RuntimeError("backend offline"))
    def test_sep_plain_low(self, _mock_extract):
        result = run_sep(
            question=TOKYO_QUESTION,
            answer=TOKYO_ANSWER,
            sampled_answers_text=TOKYO_SAMPLES,
        )

        self.assertLessEqual(result["score"], 0.33)
        self.assertEqual(result["risk_label"], "Low")
        self.assertGreater(result["metadata"]["sample_consistency_score"], 0.75)
        self.assertGreater(result["metadata"]["entity_consistency_score"], 0.75)
        self.assertGreater(result["metadata"]["numeric_consistency_score"], 0.95)
        self.assertFalse(result["metadata"]["suspicious_consensus_flag"])

    @patch("detectors.signal._extract_features", side_effect=RuntimeError("backend offline"))
    def test_sep_drift(self, _mock_extract):
        stable = run_sep(
            question=NAREV_QUESTION,
            answer=NAREV_ANSWER,
            sampled_answers_text=NAREV_STABLE_SAMPLES,
        )
        inconsistent = run_sep(
            question=NAREV_QUESTION,
            answer=NAREV_ANSWER,
            sampled_answers_text=NAREV_DRIFT_SAMPLES,
        )

        self.assertGreater(inconsistent["score"], stable["score"])
        self.assertLess(inconsistent["metadata"]["numeric_consistency_score"], stable["metadata"]["numeric_consistency_score"])
        self.assertGreater(inconsistent["metadata"]["sample_instability_penalty"], stable["metadata"]["sample_instability_penalty"])

    @patch("detectors.signal._extract_features", side_effect=RuntimeError("backend offline"))
    def test_sep_narev_tokyo(self, _mock_extract):
        narev = run_sep(
            question=NAREV_QUESTION,
            answer=NAREV_ANSWER,
            sampled_answers_text=NAREV_STABLE_SAMPLES,
        )
        tokyo = run_sep(
            question=TOKYO_QUESTION,
            answer=TOKYO_ANSWER,
            sampled_answers_text=TOKYO_SAMPLES,
        )

        self.assertGreater(narev["score"] - tokyo["score"], 0.35)
        self.assertEqual(narev["risk_label"], "High")
        self.assertEqual(tokyo["risk_label"], "Low")

    @patch("detectors.signal._extract_features", side_effect=RuntimeError("backend offline"))
    def test_table_fallback(self, _mock_extract):
        baseline = run_internal(question=MARWICK_QUESTION, answer=MARWICK_ANSWER)
        sep_result = run_sep(question=NAREV_QUESTION, answer=NAREV_ANSWER, sampled_answers_text=NAREV_STABLE_SAMPLES)
        table = build_compare_table([baseline, sep_result])
        rows = {row["Method"]: row for row in table.to_dict("records")}

        self.assertEqual(rows["Internal-Signal Baseline"]["Implementation"], "Implemented (Fallback)")
        self.assertEqual(rows["SEP-Inspired Internal Signal"]["Implementation"], "Approximate (Fallback)")
        self.assertNotEqual(rows["SEP-Inspired Internal Signal"]["Risk Score"], "Not Available")


if __name__ == "__main__":
    unittest.main()