import unittest
from unittest.mock import patch

from detectors.cove import run_cove_detector
from detectors.critic import run_critic_detector
from detectors.retrieval_check import run_retrieval_check
from methods.internal_check import run_internal
from methods.sep_check import run_sep
from methods.source_check import run_source
from methods.verify_flow import run_verify


BASELINE_CASES = [
    {"id": "tokyo_correct", "question": "What is the capital of Japan?", "answer": "Tokyo", "expected_label": "Low"},
    {"id": "ottawa_correct", "question": "What is the capital of Canada?", "answer": "Ottawa", "expected_label": "Low"},
    {"id": "jupiter_correct", "question": "What is the largest planet in the Solar System?", "answer": "Jupiter", "expected_label": "Low"},
    {"id": "austen_correct", "question": "Who wrote Pride and Prejudice?", "answer": "Jane Austen", "expected_label": "Low"},
    {"id": "dali_correct", "question": "Who painted The Persistence of Memory?", "answer": "Salvador Dali", "expected_label": "Low"},
    {"id": "tokyo_sentence", "question": "What is the capital of Japan?", "answer": "The capital of Japan is Tokyo.", "expected_label": "Low"},
    {"id": "canada_wrong_toronto", "question": "Where is the capital of Canada?", "answer": "Toronto", "expected_label": "High"},
    {"id": "japan_wrong_ottawa", "question": "What is the capital of Japan?", "answer": "Ottawa", "expected_label": "High"},
    {"id": "planet_wrong_mars", "question": "What is the largest planet in the Solar System?", "answer": "Mars", "expected_label": "High"},
    {"id": "author_wrong_dickens", "question": "Who wrote Pride and Prejudice?", "answer": "Charles Dickens", "expected_label": "High"},
    {"id": "painter_wrong_picasso", "question": "Who painted The Persistence of Memory?", "answer": "Pablo Picasso", "expected_label": "High"},
    {"id": "japan_possessive_correct_shell", "question": "What is Japan's capital?", "answer": "It is Tokyo.", "expected_label": "Low"},
    {"id": "japan_possessive_wrong_shell", "question": "What is Japan's capital?", "answer": "It is Ottawa.", "expected_label": "High"},
    {"id": "japan_city_question_correct_shell", "question": "What city is Japan's capital?", "answer": "It is Tokyo.", "expected_label": "Low"},
    {"id": "japan_city_question_wrong_shell", "question": "What city is Japan's capital?", "answer": "Ottawa.", "expected_label": "High"},
    {"id": "canada_name_question_correct_shell", "question": "Name Canada's capital city.", "answer": "The answer is Ottawa.", "expected_label": "Low"},
    {"id": "canada_name_question_wrong_shell", "question": "Name Canada's capital city.", "answer": "The answer is Toronto.", "expected_label": "High"},
    {"id": "largest_planet_paraphrase_correct_shell", "question": "Which planet is the largest in the Solar System?", "answer": "The answer is Jupiter.", "expected_label": "Low"},
    {"id": "largest_planet_paraphrase_wrong_shell", "question": "Which planet is the largest in the Solar System?", "answer": "The answer is Mars.", "expected_label": "High"},
    {"id": "biggest_planet_question_correct_shell", "question": "Which planet is the biggest in the Solar System?", "answer": "The answer is Jupiter.", "expected_label": "Low"},
    {"id": "biggest_planet_question_wrong_shell", "question": "Which planet is the biggest in the Solar System?", "answer": "Mars.", "expected_label": "High"},
    {"id": "author_paraphrase_correct_sentence", "question": "Who is the author of Pride and Prejudice?", "answer": "Jane Austen wrote Pride and Prejudice.", "expected_label": "Low"},
    {"id": "author_paraphrase_wrong_sentence", "question": "Who is the author of Pride and Prejudice?", "answer": "Charles Dickens wrote Pride and Prejudice.", "expected_label": "High"},
    {"id": "author_name_question_correct_shell", "question": "Name the author of Pride and Prejudice.", "answer": "The author was Jane Austen.", "expected_label": "Low"},
    {"id": "author_name_question_wrong_shell", "question": "Name the author of Pride and Prejudice.", "answer": "The author was Charles Dickens.", "expected_label": "High"},
    {"id": "painter_paraphrase_correct_sentence", "question": "Who is the painter of The Persistence of Memory?", "answer": "Salvador Dali painted The Persistence of Memory.", "expected_label": "Low"},
    {"id": "painter_paraphrase_wrong_sentence", "question": "Who is the painter of The Persistence of Memory?", "answer": "Pablo Picasso painted The Persistence of Memory.", "expected_label": "High"},
    {"id": "painter_paraphrase_correct_role_shell", "question": "Who is the painter of The Persistence of Memory?", "answer": "The painter was Salvador Dali.", "expected_label": "Low"},
    {"id": "painter_paraphrase_wrong_role_shell", "question": "Who is the painter of The Persistence of Memory?", "answer": "Pablo Picasso was the painter.", "expected_label": "High"},
    {"id": "painter_name_question_correct_shell", "question": "Name the painter of The Persistence of Memory.", "answer": "The painter was Salvador Dali.", "expected_label": "Low"},
    {"id": "painter_name_question_wrong_shell", "question": "Name the painter of The Persistence of Memory.", "answer": "The painter was Pablo Picasso.", "expected_label": "High"},
    {
        "id": "obscure_marwick",
        "question": "Tell me about the 2019 Marwick Islands Peace Accord.",
        "answer": (
            "The 2019 Marwick Islands Peace Accord officially ended a 14-year naval conflict between "
            "Norland and Estavia, created a jointly administered demilitarized trade port at Sel Harbor, "
            "and later became a model for regional conflict resolution."
        ),
        "expected_label": "High",
    },
]

SEP_CASES = [
    {
        "id": "tokyo_consensus",
        "question": "What is the capital of Japan?",
        "answer": "Tokyo",
        "samples": "Tokyo\n\n---\n\nTokyo\n\n---\n\nTokyo",
        "expected_label": "Low",
    },
    {
        "id": "ottawa_consensus",
        "question": "What is the capital of Canada?",
        "answer": "Ottawa",
        "samples": "Ottawa\n\n---\n\nOttawa\n\n---\n\nOttawa",
        "expected_label": "Low",
    },
    {
        "id": "jupiter_consensus",
        "question": "What is the largest planet in the Solar System?",
        "answer": "Jupiter",
        "samples": "Jupiter\n\n---\n\nJupiter\n\n---\n\nJupiter",
        "expected_label": "Low",
    },
    {
        "id": "austen_consensus",
        "question": "Who wrote Pride and Prejudice?",
        "answer": "Jane Austen",
        "samples": "Jane Austen\n\n---\n\nJane Austen\n\n---\n\nAusten",
        "expected_label": "Low",
    },
    {
        "id": "canada_plurality_disagreement",
        "question": "What is the capital of Canada?",
        "answer": "Toronto",
        "samples": "Ottawa\n\n---\n\nToronto\n\n---\n\nOttawa",
        "expected_label": "High",
    },
    {
        "id": "planet_plurality_disagreement",
        "question": "What is the largest planet in the Solar System?",
        "answer": "Mars",
        "samples": "Jupiter\n\n---\n\nMars\n\n---\n\nJupiter",
        "expected_label": "High",
    },
    {
        "id": "author_plurality_disagreement",
        "question": "Who wrote Pride and Prejudice?",
        "answer": "Charles Dickens",
        "samples": "Jane Austen\n\n---\n\nCharles Dickens\n\n---\n\nJane Austen",
        "expected_label": "High",
    },
    {
        "id": "narev_suspicious_consensus",
        "question": "What happened during the 2020 Narev medical summit?",
        "answer": (
            "The 2020 Narev medical summit introduced the first universal clinical framework for AI-assisted surgery "
            "and was signed by 42 countries in Geneva."
        ),
        "samples": (
            "The 2020 Narev medical summit created a global AI surgery protocol signed by dozens of countries.\n\n---\n\n"
            "The Narev summit established international standards for robotic surgery in Geneva in 2020.\n\n---\n\n"
            "In 2020, the Narev medical summit unified global regulation for AI surgery across 42 nations."
        ),
        "expected_label": "High",
    },
    {
        "id": "narev_inconsistent_samples",
        "question": "What happened during the 2020 Narev medical summit?",
        "answer": (
            "The 2020 Narev medical summit introduced the first universal clinical framework for AI-assisted surgery "
            "and was signed by 42 countries in Geneva."
        ),
        "samples": (
            "The 2020 Narev medical summit was signed by 11 countries in Zurich.\n\n---\n\n"
            "The Narev summit banned AI-assisted surgery and ended without agreement in Vienna.\n\n---\n\n"
            "In 2021, the Narev medical summit proposed voluntary guidance for 7 nations in Geneva."
        ),
        "expected_label": "High",
    },
    {
        "id": "tokyo_sentence_consensus",
        "question": "What is the capital of Japan?",
        "answer": "The capital of Japan is Tokyo.",
        "samples": "Tokyo is the capital of Japan.\n\n---\n\nJapan's capital city is Tokyo.\n\n---\n\nThe capital of Japan is Tokyo.",
        "expected_label": "Low",
    },
    {
        "id": "japan_wrong_plurality",
        "question": "What is the capital of Japan?",
        "answer": "Ottawa",
        "samples": "Tokyo\n\n---\n\nOttawa\n\n---\n\nTokyo",
        "expected_label": "High",
    },
    {
        "id": "painter_plurality_disagreement",
        "question": "Who painted The Persistence of Memory?",
        "answer": "Pablo Picasso",
        "samples": "Salvador Dali\n\n---\n\nPablo Picasso\n\n---\n\nDali",
        "expected_label": "High",
    },
    {
        "id": "japan_possessive_consensus",
        "question": "What is Japan's capital?",
        "answer": "It is Tokyo.",
        "samples": "Tokyo\n\n---\n\nIt is Tokyo.\n\n---\n\nThe answer is Tokyo.",
        "expected_label": "Low",
    },
    {
        "id": "japan_possessive_plurality_disagreement",
        "question": "What is Japan's capital?",
        "answer": "It is Ottawa.",
        "samples": "Tokyo\n\n---\n\nOttawa\n\n---\n\nTokyo",
        "expected_label": "High",
    },
    {
        "id": "japan_city_question_consensus",
        "question": "What city is Japan's capital?",
        "answer": "It is Tokyo.",
        "samples": "Tokyo\n\n---\n\nIt is Tokyo.\n\n---\n\nTokyo is Japan's capital city.",
        "expected_label": "Low",
    },
    {
        "id": "japan_city_question_plurality_disagreement",
        "question": "What city is Japan's capital?",
        "answer": "Ottawa.",
        "samples": "Tokyo\n\n---\n\nOttawa\n\n---\n\nTokyo",
        "expected_label": "High",
    },
    {
        "id": "canada_name_question_consensus",
        "question": "Name Canada's capital city.",
        "answer": "The answer is Ottawa.",
        "samples": "Ottawa\n\n---\n\nCanada's capital city is Ottawa.\n\n---\n\nThe answer is Ottawa.",
        "expected_label": "Low",
    },
    {
        "id": "canada_name_question_plurality_disagreement",
        "question": "Name Canada's capital city.",
        "answer": "The answer is Toronto.",
        "samples": "Ottawa\n\n---\n\nToronto\n\n---\n\nOttawa",
        "expected_label": "High",
    },
    {
        "id": "largest_planet_paraphrase_consensus",
        "question": "Which planet is the largest in the Solar System?",
        "answer": "The answer is Jupiter.",
        "samples": "Jupiter\n\n---\n\nThe answer is Jupiter.\n\n---\n\nJupiter is the biggest planet in the Solar System.",
        "expected_label": "Low",
    },
    {
        "id": "largest_planet_paraphrase_plurality_disagreement",
        "question": "Which planet is the largest in the Solar System?",
        "answer": "The answer is Mars.",
        "samples": "Jupiter\n\n---\n\nMars\n\n---\n\nJupiter",
        "expected_label": "High",
    },
    {
        "id": "biggest_planet_question_consensus",
        "question": "Which planet is the biggest in the Solar System?",
        "answer": "The answer is Jupiter.",
        "samples": "Jupiter\n\n---\n\nThe answer is Jupiter.\n\n---\n\nJupiter is the biggest planet in the Solar System.",
        "expected_label": "Low",
    },
    {
        "id": "biggest_planet_question_plurality_disagreement",
        "question": "Which planet is the biggest in the Solar System?",
        "answer": "Mars.",
        "samples": "Jupiter\n\n---\n\nMars\n\n---\n\nJupiter",
        "expected_label": "High",
    },
    {
        "id": "author_paraphrase_consensus",
        "question": "Who is the author of Pride and Prejudice?",
        "answer": "Jane Austen wrote Pride and Prejudice.",
        "samples": "Jane Austen\n\n---\n\nAusten\n\n---\n\nThe author was Jane Austen.",
        "expected_label": "Low",
    },
    {
        "id": "author_paraphrase_plurality_disagreement",
        "question": "Who is the author of Pride and Prejudice?",
        "answer": "Charles Dickens wrote Pride and Prejudice.",
        "samples": "Jane Austen\n\n---\n\nCharles Dickens\n\n---\n\nAusten",
        "expected_label": "High",
    },
    {
        "id": "author_name_question_consensus",
        "question": "Name the author of Pride and Prejudice.",
        "answer": "The author was Jane Austen.",
        "samples": "Jane Austen\n\n---\n\nThe author was Jane Austen.\n\n---\n\nAusten",
        "expected_label": "Low",
    },
    {
        "id": "author_name_question_plurality_disagreement",
        "question": "Name the author of Pride and Prejudice.",
        "answer": "The author was Charles Dickens.",
        "samples": "Jane Austen\n\n---\n\nCharles Dickens\n\n---\n\nAusten",
        "expected_label": "High",
    },
    {
        "id": "painter_paraphrase_consensus_sentence",
        "question": "Who is the painter of The Persistence of Memory?",
        "answer": "Salvador Dali painted The Persistence of Memory.",
        "samples": "Salvador Dali\n\n---\n\nDali\n\n---\n\nThe painter was Salvador Dali.",
        "expected_label": "Low",
    },
    {
        "id": "painter_paraphrase_plurality_disagreement_sentence",
        "question": "Who is the painter of The Persistence of Memory?",
        "answer": "Pablo Picasso painted The Persistence of Memory.",
        "samples": "Salvador Dali\n\n---\n\nPablo Picasso\n\n---\n\nDali",
        "expected_label": "High",
    },
    {
        "id": "painter_paraphrase_plurality_disagreement_role_shell",
        "question": "Who is the painter of The Persistence of Memory?",
        "answer": "Pablo Picasso was the painter.",
        "samples": "The painter was Salvador Dali.\n\n---\n\nPablo Picasso was the painter.\n\n---\n\nDali",
        "expected_label": "High",
    },
    {
        "id": "painter_name_question_consensus",
        "question": "Name the painter of The Persistence of Memory.",
        "answer": "The painter was Salvador Dali.",
        "samples": "Salvador Dali\n\n---\n\nThe painter was Salvador Dali.\n\n---\n\nDali",
        "expected_label": "Low",
    },
    {
        "id": "painter_name_question_plurality_disagreement",
        "question": "Name the painter of The Persistence of Memory.",
        "answer": "The painter was Pablo Picasso.",
        "samples": "Salvador Dali\n\n---\n\nPablo Picasso\n\n---\n\nDali",
        "expected_label": "High",
    },
]

GROUNDING_SCENARIOS = [
    {
        "id": "museum",
        "question": "Summarize the museum schedule.",
        "source_text": "The museum is open Tuesday through Sunday from 10 AM to 5 PM. It is closed on Mondays. General admission is $12 for adults and $8 for students.",
        "evidence_text": "",
        "low_answer": "The museum is open Tuesday through Sunday from 10 AM to 5 PM, is closed on Mondays, and charges admission with discounts for students.",
        "high_answer": "The museum is open every day from 8 AM to 8 PM, and entry is free for everyone.",
    },
    {
        "id": "greenfield",
        "question": "Summarize the Greenfield transit pilot.",
        "source_text": "Greenfield ran a four-month transit pilot in 8 neighborhoods. City staff reported average commute times fell by about 6%. Officials extended the pilot for another quarter, but no permanent rollout has been approved yet.",
        "evidence_text": "",
        "low_answer": "Greenfield ran a pilot in 8 neighborhoods, reduced commute times modestly, and extended the program rather than permanently approving it.",
        "high_answer": "Greenfield ran a six-month pilot in 12 neighborhoods, cut commute times by 18%, and the city approved a permanent rollout immediately.",
    },
    {
        "id": "riverside",
        "question": "Using the retrieved notes, what were the main outcomes of the 2021 Riverside school lunch pilot?",
        "source_text": "",
        "evidence_text": "Evidence note 1: The Riverside school lunch pilot ran from March to June 2021 in 8 public schools. Evidence note 2: District records showed average attendance rose by 1.2% during the pilot period. Evidence note 3: Teachers reported fewer afternoon fatigue complaints among students. Evidence note 4: The school board voted to extend the pilot for one additional semester, but no districtwide permanent rollout was approved in 2022.",
        "low_answer": "The 2021 Riverside school lunch pilot ran from March to June in 8 public schools. District records showed attendance rose by about 1.2%. Teachers reported fewer afternoon fatigue complaints, and the board extended the pilot for one additional semester rather than approving a permanent districtwide rollout.",
        "high_answer": "The 2021 Riverside school lunch pilot provided free lunches in 12 schools for six months, increased student attendance by about 4%, reduced nurse visits related to hunger, and convinced the district to approve a permanent citywide rollout in early 2022.",
    },
    {
        "id": "harbor",
        "question": "Summarize the Harbor cleanup update.",
        "source_text": "The Harbor cleanup entered a second remediation phase in March. Officials said contamination levels fell, but some hotspots remain under treatment. The waterfront remains closed to swimmers pending further testing.",
        "evidence_text": "Meeting note: Inspectors said the cleanup is progressing, but final clearance has not been issued.",
        "low_answer": "The Harbor cleanup entered a second remediation phase in March, contamination improved but some hotspots remain, and the waterfront is still closed pending more testing.",
        "high_answer": "The Harbor cleanup finished in March, removed all contamination, and reopened the waterfront for swimming immediately.",
    },
    {
        "id": "pinecrest",
        "question": "What did the Pinecrest housing memo approve?",
        "source_text": "The Pinecrest housing memo approved a 12-story building with 180 mixed-income units. The earliest construction start is late 2025 after permit review.",
        "evidence_text": "Board note: The January 2025 start date from an early draft was withdrawn.",
        "low_answer": "The Pinecrest housing memo approved a 12-story building with 180 mixed-income units. The earliest construction start is late 2025 after permit review.",
        "high_answer": "The Pinecrest memo approved a 20-story tower with 300 units and said construction starts in January 2025.",
    },
    {
        "id": "midtown",
        "question": "According to the clinic expansion memo, what did the board approve for Midtown Clinic?",
        "source_text": "The Midtown Clinic board approved a two-story outpatient expansion with 18 observation beds, diagnostic rooms, and a larger pharmacy area. The plan does not include a new emergency department. Administrators said the earliest opening target is late 2026, pending permits and contractor bids.",
        "evidence_text": "Meeting note: Board members emphasized that the January 2026 date from an early draft is no longer active. Meeting note: The project is still awaiting zoning review and final permit approval.",
        "low_answer": "The board approved a two-story outpatient expansion with 18 observation beds, diagnostic rooms, and a larger pharmacy area. It did not approve a new emergency department, and the earliest opening target is late 2026 pending permits.",
        "high_answer": "The board approved a three-story expansion with 24 inpatient beds, a new emergency wing, and said the building opens in January 2026.",
    },
    {
        "id": "northbridge",
        "question": "According to the renovation notes, what did the board decide about the Northbridge library project?",
        "source_text": "",
        "evidence_text": "Planning note 1: The board approved roof and HVAC replacement for the west wing rather than demolition. Planning note 2: Administrators said the earliest reopening target for the west wing is spring 2026. Planning note 3: The children's room will remain open in temporary space during construction. Planning note 4: The cafe proposal was deferred and not approved in this phase.",
        "low_answer": "The board approved roof and HVAC replacement for the west wing rather than demolition. The earliest reopening target is spring 2026, the children's room will remain open in temporary space, and the cafe proposal was deferred to a later phase.",
        "high_answer": "The board approved demolition of the west wing, said it would reopen in summer 2025, moved the children's room fully offsite, and approved a cafe in the first phase.",
    },
    {
        "id": "solaris",
        "question": "What did the product briefing claim about the Solaris X battery pack?",
        "source_text": "",
        "evidence_text": "Briefing note: The 2022 Harbor Energy test report measured the Solaris X at 12 hours of continuous output under lab conditions. Regulatory note: The battery pack has not received FAA approval and is not certified for in-flight use. Supplier note: The battery pack is manufactured by Solstice Power, while Aerodyne Systems supplies the control software.",
        "low_answer": "The 2022 Harbor Energy test report measured the Solaris X at 12 hours of continuous output under lab conditions. The battery pack has not received FAA approval for in-flight use. It is manufactured by Solstice Power, and Aerodyne Systems supplies the control software.",
        "high_answer": "The Solaris X lasted 18 hours, is FAA-approved, and was manufactured entirely by Aerodyne Systems.",
    },
]


def _combined_source_text(scenario: dict) -> str:
    return " ".join(part for part in [scenario["source_text"], scenario["evidence_text"]] if part)


@patch("utils.text_utils.get_st_model", return_value=None)
def _run_source(scenario: dict, answer: str, _mock_model=None) -> dict:
    return run_source(
        question=scenario["question"],
        answer=answer,
        source_text=_combined_source_text(scenario),
    )


def _run_retrieval(scenario: dict, answer: str) -> dict:
    return run_retrieval_check(
        question=scenario["question"],
        answer=answer,
        source_text=scenario["source_text"],
        evidence_text=scenario["evidence_text"],
        method_name="Retrieval-Grounded Checker",
        family="retrieval-grounded",
        preferred_backend="tfidf",
        impl_status="implemented",
    )


def _run_rag_grounded(scenario: dict, answer: str) -> dict:
    return run_retrieval_check(
        question=scenario["question"],
        answer=answer,
        source_text=scenario["source_text"],
        evidence_text=scenario["evidence_text"],
        method_name="RAG Grounded Check",
        family="retrieval-grounded / RAG-style",
        preferred_backend="tfidf",
        impl_status="approximate",
    )


def _run_cove(scenario: dict, answer: str) -> dict:
    return run_cove_detector(
        question=scenario["question"],
        answer=answer,
        source_text=scenario["source_text"],
        evidence_text=scenario["evidence_text"],
        method_name="CoVe-Style Verification",
        preferred_backend="tfidf",
    )


def _run_critic(scenario: dict, answer: str) -> dict:
    return run_critic_detector(
        question=scenario["question"],
        answer=answer,
        source_text=scenario["source_text"],
        evidence_text=scenario["evidence_text"],
        method_name="CRITIC-lite Tool Check",
        preferred_backend="tfidf",
    )


def _run_verification(scenario: dict, answer: str) -> dict:
    return run_verify(
        question=scenario["question"],
        answer=answer,
        source_text=scenario["source_text"],
        evidence_text=scenario["evidence_text"],
        allow_web=False,
    )


GROUNDING_RUNNERS = {
    "Source-Grounded Consistency": _run_source,
    "Retrieval-Grounded Checker": _run_retrieval,
    "RAG Grounded Check": _run_rag_grounded,
    "CoVe-Style Verification": _run_cove,
    "CRITIC-lite Tool Check": _run_critic,
    "Verification-Based Workflow": _run_verification,
}


class DetectorRegressions(unittest.TestCase):
    @patch("detectors.signal._extract_features", side_effect=RuntimeError("backend offline"))
    def test_fact_bank_errors(self, _mock_extract):
        for case in BASELINE_CASES:
            with self.subTest(method="Internal-Signal Baseline", case=case["id"]):
                result = run_internal(question=case["question"], answer=case["answer"])
                self.assertEqual(result["risk_label"], case["expected_label"])

    @patch("detectors.signal._extract_features", side_effect=RuntimeError("backend offline"))
    def test_sep_consensus(self, _mock_extract):
        for case in SEP_CASES:
            with self.subTest(method="SEP-Inspired Internal Signal", case=case["id"]):
                result = run_sep(
                    question=case["question"],
                    answer=case["answer"],
                    sampled_answers_text=case["samples"],
                )
                self.assertEqual(result["risk_label"], case["expected_label"])

    def test_grounded_split(self):
        total_asserted_cases = 0
        for method_name, runner in GROUNDING_RUNNERS.items():
            for scenario in GROUNDING_SCENARIOS:
                with self.subTest(method=method_name, scenario=scenario["id"], expected="Low"):
                    low_result = runner(scenario, scenario["low_answer"])
                    self.assertEqual(low_result["risk_label"], "Low")
                    total_asserted_cases += 1
                with self.subTest(method=method_name, scenario=scenario["id"], expected="High"):
                    high_result = runner(scenario, scenario["high_answer"])
                    self.assertEqual(high_result["risk_label"], "High")
                    total_asserted_cases += 1

        total_case_count = total_asserted_cases + len(BASELINE_CASES) + len(SEP_CASES)
        self.assertGreaterEqual(total_case_count, 100)

    @patch("detectors.signal._extract_features", side_effect=RuntimeError("backend offline"))
    def test_fact_fallbacks(self, _mock_extract):
        for question, answer in [
            ("Where is the capital of Canada?", "Toronto"),
            ("What is the largest planet in the Solar System?", "Mars"),
            ("Who wrote Pride and Prejudice?", "Charles Dickens"),
            ("What is Japan's capital?", "It is Ottawa."),
            ("What city is Japan's capital?", "Ottawa."),
            ("Name Canada's capital city.", "The answer is Toronto."),
            ("Which planet is the largest in the Solar System?", "The answer is Mars."),
            ("Which planet is the biggest in the Solar System?", "Mars."),
            ("Who is the author of Pride and Prejudice?", "Charles Dickens wrote Pride and Prejudice."),
            ("Name the author of Pride and Prejudice.", "The author was Charles Dickens."),
            ("Who is the painter of The Persistence of Memory?", "Pablo Picasso painted The Persistence of Memory."),
            ("Name the painter of The Persistence of Memory.", "The painter was Pablo Picasso."),
        ]:
            baseline = run_internal(question=question, answer=answer)
            self.assertEqual(baseline["risk_label"], "High")
            self.assertIsNotNone(baseline["metadata"]["simple_fact_sanity"])
            self.assertEqual(baseline["metadata"]["simple_fact_sanity"]["verdict"], "incorrect")

        for question, answer, samples in [
            ("What is the capital of Canada?", "Toronto", "Ottawa\n\n---\n\nToronto\n\n---\n\nOttawa"),
            ("What city is Japan's capital?", "Ottawa.", "Tokyo\n\n---\n\nOttawa\n\n---\n\nTokyo"),
            ("Name Canada's capital city.", "The answer is Toronto.", "Ottawa\n\n---\n\nToronto\n\n---\n\nOttawa"),
            ("Which planet is the largest in the Solar System?", "The answer is Mars.", "Jupiter\n\n---\n\nMars\n\n---\n\nJupiter"),
            ("Which planet is the biggest in the Solar System?", "Mars.", "Jupiter\n\n---\n\nMars\n\n---\n\nJupiter"),
            ("Name the author of Pride and Prejudice.", "The author was Charles Dickens.", "Jane Austen\n\n---\n\nCharles Dickens\n\n---\n\nAusten"),
            ("Who is the painter of The Persistence of Memory?", "Pablo Picasso painted The Persistence of Memory.", "Salvador Dali\n\n---\n\nPablo Picasso\n\n---\n\nDali"),
            ("Name the painter of The Persistence of Memory.", "The painter was Pablo Picasso.", "Salvador Dali\n\n---\n\nPablo Picasso\n\n---\n\nDali"),
        ]:
            sep_result = run_sep(
                question=question,
                answer=answer,
                sampled_answers_text=samples,
            )
            self.assertEqual(sep_result["risk_label"], "High")
            self.assertTrue(sep_result["metadata"]["main_answer_conflicts_with_sample_plurality"])
            self.assertIsNotNone(sep_result["metadata"]["simple_fact_sanity"])

    def test_conflicts(self):
        museum = next(s for s in GROUNDING_SCENARIOS if s["id"] == "museum")
        midtown = next(s for s in GROUNDING_SCENARIOS if s["id"] == "midtown")
        northbridge = next(s for s in GROUNDING_SCENARIOS if s["id"] == "northbridge")
        solaris = next(s for s in GROUNDING_SCENARIOS if s["id"] == "solaris")

        for method_name, runner in GROUNDING_RUNNERS.items():
            with self.subTest(method=method_name, scenario="museum"):
                self.assertEqual(runner(museum, museum["high_answer"])["risk_label"], "High")
            with self.subTest(method=method_name, scenario="midtown"):
                self.assertEqual(runner(midtown, midtown["high_answer"])["risk_label"], "High")
            with self.subTest(method=method_name, scenario="northbridge"):
                self.assertEqual(runner(northbridge, northbridge["high_answer"])["risk_label"], "High")
            with self.subTest(method=method_name, scenario="solaris"):
                self.assertEqual(runner(solaris, solaris["high_answer"])["risk_label"], "High")

    def test_restrictions(self):
        low_cases = [
            {
                "id": "off_peak_only",
                "question": "Summarize the ferry bicycle restriction.",
                "evidence": "Peak weekday commute sailings do not permit bicycles.",
                "answer": "Bicycles are limited to off-peak sailings.",
            },
            {
                "id": "weekdays_only",
                "question": "Summarize the archive access rule.",
                "evidence": "The archive is not available on weekends.",
                "answer": "The archive is limited to weekdays only.",
            },
            {
                "id": "members_only",
                "question": "Summarize the lounge admission policy.",
                "evidence": "Non-members are not allowed in the lounge.",
                "answer": "The lounge is members only.",
            },
            {
                "id": "not_on_weekends",
                "question": "Summarize the pool schedule.",
                "evidence": "The community pool is open on weekdays only.",
                "answer": "The community pool is not open on weekends.",
            },
            {
                "id": "not_during_peak",
                "question": "Summarize the bicycle boarding rule.",
                "evidence": "Bicycle boarding is limited to off-peak sailings.",
                "answer": "Bicycles are not allowed during peak sailings.",
            },
        ]

        high_cases = [
            {
                "id": "wrong_date",
                "question": "When does the west wing reopen?",
                "evidence": "Administrators said the earliest reopening target for the west wing is spring 2026.",
                "answer": "The west wing reopens in summer 2025.",
            },
            {
                "id": "wrong_capacity",
                "question": "What capacity did the board approve?",
                "evidence": "The Midtown Clinic board approved 18 observation beds.",
                "answer": "The board approved 24 observation beds.",
            },
            {
                "id": "wrong_location",
                "question": "Where will the satellite office open?",
                "evidence": "The satellite office will open in Northbridge.",
                "answer": "The satellite office will open in Midtown.",
            },
            {
                "id": "wrong_approval_status",
                "question": "What is the certification status?",
                "evidence": "The battery pack has not received FAA approval and is not certified for in-flight use.",
                "answer": "The battery pack is FAA-approved for in-flight use.",
            },
            {
                "id": "wrong_manufacturer",
                "question": "Who manufactures the battery pack?",
                "evidence": "The battery pack is manufactured by Solstice Power, while Aerodyne Systems supplies the control software.",
                "answer": "The battery pack is manufactured entirely by Aerodyne Systems.",
            },
        ]

        for method_name, runner in GROUNDING_RUNNERS.items():
            for case in low_cases:
                scenario = {
                    "question": case["question"],
                    "source_text": "",
                    "evidence_text": case["evidence"],
                }
                with self.subTest(method=method_name, scenario=case["id"], expected="Low"):
                    self.assertEqual(runner(scenario, case["answer"])["risk_label"], "Low")
            for case in high_cases:
                scenario = {
                    "question": case["question"],
                    "source_text": "",
                    "evidence_text": case["evidence"],
                }
                with self.subTest(method=method_name, scenario=case["id"], expected="High"):
                    self.assertEqual(runner(scenario, case["answer"])["risk_label"], "High")

    def test_cove_date(self):
        pinecrest = next(s for s in GROUNDING_SCENARIOS if s["id"] == "pinecrest")
        result = _run_cove(pinecrest, pinecrest["high_answer"])
        self.assertIn("late 2025", result["revised_answer"])


if __name__ == "__main__":
    unittest.main()
