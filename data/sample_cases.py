"""Demo cases used by the sample browser."""


SAMPLE_CASES = [
    {
        "id": "internal_baseline_low_tokyo",
        "pair_id": "internal_baseline_pair",
        "method_targets": ["Internal-Signal Baseline"],
        "risk_level": "low",
        "title": "Tokyo sanity check",
        "description": "Short common-fact answer used as a sanity check.",
        "question": "What is the capital of Japan?",
        "answer": "The capital of Japan is Tokyo.",
        "answer_samples": "",
        "source_text": "",
        "evidence_text": "",
        "expected_label": "Low",
        "notes": (
            "Backend runs and fallback runs both treat this as low risk; fallback confidence "
            "stays modest because there is no evidence check."
        ),
    },
    {
        "id": "internal_baseline_high_marwick",
        "pair_id": "internal_baseline_pair",
        "method_targets": ["Internal-Signal Baseline"],
        "risk_level": "high",
        "title": "Unsupported Marwick treaty claim",
        "description": "A confident treaty-style answer with names, dates, and outcomes but no supporting source.",
        "question": "Tell me about the 2019 Marwick Islands Peace Accord.",
        "answer": (
            "The 2019 Marwick Islands Peace Accord officially ended a 14-year naval conflict between "
            "Norland and Estavia, created a jointly administered demilitarized trade port at Sel Harbor, "
            "and later became a model for regional conflict resolution."
        ),
        "answer_samples": "",
        "source_text": "",
        "evidence_text": "",
        "expected_label": "High",
        "notes": "Useful high-risk internal-signal case: specific, confident, and unsupported.",
    },
    {
        "id": "sep_low_tokyo_consensus",
        "pair_id": "sep_pair",
        "method_targets": ["SEP-Inspired Internal Signal"],
        "risk_level": "low",
        "title": "Stable Tokyo samples",
        "description": "Several sampled answers land on the same simple fact.",
        "question": "What is the capital of Japan?",
        "answer": "The capital of Japan is Tokyo.",
        "answer_samples": (
            "Tokyo is the capital of Japan.\n\n---\n\n"
            "Japan's capital city is Tokyo.\n\n---\n\n"
            "The capital of Japan is Tokyo."
        ),
        "source_text": "",
        "evidence_text": "",
        "expected_label": "Low",
        "notes": "Expected result is low risk driven by strong sample consistency.",
    },
    {
        "id": "sep_high_narev_consensus",
        "pair_id": "sep_pair",
        "method_targets": ["SEP-Inspired Internal Signal"],
        "risk_level": "high",
        "title": "Narev consensus trap",
        "description": "The samples agree with each other while repeating a highly specific invented-sounding event.",
        "question": "What happened during the 2020 Narev medical summit?",
        "answer": (
            "The 2020 Narev medical summit introduced the first universal clinical framework for AI-assisted "
            "surgery and was signed by 42 countries in Geneva."
        ),
        "answer_samples": (
            "The 2020 Narev medical summit created a global AI surgery protocol signed by dozens of countries.\n\n---\n\n"
            "The Narev summit established international standards for robotic surgery in Geneva in 2020.\n\n---\n\n"
            "In 2020, the Narev medical summit unified global regulation for AI surgery across 42 nations."
        ),
        "source_text": "",
        "evidence_text": "",
        "expected_label": "High",
        "notes": "Useful for SEP fallback because the risk comes from suspicious consensus, not sample disagreement.",
    },
    {
        "id": "source_grounded_low_midtown",
        "pair_id": "source_grounded_pair",
        "method_targets": ["Source-Grounded Consistency"],
        "risk_level": "low",
        "title": "Midtown memo summary",
        "description": "The answer follows the memo without adding new facts.",
        "question": "According to the clinic expansion memo, what did the board approve for Midtown Clinic?",
        "answer": (
            "The board approved a two-story outpatient expansion with 18 observation beds, diagnostic rooms, "
            "and a larger pharmacy area. It did not approve a new emergency department, and the earliest "
            "opening target is late 2026 pending permits."
        ),
        "answer_samples": "",
        "source_text": (
            "The Midtown Clinic board approved a two-story outpatient expansion with 18 observation beds, "
            "diagnostic rooms, and a larger pharmacy area. The plan does not include a new emergency "
            "department. Administrators said the earliest opening target is late 2026, pending permits "
            "and contractor bids."
        ),
        "evidence_text": "",
        "expected_label": "Low",
        "notes": "Low-risk source-grounded case with direct source support.",
    },
    {
        "id": "source_grounded_high_midtown",
        "pair_id": "source_grounded_pair",
        "method_targets": ["Source-Grounded Consistency"],
        "risk_level": "high",
        "title": "Midtown unsupported additions",
        "description": "The answer adds beds, a new emergency wing, and an earlier opening date not supported by the source.",
        "question": "According to the clinic expansion memo, what did the board approve for Midtown Clinic?",
        "answer": (
            "The board approved a three-story expansion with 24 inpatient beds and a new emergency wing, "
            "and it said the new building will open in January 2026."
        ),
        "answer_samples": "",
        "source_text": (
            "The Midtown Clinic board approved a two-story outpatient expansion with 18 observation beds, "
            "diagnostic rooms, and a larger pharmacy area. The plan does not include a new emergency "
            "department. Administrators said the earliest opening target is late 2026, pending permits "
            "and contractor bids."
        ),
        "evidence_text": "",
        "expected_label": "High",
        "notes": "Contradiction and unsupported-detail findings are the important signals here.",
    },
    {
        "id": "retrieval_low_riverside",
        "pair_id": "retrieval_pair",
        "method_targets": ["Retrieval-Grounded Checker"],
        "risk_level": "low",
        "title": "Riverside notes match",
        "description": "The answer is close to the notes the retrieval path can cite.",
        "question": "Using the retrieved notes, what were the main outcomes of the 2021 Riverside school lunch pilot?",
        "answer": (
            "The 2021 Riverside school lunch pilot ran from March to June in 8 public schools. District "
            "records showed attendance rose by about 1.2%, teachers reported fewer afternoon fatigue "
            "complaints, and the board extended the pilot for one additional semester rather than "
            "approving a permanent districtwide rollout."
        ),
        "answer_samples": "",
        "source_text": "",
        "evidence_text": (
            "Evidence note 1: The Riverside school lunch pilot ran from March to June 2021 in 8 public schools.\n"
            "Evidence note 2: District records showed average attendance rose by 1.2% during the pilot period.\n"
            "Evidence note 3: Teachers reported fewer afternoon fatigue complaints among students.\n"
            "Evidence note 4: The school board voted to extend the pilot for one additional semester, but no "
            "districtwide permanent rollout was approved in 2022."
        ),
        "expected_label": "Low",
        "notes": "Citation-backed support is the main thing to inspect.",
    },
    {
        "id": "retrieval_high_riverside",
        "pair_id": "retrieval_pair",
        "method_targets": ["Retrieval-Grounded Checker"],
        "risk_level": "high",
        "title": "Riverside notes conflict",
        "description": "The answer changes counts, effects, and policy outcomes from the notes.",
        "question": "Using the retrieved notes, what were the main outcomes of the 2021 Riverside school lunch pilot?",
        "answer": (
            "The 2021 Riverside school lunch pilot provided free lunches in 12 schools for six months, "
            "increased student attendance by about 4%, reduced nurse visits related to hunger, and convinced "
            "the district to approve a permanent citywide rollout in early 2022."
        ),
        "answer_samples": "",
        "source_text": "",
        "evidence_text": (
            "Evidence note 1: The Riverside school lunch pilot ran from March to June 2021 in 8 public schools.\n"
            "Evidence note 2: District records showed average attendance rose by 1.2% during the pilot period.\n"
            "Evidence note 3: Teachers reported fewer afternoon fatigue complaints among students.\n"
            "Evidence note 4: The school board voted to extend the pilot for one additional semester, but no "
            "districtwide permanent rollout was approved in 2022."
        ),
        "expected_label": "High",
        "notes": "High-risk retrieval case with count, effect, and rollout conflicts.",
    },
    {
        "id": "rag_low_harbor",
        "pair_id": "rag_pair",
        "method_targets": ["RAG Grounded Check"],
        "risk_level": "low",
        "title": "Harbor dossier match",
        "description": "The answer follows the local dossier.",
        "question": "Using the project dossier, summarize what funding the Harbor District flood plan received and when construction begins.",
        "answer": (
            "In March 2024, Harbor District received an $8.5 million state planning grant. The grant "
            "supports floodgate design and drainage upgrades. Final construction has not yet been approved. "
            "Officials said physical construction would start no earlier than late 2025."
        ),
        "answer_samples": "",
        "source_text": "",
        "evidence_text": (
            "Dossier note 1: In March 2024, the state resilience office awarded Harbor District an $8.5 million "
            "planning grant for floodgate design and drainage upgrades.\n"
            "Dossier note 2: The project has not yet been approved for final construction; engineering and "
            "environmental review will continue through early 2025.\n"
            "Dossier note 3: Officials said no permanent seawall contract has been signed, and any physical "
            "construction would start no earlier than late 2025."
        ),
        "expected_label": "Low",
        "notes": "Low-risk grounded case with funding and timing details preserved.",
    },
    {
        "id": "rag_high_harbor",
        "pair_id": "rag_pair",
        "method_targets": ["RAG Grounded Check"],
        "risk_level": "high",
        "title": "Harbor dossier mismatch",
        "description": "The answer invents federal grant details and an earlier construction start.",
        "question": "Using the project dossier, summarize what funding the Harbor District flood plan received and when construction begins.",
        "answer": (
            "The Harbor District received a $12 million federal resilience grant in February 2024 to build a "
            "permanent seawall, and construction will begin in April 2024."
        ),
        "answer_samples": "",
        "source_text": "",
        "evidence_text": (
            "Dossier note 1: In March 2024, the state resilience office awarded Harbor District an $8.5 million "
            "planning grant for floodgate design and drainage upgrades.\n"
            "Dossier note 2: The project has not yet been approved for final construction; engineering and "
            "environmental review will continue through early 2025.\n"
            "Dossier note 3: Officials said no permanent seawall contract has been signed, and any physical "
            "construction would start no earlier than late 2025."
        ),
        "expected_label": "High",
        "notes": "High-risk RAG-style check with funding source and timing drift.",
    },
    {
        "id": "verification_low_northbridge",
        "pair_id": "verification_pair",
        "method_targets": ["Verification-Based Workflow"],
        "risk_level": "low",
        "title": "Northbridge verification pass",
        "description": "The answer matches the evidence notes.",
        "question": "According to the renovation notes, what did the board decide about the Northbridge library project?",
        "answer": (
            "The board approved roof and HVAC replacement for the west wing rather than demolition. "
            "The earliest reopening target is spring 2026, the children's room will remain open in "
            "temporary space, and the cafe proposal was deferred to a later phase."
        ),
        "answer_samples": "",
        "source_text": "",
        "evidence_text": (
            "Planning note 1: The board approved roof and HVAC replacement for the west wing rather than demolition.\n"
            "Planning note 2: Administrators said the earliest reopening target for the west wing is spring 2026.\n"
            "Planning note 3: The children's room will remain open in temporary space during construction.\n"
            "Planning note 4: The cafe proposal was deferred and not approved in this phase."
        ),
        "expected_label": "Low",
        "notes": "The staged trace resolves each evidence note cleanly.",
    },
    {
        "id": "verification_high_northbridge",
        "pair_id": "verification_pair",
        "method_targets": ["Verification-Based Workflow"],
        "risk_level": "high",
        "title": "Northbridge verification fail",
        "description": "The answer conflicts with the notes on demolition, timing, children's room access, and cafe approval.",
        "question": "According to the renovation notes, what did the board decide about the Northbridge library project?",
        "answer": (
            "The board approved a full closure and demolition of the west wing, said the project would reopen "
            "in summer 2025, moved the children's room entirely offsite, and approved a new cafe in the first phase."
        ),
        "answer_samples": "",
        "source_text": "",
        "evidence_text": (
            "Planning note 1: The board approved roof and HVAC replacement for the west wing rather than demolition.\n"
            "Planning note 2: Administrators said the earliest reopening target for the west wing is spring 2026.\n"
            "Planning note 3: The children's room will remain open in temporary space during construction.\n"
            "Planning note 4: The cafe proposal was deferred and not approved in this phase."
        ),
        "expected_label": "High",
        "notes": "Verification-baseline failure case with several independent conflicts.",
    },
    {
        "id": "cove_low_midtown",
        "pair_id": "cove_pair",
        "method_targets": ["CoVe-Style Verification"],
        "risk_level": "low",
        "title": "CoVe keeps the Midtown answer",
        "description": "The answer is already supported; the revision stage mostly leaves it alone.",
        "question": "According to the clinic expansion memo, what did the board approve for Midtown Clinic?",
        "answer": (
            "The board approved a two-story outpatient expansion with 18 observation beds, diagnostic rooms, "
            "and a larger pharmacy area. It did not approve a new emergency department, and the earliest "
            "opening target is late 2026 pending permits."
        ),
        "answer_samples": "",
        "source_text": (
            "The Midtown Clinic board approved a two-story outpatient expansion with 18 observation beds, "
            "diagnostic rooms, and a larger pharmacy area. The plan does not include a new emergency "
            "department. Administrators said the earliest opening target is late 2026, pending permits "
            "and contractor bids."
        ),
        "evidence_text": (
            "Meeting note: Board members emphasized that the January 2026 date from an early draft is no longer active.\n"
            "Meeting note: The project is still awaiting zoning review and final permit approval."
        ),
        "expected_label": "Low",
        "notes": "The useful check is whether the revision stays close to the supported answer.",
    },
    {
        "id": "cove_high_midtown",
        "pair_id": "cove_pair",
        "method_targets": ["CoVe-Style Verification"],
        "risk_level": "high",
        "title": "CoVe rewrites the Midtown answer",
        "description": "The answer has several wrong details for the revision stage to replace.",
        "question": "According to the clinic expansion memo, what did the board approve for Midtown Clinic?",
        "answer": (
            "The board approved a three-story expansion with 24 inpatient beds and a new emergency wing, "
            "and it said the new building will open in January 2026."
        ),
        "answer_samples": "",
        "source_text": (
            "The Midtown Clinic board approved a two-story outpatient expansion with 18 observation beds, "
            "diagnostic rooms, and a larger pharmacy area. The plan does not include a new emergency "
            "department. Administrators said the earliest opening target is late 2026, pending permits "
            "and contractor bids."
        ),
        "evidence_text": (
            "Meeting note: Board members emphasized that the January 2026 date from an early draft is no longer active.\n"
            "Meeting note: The project is still awaiting zoning review and final permit approval."
        ),
        "expected_label": "High",
        "notes": "The useful check is whether the revised answer stays concise and grounded.",
    },
    {
        "id": "critic_low_solaris",
        "pair_id": "critic_pair",
        "method_targets": ["CRITIC-lite Tool Check"],
        "risk_level": "low",
        "title": "Solaris tool-check pass",
        "description": "The answer matches the local product notes and passes retrieval and numeric checks.",
        "question": "What did the product briefing claim about the Solaris X battery pack?",
        "answer": (
            "The 2022 Harbor Energy test report measured the Solaris X at 12 hours of continuous output under "
            "lab conditions. The battery pack has not received FAA approval for in-flight use. It is manufactured "
            "by Solstice Power, while Aerodyne Systems supplies the control software."
        ),
        "answer_samples": "",
        "source_text": "",
        "evidence_text": (
            "Briefing note: The 2022 Harbor Energy test report measured the Solaris X at 12 hours of continuous output under lab conditions.\n"
            "Regulatory note: The battery pack has not received FAA approval and is not certified for in-flight use.\n"
            "Supplier note: The battery pack is manufactured by Solstice Power, while Aerodyne Systems supplies the control software."
        ),
        "expected_label": "Low",
        "notes": "Tool-check trace with a low-risk outcome.",
    },
    {
        "id": "critic_high_solaris",
        "pair_id": "critic_pair",
        "method_targets": ["CRITIC-lite Tool Check"],
        "risk_level": "high",
        "title": "Solaris tool-check fail",
        "description": "The answer gets battery life, FAA status, and manufacturer wrong.",
        "question": "What did the product briefing claim about the Solaris X battery pack?",
        "answer": (
            "According to the 2022 Harbor Energy study, the Solaris X battery pack lasted 18 hours. "
            "The study said the company had already received FAA approval for in-flight use. "
            "The device was manufactured by Aerodyne Systems."
        ),
        "answer_samples": "",
        "source_text": "",
        "evidence_text": (
            "Briefing note: The 2022 Harbor Energy test report measured the Solaris X at 12 hours of continuous output under lab conditions.\n"
            "Regulatory note: The battery pack has not received FAA approval and is not certified for in-flight use.\n"
            "Supplier note: The battery pack is manufactured by Solstice Power, while Aerodyne Systems supplies the control software."
        ),
        "expected_label": "High",
        "notes": "High-risk CRITIC-lite case with numeric, approval, and manufacturer errors.",
    },
]


LEGACY_CASE_ALIASES = {
    "multi_answer_instability": "sep_low_tokyo_consensus",
    "sep_uncertainty_probe": "sep_high_narev_consensus",
    "grounded_summarization": "source_grounded_high_midtown",
    "simple_no_evidence": "internal_baseline_low_tokyo",
    "overconfident_no_evidence": "internal_baseline_high_marwick",
    "retrieval_contradiction": "retrieval_high_riverside",
    "rag_grounding_corpus": "rag_high_harbor",
    "cove_verification_loop": "cove_high_midtown",
    "critic_external_checks": "critic_high_solaris",
}


def _copy_case(case: dict) -> dict:
    return {
        key: list(value) if isinstance(value, list) else value
        for key, value in case.items()
    }


def list_cases() -> list[dict]:
    """Return all curated demo cases."""
    return [_copy_case(case) for case in SAMPLE_CASES]


def case_by_id(case_id: str) -> dict | None:
    """Look up one sample case by id, with legacy aliases for older tests and notebooks."""
    resolved_id = LEGACY_CASE_ALIASES.get(case_id, case_id)
    for case in SAMPLE_CASES:
        if case["id"] == resolved_id:
            return _copy_case(case)
    return None


def cases_for_method(method_name: str) -> list[dict]:
    """Return all demo cases targeted at one method."""
    return [_copy_case(case) for case in SAMPLE_CASES if method_name in case.get("method_targets", [])]


def sample_pairs_for(method_name: str) -> dict[str, dict | None]:
    """Return the low-risk and high-risk demo pair for one method."""
    cases = {
        case["risk_level"]: _copy_case(case)
        for case in SAMPLE_CASES
        if method_name in case.get("method_targets", [])
    }
    return {"low": cases.get("low"), "high": cases.get("high")}


def get_sample_pair(method_name: str, risk_level: str) -> dict | None:
    """Return one curated demo case for a method and risk level."""
    return sample_pairs_for(method_name).get(risk_level)
