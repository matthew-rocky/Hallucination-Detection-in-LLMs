"""Method metadata used by the Streamlit UI and comparison helpers."""

PROFILE_FIELD_ORDER = [
    ("Implementation", "implementation"),
    ("Family", "family"),
    ("Evidence Source", "evidence_source"),
    ("Inference Stage", "inference_stage"),
    ("Granularity", "granularity"),
    ("Output Form", "output_form"),
    ("Action Type", "action_type"),
    ("Inputs Needed", "inputs_needed"),
    ("Computational Profile", "computational_profile"),
    ("Interpretability", "interpretability"),
    ("Cross-Task Portability", "cross_task_portability"),
    ("Main Strength", "main_strength"),
    ("Main Weakness", "main_weakness"),
    ("Main Deployment Tradeoff", "main_deployment_tradeoff"),
]

FIELD_DISPLAY_ORDER = [
    "question",
    "answer",
    "sampled_answers",
    "source_text",
    "evidence_text",
    "uploaded_documents",
]

FIELD_SPECS = {
    "question": {
        "label": "Question",
        "short_label": "Question",
        "helper": "Paste the user question or task prompt that the answer is trying to address.",
        "placeholder": "Ask a question or paste the original prompt.",
        "height": 110,
    },
    "answer": {
        "label": "LLM Answer",
        "short_label": "Answer",
        "helper": "Paste the model answer you want to evaluate for hallucination risk.",
        "placeholder": "Paste the answer produced by the model.",
        "height": 170,
    },
    "sampled_answers": {
        "label": "Sampled Answers",
        "short_label": "Sampled answers",
        "helper": "Paste alternative generated answers, separated by blank lines or --- markers.",
        "placeholder": "Alternative answer 1\n\n---\n\nAlternative answer 2",
        "height": 190,
    },
    "source_text": {
        "label": "Source Text",
        "short_label": "Source text",
        "helper": "Paste the source passage that the answer is supposed to follow.",
        "placeholder": "Paste the reference passage or source excerpt here.",
        "height": 210,
    },
    "evidence_text": {
        "label": "Evidence Text",
        "short_label": "Evidence text",
        "helper": "Paste retrieved notes, evidence snippets, or supporting material used to verify the answer.",
        "placeholder": "Paste supporting evidence, retrieval notes, or document excerpts here.",
        "height": 210,
    },
    "uploaded_documents": {
        "label": "Upload Documents",
        "short_label": "Documents",
        "helper": "Upload local files to add more evidence for retrieval-grounded methods.",
        "placeholder": "",
        "height": 0,
    },
}

METHOD_ORDER = [
    "Internal-Signal Baseline",
    "SEP-Inspired Internal Signal",
    "Source-Grounded Consistency",
    "Retrieval-Grounded Checker",
    "RAG Grounded Check",
    "Verification-Based Workflow",
    "CoVe-Style Verification",
    "CRITIC-lite Tool Check",
]

METHOD_CATALOG = {
    "Internal-Signal Baseline": {
        "short_purpose": "Quick style-only risk check from a single answer.",
        "best_for": "quick style-only risk check",
        "how_it_works": "This method looks for internal uncertainty patterns in one answer. When the local Hugging Face backend is unavailable, it falls back to a deterministic local text-only proxy instead of becoming unavailable.",
        "required_fields": ["question", "answer"],
        "optional_fields": [],
        "visible_fields": ["question", "answer"],
        "ignored_fields": ["sampled_answers", "source_text", "evidence_text", "uploaded_documents"],
        "supports_uploads": False,
        "caption": "Implemented token-uncertainty baseline using a local Hugging Face causal LM under teacher forcing.",
        "implementation": "Implemented",
        "family": "Internal-signal methods",
        "evidence_source": "Model token probabilities from one answer",
        "inference_stage": "Post-hoc uncertainty scoring",
        "granularity": "Answer level with claim-level internal-risk slices",
        "output_form": "Risk score + uncertainty sub-signals",
        "action_type": "Detection only",
        "inputs_needed": "Question + answer text",
        "computational_profile": "Moderate CPU inference",
        "interpretability": "Medium",
        "cross_task_portability": "High",
        "main_strength": "Uses real model uncertainty instead of wording heuristics when the local backend is available.",
        "main_weakness": "A small local LM can still assign low risk to confident false answers.",
        "main_deployment_tradeoff": "Portable and local, but only as informative as the chosen open-weight model.",
    },
    "SEP-Inspired Internal Signal": {
        "short_purpose": "Check whether multiple sampled answers stay stable or drift.",
        "best_for": "comparing multiple sampled answers for stability",
        "how_it_works": "This method compares the main answer with sampled alternatives, looking for stable agreement, drift, or suspiciously specific consensus. If the local model backend fails, it uses a deterministic local fallback that still keeps sampled-answer consistency front and center.",
        "required_fields": ["question", "answer", "sampled_answers"],
        "optional_fields": [],
        "visible_fields": ["question", "answer", "sampled_answers"],
        "ignored_fields": ["source_text", "evidence_text", "uploaded_documents"],
        "supports_uploads": False,
        "caption": "Approximate SEP-lite path using hidden-state statistics plus optional offline probes. This is not a paper-faithful SEP reproduction.",
        "implementation": "Approximate",
        "family": "Internal-signal / SEP-inspired",
        "evidence_source": "Model hidden states plus token uncertainty; optional sampled answers",
        "inference_stage": "Post-hoc hidden-state feature scoring",
        "granularity": "Answer level with claim-level risk slices",
        "output_form": "Risk score + hidden-state trace + optional probe",
        "action_type": "Detection only",
        "inputs_needed": "Question + answer text; sampled answers recommended",
        "computational_profile": "Moderate CPU inference",
        "interpretability": "Medium",
        "cross_task_portability": "Medium",
        "main_strength": "Uses sampled-answer consistency instead of relying on one answer alone.",
        "main_weakness": "Still SEP-lite rather than full semantic-entropy clustering at original scale.",
        "main_deployment_tradeoff": "Stronger than the baseline when samples exist, but more configuration-heavy.",
    },
    "Source-Grounded Consistency": {
        "short_purpose": "Check whether an answer matches a provided source passage.",
        "best_for": "checking an answer against a provided source passage",
        "how_it_works": "This method breaks the answer into coarse claims and compares them against a supplied source passage, distinguishing support, contradiction, unsupported additions, and summary-safe abstraction.",
        "required_fields": ["answer", "source_text"],
        "optional_fields": ["question"],
        "visible_fields": ["question", "answer", "source_text"],
        "ignored_fields": ["sampled_answers", "evidence_text", "uploaded_documents"],
        "supports_uploads": False,
        "caption": "Legacy local baseline that checks claims against a supplied source passage.",
        "implementation": "Approximate",
        "family": "Source-grounded consistency",
        "evidence_source": "User-provided source text",
        "inference_stage": "Claim-to-source comparison",
        "granularity": "Claim/chunk level",
        "output_form": "Risk score + matched source chunks",
        "action_type": "Detection only",
        "inputs_needed": "Answer text + source passage",
        "computational_profile": "Low to moderate",
        "interpretability": "High",
        "cross_task_portability": "Medium",
        "main_strength": "Interpretable baseline when a reference passage exists.",
        "main_weakness": "Still uses lightweight similarity and contradiction cues instead of full NLI.",
        "main_deployment_tradeoff": "Cheap and transparent, but limited to grounded tasks with a supplied source.",
    },
    "Retrieval-Grounded Checker": {
        "short_purpose": "Check claims against retrieved local evidence with citations.",
        "best_for": "checking claims against retrieved evidence",
        "how_it_works": "This method indexes the evidence you provide, retrieves the most relevant chunks for each claim, and scores whether the answer is supported, contradicted, or unsupported with explicit local citations.",
        "required_fields": ["answer", "evidence_text"],
        "optional_fields": ["question", "uploaded_documents"],
        "visible_fields": ["question", "answer", "evidence_text", "uploaded_documents"],
        "ignored_fields": ["sampled_answers", "source_text"],
        "supports_uploads": True,
        "caption": "Implemented local retrieval pipeline with document ingestion, chunking, embeddings, vector search, and claim-level verdicts with citations.",
        "implementation": "Implemented",
        "family": "Retrieval-grounded checking",
        "evidence_source": "Evidence text and uploaded local documents",
        "inference_stage": "Index -> retrieve -> classify support",
        "granularity": "Claim/chunk level",
        "output_form": "Risk score + evidence objects + citations",
        "action_type": "Detection and evidence checking",
        "inputs_needed": "Answer text + local evidence corpus",
        "computational_profile": "Moderate",
        "interpretability": "High",
        "cross_task_portability": "Medium",
        "main_strength": "Real local retrieval with explicit citations and per-claim verdicts.",
        "main_weakness": "Still limited to the local corpus and lightweight contradiction rules.",
        "main_deployment_tradeoff": "More credible than keyword overlap, but depends on document coverage and embedding quality.",
    },
    "RAG Grounded Check": {
        "short_purpose": "RAG-style grounded checking over retrieved evidence.",
        "best_for": "RAG-style evidence checking",
        "how_it_works": "This method uses the same local retrieval stack as the retrieval-grounded checker, but frames the analysis as a RAG-style grounded check over the retrieved evidence instead of a direct one-pass checker.",
        "required_fields": ["answer", "evidence_text"],
        "optional_fields": ["question", "uploaded_documents"],
        "visible_fields": ["question", "answer", "evidence_text", "uploaded_documents"],
        "ignored_fields": ["sampled_answers", "source_text"],
        "supports_uploads": True,
        "caption": "Implemented as a RAG-style post-hoc grounded checker over the same local retrieval stack; it does not re-generate the answer from retrieved context.",
        "implementation": "Approximate",
        "family": "RAG-style grounded checking",
        "evidence_source": "Indexed local corpus",
        "inference_stage": "Index -> retrieve -> grounded claim check",
        "granularity": "Claim/chunk level",
        "output_form": "Risk score + retrieval trace + citations",
        "action_type": "Grounding-focused detection",
        "inputs_needed": "Answer text + local evidence corpus",
        "computational_profile": "Moderate",
        "interpretability": "High",
        "cross_task_portability": "Medium",
        "main_strength": "Shows what a RAG-style grounding pass would retrieve and cite.",
        "main_weakness": "This prototype checks an existing answer instead of generating a new one from retrieved evidence.",
        "main_deployment_tradeoff": "Useful for grounded analysis, but still lighter than a full end-to-end RAG stack.",
    },
    "Verification-Based Workflow": {
        "short_purpose": "Run a staged verification workflow over supplied evidence.",
        "best_for": "step-by-step claim verification",
        "how_it_works": "This method decomposes the answer into claims, writes verification questions, checks them against the available passages, and summarizes whether the claims look verified, contradicted, or unresolved.",
        "required_fields": ["answer", "evidence_text"],
        "optional_fields": ["question"],
        "visible_fields": ["question", "answer", "evidence_text"],
        "ignored_fields": ["sampled_answers", "source_text", "uploaded_documents"],
        "supports_uploads": False,
        "caption": "Legacy staged baseline retained as a simpler comparison point beside CoVe and CRITIC-lite.",
        "implementation": "Approximate",
        "family": "Verification workflow baseline",
        "evidence_source": "Evidence text",
        "inference_stage": "Question generation + local checking",
        "granularity": "Claim/question level",
        "output_form": "Risk score + verification trace",
        "action_type": "Verification baseline",
        "inputs_needed": "Answer text + local evidence",
        "computational_profile": "Moderate",
        "interpretability": "High",
        "cross_task_portability": "Medium",
        "main_strength": "Provides a lightweight baseline for staged verification behavior.",
        "main_weakness": "Not as explicit or tool-interactive as the CoVe and CRITIC-lite paths.",
        "main_deployment_tradeoff": "Good for side-by-side comparison, but not the main backend the prototype is centered on.",
    },
    "CoVe-Style Verification": {
        "short_purpose": "Verify first, then revise the answer if needed.",
        "best_for": "verify-and-check style analysis",
        "how_it_works": "This method follows a Chain-of-Verification style flow: draft answer, verification questions, independent evidence checks, and a revision step when the evidence points somewhere else.",
        "required_fields": ["answer", "evidence_text"],
        "optional_fields": ["question", "uploaded_documents"],
        "visible_fields": ["question", "answer", "evidence_text", "uploaded_documents"],
        "ignored_fields": ["sampled_answers", "source_text"],
        "supports_uploads": True,
        "caption": "Implemented CoVe-style pipeline with draft, verification questions, independent evidence answers, revision, and final summary.",
        "implementation": "Implemented",
        "family": "Chain-of-Verification",
        "evidence_source": "Indexed local evidence corpus",
        "inference_stage": "Draft -> verify -> revise",
        "granularity": "Claim/question level",
        "output_form": "Risk score + intermediate stages + revised answer",
        "action_type": "Verification and correction",
        "inputs_needed": "Answer text + local evidence corpus",
        "computational_profile": "Moderate",
        "interpretability": "High",
        "cross_task_portability": "Medium",
        "main_strength": "Separates verification from revision and exposes every stage in the UI.",
        "main_weakness": "Uses local extractive synthesis rather than a large multi-call model pipeline.",
        "main_deployment_tradeoff": "More faithful than a one-pass checker, but slower and evidence dependent.",
    },
    "CRITIC-lite Tool Check": {
        "short_purpose": "Use explicit tool checks for numeric and grounded claims.",
        "best_for": "numeric and directly checkable claims",
        "how_it_works": "This method routes claims through lightweight local tools, such as retrieval and numeric checks, critiques the answer against those tool outputs, and proposes a more grounded revision.",
        "required_fields": ["answer", "evidence_text"],
        "optional_fields": ["question"],
        "visible_fields": ["question", "answer", "evidence_text"],
        "ignored_fields": ["sampled_answers", "source_text", "uploaded_documents"],
        "supports_uploads": False,
        "caption": "Implemented CRITIC-lite workflow with tool routing, local retrieval, numeric checks, critique, revision, and an optional second pass.",
        "implementation": "Implemented",
        "family": "Tool-augmented critique and revision",
        "evidence_source": "Tool outputs from local retrieval and numeric checks",
        "inference_stage": "Draft -> tools -> critique -> revise",
        "granularity": "Claim/tool-check level",
        "output_form": "Risk score + tool trace + revised answer",
        "action_type": "Verification and correction",
        "inputs_needed": "Answer text + local evidence corpus",
        "computational_profile": "Moderate",
        "interpretability": "High",
        "cross_task_portability": "Medium",
        "main_strength": "Critiques claims against explicit tool outputs instead of the model's text alone.",
        "main_weakness": "Uses a deliberately small tool set and local evidence only.",
        "main_deployment_tradeoff": "Stronger than pure self-verification, but still far smaller than full CRITIC systems.",
    },
}


def method_meta(method_name: str) -> dict:
    """Return the rich UI metadata for one method."""
    return METHOD_CATALOG.get(method_name, {})


def get_method_profile(method_name: str) -> dict[str, str]:
    """Return the plain-English profile fields for one method."""
    metadata = method_meta(method_name)
    if metadata:
        return {field_key: metadata.get(field_key, "N/A") for _, field_key in PROFILE_FIELD_ORDER} | {
            "caption": metadata.get("caption", "N/A"),
        }
    return {field_key: "N/A" for _, field_key in PROFILE_FIELD_ORDER} | {"caption": "N/A"}


METHOD_PROFILES = {name: get_method_profile(name) for name in METHOD_ORDER}


def get_method_caption(method_name: str) -> str:
    """Return the short caption for one method."""
    return get_method_profile(method_name).get("caption", "")


def method_desc(method_name: str) -> str:
    """Return the human-friendly method description."""
    return method_meta(method_name).get("how_it_works", "")


def visible_fields_for(method_names: list[str] | tuple[str, ...]) -> list[str]:
    """Return the union of visible UI fields in stable display order."""
    selected = set(method_names or [])
    return [
        field_key
        for field_key in FIELD_DISPLAY_ORDER
        if any(field_key in method_meta(method_name).get("visible_fields", []) for method_name in selected)
    ]


def required_fields_for(method_names: list[str] | tuple[str, ...]) -> list[str]:
    """Return required fields for the selected method set in stable order."""
    selected = set(method_names or [])
    return [
        field_key
        for field_key in FIELD_DISPLAY_ORDER
        if any(field_key in method_meta(method_name).get("required_fields", []) for method_name in selected)
    ]


def optional_fields_for(method_names: list[str] | tuple[str, ...]) -> list[str]:
    """Return optional fields for the selected method set in stable order."""
    selected = set(method_names or [])
    return [
        field_key
        for field_key in FIELD_DISPLAY_ORDER
        if any(field_key in method_meta(method_name).get("optional_fields", []) for method_name in selected)
    ]


def methods_using(method_names: list[str] | tuple[str, ...], field_key: str) -> list[str]:
    """Return selected methods that actively use one field."""
    users = []
    for method_name in method_names or []:
        metadata = method_meta(method_name)
        if field_key in metadata.get("visible_fields", []):
            users.append(method_name)
    return users


def methods_requiring(method_names: list[str] | tuple[str, ...], field_key: str) -> list[str]:
    """Return selected methods for which the field is required."""
    required_by = []
    for method_name in method_names or []:
        metadata = method_meta(method_name)
        if field_key in metadata.get("required_fields", []):
            required_by.append(method_name)
    return required_by


def supports_uploads(method_names: list[str] | tuple[str, ...]) -> bool:
    """Return whether any selected method supports uploaded documents."""
    return any(method_meta(method_name).get("supports_uploads") for method_name in method_names or [])


def get_input_summary(method_name: str) -> str:
    """Return a short input summary for method cards."""
    metadata = method_meta(method_name)
    labels = [FIELD_SPECS[field_key]["short_label"] for field_key in metadata.get("visible_fields", [])]
    return " + ".join(labels)
