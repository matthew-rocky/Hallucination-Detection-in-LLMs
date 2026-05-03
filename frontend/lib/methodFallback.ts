import type { FieldSpec, MethodInfo } from "./types";

const method = (
  name: string,
  family: string,
  short: string,
  best: string,
  fields: string[],
  required: string[],
  implementation: string,
  color: string,
): MethodInfo => ({
  id: name.toLowerCase().replaceAll(" ", "-").replaceAll("/", "-"),
  name,
  family,
  short_purpose: short,
  best_for: best,
  how_it_works: short,
  required_fields: required,
  optional_fields: fields.filter((field) => !required.includes(field)),
  visible_fields: fields,
  ignored_fields: [],
  supports_uploads: fields.includes("uploaded_documents"),
  implementation,
  caption: short,
  input_requirements: { required, optional: fields.filter((field) => !required.includes(field)), visible: fields },
  strengths: [best],
  weaknesses: ["Backend metadata unavailable until FastAPI is online."],
  recommended_use: best,
  color,
  tone: color,
  profile: {
    main_strength: best,
    main_weakness: "Backend metadata unavailable until FastAPI is online.",
    main_deployment_tradeoff: "Local dashboard fallback metadata only.",
    color
  }
});

export const fallbackMethods: MethodInfo[] = [
  method("Internal-Signal Baseline", "Internal-signal methods", "Quick uncertainty-style risk check from one answer.", "fast local screening", ["question", "answer"], ["question", "answer"], "Implemented", "cyan"),
  method("SEP-Inspired Internal Signal", "Internal-signal / SEP-inspired", "Compare sampled answers for agreement, drift, and suspicious consensus.", "sample stability analysis", ["question", "answer", "sampled_answers"], ["question", "answer", "sampled_answers"], "Approximate", "purple"),
  method("Source-Grounded Consistency", "Source-grounded consistency", "Check claims against a provided source passage.", "source-faithful summarization", ["question", "answer", "source_text"], ["answer", "source_text"], "Approximate", "green"),
  method("Retrieval-Grounded Checker", "Retrieval-grounded checking", "Retrieve local evidence and assign claim-level support labels.", "evidence-backed checking", ["question", "answer", "evidence_text", "uploaded_documents"], ["answer", "evidence_text"], "Implemented", "teal"),
  method("RAG Grounded Check", "RAG-style grounded checking", "RAG-style post-hoc grounding over retrieved evidence.", "RAG answer auditing", ["question", "answer", "evidence_text", "uploaded_documents"], ["answer", "evidence_text"], "Approximate", "amber"),
  method("Verification-Based Workflow", "Verification workflow baseline", "Extract claims, ask verification questions, and aggregate verdicts.", "step-by-step verification", ["question", "answer", "evidence_text"], ["answer", "evidence_text"], "Approximate", "orange"),
  method("CoVe-Style Verification", "Chain-of-Verification", "Verify first, then revise the answer if evidence disagrees.", "verify-and-revise workflows", ["question", "answer", "evidence_text", "uploaded_documents"], ["answer", "evidence_text"], "Implemented", "indigo"),
  method("CRITIC-lite Tool Check", "Tool-augmented critique and revision", "Use local retrieval and numeric checks to critique answer claims.", "tool-checkable claims", ["question", "answer", "evidence_text"], ["answer", "evidence_text"], "Implemented", "rose")
];

export const fallbackFields: Record<string, FieldSpec> = {
  question: { label: "Question / Prompt", short_label: "Question", helper: "The user question or task prompt.", placeholder: "What did the briefing claim?", height: 120 },
  answer: { label: "LLM Answer To Evaluate", short_label: "Answer", helper: "The model answer being audited.", placeholder: "Paste the answer you want to check.", height: 190 },
  sampled_answers: { label: "Sampled Answers", short_label: "Samples", helper: "Alternative model answers separated by blank lines or ---.", placeholder: "Alternative answer 1\n\n---\n\nAlternative answer 2", height: 190 },
  source_text: { label: "Source Text", short_label: "Source", helper: "Reference passage the answer should follow.", placeholder: "Paste source passage.", height: 210 },
  evidence_text: { label: "Evidence Text", short_label: "Evidence", helper: "Retrieved notes or local evidence.", placeholder: "Paste evidence snippets.", height: 210 },
  uploaded_documents: { label: "Uploaded Documents", short_label: "Files", helper: "Upload txt, md, json, jsonl, or pdf evidence.", placeholder: "", height: 0 }
};
