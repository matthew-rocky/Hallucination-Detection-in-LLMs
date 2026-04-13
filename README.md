# Hallucination Detector Family Comparison Prototype

This is a Streamlit prototype for comparing several hallucination-detection approaches on the same inputs. It is meant for local experimentation and teaching, not for benchmarking detector quality or deploying a safety system.

The app focuses on method shape: what each detector needs as input, what it can inspect, and what kind of trace it returns. Some paths use a real local backend. Others are lightweight approximations that keep the interface and result schema comparable while leaving the limitations visible.

## What Runs

The current app supports two modes:

- `Quick Check`: run one selected method on the current input.
- `Compare Methods`: run several methods side by side and compare their scores, labels, citations, and traces.

Each method returns a shared result dictionary with a risk score, label, confidence, explanation, evidence/citation fields, optional intermediate steps, and metadata about the runtime path.

## Detector Paths

### Implemented Backends

1. `Internal-Signal Baseline`
   - Uses teacher-forced token scoring from a local Hugging Face causal language model when that backend is available.
   - Falls back to a deterministic text proxy when model loading fails.

2. `Retrieval-Grounded Checker`
   - Chunks local evidence, embeds/indexes it, retrieves relevant passages, and assigns claim-level support or contradiction labels with citations.

3. `CoVe-Style Verification`
   - Runs a local verify-and-revise workflow over supplied evidence.
   - The revision step is extractive/deterministic rather than a full LLM rewrite.

4. `CRITIC-lite Tool Check`
   - Routes claims through local retrieval and numeric checks, then builds a short grounded revision from the tool outputs.

### Approximate Baselines

1. `SEP-Inspired Internal Signal`
   - Uses a SEP-lite hidden-state and sampled-answer workflow.
   - Can use a trained probe bundle if one is available.
   - Otherwise falls back to deterministic sampled-answer consistency and suspicious-specificity cues.

2. `RAG Grounded Check`
   - Checks an existing answer against retrieved evidence.
   - It does not generate the answer from retrieved context.

3. `Source-Grounded Consistency`
   - Compares answer claims against a supplied source passage with local similarity and contradiction heuristics.

4. `Verification-Based Workflow`
   - A simpler staged verification baseline: extract claims, write checking questions, retrieve passages, and aggregate verdicts.

## Project Layout

```text
app.py                         Streamlit entry point and method runner glue
data/sample_cases.py           Curated low/high examples for each method
detectors/                     Lower-level detector implementations
methods/                       Public method wrappers used by the UI
retrieval/                     Local document ingestion, chunking, indexing
ui/                            Streamlit layout, forms, result rendering
utils/                         Text, scoring, grounding, and revision helpers
scripts/                       Retrieval-index and internal-probe utilities
tests/                         Regression and behavior tests
```

The code is local-first. If optional ML dependencies or model weights are unavailable, the app should still run where a deterministic fallback exists, and the result metadata should identify that path.

## Install

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

The internal-signal backends depend on local Hugging Face model loading. The default model name is `distilgpt2`; override it with:

```bash
set HD_INTERNAL_MODEL=distilgpt2
set HD_INTERNAL_LAYERS=-1,-3,-5
set HD_HF_LOCAL_ONLY=1
```

## Run

```bash
streamlit run app.py
```

The app can also be launched by running `app.py` directly; the bottom of the file delegates to Streamlit bootstrap when needed.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover the shared result schema, curated demo pairs, fallback paths, grounded contradiction behavior, citation traces, and revision quality.

## Retrieval Utilities

Build a local retrieval index from one evidence file:

```bash
python scripts/build_retrieval_index.py --evidence-file path\to\evidence.txt --output artifacts\retrieval_index.pkl --backend tfidf
```

Build from multiple local documents:

```bash
python scripts/build_retrieval_index.py --document docs\memo.txt --document docs\notes.json --output artifacts\retrieval_index.pkl
```

Supported document inputs are plain text-like files, JSON/JSONL, and PDFs when `pypdf` is installed.

## SEP-lite Probe Workflow

Prepare a JSONL file with labels and optional sampled answers:

```json
{"question": "What is the capital of Canada?", "answer": "Ottawa", "label": 0}
{"question": "What is the capital of Canada?", "answer": "Toronto", "label": 1, "sampled_answers_text": "Ottawa\n\n---\n\nToronto"}
```

Export feature rows:

```bash
python scripts/export_internal_probe_features.py --input-jsonl data\probe_train.jsonl --output-jsonl artifacts\internal_probe_features.jsonl --overwrite
```

Train the probe:

```bash
python scripts/train_internal_probe.py --feature-jsonl artifacts\internal_probe_features.jsonl --output artifacts\internal_probe.pkl
```

Point the app at the trained bundle:

```bash
set HD_INTERNAL_PROBE_PATH=artifacts\internal_probe.pkl
```

## Result Metadata

All detectors return the same top-level shape. The fields used most often in the UI and tests are:

- `method_name`, `family`
- `score`, `label`, `confidence`
- `risk_score`, `risk_label`
- `explanation`, `summary`, `limitations`
- `evidence`, `citations`, `claim_findings`
- `intermediate_steps`, `revised_answer`
- `implementation_status`
- `metadata`

Internal methods also report backend status and whether the run used full model scoring, a SEP-lite probe path, or a deterministic fallback approximation.

## Current Limits

- The SEP path is a compact local approximation, not a full semantic-entropy reproduction.
- RAG checking is post-hoc grounding over an existing answer, not retrieved generation.
- CoVe and CRITIC-lite revisions are deterministic summaries of supported evidence, not free-form LLM rewrites.
- Retrieval is only as good as the local evidence provided to the app.
- Fallback text heuristics can flag suspicious answers, but they cannot prove factual correctness on their own.