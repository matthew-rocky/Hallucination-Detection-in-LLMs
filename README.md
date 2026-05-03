# Hallucination Detector Family Comparison Dashboard

This project now includes a full-stack dashboard for comparing several hallucination-detection approaches on the same inputs. It is meant for local experimentation and teaching, not for benchmarking detector quality or deploying a safety system.

The detector logic remains in the original Python modules. A FastAPI backend wraps those methods, and a Next.js dashboard renders the analysis workflow, comparison charts, traces, evidence, citations, and curated sample cases.

## What Runs

The dashboard supports two modes:

- `Quick Check`: run one selected method on the current input.
- `Compare Methods`: run several methods side by side and compare their scores, labels, citations, and traces.

Each method returns a shared result dictionary with a risk score, label, confidence, explanation, evidence/citation fields, optional intermediate steps, and metadata about the runtime path.

The original Streamlit app is still available through `app.py`. The recommended app is the new FastAPI + Next.js dashboard.

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
backend/main.py                FastAPI entry point
backend/schemas.py             API request and response models
backend/services/              Service wrapper around existing method runners
data/sample_cases.py           Curated low/high examples for each method
detectors/                     Lower-level detector implementations
methods/                       Public method wrappers used by the UI
retrieval/                     Local document ingestion, chunking, indexing
frontend/app/                  Next.js app router pages and global styles
frontend/components/           Dashboard UI components
frontend/lib/                  Typed API client and TypeScript interfaces
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

Install frontend dependencies from the `frontend` directory:

```bash
cd frontend
npm install
```

The internal-signal backends depend on local Hugging Face model loading. The default model name is `distilgpt2`; override it with:

```bash
set HD_INTERNAL_MODEL=distilgpt2
set HD_INTERNAL_LAYERS=-1,-3,-5
set HD_HF_LOCAL_ONLY=1
```

## Run the Full-Stack Dashboard

Start the FastAPI backend from the repository root:

```bash
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Start the Next.js frontend in a second terminal:

```bash
cd frontend
npm run dev
```

Open `http://localhost:3000`. The frontend expects the backend at `http://127.0.0.1:8000` by default. Override this with `NEXT_PUBLIC_API_BASE_URL` if needed.

Optional environment file:

```bash
copy frontend\.env.local.example frontend\.env.local
```

Root helper scripts are also available:

```bash
npm run dev:backend
npm run dev:frontend
npm run build:frontend
npm run test:python
```

On Windows, `START_FULLSTACK.bat` starts both services in separate terminals. It activates `.venv` when present, then launches:

```bat
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
cd frontend
npm run dev
```

Use Node.js 18.18 or newer for the Next.js frontend. The FastAPI backend keeps the detector logic in Python and calls the existing modules in `methods/`, `detectors/`, `retrieval/`, `utils/`, and `data/sample_cases.py`.

The dashboard has these working tabs:

- `Overview`: status, method counts, risk metrics, and quick actions.
- `Ask Studio`: chatbot-style question/answer audit interface.
- `Analyze`: advanced method-aware form with dynamic required and optional fields.
- `Samples`: curated low/high sample browser backed by `data/sample_cases.py`.
- `Results`: detailed risk card, explanation, claims, evidence, citations, trace, and metadata.
- `Compare`: Recharts risk/confidence comparison and method ranking.
- `Method Flow`: animated React Flow detector pipeline.
- `Method Library`: all eight method cards, using backend metadata when online and local fallback metadata when offline.
- `Report / Export`: Markdown and JSON export from the latest analysis.

If the dashboard shows `Backend offline`, start or restart the backend with:

```bash
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

The frontend keeps showing fallback method metadata while offline, but samples and real analysis require FastAPI. `frontend/node_modules/`, `frontend/.next/`, Python cache folders, runtime artifacts, and zip backups are intentionally ignored by git because they are generated locally and can be recreated.

## Run the Streamlit App

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
