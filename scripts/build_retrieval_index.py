"""Build and persist a local retrieval index from text documents."""

import argparse
from pathlib import Path

from retrieval.chunking import ingest_docs, load_doc_path
from retrieval.indexing import VectorIndex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Path to the output .pkl index file")
    parser.add_argument("--source-file", help="Optional source text file")
    parser.add_argument("--evidence-file", help="Optional evidence text file")
    parser.add_argument("--document", action="append", default=[], help="Additional document path. Repeat for multiple files.")
    parser.add_argument("--backend", default="sentence-transformer", choices=["sentence-transformer", "tfidf"], help="Embedding backend to use")
    parser.add_argument("--model-name", default=None, help="Optional embedding model name override")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_text = Path(args.source_file).read_text(encoding="utf-8") if args.source_file else ""
    evidence_text = Path(args.evidence_file).read_text(encoding="utf-8") if args.evidence_file else ""
    extra_documents = [load_doc_path(path) for path in args.document]
    documents = ingest_docs(
        source_text=source_text,
        evidence_text=evidence_text,
        extra_documents=extra_documents,
    )
    if not documents:
        raise SystemExit("No documents were provided. Supply --source-file, --evidence-file, or --document.")

    index = VectorIndex.from_documents(
        documents,
        preferred_backend=args.backend,
        model_name=args.model_name,
        max_sentences=1,
        max_chars=320,
        overlap=0,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    index.save(output_path)
    print(f"Saved retrieval index with {len(index.chunks)} chunks to {output_path}")


if __name__ == "__main__":
    main()
