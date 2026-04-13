"""Export internal-signal probe features from a JSONL dataset."""

import argparse
import json
from pathlib import Path

from detectors.signal import append_probe_example


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, help="JSONL rows with question, answer, label, and optional sampled_answers_text")
    parser.add_argument("--output-jsonl", required=True, help="Where to append exported feature rows")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output file before exporting")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and output_path.exists():
        output_path.unlink()

    count = 0
    with open(args.input_jsonl, "r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            append_probe_example(
                output_path=str(output_path),
                question=row.get("question", ""),
                answer=row.get("answer", ""),
                label=int(row["label"]),
                sampled_answers_text=row.get("sampled_answers_text", ""),
            )
            count += 1
    print(f"Exported {count} feature row(s) to {output_path}")


if __name__ == "__main__":
    main()
