"""Train a logistic SEP-lite probe from exported feature rows."""

import argparse

from detectors.signal import train_probe_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-jsonl", required=True, help="Path to exported feature rows")
    parser.add_argument("--output", required=True, help="Path to the output .pkl probe bundle")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    info = train_probe_jsonl(feature_path=args.feature_jsonl, output_path=args.output)
    print(f"Trained probe on {info['trained_rows']} row(s) and saved it to {info['output_path']}")


if __name__ == "__main__":
    main()
