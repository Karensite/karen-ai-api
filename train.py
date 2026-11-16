"""Train the tiny toy model from the checked-in, hand-written sample dialogues."""

import argparse
import json
from pathlib import Path

from model import TinyNgramModel

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/sample_dialogues.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/model.json")
    args = parser.parse_args()
    dialogues = json.loads(args.data.read_text(encoding="utf-8"))
    model = TinyNgramModel.train(dialogues)
    model.save(args.output)
    print(f"Trained toy model from {len(dialogues)} dialogue examples; saved to {args.output}")


if __name__ == "__main__":
    main()
