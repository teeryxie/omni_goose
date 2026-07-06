from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from socialomni_annotation.omni_goose.decrypto_diagnostics import score_decrypto_diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score Decrypto-style SocialOmni-Goose diagnostics.")
    parser.add_argument("--responses", type=Path, required=True, help="JSONL model responses.")
    parser.add_argument("--hidden-gold", type=Path, required=True, help="JSONL hidden gold file.")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aggregate = score_decrypto_diagnostics(args.responses, args.hidden_gold, args.output)
    print(json.dumps({"ok": True, "aggregate": aggregate}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
