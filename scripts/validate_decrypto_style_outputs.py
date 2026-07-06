from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from socialomni_annotation.omni_goose.decrypto_diagnostics import validate_decrypto_outputs, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate SocialOmni-Goose Decrypto-style outputs.")
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_decrypto_outputs(args.annotation_root, args.benchmark_root)
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
