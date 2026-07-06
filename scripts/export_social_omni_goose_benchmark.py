from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from socialomni_annotation.omni_goose.decrypto_diagnostics import export_social_omni_goose_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SocialOmni-Goose-v1 benchmark files.")
    parser.add_argument("--annotation-root", type=Path, required=True, help="Root containing diagnostics/.")
    parser.add_argument("--benchmark-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = export_social_omni_goose_benchmark(args.annotation_root, args.benchmark_root)
    print(json.dumps({"ok": True, "counts": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
