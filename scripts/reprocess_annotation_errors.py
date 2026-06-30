from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from socialomni_annotation.runner import reprocess_error_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-parse saved Qwen raw responses.")
    parser.add_argument("--error-dir", default="annotations_qwen/errors", type=Path)
    parser.add_argument("--output-dir", default="annotations_qwen/pov_events", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(reprocess_error_files(args.error_dir, args.output_dir))


if __name__ == "__main__":
    main()
