from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from socialomni_annotation.postprocess import build_information_states


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build initial information states.")
    parser.add_argument("--pov-events-dir", default="annotations_qwen/pov_events", type=Path)
    parser.add_argument(
        "--output-path",
        default="data/processed/information_states.json",
        type=Path,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    states = build_information_states(args.pov_events_dir, args.output_path)
    print(f"information_states={len(states)}")


if __name__ == "__main__":
    main()
