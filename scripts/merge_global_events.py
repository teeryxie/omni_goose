from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from socialomni_annotation.postprocess import merge_global_events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge POV events into initial global events.")
    parser.add_argument("--pov-events-dir", default="annotations_qwen/pov_events", type=Path)
    parser.add_argument(
        "--output-path",
        default="data/processed/global_events.json",
        type=Path,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events = merge_global_events(args.pov_events_dir, args.output_path)
    print(f"global_events={len(events)}")


if __name__ == "__main__":
    main()
