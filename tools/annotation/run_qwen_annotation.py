from __future__ import annotations
import sys
from pathlib import Path as _Path

_REPO_ROOT = next(
    _parent for _parent in _Path(__file__).resolve().parents if (_parent / "pyproject.toml").exists()
)
for _path in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from socialomni_annotation.backends.qwen_omni import QwenBackendConfig, create_backend
from socialomni_annotation.runner import annotate_pov_events, filter_clips, load_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen3-Omni initial POV annotation.")
    parser.add_argument("--manifest-path", default="data/processed/clip_manifest.jsonl", type=Path)
    parser.add_argument("--output-dir", default="annotations_qwen/pov_events", type=Path)
    parser.add_argument("--error-dir", default="annotations_qwen/errors", type=Path)
    parser.add_argument("--backend", default="mock", choices=["mock", "local", "openai"])
    parser.add_argument("--server-url", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default="qwen3-omni")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--game-id", default=None)
    parser.add_argument("--player-id", default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clips = filter_clips(
        load_manifest(args.manifest_path),
        game_id=args.game_id,
        player_id=args.player_id,
        limit=args.limit,
    )
    backend = create_backend(
        QwenBackendConfig(
            backend=args.backend,
            model=args.model,
            server_url=args.server_url,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
        )
    )
    stats = annotate_pov_events(
        clips=clips,
        backend=backend,
        output_dir=args.output_dir,
        error_dir=args.error_dir,
        resume=args.resume,
    )
    print(stats)


if __name__ == "__main__":
    main()
