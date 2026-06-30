from __future__ import annotations

from pathlib import Path
from typing import Any

from .backends.qwen_omni import ModelBackend
from .json_utils import parse_json, validation_error_payload, write_json
from .prompts import pov_event_prompt
from .schemas import Clip, POVEvent


def _coerce_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _normalize_event_time_window(normalized: dict, clip: Clip) -> None:
    start_sec = _coerce_float(normalized.get("start_sec"), clip.start_sec)
    end_sec = _coerce_float(normalized.get("end_sec"), clip.end_sec)

    start_sec = min(max(start_sec, clip.start_sec), clip.end_sec)
    end_sec = min(max(end_sec, clip.start_sec), clip.end_sec)
    if end_sec <= start_sec:
        end_sec = min(clip.end_sec, start_sec + 1.0)
        if end_sec <= start_sec:
            start_sec = clip.start_sec
            end_sec = clip.end_sec

    normalized["start_sec"] = start_sec
    normalized["end_sec"] = end_sec


def load_manifest(path: Path) -> list[Clip]:
    clips: list[Clip] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                clips.append(Clip.model_validate_json(line))
    return clips


def filter_clips(
    clips: list[Clip],
    game_id: str | None = None,
    player_id: str | None = None,
    limit: int | None = None,
) -> list[Clip]:
    selected = [
        clip
        for clip in clips
        if (game_id is None or clip.game_id == game_id)
        and (player_id is None or clip.player_id == player_id)
    ]
    return selected[:limit] if limit is not None else selected


def normalize_pov_event_payload(item: dict, clip: Clip) -> dict:
    normalized = dict(item)
    normalized["clip_id"] = clip.clip_id
    normalized["game_id"] = clip.game_id
    normalized["player_id"] = clip.player_id
    normalized.setdefault("start_sec", clip.start_sec)
    normalized.setdefault("end_sec", clip.end_sec)
    normalized.setdefault("event_type", "observation")
    normalized.setdefault("description", "")
    normalized.setdefault("visible_players", [])
    normalized.setdefault("mentioned_players", [])
    normalized.setdefault("location", None)
    normalized.setdefault("evidence", None)
    if isinstance(normalized["location"], list):
        normalized["location"] = "、".join(str(item) for item in normalized["location"])
    _normalize_event_time_window(normalized, clip)

    confidence = normalized.get("confidence", 0.0)
    if isinstance(confidence, str):
        confidence_map = {
            "高": 0.9,
            "中": 0.6,
            "低": 0.3,
            "high": 0.9,
            "medium": 0.6,
            "low": 0.3,
        }
        normalized["confidence"] = confidence_map.get(confidence.strip().lower(), 0.0)
    return normalized


def annotate_pov_events(
    clips: list[Clip],
    backend: ModelBackend,
    output_dir: Path,
    error_dir: Path,
    resume: bool,
) -> dict[str, int]:
    stats = {"ok": 0, "error": 0, "skipped": 0}
    for clip in clips:
        output_path = output_dir / f"{clip.clip_id}.json"
        if resume and output_path.exists():
            stats["skipped"] += 1
            continue

        prompt = pov_event_prompt(clip)
        try:
            raw_response = backend.generate(prompt, clip.clip_path)
            payload = parse_json(raw_response)
            events = [
                POVEvent.model_validate(normalize_pov_event_payload(item, clip))
                for item in payload
            ]
            write_json(
                output_path,
                {
                    "clip": clip.model_dump(),
                    "events": [event.model_dump() for event in events],
                    "raw_response": raw_response,
                },
            )
            stats["ok"] += 1
        except Exception as exc:  # noqa: BLE001
            stats["error"] += 1
            raw = locals().get("raw_response", "")
            write_json(
                error_dir / f"{clip.clip_id}.json",
                {
                    "clip": clip.model_dump(),
                    **validation_error_payload(exc, raw),
                },
            )
    return stats


def reprocess_error_files(error_dir: Path, output_dir: Path) -> dict[str, int]:
    stats = {"ok": 0, "error": 0}
    for error_path in sorted(error_dir.glob("*.json")):
        try:
            import json

            record = json.loads(error_path.read_text(encoding="utf-8"))
            clip = Clip.model_validate(record["clip"])
            raw_response = record.get("raw_response", "")
            payload = parse_json(raw_response)
            events = [
                POVEvent.model_validate(normalize_pov_event_payload(item, clip))
                for item in payload
            ]
            write_json(
                output_dir / f"{clip.clip_id}.json",
                {
                    "clip": clip.model_dump(),
                    "events": [event.model_dump() for event in events],
                    "raw_response": raw_response,
                    "reprocessed_from_error": str(error_path),
                },
            )
            error_path.unlink()
            stats["ok"] += 1
        except Exception:  # noqa: BLE001
            stats["error"] += 1
    return stats
