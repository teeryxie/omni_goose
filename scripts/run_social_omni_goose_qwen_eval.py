#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from typing import Any

from models.utils.omni_http_client import OmniHttpClient
from socialomni_annotation.omni_goose.backends import create_backend


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def try_parse_json(raw: str) -> tuple[dict[str, Any], bool]:
    try:
        value = json.loads(raw)
        return (value if isinstance(value, dict) else {"value": value}), True
    except Exception:
        pass
    match = re.search(r"\{.*\}", raw or "", flags=re.DOTALL)
    if match:
        try:
            value = json.loads(match.group(0))
            return (value if isinstance(value, dict) else {"value": value}), True
        except Exception:
            pass
    return {}, False


def prompt_with_schema(row: dict[str, Any]) -> str:
    schema = row.get("expected_output_schema", {})
    suffix = (
        "\n\nOUTPUT_REQUIREMENTS:\n"
        "Return strict JSON only. Do not include markdown fences or commentary. "
        "Use only evidence available under the stated input condition. "
        "Do not use hidden oracle facts, other-POV information, forbidden evidence, or later reveal information unless the prompt explicitly reveals oracle truth for reconstruction. "
        "For death/body/blood/death-animation evidence, certify only visible death/body/status; do not infer duck skill, killer identity, role/alignment, causal chain, or kill mechanism without independent evidence.\n"
        f"EXPECTED_OUTPUT_SCHEMA_JSON:\n{json.dumps(schema, ensure_ascii=False, indent=2)}"
    )
    return str(row.get("prompt", "")) + suffix


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not row.get("error") and row.get("trial_id"):
                ids.add(str(row["trial_id"]))
        except Exception:
            continue
    return ids


def resolve_video_path(args: argparse.Namespace, value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return path.as_posix()
    base = args.input.parent.parent
    candidate = base / path
    if candidate.exists():
        return candidate.as_posix()
    data_candidate = base / "data" / path
    if data_candidate.exists():
        return data_candidate.as_posix()
    return candidate.as_posix()


def call_model(args: argparse.Namespace, row: dict[str, Any], prompt: str) -> str:
    backend_name = args.backend.lower()
    if backend_name == "mock":
        probe_type = row.get("probe_type", "")
        if probe_type.startswith("A_"):
            return json.dumps({"knows_truth": False, "belief_label": "does_not_know", "likely_belief": "mock limited-perspective answer", "evidence_ids": [], "confidence": 0.5, "suspicion_update": "unknown"}, ensure_ascii=False)
        if probe_type.startswith("B_"):
            return json.dumps({"target_knew_truth_at_cutoff": False, "reconstructed_prior_belief": "does_not_know", "evidence_ids_available_at_cutoff": [], "confidence": 0.5, "must_not_use_revealed_truth_as_prior_evidence": True}, ensure_ascii=False)
        if probe_type.startswith("C_"):
            return json.dumps({"other_player_knew_truth_at_cutoff": False, "other_player_likely_belief": "uncertain", "evidence_ids": [], "confidence": 0.5}, ensure_ascii=False)
        return json.dumps({"speaker": "unknown", "listener": row.get("target_player", "unknown"), "predicted_listener_trust_update": "unknown", "predicted_listener_next_action": "unknown", "reason_from_speaker_perspective": "mock", "confidence": 0.5}, ensure_ascii=False)
    if backend_name in {"local", "local-server", "server"}:
        server_url = args.server_url or os.getenv("QWEN3_OMNI_SERVER_URL")
        if not server_url:
            raise ValueError("local backend requires --server-url or QWEN3_OMNI_SERVER_URL")
        video_file = row.get("video_file")
        if video_file:
            return OmniHttpClient(server_url).call_api(resolve_video_path(args, str(video_file)), prompt, use_video=True, use_audio=True) or ""
        if not args.text_context_video:
            raise ValueError("local structured evaluation requires --text-context-video because the local server upload endpoint requires a file")
        return OmniHttpClient(server_url).call_api(args.text_context_video, prompt, use_video=False, use_audio=False) or ""
    backend = create_backend(args.backend, model=args.model, api_key_env=args.api_key_env, base_url=args.base_url, server_url=args.server_url)
    video_file = row.get("video_file")
    if video_file:
        return backend.annotate_video(Path(resolve_video_path(args, str(video_file))), prompt)
    return backend.annotate_text(prompt)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen3-Omni as evaluated model on SocialOmni-Goose trials.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--track", choices=["structured", "raw_video_smoke"], required=True)
    parser.add_argument("--backend", choices=["mock", "local", "qwen", "openai"], default="mock")
    parser.add_argument("--server-url", default=None)
    parser.add_argument("--model", default="Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--text-context-video", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--job-manifest", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    selected = [row for idx, row in enumerate(rows) if idx >= args.skip and (idx - args.skip) % args.stride == 0]
    if args.limit is not None:
        selected = selected[: args.limit]
    if args.overwrite and args.output.exists():
        args.output.unlink()
    done = completed_ids(args.output) if args.resume and not args.overwrite else set()
    stats = {"input_rows": len(rows), "selected_rows": len(selected), "ok": 0, "error": 0, "skipped": 0, "output": args.output.as_posix()}
    if args.job_manifest:
        append_jsonl(args.job_manifest, {"event": "start", "track": args.track, "input": args.input.as_posix(), "output": args.output.as_posix(), "backend": args.backend, "model": args.model, "skip": args.skip, "stride": args.stride, "limit": args.limit, "time_unix": time.time()})
    for row in selected:
        trial_id = str(row.get("trial_id") or row.get("probe_id") or "")
        if args.resume and trial_id in done:
            stats["skipped"] += 1
            continue
        start = time.time()
        raw = ""
        parsed: dict[str, Any] = {}
        parse_ok = False
        error = None
        try:
            raw = call_model(args, row, prompt_with_schema(row))
            parsed, parse_ok = try_parse_json(raw)
            stats["ok"] += 1
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            stats["error"] += 1
        out = {"trial_id": trial_id, "probe_group_id": row.get("probe_group_id"), "probe_type": row.get("probe_type"), "input_condition": row.get("input_condition"), "track": args.track, "model": args.model, "raw_response": raw, "parsed": parsed, "parse_ok": parse_ok, "latency_sec": round(time.time() - start, 3), "error": error}
        if row.get("video_file"):
            out["video_file"] = row["video_file"]
        append_jsonl(args.output, out)
    write_json(args.output.with_suffix(".summary.json"), stats)
    if args.job_manifest:
        append_jsonl(args.job_manifest, {"event": "finish", **stats, "time_unix": time.time()})
    print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
