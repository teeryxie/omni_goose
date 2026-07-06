#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from typing import Any

WEIGHTS = {"RC_strong": 0.20, "FB_strong": 0.20, "PT_strong": 0.20, "claim_verification_local": 0.15, "evidence_support": 0.10, "no_perspective_leakage": 0.10, "no_death_skill_overclaim": 0.05}
DEATH_OVERCLAIM_PATTERNS = [r"鸭子技能", r"鸭阵营", r"坏人技能", r"技能杀", r"用刀击杀", r"发射了攻击", r"杀手身份", r"凶手身份", r"\bduck skill\b", r"\bkill skill\b", r"\bassassin\b", r"\bkiller identity\b"]
DEATH_CONTEXT_PATTERNS = [r"死亡", r"倒地", r"尸体", r"血迹", r"death", r"body", r"blood"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def norm(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return {"false": "contradicted", "true": "supported", "not_enough_info": "not_enough_information", "not_enough_evidence": "not_enough_information"}.get(text, text)


def nested_gold(row: dict[str, Any]) -> dict[str, Any]:
    hg = row.get("hidden_gold")
    return hg if isinstance(hg, dict) else row


def evidence_ids(obj: Any) -> set[str]:
    ids: set[str] = set()
    stack = [obj]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            for key, value in item.items():
                if key in {"evidence_id", "event_id", "claim_id", "world_event_id"} and isinstance(value, str):
                    ids.add(value)
                elif key.endswith("_ids") and isinstance(value, list):
                    ids.update(str(v) for v in value if isinstance(v, str))
                else:
                    stack.append(value)
        elif isinstance(item, list):
            stack.extend(item)
    return ids


def contains_forbidden(parsed: dict[str, Any], raw: str, forbidden: list[str]) -> bool:
    text = json.dumps(parsed, ensure_ascii=False) + "\n" + (raw or "")
    return any(fid and fid in text for fid in forbidden)


def death_skill_overclaim(parsed: dict[str, Any], raw: str) -> bool:
    text = json.dumps(parsed, ensure_ascii=False) + "\n" + (raw or "")
    has_death = any(re.search(p, text, flags=re.I) for p in DEATH_CONTEXT_PATTERNS)
    has_overclaim = any(re.search(p, text, flags=re.I) for p in DEATH_OVERCLAIM_PATTERNS)
    return bool(has_death and has_overclaim)


def schema_ok(probe_type: str, parsed: dict[str, Any]) -> bool:
    if not parsed:
        return False
    if probe_type.startswith("A_"):
        return "knows_truth" in parsed and bool(parsed.get("belief_label") or parsed.get("likely_belief"))
    if probe_type.startswith("B_"):
        return "target_knew_truth_at_cutoff" in parsed and bool(parsed.get("reconstructed_prior_belief") or parsed.get("belief_label"))
    if probe_type.startswith("C_"):
        return "other_player_knew_truth_at_cutoff" in parsed and bool(parsed.get("other_player_likely_belief") or parsed.get("belief_label"))
    if probe_type.startswith("D_"):
        return bool(parsed.get("listener")) and bool(parsed.get("predicted_listener_trust_update") or parsed.get("predicted_trust_update"))
    return True


def first_value(parsed: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in parsed:
            return parsed[key]
    return None


def text_of(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False).lower()


def score_trial(response: dict[str, Any], hidden_row: dict[str, Any]) -> dict[str, Any]:
    parsed = response.get("parsed") if isinstance(response.get("parsed"), dict) else {}
    raw = str(response.get("raw_response") or "")
    probe_type = str(response.get("probe_type") or "")
    hg = nested_gold(hidden_row)
    forbidden = list(hidden_row.get("forbidden_event_ids", []) or hg.get("forbidden_event_ids_for_target", []) or [])
    acceptable = set(hidden_row.get("acceptable_evidence_ids", []) or hg.get("acceptable_evidence_ids_for_target", []) or [])
    cited = evidence_ids(parsed)
    leakage = contains_forbidden(parsed, raw, forbidden)
    unsupported = (cited - acceptable - set(forbidden)) if acceptable else set()
    parse_success = bool(response.get("parse_ok") and parsed)
    schema_success = schema_ok(probe_type, parsed) if parse_success else False
    overclaim = death_skill_overclaim(parsed, raw)
    score = {"trial_id": response.get("trial_id"), "probe_group_id": response.get("probe_group_id"), "probe_type": probe_type, "input_condition": response.get("input_condition"), "track": response.get("track"), "json_parse_success": parse_success, "schema_validation_success": schema_success, "perspective_leakage": leakage, "forbidden_evidence_usage": leakage, "death_skill_overclaim": overclaim, "evidence_support": bool(parse_success and not leakage and not unsupported), "unsupported_evidence_ids": sorted(unsupported), "error": response.get("error"), "RC_weak": None, "RC_strong": None, "FB_weak": None, "FB_strong": None, "PT_weak": None, "PT_strong": None, "claim_verification_global": None, "claim_verification_local": None}
    expected_global = norm(hg.get("claim_truth_global"))
    expected_local = norm(hg.get("claim_awareness_local_target"))
    actual_global = norm(first_value(parsed, ["claim_truth_global", "global_claim_truth", "truth_status_global", "claim_global_truth"]))
    actual_local_raw = first_value(parsed, ["claim_awareness_local_target", "local_claim_awareness", "target_claim_awareness", "target_can_know_claim_truth", "target_has_evidence_for_claim_truth"])
    actual_local = ("has_evidence" if actual_local_raw else "not_enough_information") if isinstance(actual_local_raw, bool) else norm(actual_local_raw)
    if expected_global:
        score["claim_verification_global"] = actual_global == expected_global
    if expected_local:
        score["claim_verification_local"] = actual_local == expected_local
    if probe_type.startswith("A_"):
        expected = hg.get("A_expected_weak", {}) if isinstance(hg.get("A_expected_weak"), dict) else {}
        if "knows_truth" in expected:
            score["A_expected_weak"] = parsed.get("knows_truth") == expected.get("knows_truth") and not leakage
    elif probe_type.startswith("B_"):
        weak_ref = hg.get("B_RC_weak", {}) if isinstance(hg.get("B_RC_weak"), dict) else {}
        strong_ref = hg.get("B_RC_strong_reference", {}) if isinstance(hg.get("B_RC_strong_reference"), dict) else {}
        target_knew = parsed.get("target_knew_truth_at_cutoff")
        if weak_ref.get("must_not_answer_as_if_target_saw_hidden_events"):
            score["RC_weak"] = target_knew is False and not leakage
        elif target_knew is not None:
            score["RC_weak"] = not leakage
        ref_label = norm(strong_ref.get("belief_label"))
        answer_label = norm(parsed.get("belief_label")) or norm(parsed.get("reconstructed_prior_belief"))
        if ref_label:
            score["RC_strong"] = bool(score["RC_weak"] and answer_label and (answer_label == ref_label or ref_label in answer_label or answer_label in ref_label))
        elif score["RC_weak"] is not None:
            score["RC_strong"] = bool(score["RC_weak"] and parsed.get("reconstructed_prior_belief"))
    elif probe_type.startswith("C_"):
        weak_ref = hg.get("C_FB_weak", {}) if isinstance(hg.get("C_FB_weak"), dict) else {}
        expected_knows = weak_ref.get("other_player_knows_truth")
        answer_knows = parsed.get("other_player_knew_truth_at_cutoff")
        if expected_knows is not None:
            score["FB_weak"] = answer_knows == expected_knows and not leakage
        elif answer_knows is not None:
            score["FB_weak"] = not leakage
        score["FB_strong"] = bool(score["FB_weak"] and (parsed.get("other_player_likely_belief") or parsed.get("belief_label")))
    elif probe_type.startswith("D_"):
        ref = hg.get("D_PT_reference", {}) if isinstance(hg.get("D_PT_reference"), dict) else {}
        listener = parsed.get("listener")
        expected_listener = ref.get("listener")
        listener_ok = bool(listener and (not expected_listener or listener == expected_listener))
        score["PT_weak"] = listener_ok and not leakage
        answer_text = text_of(parsed) + "\n" + raw.lower()
        has_followup_signal = any(token in answer_text for token in ["follow", "followup", "corrobor", "question", "challenge", "质疑", "追问", "跟进", "附和", "认同", "补充"])
        has_next_action = bool(parsed.get("predicted_listener_next_action") or parsed.get("predicted_next_action"))
        score["PT_strong"] = bool(score["PT_weak"] and (has_followup_signal or has_next_action) and not overclaim)
    return score


def rate(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [row.get(key) for row in rows if row.get(key) is not None]
    if not vals:
        return None
    return sum(1 for v in vals if bool(v)) / len(vals)


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    base = {"trials_scored": len(rows), "error_count": sum(1 for row in rows if row.get("error")), "json_parse_success": rate(rows, "json_parse_success") or 0.0, "schema_validation_success": rate(rows, "schema_validation_success") or 0.0, "RC_weak": rate(rows, "RC_weak"), "RC_strong": rate(rows, "RC_strong"), "FB_weak": rate(rows, "FB_weak"), "FB_strong": rate(rows, "FB_strong"), "PT_weak": rate(rows, "PT_weak"), "PT_strong": rate(rows, "PT_strong"), "claim_verification_global": rate(rows, "claim_verification_global"), "claim_verification_local": rate(rows, "claim_verification_local"), "perspective_leakage_rate": rate(rows, "perspective_leakage") or 0.0, "forbidden_evidence_usage_rate": rate(rows, "forbidden_evidence_usage") or 0.0, "death_skill_overclaim_rate": rate(rows, "death_skill_overclaim") or 0.0, "evidence_support_rate": rate(rows, "evidence_support") or 0.0}
    components = {"RC_strong": base["RC_strong"] or 0.0, "FB_strong": base["FB_strong"] or 0.0, "PT_strong": base["PT_strong"] or 0.0, "claim_verification_local": base["claim_verification_local"] or 0.0, "evidence_support": base["evidence_support_rate"], "no_perspective_leakage": 1.0 - base["perspective_leakage_rate"], "no_death_skill_overclaim": 1.0 - base["death_skill_overclaim_rate"]}
    base["SoG_ToM_Core"] = sum(WEIGHTS[k] * components[k] for k in WEIGHTS)
    return base


def grouped(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get(key) or "unknown")].append(row)
    return {name: aggregate(items) for name, items in sorted(buckets.items())}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score SocialOmni-Goose Qwen eval responses.")
    parser.add_argument("--responses", type=Path, required=True, nargs="+")
    parser.add_argument("--hidden-gold", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--probe-groups", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, default=None)
    parser.add_argument("--track", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    responses = []
    for path in args.responses:
        responses.extend(read_jsonl(path))
    hidden_by_trial = {row["trial_id"]: row for row in read_jsonl(args.hidden_gold)}
    template_by_group = {row["probe_group_id"]: str(row.get("template") or "unknown") for row in read_jsonl(args.probe_groups)} if args.probe_groups and args.probe_groups.exists() else {}
    scores = []
    for response in responses:
        score = score_trial(response, hidden_by_trial.get(response.get("trial_id"), {}))
        score["template"] = template_by_group.get(str(response.get("probe_group_id")), "unknown")
        scores.append(score)
    report = {"track": args.track, "aggregate": aggregate(scores), "by_probe_type": grouped(scores, "probe_type"), "by_input_condition": grouped(scores, "input_condition"), "by_template": grouped(scores, "template"), "by_track": grouped(scores, "track"), "scores": scores, "worst_examples": [row for row in scores if row.get("error") or row.get("perspective_leakage") or row.get("death_skill_overclaim") or not row.get("json_parse_success")][:10], "best_examples": [row for row in scores if row.get("schema_validation_success") and row.get("evidence_support") and not row.get("perspective_leakage")][:10]}
    write_json(args.output, report)
    if args.summary_md:
        agg = report["aggregate"]
        lines = ["# Qwen3-Omni SocialOmni-Goose Evaluation Summary", "", f"Track: `{args.track or 'mixed'}`", f"Trials scored: {agg['trials_scored']}", f"SoG-ToM-Core: {agg['SoG_ToM_Core']:.4f}", "", "## Metrics"]
        for key in ["RC_weak", "RC_strong", "FB_weak", "FB_strong", "PT_weak", "PT_strong", "claim_verification_global", "claim_verification_local", "perspective_leakage_rate", "forbidden_evidence_usage_rate", "death_skill_overclaim_rate", "evidence_support_rate", "json_parse_success", "schema_validation_success"]:
            value = agg.get(key)
            lines.append(f"- {key}: {'n/a' if value is None else f'{value:.4f}'}")
        lines.extend(["", "## Notes", "- Qwen3-Omni is evaluated model only; hidden gold is scorer-only.", "- Death/body/blood frames must not certify skill, killer identity, alignment, or mechanism."])
        args.summary_md.parent.mkdir(parents=True, exist_ok=True)
        args.summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "aggregate": report["aggregate"]}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
