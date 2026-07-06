from __future__ import annotations

import collections
import json
import re
from pathlib import Path
from typing import Any

PLAYERS = ["Gemini", "baile", "beigang", "mojiang", "saoyi", "xiaolu"]
LOW_CERTAINTY = 0.55


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def certainty(value: Any) -> float:
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    text = str(value or "").strip().lower()
    if text in {"high", "高", "很高"}:
        return 0.85
    if text in {"medium", "中", "中等"}:
        return 0.6
    if text in {"low", "低"}:
        return 0.35
    return 0.5


def normalized_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    return re.sub(r"\s+", "", text)


def phase_id_from_path(path: Path) -> str:
    return path.parent.name


def interval_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 0 else 0.0


def event_key(event: dict[str, Any]) -> tuple[str, str, str]:
    actor = ",".join(sorted(str(x) for x in event.get("actors", []))) or "unknown"
    return (str(event.get("event_type", "other")), actor, str(event.get("location", "unknown")))


def should_merge_event(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if event_key(a) != event_key(b):
        return False
    if interval_iou(float(a["abs_start_sec"]), float(a["abs_end_sec"]), float(b["abs_start_sec"]), float(b["abs_end_sec"])) > 0.5:
        return True
    return abs(float(a["abs_start_sec"]) - float(b["abs_start_sec"])) <= 3.0 and abs(float(a["abs_end_sec"]) - float(b["abs_end_sec"])) <= 3.0


def claim_key(claim: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(claim.get("speaker", "unknown")),
        str(claim.get("claim_type", "other")),
        normalized_text(claim.get("normalized_content") or claim.get("content"))[:80],
    )


def should_merge_claim(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if claim_key(a) != claim_key(b):
        return False
    return abs(float(a["abs_start_sec"]) - float(b["abs_start_sec"])) <= 3.0


def infer_claim_type(text: str) -> str:
    if any(token in text for token in ["身份", "角色", "刺客", "鸭", "鹅", "警长", "拆弹", "验尸"]):
        return "role"
    if any(token in text for token in ["在", "上面", "下面", "右边", "左边", "大厅", "下水道", "街道", "位置"]):
        return "location"
    if any(token in text for token in ["杀", "刀", "打", "投", "票", "放逐"]):
        return "accusation"
    if any(token in text for token in ["我没", "不是我", "一直", "没有", "别投"]):
        return "defense"
    if any(token in text for token in ["看到", "看见", "目击"]):
        return "sighting"
    return "other"


def infer_strategic_role(claim_type: str, text: str) -> str:
    if claim_type in {"defense", "location"} or any(token in text for token in ["我没", "不是我", "一直", "没有"]):
        return "defense"
    if claim_type == "accusation" or any(token in text for token in ["投", "杀", "刀"]):
        return "accusation"
    if claim_type == "sighting":
        return "information_sharing"
    return "other"


def mentioned_players(text: str) -> list[str]:
    return [player for player in PLAYERS if player in text]


def load_gold_annotations(release_root: Path, game_id: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((release_root / "gold_annotations" / game_id).glob("*/*.json")):
        row = read_json(path)
        row["_path"] = path.as_posix()
        rows.append(row)
    return rows


def build_candidate_events(annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    counter = 0
    for ann in annotations:
        for obs in ann.get("observations", []):
            counter += 1
            source_pov = ann["player_id"]
            actor = obs.get("actor")
            actors = [actor] if actor in PLAYERS else []
            source_types = set(obs.get("source_types", []))
            events.append(
                {
                    "local_event_id": f"{ann['phase_id']}_{source_pov}_le_{counter:06d}",
                    "game_id": ann["game_id"],
                    "source_segment_ids": [ann["phase_id"]],
                    "source_povs": [source_pov],
                    "abs_start_sec": float(obs["abs_start_sec"]),
                    "abs_end_sec": float(obs["abs_end_sec"]),
                    "phase_type": ann["phase_type"],
                    "event_type": obs.get("event_type", "other"),
                    "actors": actors,
                    "patients": [],
                    "location": obs.get("location", "unknown"),
                    "description": obs.get("description", ""),
                    "direct_visual_evidence": [obs.get("evidence", "")] if "direct_visual_observation" in source_types else [],
                    "direct_audio_evidence": [obs.get("evidence", "")] if "speech_claim" in source_types else [],
                    "public_evidence": [obs.get("public_result_text", "")] if "public_result" in source_types else [],
                    "inferred_fields": [obs.get("inferred_belief_text", "")] if obs.get("inferred_belief_text") else [],
                    "certainty": certainty(obs.get("certainty")),
                    "needs_human_review": bool(obs.get("needs_human_review", False)),
                }
            )
    return events


def build_candidate_claims(annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    counter = 0
    for ann in annotations:
        for utt in ann.get("utterances", []):
            text = next(
                (
                    value.strip()
                    for value in [utt.get("claim_text"), utt.get("transcript"), utt.get("text"), utt.get("content")]
                    if isinstance(value, str) and value.strip()
                ),
                "",
            )
            if not normalized_text(text):
                continue
            speaker = utt.get("speaker", "unknown")
            if speaker not in PLAYERS:
                speaker = "unknown"
            claim_type = infer_claim_type(text)
            counter += 1
            claims.append(
                {
                    "local_claim_id": f"{ann['phase_id']}_{ann['player_id']}_lc_{counter:06d}",
                    "game_id": ann["game_id"],
                    "source_segment_ids": [ann["phase_id"]],
                    "speaker": speaker,
                    "heard_by": PLAYERS[:] if ann["phase_type"] in {"meeting", "final"} else [ann["player_id"]],
                    "abs_start_sec": float(utt["abs_start_sec"]),
                    "abs_end_sec": float(utt["abs_end_sec"]),
                    "claim_type": claim_type,
                    "content": next((value.strip() for value in [utt.get("transcript"), text] if isinstance(value, str) and value.strip()), text),
                    "normalized_content": text,
                    "time_referred": {"type": "relative_recent" if any(t in text for t in ["刚", "刚才", "上一轮"]) else "unknown"},
                    "target_entities": mentioned_players(text),
                    "related_event_ids": [],
                    "strategic_role": infer_strategic_role(claim_type, text),
                    "certainty": certainty(utt.get("certainty")),
                    "needs_human_review": bool(utt.get("needs_human_review", False)) or speaker == "unknown",
                }
            )
    return claims


def canonicalize_events(candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    canonical: list[dict[str, Any]] = []
    maps: list[dict[str, Any]] = []
    local_to_world: dict[str, str] = {}
    for cand in sorted(candidates, key=lambda row: (row["abs_start_sec"], row["event_type"])):
        match = next((event for event in canonical if should_merge_event(event, cand)), None)
        if match is None:
            world_id = f"ge_{len(canonical) + 1:06d}"
            match = dict(cand)
            match["world_event_id"] = world_id
            match.pop("local_event_id", None)
            match["_duplicates"] = [cand["local_event_id"]]
            canonical.append(match)
        else:
            match["abs_start_sec"] = min(float(match["abs_start_sec"]), float(cand["abs_start_sec"]))
            match["abs_end_sec"] = max(float(match["abs_end_sec"]), float(cand["abs_end_sec"]))
            match["source_segment_ids"] = sorted(set(match.get("source_segment_ids", []) + cand.get("source_segment_ids", [])))
            match["source_povs"] = sorted(set(match.get("source_povs", []) + cand.get("source_povs", [])))
            for key in ["direct_visual_evidence", "direct_audio_evidence", "public_evidence", "inferred_fields"]:
                match[key] = list(dict.fromkeys(match.get(key, []) + cand.get(key, [])))
            match["certainty"] = max(float(match.get("certainty", 0.0)), float(cand.get("certainty", 0.0)))
            match["needs_human_review"] = bool(match.get("needs_human_review", False) or cand.get("needs_human_review", False))
            match.setdefault("_duplicates", []).append(cand["local_event_id"])
        local_to_world[cand["local_event_id"]] = match["world_event_id"]

    for event in canonical:
        duplicates = event.pop("_duplicates", [])
        maps.append(
            {
                "canonical_event_id": event["world_event_id"],
                "duplicate_local_event_ids": duplicates,
                "abs_time_cluster": [event["abs_start_sec"], event["abs_end_sec"]],
                "canonical_source": duplicates[0] if duplicates else event["world_event_id"],
                "merge_reason": "merged by event_type/actor/location and temporal overlap or close absolute times",
            }
        )
    return canonical, maps, local_to_world


def canonicalize_claims(candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    canonical: list[dict[str, Any]] = []
    local_to_claim: dict[str, str] = {}
    for cand in sorted(candidates, key=lambda row: (row["abs_start_sec"], row["speaker"], row["claim_type"])):
        match = next((claim for claim in canonical if should_merge_claim(claim, cand)), None)
        if match is None:
            claim_id = f"claim_{len(canonical) + 1:06d}"
            match = dict(cand)
            match["claim_id"] = claim_id
            match.pop("local_claim_id", None)
            match["_duplicates"] = [cand["local_claim_id"]]
            canonical.append(match)
        else:
            match["abs_start_sec"] = min(float(match["abs_start_sec"]), float(cand["abs_start_sec"]))
            match["abs_end_sec"] = max(float(match["abs_end_sec"]), float(cand["abs_end_sec"]))
            match["source_segment_ids"] = sorted(set(match.get("source_segment_ids", []) + cand.get("source_segment_ids", [])))
            match["heard_by"] = sorted(set(match.get("heard_by", []) + cand.get("heard_by", [])))
            match["target_entities"] = sorted(set(match.get("target_entities", []) + cand.get("target_entities", [])))
            match["certainty"] = max(float(match.get("certainty", 0.0)), float(cand.get("certainty", 0.0)))
            match["needs_human_review"] = bool(match.get("needs_human_review", False) or cand.get("needs_human_review", False))
            match.setdefault("_duplicates", []).append(cand["local_claim_id"])
        local_to_claim[cand["local_claim_id"]] = match["claim_id"]
    for claim in canonical:
        claim.pop("_duplicates", None)
    return canonical, local_to_claim


def build_phase_events(annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for ann in annotations:
        if ann["phase_id"] not in seen:
            seen[ann["phase_id"]] = {
                "phase_event_id": f"phase_{ann['phase_index_global']:06d}",
                "game_id": ann["game_id"],
                "episode_id": ann["episode_id"],
                "phase_id": ann["phase_id"],
                "phase_type": ann["phase_type"],
                "phase_order_label_zh": ann["phase_order_label_zh"],
                "abs_start_sec": ann["aligned_start_sec"],
                "abs_end_sec": ann["aligned_end_sec"],
                "previous_phase_id": ann.get("previous_phase_id"),
                "next_phase_id": ann.get("next_phase_id"),
            }
    return sorted(seen.values(), key=lambda row: row["abs_start_sec"])


def build_visibility_edges(events: list[dict[str, Any]], claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    edges = []
    for event in events:
        source_povs = set(event.get("source_povs", []))
        is_public = event.get("phase_type") in {"meeting", "final"} or bool(event.get("public_evidence"))
        for player in PLAYERS:
            if player in source_povs:
                visibility = "direct_visual"
                evidence_ids = [event["world_event_id"]]
                explanation = f"{player} POV directly observed this event."
                confidence = event.get("certainty", 0.5)
            elif is_public:
                visibility = "public_ui"
                evidence_ids = [event["world_event_id"]]
                explanation = "The event is public in meeting/final UI or public evidence."
                confidence = min(0.8, event.get("certainty", 0.5))
            else:
                visibility = "not_visible"
                evidence_ids = []
                explanation = f"No evidence that {player} saw or heard this event before cutoff."
                confidence = 0.75
            edges.append(
                {
                    "edge_id": f"vis_{event['world_event_id']}_{player}",
                    "game_id": event["game_id"],
                    "event_id": event["world_event_id"],
                    "player_id": player,
                    "cutoff_abs_sec": event["abs_end_sec"],
                    "visibility": visibility,
                    "evidence_ids": evidence_ids,
                    "explanation": explanation,
                    "confidence": confidence,
                }
            )
    for claim in claims:
        for player in PLAYERS:
            edges.append(
                {
                    "edge_id": f"vis_{claim['claim_id']}_{player}",
                    "game_id": claim["game_id"],
                    "event_id": claim["claim_id"],
                    "player_id": player,
                    "cutoff_abs_sec": claim["abs_end_sec"],
                    "visibility": "heard_claim" if player in claim.get("heard_by", []) else "not_visible",
                    "evidence_ids": [claim["claim_id"]] if player in claim.get("heard_by", []) else [],
                    "explanation": "Player heard the claim." if player in claim.get("heard_by", []) else "No evidence the player heard this claim.",
                    "confidence": claim.get("certainty", 0.5),
                }
            )
    return edges


def event_claim_similarity(event: dict[str, Any], claim: dict[str, Any]) -> bool:
    text = normalized_text(" ".join([claim.get("content", ""), claim.get("normalized_content", "")]))
    desc = normalized_text(event.get("description", ""))
    if claim.get("speaker") in event.get("actors", []):
        return True
    if any(player in event.get("actors", []) for player in claim.get("target_entities", [])):
        return True
    if event.get("location") and normalized_text(event.get("location")) in text:
        return True
    if any(token in text and token in desc for token in ["杀", "刀", "尸体", "投", "右", "左", "上", "下", "任务"]):
        return True
    return False


def build_claim_truth_links(claims: list[dict[str, Any]], events: list[dict[str, Any]], edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_event_player = {(e["event_id"], e["player_id"]): e for e in edges}
    links = []
    for claim in claims:
        nearby = [
            event
            for event in events
            if event["abs_start_sec"] <= claim["abs_end_sec"] + 180
            and event["abs_end_sec"] >= claim["abs_start_sec"] - 180
            and event_claim_similarity(event, claim)
        ][:4]
        if not nearby:
            continue
        truth = "unverified"
        text = normalized_text(claim.get("content"))
        if claim.get("claim_type") in {"defense", "location"} and any(claim.get("speaker") in event.get("actors", []) for event in nearby):
            truth = "ambiguous"
        if any(token in text for token in ["没有", "没去", "不是", "一直"]) and any(claim.get("speaker") in event.get("actors", []) for event in nearby):
            truth = "contradicted"
        elif any(claim.get("speaker") in event.get("actors", []) for event in nearby):
            truth = "supported"
        local_awareness = {}
        for player in PLAYERS:
            player_edges = [by_event_player.get((event["world_event_id"], player), {}) for event in nearby]
            if any(edge.get("visibility") == "direct_visual" for edge in player_edges):
                local_awareness[player] = "has_contradictory_visual_evidence" if truth == "contradicted" else "has_supporting_visual_evidence"
            elif player in claim.get("heard_by", []):
                local_awareness[player] = "not_enough_information"
            else:
                local_awareness[player] = "unknown"
        links.append(
            {
                "claim_truth_link_id": f"ctl_{claim['claim_id']}_{nearby[0]['world_event_id']}",
                "claim_id": claim["claim_id"],
                "world_event_ids": [event["world_event_id"] for event in nearby],
                "truth_status_global": truth,
                "local_awareness_by_player": local_awareness,
                "explanation": "Heuristic link between claim and nearby canonical world events sharing speaker/actor/location/action cues.",
                "confidence": min(claim.get("certainty", 0.5), max(event.get("certainty", 0.5) for event in nearby)),
                "needs_human_review": truth in {"ambiguous", "unverified"},
            }
        )
    return links


def build_belief_snapshots(
    events: list[dict[str, Any]], claims: list[dict[str, Any]], edges: list[dict[str, Any]], probe_cutoffs: list[tuple[str, float]]
) -> list[dict[str, Any]]:
    edge_by_event_player = collections.defaultdict(list)
    for edge in edges:
        edge_by_event_player[(edge["event_id"], edge["player_id"])].append(edge)
    snapshots = []
    for target_player, cutoff in probe_cutoffs:
        public_history = []
        private_observations = []
        heard_claims = []
        hidden = []
        available_ids = []
        forbidden_ids = []
        for event in events:
            if event["abs_start_sec"] > cutoff:
                continue
            player_edges = edge_by_event_player.get((event["world_event_id"], target_player), [])
            visibility = player_edges[0]["visibility"] if player_edges else "unknown"
            if visibility in {"direct_visual", "direct_audio", "public_ui"}:
                evidence_id = f"obs_{target_player}_{event['world_event_id']}"
                private_observations.append(
                    {
                        "evidence_id": evidence_id,
                        "world_event_id": event["world_event_id"],
                        "visibility": visibility,
                        "content": event.get("description", ""),
                        "confidence": event.get("certainty", 0.5),
                    }
                )
                available_ids.append(evidence_id)
                if visibility == "public_ui":
                    public_history.append({"evidence_id": evidence_id, "abs_sec": event["abs_start_sec"], "type": "public_ui", "content": event.get("description", "")})
            elif visibility == "not_visible":
                hidden.append(
                    {
                        "world_event_id": event["world_event_id"],
                        "reason_hidden": "not_visible",
                        "visible_to": event.get("source_povs", []),
                    }
                )
                forbidden_ids.append(event["world_event_id"])
        for claim in claims:
            if claim["abs_start_sec"] <= cutoff and target_player in claim.get("heard_by", []):
                heard_claims.append(
                    {
                        "evidence_id": claim["claim_id"],
                        "claim_id": claim["claim_id"],
                        "speaker": claim.get("speaker", "unknown"),
                        "content": claim.get("content", ""),
                        "local_truth_awareness": "not_enough_information",
                    }
                )
                available_ids.append(claim["claim_id"])
                public_history.append({"evidence_id": claim["claim_id"], "abs_sec": claim["abs_start_sec"], "type": "meeting_statement", "content": claim.get("content", "")})
        inferred = [
            {
                "belief_id": f"belief_{target_player}_{int(cutoff):06d}",
                "about": "current hidden events and heard claims",
                "belief_label": "uncertain" if hidden else "unknown",
                "basis_evidence_ids": available_ids[:10],
                "confidence": 0.5,
            }
        ]
        snapshots.append(
            {
                "snapshot_id": f"mem_g001_{target_player}_{int(cutoff):06d}",
                "game_id": "g001",
                "target_player": target_player,
                "cutoff_abs_sec": cutoff,
                "public_history": public_history,
                "private_observations": private_observations,
                "heard_claims": heard_claims,
                "inferred_beliefs": inferred,
                "hidden_events_for_target": hidden,
                "forbidden_event_ids": sorted(set(forbidden_ids)),
                "available_evidence_ids": sorted(set(available_ids)),
            }
        )
    return snapshots


def build_oracle_ledger(release_root: Path, output_root: Path, game_id: str = "g001") -> dict[str, int]:
    annotations = load_gold_annotations(release_root, game_id)
    candidate_events = build_candidate_events(annotations)
    candidate_claims = build_candidate_claims(annotations)
    world_events, event_maps, _ = canonicalize_events(candidate_events)
    claims, _ = canonicalize_claims(candidate_claims)
    phase_events = build_phase_events(annotations)
    visibility_edges = build_visibility_edges(world_events, claims)
    claim_truth_links = build_claim_truth_links(claims, world_events, visibility_edges)

    cutoffs = []
    for link in claim_truth_links[:200]:
        claim = next((c for c in claims if c["claim_id"] == link["claim_id"]), None)
        if not claim:
            continue
        for player in PLAYERS:
            cutoffs.append((player, float(claim["abs_end_sec"]) + 1.0))
    for event in world_events[:200]:
        for player in PLAYERS:
            if player not in event.get("source_povs", []):
                cutoffs.append((player, float(event["abs_end_sec"]) + 3.0))
    seen_cutoffs = []
    seen = set()
    for player, cutoff in sorted(cutoffs, key=lambda x: (x[1], x[0])):
        key = (player, round(cutoff, 1))
        if key not in seen:
            seen.add(key)
            seen_cutoffs.append((player, cutoff))
    snapshots = build_belief_snapshots(world_events, claims, visibility_edges, seen_cutoffs[:360])

    ledger = output_root / "oracle_ledger"
    write_jsonl(ledger / "world_events.jsonl", world_events)
    write_jsonl(ledger / "claims.jsonl", claims)
    write_jsonl(ledger / "phase_events.jsonl", phase_events)
    write_jsonl(ledger / "visibility_edges.jsonl", visibility_edges)
    write_jsonl(ledger / "belief_memory_snapshots.jsonl", snapshots)
    write_jsonl(ledger / "claim_truth_links.jsonl", claim_truth_links)
    write_jsonl(ledger / "canonical_event_map.jsonl", event_maps)
    return {
        "world_events": len(world_events),
        "claims": len(claims),
        "phase_events": len(phase_events),
        "visibility_edges": len(visibility_edges),
        "belief_memory_snapshots": len(snapshots),
        "claim_truth_links": len(claim_truth_links),
        "canonical_event_map": len(event_maps),
    }


def load_ledger(root: Path) -> dict[str, list[dict[str, Any]]]:
    ledger = root / "oracle_ledger"
    return {
        "world_events": read_jsonl(ledger / "world_events.jsonl"),
        "claims": read_jsonl(ledger / "claims.jsonl"),
        "phase_events": read_jsonl(ledger / "phase_events.jsonl"),
        "visibility_edges": read_jsonl(ledger / "visibility_edges.jsonl"),
        "belief_memory_snapshots": read_jsonl(ledger / "belief_memory_snapshots.jsonl"),
        "claim_truth_links": read_jsonl(ledger / "claim_truth_links.jsonl"),
        "canonical_event_map": read_jsonl(ledger / "canonical_event_map.jsonl"),
    }


def edge_lookup(edges: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(edge["event_id"], edge["player_id"]): edge for edge in edges}


def claim_by_id(claims: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {claim["claim_id"]: claim for claim in claims}


def event_by_id(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {event["world_event_id"]: event for event in events}


def pick_other_player(event: dict[str, Any], target_player: str) -> str:
    for player in event.get("source_povs", []):
        if player != target_player:
            return player
    for player in PLAYERS:
        if player != target_player:
            return player
    return target_player


def make_probe_group(
    idx: int,
    event: dict[str, Any],
    target_player: str,
    template: str,
    related_claims: list[dict[str, Any]],
    quality_review: bool = False,
) -> dict[str, Any]:
    cutoff = float(event["abs_end_sec"]) + 3.0
    available = [claim["claim_id"] for claim in related_claims if target_player in claim.get("heard_by", [])]
    if template == "vote_influence":
        qtype = "trust_update"
    elif template == "private_witness":
        qtype = "hidden_event_awareness"
    elif template == "delayed_public_reveal":
        qtype = "hidden_event_awareness"
    else:
        qtype = "claim_truth_vs_claim_awareness" if related_claims else "hidden_event_awareness"
    families = ["false_belief", "representational_change"]
    if related_claims:
        families.append("claim_verification")
    if template in {"vote_influence", "contradicted_alibi", "private_witness"}:
        families.append("perspective_taking")
    if template == "vote_influence":
        families.append("strategy_communication")
    if template == "delayed_public_reveal":
        families.append("delayed_public_reveal")
    return {
        "probe_group_id": f"g001_pg_{idx:06d}_{target_player}",
        "game_id": event["game_id"],
        "source_segment_ids": event.get("source_segment_ids", []),
        "cutoff_abs_sec": cutoff,
        "target_player": target_player,
        "query_variable": {
            "type": qtype,
            "description": f"Whether {target_player} knows or can verify: {event.get('description', '')}",
        },
        "anchor_event_ids": [event["world_event_id"]],
        "related_claim_ids": [claim["claim_id"] for claim in related_claims],
        "hidden_event_ids_for_target": [event["world_event_id"]],
        "available_evidence_ids_for_target": available,
        "selection_reason": f"{event['world_event_id']} is visible to {event.get('source_povs', [])} but hidden from {target_player}; related claims={available}.",
        "diagnostic_families": list(dict.fromkeys(families)),
        "template": template,
        "quality": {
            "visibility_confidence": 0.75,
            "claim_truth_confidence": min([claim.get("certainty", 0.5) for claim in related_claims] + [0.6]),
            "timestamp_confidence": event.get("certainty", 0.5),
            "needs_human_review": quality_review or event.get("needs_human_review", False),
        },
        "needs_human_review": quality_review or event.get("needs_human_review", False),
        "gold_source": "qwen_weak",
    }


def select_probe_groups(ledger: dict[str, list[dict[str, Any]]], limit: int = 240) -> list[dict[str, Any]]:
    events = ledger["world_events"]
    claims = ledger["claims"]
    links = ledger["claim_truth_links"]
    edges = edge_lookup(ledger["visibility_edges"])
    claims_by_id = claim_by_id(claims)
    events_by_id = event_by_id(events)
    claims_for_event: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    truth_for_event: dict[str, str] = {}
    for link in links:
        claim = claims_by_id.get(link["claim_id"])
        if not claim:
            continue
        for event_id in link.get("world_event_ids", []):
            claims_for_event[event_id].append(claim)
            truth_for_event[event_id] = link.get("truth_status_global", "unverified")

    candidates: dict[str, list[tuple[dict[str, Any], str, list[dict[str, Any]], bool]]] = collections.defaultdict(list)
    for event in sorted(events, key=lambda row: (-len(row.get("source_povs", [])), row["abs_start_sec"])):
        source_povs = set(event.get("source_povs", []))
        if not source_povs or event.get("phase_type") in {"meeting", "final"}:
            continue
        hidden_targets = [
            player
            for player in PLAYERS
            if player not in source_povs and edges.get((event["world_event_id"], player), {}).get("visibility") == "not_visible"
        ]
        related = claims_for_event.get(event["world_event_id"], [])
        for target in hidden_targets[:3]:
            heard_related = [claim for claim in related if target in claim.get("heard_by", [])]
            if related and not heard_related:
                continue
            template = "contradicted_alibi" if truth_for_event.get(event["world_event_id"]) == "contradicted" else "hidden_event_awareness"
            candidates[template].append((event, target, heard_related, False))

    for event in sorted(events, key=lambda row: row["abs_start_sec"]):
        if len(event.get("source_povs", [])) != 1:
            continue
        witness = event["source_povs"][0]
        candidates["private_witness"].append((event, witness, claims_for_event.get(event["world_event_id"], []), True))

    for link in links:
        if link.get("truth_status_global") not in {"supported", "contradicted", "ambiguous"}:
            continue
        claim = claims_by_id.get(link["claim_id"])
        if not claim or claim.get("strategic_role") not in {"accusation", "defense", "information_sharing"}:
            continue
        speaker = claim.get("speaker")
        if speaker not in PLAYERS:
            continue
        for event_id in link.get("world_event_ids", [])[:2]:
            event = events_by_id.get(event_id)
            if not event or event.get("phase_type") in {"meeting", "final"}:
                continue
            for listener in claim.get("heard_by", []):
                if listener == speaker or listener not in PLAYERS:
                    continue
                listener_edge = edges.get((event_id, listener), {})
                speaker_edge = edges.get((event_id, speaker), {})
                if listener_edge.get("visibility") == speaker_edge.get("visibility"):
                    continue
                candidates["vote_influence"].append((event, listener, [claim], bool(link.get("needs_human_review", False))))
                break

    meeting_claims = [claim for claim in claims if len(claim.get("heard_by", [])) >= 4]
    for event in sorted(events, key=lambda row: row["abs_start_sec"]):
        source_povs = set(event.get("source_povs", []))
        if not source_povs or len(source_povs) >= len(PLAYERS) or event.get("phase_type") in {"meeting", "final"}:
            continue
        later_public_claims = [
            claim
            for claim in meeting_claims
            if event["abs_end_sec"] < claim["abs_start_sec"] <= event["abs_end_sec"] + 900
        ][:3]
        if not later_public_claims:
            continue
        for target in PLAYERS:
            if target in source_povs:
                continue
            if edges.get((event["world_event_id"], target), {}).get("visibility") != "not_visible":
                continue
            candidates["delayed_public_reveal"].append((event, target, later_public_claims, True))
            break

    template_order = ["contradicted_alibi", "hidden_event_awareness", "private_witness", "vote_influence", "delayed_public_reveal"]
    minimums = {
        "contradicted_alibi": max(1, int(limit * 0.10)),
        "hidden_event_awareness": max(1, int(limit * 0.35)),
        "private_witness": max(1, int(limit * 0.15)),
        "vote_influence": max(1, int(limit * 0.15)),
        "delayed_public_reveal": max(1, int(limit * 0.10)),
    }
    groups = []
    idx = 0
    used_keys: set[tuple[str, str, str]] = set()

    def append_candidate(template: str, candidate: tuple[dict[str, Any], str, list[dict[str, Any]], bool]) -> None:
        nonlocal idx
        event, target, related_claims, quality_review = candidate
        key = (template, event["world_event_id"], target)
        if key in used_keys or len(groups) >= limit:
            return
        used_keys.add(key)
        idx += 1
        group = make_probe_group(idx, event, target, template, related_claims, quality_review=quality_review)
        if template == "private_witness":
            group["hidden_event_ids_for_target"] = []
            group["available_evidence_ids_for_target"] = [event["world_event_id"]]
            group["selection_reason"] = f"{target} directly saw {event['world_event_id']} while most other players did not."
        elif template == "vote_influence":
            claim_ids = [claim["claim_id"] for claim in related_claims]
            group["cutoff_abs_sec"] = max(float(claim["abs_end_sec"]) for claim in related_claims) + 2.0
            edge = edges.get((event["world_event_id"], target), {})
            if edge.get("visibility") in {"direct_visual", "direct_audio", "public_ui"}:
                group["hidden_event_ids_for_target"] = []
                group["available_evidence_ids_for_target"] = sorted(
                    set(group.get("available_evidence_ids_for_target", []) + edge.get("evidence_ids", []))
                )
            group["selection_reason"] = f"Strategic claim(s) {claim_ids} may influence listener {target}, whose information state differs from the speaker."
        elif template == "delayed_public_reveal":
            claim_ids = [claim["claim_id"] for claim in related_claims]
            group["selection_reason"] = f"{event['world_event_id']} is private before cutoff and later enters public discussion via claim(s) {claim_ids}."
        groups.append(group)

    for template in template_order:
        for candidate in candidates.get(template, [])[: minimums[template]]:
            append_candidate(template, candidate)
            if len(groups) >= limit:
                break
    for template in template_order:
        for candidate in candidates.get(template, []):
            append_candidate(template, candidate)
            if len(groups) >= limit:
                break
        if len(groups) >= limit:
            break

    return groups[:limit]


def snapshot_for(snapshots: list[dict[str, Any]], player: str, cutoff: float) -> dict[str, Any]:
    candidates = [s for s in snapshots if s["target_player"] == player and s["cutoff_abs_sec"] <= cutoff + 1e-6]
    if not candidates:
        return {"available_evidence_ids": [], "forbidden_event_ids": [], "public_history": [], "private_observations": [], "heard_claims": []}
    return max(candidates, key=lambda row: row["cutoff_abs_sec"])


def compact_context(snapshot: dict[str, Any]) -> str:
    return json.dumps(
        {
            "public_history": snapshot.get("public_history", [])[:12],
            "private_observations": snapshot.get("private_observations", [])[:12],
            "heard_claims": snapshot.get("heard_claims", [])[:12],
            "inferred_beliefs": snapshot.get("inferred_beliefs", [])[:6],
        },
        ensure_ascii=False,
    )


def public_query_form(group: dict[str, Any]) -> str:
    query = group.get("query_variable", {}) if isinstance(group.get("query_variable"), dict) else {}
    return json.dumps(
        {
            "query_type": query.get("type", "unknown"),
            "target_player": group.get("target_player"),
            "cutoff_abs_sec": group.get("cutoff_abs_sec"),
            "related_claim_ids": group.get("related_claim_ids", []),
            "available_evidence_ids_for_target": group.get("available_evidence_ids_for_target", []),
            "task": "Judge what the target player could know, verify, doubt, or remain uncertain about from target-available evidence only.",
        },
        ensure_ascii=False,
        indent=2,
    )


def speaker_listener_public_model(listener_snapshot: dict[str, Any], listener: str) -> str:
    return json.dumps(
        {
            "listener": listener,
            "shared_public_history": listener_snapshot.get("public_history", [])[:12],
            "public_or_heard_claims": listener_snapshot.get("heard_claims", [])[:12],
            "private_observations_excluded": True,
        },
        ensure_ascii=False,
    )


def expected_schema_for(probe_type: str) -> dict[str, Any]:
    if probe_type == "A_pre_reveal_belief":
        return {"knows_truth": "boolean", "belief_label": "believes_true|believes_false|uncertain|does_not_know|unknown", "likely_belief": "string", "suspicion_update": "increase|decrease|unchanged|unknown", "evidence_ids": "array[string]", "confidence": "number"}
    if probe_type == "B_post_reveal_reconstruct_previous_belief":
        return {"target_knew_truth_at_cutoff": "boolean", "reconstructed_prior_belief": "string", "must_not_use_revealed_truth_as_prior_evidence": "boolean", "evidence_ids_available_at_cutoff": "array[string]", "confidence": "number"}
    if probe_type == "C_other_agent_false_belief":
        return {"other_player": "string", "other_player_knew_truth_at_cutoff": "boolean", "other_player_likely_belief": "string", "evidence_ids_available_to_other_player": "array[string]", "confidence": "number"}
    return {"speaker": "string", "listener": "string", "predicted_listener_trust_update": "increase|decrease|unchanged|unknown", "predicted_listener_next_action": "accuse|vote|defend|ignore|follow|avoid|unknown", "reason_from_speaker_perspective": "string", "confidence": "number"}


def probe_prompt_header(group: dict[str, Any]) -> str:
    return (
        "Dataset: SocialOmni-Goose\n"
        "Game: Goose Goose Duck / 鹅鸭杀风格多人社交博弈\n"
        "Players: Gemini, baile, beigang, mojiang, saoyi, xiaolu\n"
        "Time rule: abs_sec = aligned_start_sec + local_sec.\n"
        "Epistemic rule: distinguish oracle truth from what each player could know at cutoff.\n"
        f"Probe group: {group['probe_group_id']}\n"
        f"Target player: {group['target_player']}\n"
        f"Cutoff abs sec: {group['cutoff_abs_sec']}\n"
    )


def generate_probes_for_group(group: dict[str, Any], ledger: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    events = event_by_id(ledger["world_events"])
    claims = claim_by_id(ledger["claims"])
    snapshots = ledger["belief_memory_snapshots"]
    target = group["target_player"]
    cutoff = group["cutoff_abs_sec"]
    target_snapshot = snapshot_for(snapshots, target, cutoff)
    anchor_events = [events[event_id] for event_id in group.get("anchor_event_ids", []) if event_id in events]
    related_claims = [claims[claim_id] for claim_id in group.get("related_claim_ids", []) if claim_id in claims]
    other_player = pick_other_player(anchor_events[0], target) if anchor_events else next(player for player in PLAYERS if player != target)
    other_snapshot = snapshot_for(snapshots, other_player, cutoff)
    oracle_truth = json.dumps(anchor_events, ensure_ascii=False)
    related_claim_text = json.dumps(related_claims, ensure_ascii=False)
    header = probe_prompt_header(group)
    target_context = compact_context(target_snapshot)
    other_context = compact_context(other_snapshot)
    query_form = public_query_form(group)
    forbidden = group.get("hidden_event_ids_for_target", [])
    acceptable = group.get("available_evidence_ids_for_target", [])
    anchor_event_ids = [event["world_event_id"] for event in anchor_events]
    target_has_anchor_evidence = bool(set(anchor_event_ids) & set(acceptable)) or any(
        target in event.get("source_povs", []) for event in anchor_events
    )
    target_knows_truth_at_cutoff = target_has_anchor_evidence and not forbidden
    target_belief_label = "knows_truth" if target_knows_truth_at_cutoff else ("does_not_know" if forbidden else "uncertain")

    probes = [
        {
            "probe_id": f"{group['probe_group_id']}_A",
            "probe_group_id": group["probe_group_id"],
            "probe_type": "A_pre_reveal_belief",
            "input_condition": "target_available_events",
            "target_player": target,
            "cutoff_abs_sec": cutoff,
            "prompt": header + f"\nOnly use the target player's available information before cutoff. Do not use hidden oracle events.\nQUERY_VARIABLE_PUBLIC_FORM_JSON:\n{query_form}\nTARGET_AVAILABLE_CONTEXT_JSON:\n{target_context}\nQUESTION: From {target}'s perspective at cutoff, can {target} verify, doubt, believe, or remain uncertain about the public-form query using only target-available evidence?\nReturn strict JSON matching the schema.",
            "expected_output_schema": expected_schema_for("A_pre_reveal_belief"),
            "forbidden_event_ids": forbidden,
            "acceptable_evidence_ids": acceptable,
            "gold_source": "qwen_weak",
        },
        {
            "probe_id": f"{group['probe_group_id']}_B",
            "probe_group_id": group["probe_group_id"],
            "probe_type": "B_post_reveal_reconstruct_previous_belief",
            "input_condition": "oracle_truth_revealed",
            "target_player": target,
            "cutoff_abs_sec": cutoff,
            "prompt": header + f"\nORACLE_TRUTH_JSON:\n{oracle_truth}\nTARGET_AVAILABLE_CONTEXT_BEFORE_REVEAL_JSON:\n{target_context}\nQUESTION: Now that oracle truth is revealed, reconstruct {target}'s belief at the earlier cutoff. Do not treat revealed truth as evidence {target} had at cutoff. Do not rely on prior probe responses.\nReturn strict JSON matching the schema.",
            "expected_output_schema": expected_schema_for("B_post_reveal_reconstruct_previous_belief"),
            "forbidden_event_ids": forbidden,
            "acceptable_evidence_ids": acceptable,
            "gold_source": "qwen_weak",
        },
        {
            "probe_id": f"{group['probe_group_id']}_C",
            "probe_group_id": group["probe_group_id"],
            "probe_type": "C_other_agent_false_belief",
            "input_condition": "global_to_perspective",
            "target_player": target,
            "cutoff_abs_sec": cutoff,
            "prompt": header + f"\nORACLE_TRUTH_JSON:\n{oracle_truth}\nOTHER_PLAYER: {other_player}\nOTHER_PLAYER_AVAILABLE_CONTEXT_BEFORE_REVEAL_JSON:\n{other_context}\nQUESTION: From {other_player}'s perspective at cutoff, did {other_player} know the oracle truth? What would {other_player} likely believe before reveal?\nReturn strict JSON matching the schema.",
            "expected_output_schema": expected_schema_for("C_other_agent_false_belief"),
            "forbidden_event_ids": [],
            "acceptable_evidence_ids": other_snapshot.get("available_evidence_ids", []),
            "gold_source": "qwen_weak",
        },
    ]

    if related_claims:
        speaker = related_claims[0].get("speaker", "unknown")
        speaker_snapshot = snapshot_for(snapshots, speaker, cutoff) if speaker in PLAYERS else {"available_evidence_ids": [], "public_history": [], "private_observations": [], "heard_claims": []}
        speaker_context = compact_context(speaker_snapshot)
        listener_public_model = speaker_listener_public_model(target_snapshot, target)
        probes.append(
            {
                "probe_id": f"{group['probe_group_id']}_D",
                "probe_group_id": group["probe_group_id"],
                "probe_type": "D_perspective_taking_prediction",
                "input_condition": "speaker_perspective",
                "target_player": target,
                "cutoff_abs_sec": cutoff,
                "prompt": header + f"\nRELATED_CLAIMS_JSON:\n{related_claim_text}\nSPEAKER_AVAILABLE_CONTEXT_JSON:\n{speaker_context}\nSPEAKER_MODEL_OF_LISTENER_PUBLIC_HISTORY_JSON:\n{listener_public_model}\nQUESTION: From speaker {speaker}'s perspective after making the strategic claim, predict how listener {target} would interpret the claim and how their trust/action may change. Use only what the speaker could know or reasonably model from public listener history.\nReturn strict JSON matching the schema.",
                "expected_output_schema": expected_schema_for("D_perspective_taking_prediction"),
                "forbidden_event_ids": forbidden,
                "acceptable_evidence_ids": acceptable,
                "gold_source": "qwen_weak",
            }
        )

    hidden_gold = {
        "probe_group_id": group["probe_group_id"],
        "A_expected_weak": {"knows_truth": target_knows_truth_at_cutoff, "must_not_use_event_ids": forbidden},
        "B_RC_weak": {
            "must_not_answer_as_if_target_saw_hidden_events": bool(forbidden),
            "target_had_anchor_evidence_at_cutoff": target_has_anchor_evidence,
        },
        "B_RC_strong_reference": {"belief_label": target_belief_label, "key_evidence_class": "available_claim_or_partial_observation"},
        "C_FB_weak": {"other_player": other_player, "other_player_knows_truth": other_player in (anchor_events[0].get("source_povs", []) if anchor_events else [])},
        "C_FB_strong_reference": {"other_player": other_player, "key_evidence_class": "other_player_available_context"},
        "D_PT_reference": {"listener": target, "requires_information_state_difference": bool(related_claims)},
        "forbidden_event_ids_for_target": forbidden,
        "acceptable_evidence_ids_for_target": acceptable,
        "claim_truth_global": "unverified",
        "claim_awareness_local_target": "can_verify_from_direct_evidence"
        if target_knows_truth_at_cutoff
        else ("not_enough_information" if related_claims else "unknown"),
        "gold_source": "qwen_weak",
    }
    quality = {
        "probe_group_id": group["probe_group_id"],
        "keep": len(probes) >= 3 and bool(group.get("hidden_event_ids_for_target") or group.get("available_evidence_ids_for_target")),
        "diagnostic_score": 0.8 if len(probes) >= 4 else 0.7,
        "failure_reasons": [],
        "needs_human_review": bool(group.get("needs_human_review", False) or group.get("quality", {}).get("needs_human_review", False)),
        "recommended_gold_source": "qwen_weak",
    }
    return probes, hidden_gold, quality


def build_decrypto_diagnostics(ledger_root: Path, output_root: Path, limit: int = 240) -> dict[str, int]:
    ledger = load_ledger(ledger_root)
    groups = select_probe_groups(ledger, limit=limit)
    all_probes: list[dict[str, Any]] = []
    by_type: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    hidden_gold: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    for group in groups:
        probes, gold, quality = generate_probes_for_group(group, ledger)
        all_probes.extend(probes)
        for probe in probes:
            by_type[probe["probe_type"]].append(probe)
        hidden_gold.append(gold)
        quality_rows.append(quality)

    diag = output_root / "diagnostics"
    write_jsonl(diag / "probe_groups.jsonl", groups)
    write_jsonl(diag / "probes_A_pre_reveal.jsonl", by_type["A_pre_reveal_belief"])
    write_jsonl(diag / "probes_B_reconstruct.jsonl", by_type["B_post_reveal_reconstruct_previous_belief"])
    write_jsonl(diag / "probes_C_false_belief.jsonl", by_type["C_other_agent_false_belief"])
    write_jsonl(diag / "probes_D_perspective_taking.jsonl", by_type["D_perspective_taking_prediction"])
    write_jsonl(diag / "hidden_gold.jsonl", hidden_gold)
    write_jsonl(diag / "diagnostic_quality.jsonl", quality_rows)
    return {
        "probe_groups": len(groups),
        "probes": len(all_probes),
        "A": len(by_type["A_pre_reveal_belief"]),
        "B": len(by_type["B_post_reveal_reconstruct_previous_belief"]),
        "C": len(by_type["C_other_agent_false_belief"]),
        "D": len(by_type["D_perspective_taking_prediction"]),
        "hidden_gold": len(hidden_gold),
    }


def export_social_omni_goose_benchmark(annotation_root: Path, benchmark_root: Path) -> dict[str, int]:
    diag = annotation_root / "diagnostics"
    groups = read_jsonl(diag / "probe_groups.jsonl")
    probes = (
        read_jsonl(diag / "probes_A_pre_reveal.jsonl")
        + read_jsonl(diag / "probes_B_reconstruct.jsonl")
        + read_jsonl(diag / "probes_C_false_belief.jsonl")
        + read_jsonl(diag / "probes_D_perspective_taking.jsonl")
    )
    hidden_gold = read_jsonl(diag / "hidden_gold.jsonl")
    quality = read_jsonl(diag / "diagnostic_quality.jsonl")
    hidden_by_group = {row["probe_group_id"]: row for row in hidden_gold}

    interactive = benchmark_root / "interactive_diagnostics"
    static = benchmark_root / "static_trials"
    reports = benchmark_root / "reports"
    write_jsonl(interactive / "probe_groups.jsonl", groups)
    write_jsonl(interactive / "prompts.jsonl", probes)
    write_jsonl(interactive / "hidden_gold.jsonl", hidden_gold)
    scoring_rules = {
        "RC_weak": "B must not attribute hidden revealed truth to target_player at cutoff.",
        "RC_strong": "B must match A on main belief label and key evidence class.",
        "FB_weak": "C must identify whether other player knew hidden truth at cutoff.",
        "FB_strong": "C must reconstruct other player's concrete pre-reveal belief.",
        "PT_weak": "D must distinguish speaker and listener information states.",
        "PT_strong": "D must predict listener reaction consistent with later evidence when available.",
        "claim_verification_global": "Model identifies whether a claim is globally supported, contradicted, unverified, or ambiguous.",
        "claim_verification_local": "Model distinguishes global claim truth from what the target player could know at cutoff.",
        "perspective_leakage": "Model references forbidden_event_ids while simulating a limited perspective.",
        "forbidden_evidence_usage_rate": "Fraction of scored groups whose answers cite hidden or forbidden event IDs.",
        "evidence_support_rate": "Fraction of scored groups that avoid forbidden evidence and cite only allowed evidence classes.",
        "json_parse_success": "Response can be parsed as JSON.",
        "schema_validation_success": "Parsed response matches the probe's expected output schema sufficiently for scoring.",
    }
    write_json(interactive / "scoring_rules.json", scoring_rules)

    trials = []
    gold_rows = []
    hidden_rows = []
    input_conditions: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for probe in probes:
        group_gold = hidden_by_group.get(probe["probe_group_id"], {})
        public_trial = {
            "trial_id": probe["probe_id"],
            "probe_group_id": probe["probe_group_id"],
            "game_id": "g001",
            "probe_type": probe["probe_type"],
            "target_player": probe["target_player"],
            "cutoff_abs_sec": probe["cutoff_abs_sec"],
            "input_condition": probe["input_condition"],
            "prompt": probe["prompt"],
            "expected_output_schema": probe["expected_output_schema"],
            "gold_source": probe["gold_source"],
        }
        trials.append(public_trial)
        input_conditions[probe["input_condition"]].append(public_trial)
        gold_rows.append(
            {
                "trial_id": probe["probe_id"],
                "probe_group_id": probe["probe_group_id"],
                "gold_source": probe["gold_source"],
                "acceptable_evidence_ids": probe.get("acceptable_evidence_ids", []),
                "metrics": [
                    "RC_weak",
                    "RC_strong",
                    "FB_weak",
                    "FB_strong",
                    "PT_weak",
                    "PT_strong",
                    "claim_verification_global",
                    "claim_verification_local",
                    "perspective_leakage",
                    "forbidden_evidence_usage_rate",
                    "evidence_support_rate",
                    "json_parse_success",
                    "schema_validation_success",
                ],
            }
        )
        hidden_rows.append(
            {
                "trial_id": probe["probe_id"],
                "probe_group_id": probe["probe_group_id"],
                "gold_source": group_gold.get("gold_source", probe["gold_source"]),
                "hidden_gold": group_gold,
                "forbidden_event_ids": probe.get("forbidden_event_ids", []),
                "acceptable_evidence_ids": probe.get("acceptable_evidence_ids", []),
            }
        )

    write_jsonl(static / "trials.jsonl", trials)
    write_jsonl(static / "gold.jsonl", gold_rows)
    write_jsonl(static / "hidden_gold.jsonl", hidden_rows)
    for condition, rows in input_conditions.items():
        write_jsonl(static / "input_conditions" / f"{condition}.jsonl", rows)
    track_rows = {
        "raw_pov_video": [
            {
                "trial_id": trial["trial_id"],
                "probe_group_id": trial["probe_group_id"],
                "track": "raw_pov_video_generalist",
                "target_player": trial["target_player"],
                "cutoff_abs_sec": trial["cutoff_abs_sec"],
                "input_condition": trial["input_condition"],
                "requires_target_pov_video": True,
                "prompt": trial["prompt"],
            }
            for trial in trials
        ],
        "structured_perspective": [
            trial
            for trial in trials
            if trial["input_condition"] in {"target_available_events", "public_history_only", "speaker_perspective"}
        ],
        "global_to_perspective": [
            trial
            for trial in trials
            if trial["input_condition"] in {"oracle_truth_revealed", "global_to_perspective"}
        ],
        "specialist_agent": [
            {
                **trial,
                "allowed_resources": ["oracle_ledger", "visibility_edges", "belief_memory_snapshots", "claim_truth_links"],
            }
            for trial in trials
        ],
    }
    for track_name, rows in track_rows.items():
        write_jsonl(static / "input_conditions" / f"{track_name}.jsonl", rows)
    review_queue = [q for q in quality if q.get("needs_human_review") or q.get("diagnostic_score", 1.0) < 0.65]
    write_jsonl(benchmark_root / "human_review_queue.jsonl", review_queue)
    (reports / "diagnostic_quality.md").parent.mkdir(parents=True, exist_ok=True)
    (reports / "benchmark_card.md").write_text(
        "# SocialOmni-Goose-v1 Benchmark Card\n\n"
        "SocialOmni-Goose-v1 is a trajectory-derived Theory-of-Mind diagnostic benchmark "
        "built from a strictly aligned 6-POV Goose Goose Duck replay. It uses oracle "
        "trajectory ledgers, visibility projections, claim-truth links, and Decrypto-style "
        "A/B/C/D probe groups to evaluate representational change, false belief, claim "
        "verification, perspective taking, strategy communication, and perspective leakage.\n\n"
        "Segments are storage units only. The benchmark unit is a trajectory node: "
        "`cutoff_abs_sec + target_player + query_variable + information_gap`. "
        "Each node is selected from an oracle trajectory ledger and projected into player-local "
        "knowledge states before probe generation.\n\n"
        "Probe groups follow a Decrypto-style diagnostic mechanism. A asks for the target player's "
        "pre-reveal belief using only target-available evidence. B reveals oracle truth but asks the "
        "model to reconstruct the target's earlier belief without truth contamination. C asks for "
        "another player's false or incomplete belief. D asks a speaker to predict how a listener "
        "will interpret a strategic claim from speaker-safe context.\n\n"
        "Interactive diagnostics evaluate A/B/C/D consistency and representational change across "
        "multiple prompts. Static trials provide frozen public prompts and separate hidden gold for "
        "leaderboard-style scoring. `trials.jsonl` must not expose hidden gold or forbidden event IDs; "
        "`hidden_gold.jsonl` is scorer-only.\n\n"
        "Perspective leakage means that an answer simulating a limited player perspective cites "
        "hidden oracle facts, forbidden event IDs, or evidence only available to another POV. "
        "`gold_source` is `qwen_weak` by default, `qwen_checked` only after high-quality video review "
        "and Codex merge gating, and `human_verified` only after explicit human review.\n\n"
        f"- probe_groups: {len(groups)}\n"
        f"- prompts: {len(probes)}\n"
        f"- human_review_queue: {len(review_queue)}\n"
        "- gold_source: qwen_weak unless later promoted by explicit checking or human review\n",
        encoding="utf-8",
    )
    (reports / "diagnostic_quality.md").write_text(
        f"# Diagnostic Quality\n\n- probe_groups: {len(groups)}\n- prompts: {len(probes)}\n- review_queue: {len(review_queue)}\n",
        encoding="utf-8",
    )
    (reports / "annotation_quality.md").write_text(
        "# Annotation Quality\n\nAll labels are weak automatic labels unless promoted to qwen_checked or human_verified.\n",
        encoding="utf-8",
    )
    return {"probe_groups": len(groups), "prompts": len(probes), "static_trials": len(trials), "review_queue": len(review_queue)}


def parse_answer(raw: str) -> tuple[dict[str, Any], bool]:
    try:
        return json.loads(raw), True
    except Exception:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0)), True
            except Exception:
                return {}, False
    return {}, False


def uses_forbidden_evidence(parsed: dict[str, Any], forbidden: list[str]) -> bool:
    text = json.dumps(parsed, ensure_ascii=False)
    return any(item and item in text for item in forbidden)


def first_present(mapping: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def normalize_enum(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "true": "supported",
        "false": "contradicted",
        "not_enough_info": "not_enough_information",
        "not_enough_evidence": "not_enough_information",
        "unknown": "unknown",
    }
    return aliases.get(text, text)


def answer_claim_truth(answers: dict[str, dict[str, Any]]) -> str | None:
    keys = ["claim_truth_global", "global_claim_truth", "truth_status_global", "claim_global_truth"]
    for letter in ["A", "B", "C", "D"]:
        value = first_present(answers.get(letter, {}), keys)
        normalized = normalize_enum(value)
        if normalized:
            return normalized
    return None


def answer_local_awareness(answers: dict[str, dict[str, Any]]) -> str | None:
    keys = [
        "claim_awareness_local_target",
        "local_claim_awareness",
        "target_claim_awareness",
        "target_can_know_claim_truth",
        "target_has_evidence_for_claim_truth",
    ]
    for letter in ["A", "B", "C", "D"]:
        answer = answers.get(letter, {})
        value = first_present(answer, keys)
        if isinstance(value, bool):
            return "has_evidence" if value else "not_enough_information"
        normalized = normalize_enum(value)
        if normalized:
            return normalized
    return None


def evidence_ids_from_answer(parsed: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    stack: list[Any] = [parsed]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            for key, value in item.items():
                if key in {"evidence_id", "event_id", "claim_id", "world_event_id"} and isinstance(value, str):
                    ids.add(value)
                elif key.endswith("_ids") and isinstance(value, list):
                    ids.update(str(entry) for entry in value if isinstance(entry, str))
                else:
                    stack.append(value)
        elif isinstance(item, list):
            stack.extend(item)
    return ids


def schema_ok_for_answer(letter: str, parsed: dict[str, Any]) -> bool:
    if not parsed:
        return False
    if letter == "A":
        return "knows_truth" in parsed and bool(parsed.get("belief_label") or parsed.get("likely_belief"))
    if letter == "B":
        return "target_knew_truth_at_cutoff" in parsed and bool(parsed.get("reconstructed_prior_belief") or parsed.get("belief_label"))
    if letter == "C":
        return "other_player_knew_truth_at_cutoff" in parsed and bool(parsed.get("other_player_likely_belief") or parsed.get("belief_label"))
    if letter == "D":
        return bool(parsed.get("listener")) and bool(
            parsed.get("predicted_listener_trust_update") or parsed.get("predicted_trust_update")
        )
    return bool(parsed)


def d_reference(hidden_gold: dict[str, Any]) -> dict[str, Any]:
    value = hidden_gold.get("D_PT_reference")
    if isinstance(value, dict):
        return value
    nested_gold = hidden_gold.get("hidden_gold")
    if isinstance(nested_gold, dict) and isinstance(nested_gold.get("D_PT_reference"), dict):
        return nested_gold["D_PT_reference"]
    return {}


def d_perspective_scores(answer: dict[str, Any], hidden_gold: dict[str, Any]) -> tuple[bool | None, bool | None]:
    if not answer:
        return None, None
    trust_update = answer.get("predicted_listener_trust_update") or answer.get("predicted_trust_update")
    next_action = answer.get("predicted_listener_next_action") or answer.get("predicted_next_action")
    if not trust_update:
        return False, False
    reference = d_reference(hidden_gold)
    expected_listener = reference.get("listener")
    expected_speaker = reference.get("speaker")
    listener_ok = not expected_listener or answer.get("listener") == expected_listener
    speaker_ok = not expected_speaker or answer.get("speaker") == expected_speaker
    weak = bool(answer.get("listener") and listener_ok)
    strong = bool(weak and speaker_ok and next_action not in {None, "unknown"})
    return weak, strong


def score_group_answers(group_id: str, answers: dict[str, dict[str, Any]], hidden_gold: dict[str, Any], forbidden: list[str]) -> dict[str, Any]:
    a = answers.get("A", {})
    b = answers.get("B", {})
    c = answers.get("C", {})
    d = answers.get("D", {})
    acceptable = hidden_gold.get("acceptable_evidence_ids_for_target", []) or hidden_gold.get("hidden_gold", {}).get("acceptable_evidence_ids_for_target", [])
    leakage = any(uses_forbidden_evidence(ans, forbidden) for ans in [a, b, c, d])
    cited_ids: set[str] = set()
    for answer in [a, b, c, d]:
        cited_ids.update(evidence_ids_from_answer(answer))
    acceptable_set = set(acceptable)
    forbidden_set = set(forbidden)
    unsupported_ids = cited_ids - acceptable_set - forbidden_set if acceptable_set else set()
    a_knows = a.get("knows_truth")
    b_knows = b.get("target_knew_truth_at_cutoff")
    c_knows = c.get("other_player_knew_truth_at_cutoff")
    a_label = a.get("belief_label") or a.get("likely_belief")
    b_label = b.get("belief_label") or b.get("reconstructed_prior_belief")
    expected_global = normalize_enum(hidden_gold.get("claim_truth_global"))
    expected_local = normalize_enum(hidden_gold.get("claim_awareness_local_target"))
    actual_global = answer_claim_truth(answers)
    actual_local = answer_local_awareness(answers)
    pt_weak, pt_strong = d_perspective_scores(d, hidden_gold)
    return {
        "probe_group_id": group_id,
        "RC_weak": b_knows is False if b else None,
        "RC_strong": bool(b and a and b_knows is False and str(a_label)[:40] in str(b_label)),
        "FB_weak": c_knows is not True if c else None,
        "FB_strong": bool(c and c_knows is not True and c.get("other_player_likely_belief")),
        "PT_weak": pt_weak,
        "PT_strong": pt_strong,
        "claim_verification_global": actual_global == expected_global if expected_global else None,
        "claim_verification_local": actual_local == expected_local if expected_local else None,
        "perspective_leakage": leakage,
        "forbidden_evidence_usage": leakage,
        "evidence_support": not leakage and not unsupported_ids,
        "json_parse_success": all(answer.get("_parse_ok", True) for answer in answers.values()),
        "schema_validation_success": all(schema_ok_for_answer(letter, answer) for letter, answer in answers.items()),
    }


def score_decrypto_diagnostics(responses_path: Path, hidden_gold_path: Path, output_path: Path) -> dict[str, Any]:
    responses = read_jsonl(responses_path)
    hidden_rows = read_jsonl(hidden_gold_path)
    hidden_by_group = {row["probe_group_id"]: row for row in hidden_rows if "probe_group_id" in row}
    answers_by_group: dict[str, dict[str, dict[str, Any]]] = collections.defaultdict(dict)
    parse_success = 0
    for row in responses:
        parsed = row.get("parsed")
        ok = isinstance(parsed, dict)
        if not ok:
            parsed, ok = parse_answer(row.get("raw_response", ""))
        parse_success += int(ok)
        parsed["_parse_ok"] = ok
        probe_type = row.get("probe_type", "")
        letter = probe_type[:1] if probe_type else row.get("probe_id", "")[-1:]
        answers_by_group[row["probe_group_id"]][letter] = parsed
    scores = []
    for group_id, answers in answers_by_group.items():
        hidden = hidden_by_group.get(group_id, {})
        forbidden = hidden.get("forbidden_event_ids_for_target", []) or hidden.get("hidden_gold", {}).get("forbidden_event_ids_for_target", [])
        scores.append(score_group_answers(group_id, answers, hidden, forbidden))
    aggregate = {
        "groups_scored": len(scores),
        "json_parse_success": parse_success / len(responses) if responses else 0.0,
        "perspective_leakage_rate": sum(1 for row in scores if row["perspective_leakage"]) / len(scores) if scores else 0.0,
        "RC_weak": sum(1 for row in scores if row["RC_weak"]) / len(scores) if scores else 0.0,
        "FB_weak": sum(1 for row in scores if row["FB_weak"]) / len(scores) if scores else 0.0,
        "claim_verification_global": sum(1 for row in scores if row["claim_verification_global"]) / len(scores) if scores else 0.0,
        "claim_verification_local": sum(1 for row in scores if row["claim_verification_local"]) / len(scores) if scores else 0.0,
        "forbidden_evidence_usage_rate": sum(1 for row in scores if row["forbidden_evidence_usage"]) / len(scores) if scores else 0.0,
        "evidence_support_rate": sum(1 for row in scores if row["evidence_support"]) / len(scores) if scores else 0.0,
        "schema_validation_success": sum(1 for row in scores if row["schema_validation_success"]) / len(scores) if scores else 0.0,
    }
    write_json(output_path, {"aggregate": aggregate, "scores": scores})
    return aggregate


def validate_decrypto_outputs(annotation_root: Path, benchmark_root: Path) -> dict[str, Any]:
    issues = []
    ledger = load_ledger(annotation_root)
    for row in ledger["world_events"]:
        if row.get("abs_start_sec") is None or row.get("abs_end_sec") is None:
            issues.append({"code": "event_missing_abs_time", "id": row.get("world_event_id")})
    for row in ledger["claims"]:
        if row.get("abs_start_sec") is None or row.get("abs_end_sec") is None:
            issues.append({"code": "claim_missing_abs_time", "id": row.get("claim_id")})
    edge_counts = collections.Counter(edge["event_id"] for edge in ledger["visibility_edges"] if edge["event_id"].startswith("ge_"))
    for event in ledger["world_events"]:
        if edge_counts[event["world_event_id"]] < len(PLAYERS):
            issues.append({"code": "visibility_not_6pov", "id": event["world_event_id"], "count": edge_counts[event["world_event_id"]]})
    groups = read_jsonl(annotation_root / "diagnostics" / "probe_groups.jsonl")
    hidden_gold = {row.get("probe_group_id"): row for row in read_jsonl(annotation_root / "diagnostics" / "hidden_gold.jsonl")}
    probes = read_jsonl(benchmark_root / "interactive_diagnostics" / "prompts.jsonl")
    probes_by_group = collections.defaultdict(set)
    for probe in probes:
        probes_by_group[probe["probe_group_id"]].add(probe["probe_type"])
        prompt = probe.get("prompt", "")
        if probe["probe_type"] == "A_pre_reveal_belief":
            if "QUERY_VARIABLE_PUBLIC_FORM_JSON" not in prompt:
                issues.append({"code": "A_prompt_missing_public_query_form", "id": probe["probe_id"]})
            if "ORACLE_TRUTH_JSON" in prompt:
                issues.append({"code": "A_prompt_contains_oracle_truth", "id": probe["probe_id"]})
            if any(fid in prompt for fid in probe.get("forbidden_event_ids", [])):
                issues.append({"code": "A_prompt_leaks_hidden_truth", "id": probe["probe_id"]})
        if probe["probe_type"] == "B_post_reveal_reconstruct_previous_belief" and "answer to probe a" in prompt.lower():
            issues.append({"code": "B_prompt_mentions_A_answer", "id": probe["probe_id"]})
        if probe["probe_type"] == "D_perspective_taking_prediction":
            if "TARGET_LISTENER_CONTEXT_JSON" in prompt:
                issues.append({"code": "D_prompt_contains_listener_private_context", "id": probe["probe_id"]})
            if "SPEAKER_AVAILABLE_CONTEXT_JSON" not in prompt:
                issues.append({"code": "D_prompt_missing_speaker_context", "id": probe["probe_id"]})
            if "SPEAKER_MODEL_OF_LISTENER_PUBLIC_HISTORY_JSON" not in prompt:
                issues.append({"code": "D_prompt_missing_listener_public_model", "id": probe["probe_id"]})
    for group in groups:
        types = probes_by_group[group["probe_group_id"]]
        for required in ["A_pre_reveal_belief", "B_post_reveal_reconstruct_previous_belief", "C_other_agent_false_belief"]:
            if required not in types:
                issues.append({"code": "probe_group_missing_required_probe", "id": group["probe_group_id"], "missing": required})
        if group.get("gold_source") == "human_verified" and "D_perspective_taking_prediction" in types:
            group_hidden = hidden_gold.get(group["probe_group_id"], {})
            d_ref = group_hidden.get("D_PT_reference", {})
            if not group_hidden.get("D_PT_reference_enriched_by") or not d_ref.get("human_verified_scope"):
                issues.append({"code": "human_verified_D_missing_enriched_gold", "id": group["probe_group_id"]})
    trials_text = (benchmark_root / "static_trials" / "trials.jsonl").read_text(encoding="utf-8") if (benchmark_root / "static_trials" / "trials.jsonl").exists() else ""
    if "hidden_gold" in trials_text or "forbidden_event_ids" in trials_text:
        issues.append({"code": "static_trials_leak_hidden_gold"})
    return {
        "ok": not issues,
        "issue_count": len(issues),
        "issues": issues,
        "counts": {
            "world_events": len(ledger["world_events"]),
            "claims": len(ledger["claims"]),
            "visibility_edges": len(ledger["visibility_edges"]),
            "probe_groups": len(groups),
            "prompts": len(probes),
        },
    }
