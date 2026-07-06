from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


PLAYERS = ("Gemini", "baile", "beigang", "mojiang", "saoyi", "xiaolu")
UPSTREAM_STAGES = ("pov_events", "utterances", "phase_events")
UPSTREAM_CHAIN_STAGE = "upstream_chain"
UPSTREAM_ACTIVE_STAGES = UPSTREAM_STAGES + (UPSTREAM_CHAIN_STAGE,)
DOWNSTREAM_STAGES = ("global_events", "information_states", "memory_states", "belief_states", "candidate_trials")
DOWNSTREAM_CHAIN_STAGE = "downstream_chain"
DOWNSTREAM_ACTIVE_STAGES = DOWNSTREAM_STAGES + (DOWNSTREAM_CHAIN_STAGE,)
STAGE_ABBREVIATIONS = {
    "pov_events": "pov",
    "utterances": "utt",
    "phase_events": "phase",
    "upstream_chain": "up",
    "global_events": "glob",
    "information_states": "info",
    "memory_states": "mem",
    "belief_states": "belief",
    "candidate_trials": "trial",
    "downstream_chain": "chain",
}

SUBMIT_LIMIT = "SUBMIT_LIMIT"


@dataclass(frozen=True)
class SegmentStatus:
    segment_id: str
    pov_events: int
    utterances: int
    phase_events: bool
    global_events: bool
    information_states: int
    memory_states: int
    belief_states: int
    candidate_trials: int

    @property
    def upstream_complete(self) -> bool:
        return self.pov_events == len(PLAYERS) and self.utterances == len(PLAYERS) and self.phase_events

    @property
    def ready_for_global(self) -> bool:
        return self.upstream_complete and not self.global_events

    @property
    def downstream_complete(self) -> bool:
        return (
            self.global_events
            and self.information_states == len(PLAYERS)
            and self.memory_states == len(PLAYERS)
            and self.belief_states == len(PLAYERS)
            and self.candidate_trials > 0
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit Qwen3-Omni Omni Goose oracle annotation jobs.")
    parser.add_argument("--dataset-root", default=Path("data/omni_goose"), type=Path)
    parser.add_argument("--segments-jsonl", default=None, type=Path)
    parser.add_argument("--annotation-root", default=Path("runs/omni_goose_oracle_pass1/annotations_qwen"), type=Path)
    parser.add_argument("--slurm-script", default=Path("slurm/qwen3_omni_oracle_2gpu_stage.slurm"), type=Path)
    parser.add_argument(
        "--multi-worker-slurm-script",
        default=Path("slurm/qwen3_omni_oracle_4x2_local.slurm"),
        type=Path,
        help="8-GPU script that runs multiple 2-GPU Qwen3-Omni workers inside one Slurm job.",
    )
    parser.add_argument("--game-id", default="g001")
    parser.add_argument(
        "--mode",
        choices=[
            "status",
            "submit-upstream",
            "promote-downstream",
            "repair-downstream",
            "submit-balanced",
            "submit-4x2-balanced",
            "export",
        ],
        default="status",
    )
    parser.add_argument("--start-index", default=1, type=int, help="1-based segment index.")
    parser.add_argument("--end-index", default=None, type=int, help="Inclusive 1-based segment index.")
    parser.add_argument(
        "--upstream-start-index",
        default=None,
        type=int,
        help="Optional upstream-only start index for submit-balanced mode.",
    )
    parser.add_argument(
        "--upstream-end-index",
        default=None,
        type=int,
        help="Optional upstream-only inclusive end index for submit-balanced mode.",
    )
    parser.add_argument("--stage", choices=list(UPSTREAM_STAGES), default=None)
    parser.add_argument("--max-jobs", default=12, type=int)
    parser.add_argument(
        "--max-total-jobs",
        default=60,
        type=int,
        help="Skip new Slurm submissions when this user already has at least this many queued/running jobs. Use 0 to disable.",
    )
    parser.add_argument(
        "--max-pending-jobs",
        default=55,
        type=int,
        help="Skip new Slurm submissions when this user already has at least this many pending jobs. Use 0 to disable.",
    )
    parser.add_argument(
        "--balanced-upstream-jobs",
        default=4,
        type=int,
        help="Maximum upstream jobs to submit after downstream/repair attempts in submit-balanced mode.",
    )
    parser.add_argument(
        "--balanced-downstream-jobs",
        default=8,
        type=int,
        help="Maximum downstream chain jobs to submit before upstream attempts in submit-balanced mode.",
    )
    parser.add_argument(
        "--balanced-repair-jobs",
        default=4,
        type=int,
        help="Maximum repair jobs to submit before upstream attempts in submit-balanced mode.",
    )
    parser.add_argument("--batch-size", default=4, type=int, help="Number of upstream segments per 2-GPU job.")
    parser.add_argument("--multi-worker-workers", default=4, type=int)
    parser.add_argument(
        "--multi-worker-stage-plan",
        default="upstream_chain,downstream_chain,upstream_chain,downstream_chain",
        help="Comma-separated worker stage plan for submit-4x2-balanced mode.",
    )
    parser.add_argument(
        "--multi-worker-limit-per-worker",
        default="",
        help="Optional per-worker segment limit. Empty means each worker scans all remaining resumable items.",
    )
    parser.add_argument(
        "--allow-small-job-fallback-on-submit-limit",
        action="store_true",
        help="When submit-balanced cannot submit the 4x2 job due to the Slurm submit limit, allow fallback submission of small 2-GPU jobs.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Ignore existing submission markers.")
    parser.add_argument("--qwen-max-tokens", default="16384")
    parser.add_argument("--qwen-text-merge-max-tokens", default="32768")
    parser.add_argument("--qwen-video-fps", default="1.0")
    parser.add_argument("--qwen-video-max-frames", default="48")
    parser.add_argument("--qwen-video-max-pixels", default="401408")
    parser.add_argument("--omni-http-timeout-sec", default="900")
    return parser.parse_args()


def load_segments(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def count_json(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    for item in path.glob("*.json"):
        try:
            text = item.read_text(encoding="utf-8")
            if not text.strip():
                continue
            json.loads(text)
        except Exception:
            continue
        count += 1
    return count


def load_candidate_counts(annotation_root: Path) -> dict[str, int]:
    path = annotation_root / "candidate_trials" / "g001_candidate_trials.jsonl"
    counts: dict[str, int] = {}
    if not path.exists():
        return counts
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        segment_id = payload.get("segment_id")
        if segment_id:
            counts[segment_id] = counts.get(segment_id, 0) + 1
    return counts


def segment_status(annotation_root: Path, segment_id: str, candidate_counts: dict[str, int]) -> SegmentStatus:
    game_root = annotation_root
    return SegmentStatus(
        segment_id=segment_id,
        pov_events=count_json(game_root / "pov_events" / "g001" / segment_id),
        utterances=count_json(game_root / "utterances" / "g001" / segment_id),
        phase_events=valid_json_file(game_root / "phase_events" / "g001" / f"{segment_id}.json"),
        global_events=valid_json_file(game_root / "global_events" / "g001" / f"{segment_id}.json"),
        information_states=count_json(game_root / "information_states" / "g001" / segment_id),
        memory_states=count_json(game_root / "memory_states" / "g001" / segment_id),
        belief_states=count_json(game_root / "belief_states" / "g001" / segment_id),
        candidate_trials=candidate_counts.get(segment_id, 0),
    )


def valid_json_file(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        text = path.read_text(encoding="utf-8")
        return bool(text.strip()) and isinstance(json.loads(text), dict)
    except Exception:
        return False


def selected_segments(args: argparse.Namespace) -> list[tuple[int, str]]:
    segments_path = args.segments_jsonl or args.dataset_root / "segments.jsonl"
    rows = load_segments(segments_path)
    end = args.end_index or len(rows)
    selected = []
    for index, row in enumerate(rows, start=1):
        if index < args.start_index or index > end:
            continue
        if row.get("game_id") != args.game_id:
            continue
        selected.append((index, row["segment_id"]))
    return selected


def stage_complete(status: SegmentStatus, stage: str) -> bool:
    if stage == "pov_events":
        return status.pov_events == len(PLAYERS)
    if stage == "utterances":
        return status.utterances == len(PLAYERS)
    if stage == "phase_events":
        return status.phase_events
    raise ValueError(f"unsupported upstream stage: {stage}")


def run_command(cmd: list[str], *, dry_run: bool, env: dict[str, str] | None = None) -> str:
    env_prefix = ""
    if env:
        env_prefix = " ".join(f"{key}={value}" for key, value in sorted(env.items())) + " "
    print(env_prefix + " ".join(cmd))
    if dry_run:
        return "DRY_RUN"
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    result = subprocess.run(cmd, check=False, capture_output=True, text=True, env=run_env)
    if result.returncode == 0:
        return result.stdout.strip()
    stderr = result.stderr.strip()
    stdout = result.stdout.strip()
    message = stderr or stdout or f"command failed with returncode={result.returncode}"
    if "AssocGrpSubmitJobsLimit" in message or "job submit limit" in message:
        print(f"submit_limit_reached: {message}")
        return SUBMIT_LIMIT
    raise subprocess.CalledProcessError(result.returncode, cmd, output=result.stdout, stderr=result.stderr)


def sbatch_env(
    args: argparse.Namespace,
    stage: str,
    segment_id: str,
    target_player: str | None = None,
    segment_ids: list[str] | None = None,
) -> dict[str, str]:
    env = {
        "STAGE": stage,
        "SEGMENT_ID": segment_id,
        "LIMIT": "1",
        "RESUME": "1",
        "OVERWRITE": "0",
        "ANNOTATION_ROOT": args.annotation_root.as_posix(),
        "QWEN3_OMNI_MAX_TOKENS": args.qwen_max_tokens,
        "QWEN3_OMNI_TEXT_MERGE_MAX_TOKENS": args.qwen_text_merge_max_tokens,
        "QWEN3_OMNI_VIDEO_FPS": args.qwen_video_fps,
        "QWEN3_OMNI_VIDEO_MAX_FRAMES": args.qwen_video_max_frames,
        "QWEN3_OMNI_VIDEO_MAX_PIXELS": args.qwen_video_max_pixels,
        "OMNI_HTTP_TIMEOUT_SEC": args.omni_http_timeout_sec,
    }
    if segment_ids:
        env["SEGMENT_IDS"] = ",".join(segment_ids)
    if target_player is not None:
        env["TARGET_PLAYER"] = target_player
    return env


def submit_stage(
    args: argparse.Namespace,
    stage: str,
    segment_id: str,
    dependency: str | None = None,
    target_player: str | None = None,
    segment_ids: list[str] | None = None,
) -> str:
    env = sbatch_env(args, stage, segment_id, target_player, segment_ids)
    cmd = ["sbatch", "--parsable", f"--job-name={job_name(stage, segment_id, target_player, segment_ids)}"]
    if dependency:
        cmd.append(f"--dependency=afterok:{dependency}")
    cmd.append(args.slurm_script.as_posix())
    return run_command(cmd, dry_run=args.dry_run, env=env)


def segment_number(segment_id: str) -> str:
    match = re.search(r"_seg_(\d{4})_", segment_id)
    return match.group(1) if match else segment_id[-8:]


def job_name(
    stage: str,
    segment_id: str,
    target_player: str | None = None,
    segment_ids: list[str] | None = None,
) -> str:
    if segment_ids and len(segment_ids) > 1:
        first = segment_number(segment_ids[0])
        last = segment_number(segment_ids[-1])
        segment_index = f"{first}-{last}"
    else:
        segment_index = segment_number(segment_id)
    stage_name = STAGE_ABBREVIATIONS.get(stage, stage[:8])
    suffix = f"-{target_player[:3].lower()}" if target_player else ""
    return f"og-{stage_name}-{segment_index}{suffix}"


def queue_counts() -> dict[str, int] | None:
    try:
        result = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%T"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        print(f"skip submit queue_cap reason=squeue_failed error={exc}")
        return None
    states = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return {
        "total": len(states),
        "pending": sum(1 for state in states if state == "PENDING"),
    }


def queue_submit_allowed(args: argparse.Namespace) -> bool:
    if args.force:
        return True
    counts = queue_counts()
    if counts is None:
        return False
    total_cap_hit = args.max_total_jobs > 0 and counts["total"] >= args.max_total_jobs
    pending_cap_hit = args.max_pending_jobs > 0 and counts["pending"] >= args.max_pending_jobs
    if total_cap_hit or pending_cap_hit:
        print(
            "skip submit queue_cap "
            f"total={counts['total']} pending={counts['pending']} "
            f"max_total={args.max_total_jobs} max_pending={args.max_pending_jobs}"
        )
        return False
    print(
        "queue submit allowed "
        f"total={counts['total']} pending={counts['pending']} "
        f"max_total={args.max_total_jobs} max_pending={args.max_pending_jobs}"
    )
    return True


def active_job_names() -> set[str]:
    try:
        result = subprocess.run(
            ["squeue", "-h", "-o", "%j"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return set()
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def active_job_ids_by_name() -> dict[str, str]:
    try:
        result = subprocess.run(
            ["squeue", "-h", "-o", "%i %j"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return {}
    jobs: dict[str, str] = {}
    for line in result.stdout.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            jobs[parts[1]] = parts[0]
    return jobs


def active_job_ids() -> set[str]:
    try:
        result = subprocess.run(
            ["squeue", "-h", "-o", "%i"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return set()
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def downstream_marker_active(marker_path: Path, active_jobs: set[str], active_ids: dict[str, str]) -> bool:
    try:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    active_id_values = set(active_ids.values()) | active_job_ids()
    jobs = payload.get("jobs")
    if not isinstance(jobs, dict):
        return False
    for stage, value in jobs.items():
        job_id = str(value)
        if job_id in active_id_values:
            return True
        segment_id = str(payload.get("segment_id", ""))
        if segment_id and job_name(str(stage), segment_id) in active_jobs:
            return True
    return False


def active_segment_numbers_for_stage(stage: str, active_jobs: set[str]) -> set[int]:
    stage_name = STAGE_ABBREVIATIONS.get(stage, stage[:8])
    prefix = f"og-{stage_name}-"
    active: set[int] = set()
    for name in active_jobs:
        if not name.startswith(prefix):
            continue
        suffix = name.removeprefix(prefix)
        match = re.match(r"(\d{4})(?:-(\d{4}))?", suffix)
        if not match:
            continue
        start = int(match.group(1))
        end = int(match.group(2) or match.group(1))
        active.update(range(start, end + 1))
    return active


def active_segment_ranges_for_stage(stage: str, active_jobs: set[str]) -> list[tuple[int, int]]:
    stage_name = STAGE_ABBREVIATIONS.get(stage, stage[:8])
    prefix = f"og-{stage_name}-"
    ranges: list[tuple[int, int]] = []
    for name in active_jobs:
        if not name.startswith(prefix):
            continue
        suffix = name.removeprefix(prefix)
        match = re.match(r"(\d{4})(?:-(\d{4}))?", suffix)
        if not match:
            continue
        start = int(match.group(1))
        end = int(match.group(2) or match.group(1))
        ranges.append((start, end))
    return sorted(set(ranges))


def chunked(items: list[str], size: int) -> list[list[str]]:
    if size <= 1:
        return [[item] for item in items]
    return [items[index : index + size] for index in range(0, len(items), size)]


def partial_downstream_batches(
    ready_segment_ids: list[str],
    active_jobs: set[str],
    batch_size: int,
) -> list[list[str]]:
    by_index = {int(segment_number(segment_id)): segment_id for segment_id in ready_segment_ids}
    batches: list[list[str]] = []
    seen_batches: set[str] = set()
    covered_segments: set[str] = set()
    for stage in ("global_events", "information_states", "memory_states"):
        for start, end in active_segment_ranges_for_stage(stage, active_jobs):
            segment_ids = [
                by_index[index]
                for index in range(start, end + 1)
                if index in by_index
            ]
            if not segment_ids:
                continue
            key = ",".join(segment_ids)
            if key not in seen_batches:
                batches.append(segment_ids)
                seen_batches.add(key)
                covered_segments.update(segment_ids)
    for segment_ids in chunked([item for item in ready_segment_ids if item not in covered_segments], batch_size):
        key = ",".join(segment_ids)
        if key not in seen_batches:
            batches.append(segment_ids)
            seen_batches.add(key)
    return batches


def active_batch_dependency(
    stage: str,
    segment_ids: list[str],
    active_ids: dict[str, str],
) -> str | None:
    name = job_name(stage, segment_ids[0], segment_ids=segment_ids)
    if name in active_ids:
        return active_ids[name]
    if len(segment_ids) > 1:
        return None
    single_name = job_name(stage, segment_ids[0])
    return active_ids.get(single_name)


def submit_partial_downstream_batch(
    args: argparse.Namespace,
    segment_ids: list[str],
    active_ids: dict[str, str],
) -> int | str:
    global_dependency = active_batch_dependency("global_events", segment_ids, active_ids)
    info_dependency = active_batch_dependency("information_states", segment_ids, active_ids)
    memory_dependency = active_batch_dependency("memory_states", segment_ids, active_ids)
    if not global_dependency and not info_dependency and not memory_dependency:
        return 0
    submitted = 0
    segment_id = segment_ids[0]
    info_name = job_name("information_states", segment_id, segment_ids=segment_ids)
    memory_name = job_name("memory_states", segment_id, segment_ids=segment_ids)
    belief_name = job_name("belief_states", segment_id, segment_ids=segment_ids)
    trial_name = job_name("candidate_trials", segment_id, segment_ids=segment_ids)
    active_jobs = set(active_ids)
    active_info_segments = active_segment_numbers_for_stage("information_states", active_jobs)
    active_memory_segments = active_segment_numbers_for_stage("memory_states", active_jobs)
    active_belief_segments = active_segment_numbers_for_stage("belief_states", active_jobs)
    active_trial_segments = active_segment_numbers_for_stage("candidate_trials", active_jobs)
    segment_numbers = {int(segment_number(item)) for item in segment_ids}
    if global_dependency and not segment_numbers.issubset(active_info_segments) and info_name not in active_ids:
        info_job = submit_stage(args, "information_states", segment_id, global_dependency, segment_ids=segment_ids)
        if info_job == SUBMIT_LIMIT:
            print(f"stop downstream submit_limit stage=information_states segment_ids={','.join(segment_ids)}")
            return SUBMIT_LIMIT
        active_ids[info_name] = info_job
        info_dependency = info_job
        submitted += 1
        print(f"submitted downstream completion stage=information_states segment_ids={','.join(segment_ids)} job={info_job}")
    if info_dependency and not segment_numbers.issubset(active_memory_segments) and memory_name not in active_ids:
        memory_job = submit_stage(args, "memory_states", segment_id, info_dependency, segment_ids=segment_ids)
        if memory_job == SUBMIT_LIMIT:
            print(f"stop downstream submit_limit stage=memory_states segment_ids={','.join(segment_ids)}")
            return SUBMIT_LIMIT
        active_ids[memory_name] = memory_job
        memory_dependency = memory_job
        submitted += 1
        print(f"submitted downstream completion stage=memory_states segment_ids={','.join(segment_ids)} job={memory_job}")
    if memory_dependency and not segment_numbers.issubset(active_belief_segments) and belief_name not in active_ids:
        belief_job = submit_stage(args, "belief_states", segment_id, memory_dependency, segment_ids=segment_ids)
        if belief_job == SUBMIT_LIMIT:
            print(f"stop downstream submit_limit stage=belief_states segment_ids={','.join(segment_ids)}")
            return SUBMIT_LIMIT
        active_ids[belief_name] = belief_job
        submitted += 1
        print(f"submitted downstream completion stage=belief_states segment_ids={','.join(segment_ids)} job={belief_job}")
    if info_dependency and not segment_numbers.issubset(active_trial_segments) and trial_name not in active_ids:
        trial_job = submit_stage(args, "candidate_trials", segment_id, info_dependency, segment_ids=segment_ids)
        if trial_job == SUBMIT_LIMIT:
            print(f"stop downstream submit_limit stage=candidate_trials segment_ids={','.join(segment_ids)}")
            return SUBMIT_LIMIT
        active_ids[trial_name] = trial_job
        submitted += 1
        print(f"submitted downstream completion stage=candidate_trials segment_ids={','.join(segment_ids)} job={trial_job}")
    return submitted


def has_unsubmitted_ready_downstream(args: argparse.Namespace, statuses: dict[str, SegmentStatus]) -> bool:
    active_jobs = active_job_names()
    active_segments_by_stage = {
        stage: active_segment_numbers_for_stage(stage, active_jobs)
        for stage in DOWNSTREAM_ACTIVE_STAGES
    }
    all_args = argparse.Namespace(**vars(args))
    all_args.start_index = 1
    all_args.end_index = None
    candidate_counts = load_candidate_counts(args.annotation_root)
    for _, segment_id in selected_segments(all_args):
        status = segment_status(args.annotation_root, segment_id, candidate_counts)
        if not status.ready_for_global:
            continue
        segment_index = int(segment_number(segment_id))
        active_stages = {
            stage
            for stage in DOWNSTREAM_STAGES
            if segment_index in active_segments_by_stage[stage] or job_name(stage, segment_id) in active_jobs
        }
        if active_stages and active_stages != set(DOWNSTREAM_STAGES):
            return True
        if active_stages == set(DOWNSTREAM_STAGES):
            continue
        if downstream_marker_path(args.annotation_root, segment_id).exists() and not args.force:
            continue
        return True
    return False


def submit_upstream(args: argparse.Namespace, statuses: dict[str, SegmentStatus]) -> int:
    if not args.force and args.stage is None and has_unsubmitted_ready_downstream(args, statuses):
        print("defer upstream: unsubmitted ready downstream segments exist")
        print("submitted_upstream_total=0")
        return 0
    active_jobs = active_job_names()
    submitted = 0
    rows = selected_segments(args)
    if args.stage is None:
        active_segments_by_stage = {
            stage: active_segment_numbers_for_stage(stage, active_jobs)
            for stage in UPSTREAM_ACTIVE_STAGES
        }
        missing_chain_segments = []
        for _, segment_id in rows:
            status = statuses[segment_id]
            if status.upstream_complete:
                continue
            segment_index = int(segment_number(segment_id))
            if any(segment_index in active_segments_by_stage[stage] for stage in UPSTREAM_ACTIVE_STAGES) and not args.force:
                print(f"skip upstream_chain already_active segment_id={segment_id} segment_index={segment_index:04d}")
                continue
            missing_chain_segments.append(segment_id)
        for segment_ids in chunked(missing_chain_segments, args.batch_size):
            segment_id = segment_ids[0]
            name = job_name(UPSTREAM_CHAIN_STAGE, segment_id, segment_ids=segment_ids)
            if name in active_jobs and not args.force:
                print(
                    "skip upstream_chain batch already_active "
                    f"segment_ids={','.join(segment_ids)} job_name={name}"
                )
                continue
            job_id = submit_stage(args, UPSTREAM_CHAIN_STAGE, segment_id, segment_ids=segment_ids)
            if job_id == SUBMIT_LIMIT:
                print(f"stop upstream submit_limit stage={UPSTREAM_CHAIN_STAGE} segment_ids={','.join(segment_ids)}")
                print(f"submitted_upstream_total={submitted}")
                return submitted
            print(f"submitted upstream_chain segment_ids={','.join(segment_ids)} job={job_id}")
            submitted += 1
            if submitted >= args.max_jobs:
                print(f"max_jobs_reached={args.max_jobs}")
                return submitted
        print(f"submitted_upstream_total={submitted}")
        return submitted

    stage_batches: list[tuple[str, list[str]]] = []
    for stage in ([args.stage] if args.stage else UPSTREAM_STAGES):
        active_segments = active_segment_numbers_for_stage(stage, active_jobs)
        missing = []
        for _, segment_id in rows:
            status = statuses[segment_id]
            if stage_complete(status, stage):
                continue
            segment_index = int(segment_number(segment_id))
            if segment_index in active_segments and not args.force:
                print(f"skip upstream already_active stage={stage} segment_id={segment_id} segment_index={segment_index:04d}")
                continue
            name = job_name(stage, segment_id)
            if name in active_jobs and not args.force:
                print(f"skip upstream already_active stage={stage} segment_id={segment_id} job_name={name}")
                continue
            missing.append(segment_id)
        stage_batches.extend((stage, segment_ids) for segment_ids in chunked(missing, args.batch_size))
    if args.stage is None:
        stage_order = {stage: index for index, stage in enumerate(UPSTREAM_STAGES)}
        stage_batches.sort(
            key=lambda item: (
                int(segment_number(item[1][0])),
                stage_order.get(item[0], len(stage_order)),
            )
        )
    for stage, segment_ids in stage_batches:
        segment_id = segment_ids[0]
        name = job_name(stage, segment_id, segment_ids=segment_ids)
        if name in active_jobs and not args.force:
            print(f"skip upstream batch already_active stage={stage} segment_ids={','.join(segment_ids)} job_name={name}")
            continue
        job_id = submit_stage(args, stage, segment_id, segment_ids=segment_ids)
        if job_id == SUBMIT_LIMIT:
            print(f"stop upstream submit_limit stage={stage} segment_ids={','.join(segment_ids)}")
            print(f"submitted_upstream_total={submitted}")
            return submitted
        print(f"submitted upstream stage={stage} segment_ids={','.join(segment_ids)} job={job_id}")
        submitted += 1
        if submitted >= args.max_jobs:
            print(f"max_jobs_reached={args.max_jobs}")
            return submitted
    print(f"submitted_upstream_total={submitted}")
    return submitted


def promote_downstream(args: argparse.Namespace, statuses: dict[str, SegmentStatus]) -> int:
    active_jobs = active_job_names()
    active_ids = active_job_ids_by_name()
    submitted = 0
    active_segments_by_stage = {
        stage: active_segment_numbers_for_stage(stage, active_jobs)
        for stage in DOWNSTREAM_ACTIVE_STAGES
    }
    ready_segments = []
    for _, segment_id in selected_segments(args):
        status = statuses[segment_id]
        if not status.ready_for_global:
            continue
        segment_index = int(segment_number(segment_id))
        active_downstream = [
            job_name(stage, segment_id)
            for stage in DOWNSTREAM_ACTIVE_STAGES
            if segment_index in active_segments_by_stage[stage] or job_name(stage, segment_id) in active_jobs
        ]
        if active_downstream and not args.force:
            print(f"skip downstream already_active segment_id={segment_id} jobs={','.join(active_downstream)}")
            continue
        marker_path = downstream_marker_path(args.annotation_root, segment_id)
        if marker_path.exists() and not args.force:
            if downstream_marker_active(marker_path, active_jobs, active_ids):
                print(f"skip downstream already_submitted segment_id={segment_id} marker={marker_path.as_posix()}")
                continue
            print(f"ignore stale downstream marker segment_id={segment_id} marker={marker_path.as_posix()}")
        ready_segments.append(segment_id)

    ready_segment_ids = [
        segment_id
        for _, segment_id in selected_segments(args)
        if statuses[segment_id].ready_for_global
    ]
    for segment_ids in partial_downstream_batches(ready_segment_ids, active_jobs, args.batch_size):
        added = submit_partial_downstream_batch(args, segment_ids, active_ids)
        if added == SUBMIT_LIMIT:
            print(f"submitted_downstream_total={submitted}")
            return submitted
        submitted += added
        if added and submitted >= args.max_jobs:
            print(f"max_jobs_reached={args.max_jobs}")
            return submitted

    for segment_ids in chunked(ready_segments, args.batch_size):
        segment_id = segment_ids[0]
        chain_job = submit_stage(args, DOWNSTREAM_CHAIN_STAGE, segment_id, segment_ids=segment_ids)
        if chain_job == SUBMIT_LIMIT:
            print(f"stop downstream submit_limit stage={DOWNSTREAM_CHAIN_STAGE} segment_ids={','.join(segment_ids)}")
            print(f"submitted_downstream_total={submitted}")
            return submitted
        jobs = {DOWNSTREAM_CHAIN_STAGE: chain_job}
        for marker_segment_id in segment_ids:
            write_downstream_marker(
                downstream_marker_path(args.annotation_root, marker_segment_id),
                {
                    "segment_id": marker_segment_id,
                    "batch_segment_ids": segment_ids,
                    "submitted_at_unix": time.time(),
                    "jobs": jobs,
                },
                dry_run=args.dry_run,
            )
        print(f"submitted downstream_chain segment_ids={','.join(segment_ids)} job={chain_job}")
        submitted += 1
        if submitted >= args.max_jobs:
            print(f"max_jobs_reached={args.max_jobs}")
            return submitted
    print(f"submitted_downstream_total={submitted}")
    return submitted


def repair_downstream(args: argparse.Namespace, statuses: dict[str, SegmentStatus]) -> int:
    active_jobs = active_job_names()
    active_segments_by_stage = {
        stage: active_segment_numbers_for_stage(stage, active_jobs)
        for stage in DOWNSTREAM_ACTIVE_STAGES
    }
    submitted = 0
    for _, segment_id in selected_segments(args):
        segment_index = int(segment_number(segment_id))
        status = statuses[segment_id]
        if not status.global_events:
            continue
        if status.information_states < len(PLAYERS):
            name = job_name("information_states", segment_id)
            if (segment_index in active_segments_by_stage["information_states"] or name in active_jobs) and not args.force:
                print(f"skip repair already_active stage=information_states segment_id={segment_id}")
            else:
                job_id = submit_stage(args, "information_states", segment_id)
                if job_id == SUBMIT_LIMIT:
                    print(f"stop repair submit_limit stage=information_states segment_id={segment_id}")
                    print(f"submitted_repair_total={submitted}")
                    return submitted
                print(f"submitted repair stage=information_states segment_id={segment_id} job={job_id}")
                submitted += 1
        if status.memory_states < len(PLAYERS) and status.information_states == len(PLAYERS):
            stage_active = segment_index in active_segments_by_stage["memory_states"]
            if stage_active and not args.force:
                print(f"skip repair already_active stage=memory_states segment_id={segment_id}")
                continue
            for player in missing_players(args.annotation_root, "memory_states", segment_id):
                name = job_name("memory_states", segment_id, player)
                if name in active_jobs and not args.force:
                    print(f"skip repair already_active stage=memory_states segment_id={segment_id} target={player}")
                    continue
                job_id = submit_stage(args, "memory_states", segment_id, target_player=player)
                if job_id == SUBMIT_LIMIT:
                    print(f"stop repair submit_limit stage=memory_states segment_id={segment_id} target={player}")
                    print(f"submitted_repair_total={submitted}")
                    return submitted
                print(f"submitted repair stage=memory_states segment_id={segment_id} target={player} job={job_id}")
                submitted += 1
                if submitted >= args.max_jobs:
                    print(f"max_jobs_reached={args.max_jobs}")
                    return submitted
        if status.belief_states < len(PLAYERS) and status.memory_states == len(PLAYERS):
            stage_active = segment_index in active_segments_by_stage["belief_states"]
            if stage_active and not args.force:
                print(f"skip repair already_active stage=belief_states segment_id={segment_id}")
                continue
            for player in missing_players(args.annotation_root, "belief_states", segment_id):
                name = job_name("belief_states", segment_id, player)
                if name in active_jobs and not args.force:
                    print(f"skip repair already_active stage=belief_states segment_id={segment_id} target={player}")
                    continue
                job_id = submit_stage(args, "belief_states", segment_id, target_player=player)
                if job_id == SUBMIT_LIMIT:
                    print(f"stop repair submit_limit stage=belief_states segment_id={segment_id} target={player}")
                    print(f"submitted_repair_total={submitted}")
                    return submitted
                print(f"submitted repair stage=belief_states segment_id={segment_id} target={player} job={job_id}")
                submitted += 1
                if submitted >= args.max_jobs:
                    print(f"max_jobs_reached={args.max_jobs}")
                    return submitted
        if status.candidate_trials == 0 and status.information_states == len(PLAYERS):
            name = job_name("candidate_trials", segment_id)
            if (segment_index in active_segments_by_stage["candidate_trials"] or name in active_jobs) and not args.force:
                print(f"skip repair already_active stage=candidate_trials segment_id={segment_id}")
            else:
                job_id = submit_stage(args, "candidate_trials", segment_id)
                if job_id == SUBMIT_LIMIT:
                    print(f"stop repair submit_limit stage=candidate_trials segment_id={segment_id}")
                    print(f"submitted_repair_total={submitted}")
                    return submitted
                print(f"submitted repair stage=candidate_trials segment_id={segment_id} job={job_id}")
                submitted += 1
        if submitted >= args.max_jobs:
            print(f"max_jobs_reached={args.max_jobs}")
            return submitted
    print(f"submitted_repair_total={submitted}")
    return submitted


def missing_players(annotation_root: Path, stage: str, segment_id: str) -> list[str]:
    path = annotation_root / stage / "g001" / segment_id
    existing = {item.stem for item in path.glob("*.json")} if path.exists() else set()
    return [player for player in PLAYERS if player not in existing]


def downstream_marker_path(annotation_root: Path, segment_id: str) -> Path:
    return annotation_root / "job_markers" / "downstream" / f"{segment_id}.json"


def write_downstream_marker(path: Path, payload: dict[str, object], *, dry_run: bool) -> None:
    print(f"write_marker {path.as_posix()}")
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def refresh_statuses(args: argparse.Namespace, rows: list[tuple[int, str]]) -> dict[str, SegmentStatus]:
    candidate_counts = load_candidate_counts(args.annotation_root)
    return {
        segment_id: segment_status(args.annotation_root, segment_id, candidate_counts)
        for _, segment_id in rows
    }


def with_max_jobs(args: argparse.Namespace, max_jobs: int) -> argparse.Namespace:
    copied = argparse.Namespace(**vars(args))
    copied.max_jobs = max_jobs
    return copied


def upstream_range_args(args: argparse.Namespace) -> argparse.Namespace:
    copied = argparse.Namespace(**vars(args))
    if args.upstream_start_index is not None:
        copied.start_index = args.upstream_start_index
    if args.upstream_end_index is not None:
        copied.end_index = args.upstream_end_index
    return copied


def submit_balanced(
    args: argparse.Namespace,
    rows: list[tuple[int, str]],
    statuses: dict[str, SegmentStatus],
) -> None:
    if not queue_submit_allowed(args):
        print("submitted_balanced_total=0 downstream=0 repair=0 upstream=0 queue_cap=1")
        return
    multi_worker_status = submit_4x2_balanced(args)
    if multi_worker_status in {"submitted", "already_active"}:
        print(
            "submitted_balanced_total=0 "
            f"downstream=0 repair=0 upstream=0 multi_worker={multi_worker_status}"
        )
        return
    if multi_worker_status == "submit_limit" and not args.allow_small_job_fallback_on_submit_limit:
        print(
            "submitted_balanced_total=0 "
            "downstream=0 repair=0 upstream=0 "
            "multi_worker=submit_limit small_job_fallback=disabled"
        )
        return

    downstream_jobs = promote_downstream(
        with_max_jobs(args, args.balanced_downstream_jobs),
        statuses,
    )
    statuses = refresh_statuses(args, rows)
    repair_jobs = repair_downstream(
        with_max_jobs(args, args.balanced_repair_jobs),
        statuses,
    )
    upstream_args = upstream_range_args(args)
    upstream_rows = selected_segments(upstream_args)
    upstream_jobs = submit_upstream(
        with_max_jobs(upstream_args, args.balanced_upstream_jobs),
        refresh_statuses(upstream_args, upstream_rows),
    )
    print(
        "submitted_balanced_total="
        f"{downstream_jobs + repair_jobs + upstream_jobs} "
        f"downstream={downstream_jobs} repair={repair_jobs} upstream={upstream_jobs}"
    )



def effective_multi_worker_stage_plan(args: argparse.Namespace) -> str:
    rows = selected_segments(args)
    statuses = refresh_statuses(args, rows)
    if statuses and all(status.upstream_complete for status in statuses.values()):
        return ",".join([DOWNSTREAM_CHAIN_STAGE] * args.multi_worker_workers)
    if statuses and all(status.downstream_complete for status in statuses.values()):
        return ",".join([UPSTREAM_CHAIN_STAGE] * args.multi_worker_workers)
    return args.multi_worker_stage_plan


def submit_4x2_balanced(args: argparse.Namespace) -> str:
    if not queue_submit_allowed(args):
        print("submitted_4x2_balanced=0 reason=queue_cap")
        return "queue_cap"
    active_jobs = active_job_names()
    if "og-4x2-balanced" in active_jobs and not args.force:
        print("submitted_4x2_balanced=0 reason=already_active")
        return "already_active"
    stage_plan = effective_multi_worker_stage_plan(args)
    env = {
        "DATASET_ROOT": args.dataset_root.as_posix(),
        "SEGMENTS_JSONL": (args.segments_jsonl or (args.dataset_root / "segments.jsonl")).as_posix(),
        "ANNOTATION_ROOT": args.annotation_root.as_posix(),
        "GAME_ID": args.game_id,
        "WORKERS": str(args.multi_worker_workers),
        "STAGE_SET": "balanced",
        "STAGE_PLAN": stage_plan,
        "LIMIT_PER_WORKER": args.multi_worker_limit_per_worker,
        "RESUME": "1",
        "OVERWRITE": "0",
        "QWEN3_OMNI_MAX_TOKENS": args.qwen_max_tokens,
        "QWEN3_OMNI_TEXT_MERGE_MAX_TOKENS": args.qwen_text_merge_max_tokens,
        "QWEN3_OMNI_VIDEO_FPS": args.qwen_video_fps,
        "QWEN3_OMNI_VIDEO_MAX_FRAMES": args.qwen_video_max_frames,
        "QWEN3_OMNI_VIDEO_MAX_PIXELS": args.qwen_video_max_pixels,
        "OMNI_HTTP_TIMEOUT_SEC": args.omni_http_timeout_sec,
    }
    cmd = [
        "sbatch",
        "--parsable",
        "--job-name=og-4x2-balanced",
        args.multi_worker_slurm_script.as_posix(),
    ]
    job_id = run_command(cmd, dry_run=args.dry_run, env=env)
    if job_id == SUBMIT_LIMIT:
        print("submitted_4x2_balanced=0 reason=submit_limit")
        return "submit_limit"
    else:
        print(f"submitted_4x2_balanced=1 job={job_id} stage_plan={stage_plan}")
        return "submitted"


def export_benchmark(args: argparse.Namespace) -> None:
    benchmark_root = args.annotation_root.parent / "benchmark"
    run_command(
        [
            ".venv/bin/python",
            "-u",
            "scripts/export_tom_benchmark.py",
            "--dataset-root",
            args.dataset_root.as_posix(),
            "--annotation-root",
            args.annotation_root.as_posix(),
            "--benchmark-root",
            benchmark_root.as_posix(),
        ],
        dry_run=args.dry_run,
    )
    run_command(
        [
            ".venv/bin/python",
            "-u",
            "scripts/report_benchmark_quality.py",
            "--dataset-root",
            args.dataset_root.as_posix(),
            "--benchmark-root",
            benchmark_root.as_posix(),
        ],
        dry_run=args.dry_run,
    )


def print_status(rows: list[tuple[int, str]], statuses: dict[str, SegmentStatus]) -> None:
    totals = {
        "upstream_complete": 0,
        "global_complete": 0,
        "downstream_complete": 0,
        "ready_for_global": 0,
    }
    for _, segment_id in rows:
        status = statuses[segment_id]
        totals["upstream_complete"] += int(status.upstream_complete)
        totals["global_complete"] += int(status.global_events)
        totals["downstream_complete"] += int(status.downstream_complete)
        totals["ready_for_global"] += int(status.ready_for_global)
    print(json.dumps(totals, ensure_ascii=False, sort_keys=True))
    for index, segment_id in rows:
        status = statuses[segment_id]
        print(
            f"{index:03d} {segment_id} "
            f"pov={status.pov_events}/6 utt={status.utterances}/6 phase={int(status.phase_events)} "
            f"global={int(status.global_events)} info={status.information_states}/6 "
            f"memory={status.memory_states}/6 belief={status.belief_states}/6 trials={status.candidate_trials}"
        )


def main() -> None:
    args = parse_args()
    rows = selected_segments(args)
    candidate_counts = load_candidate_counts(args.annotation_root)
    statuses = {
        segment_id: segment_status(args.annotation_root, segment_id, candidate_counts)
        for _, segment_id in rows
    }
    if args.mode == "status":
        print_status(rows, statuses)
    elif args.mode == "submit-upstream":
        if not queue_submit_allowed(args):
            print("submitted_upstream_total=0 queue_cap=1")
            return
        submit_upstream(args, statuses)
    elif args.mode == "promote-downstream":
        if not queue_submit_allowed(args):
            print("submitted_downstream_total=0 queue_cap=1")
            return
        promote_downstream(args, statuses)
    elif args.mode == "repair-downstream":
        if not queue_submit_allowed(args):
            print("submitted_repair_total=0 queue_cap=1")
            return
        repair_downstream(args, statuses)
    elif args.mode == "submit-balanced":
        submit_balanced(args, rows, statuses)
    elif args.mode == "submit-4x2-balanced":
        submit_4x2_balanced(args)
    elif args.mode == "export":
        export_benchmark(args)


if __name__ == "__main__":
    main()
