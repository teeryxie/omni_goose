# SocialOmni-Goose

SocialOmni-Goose is a trajectory-derived Theory-of-Mind benchmark built from a six-POV aligned Goose Goose Duck replay. The benchmark turns a fixed multiplayer social-game trajectory into grouped diagnostic probes that test whether an omni model can distinguish oracle truth from the limited perspective of each player.

The core benchmark question is:

```text
Given a real multi-agent social-game trajectory, can a model distinguish:
1. what globally happened;
2. what a specific player saw or heard at a cutoff;
3. what that player did not see, but another player did;
4. whether a heard claim conflicts with oracle events;
5. whether the target player can know that conflict;
6. what false or incomplete belief existed before reveal;
7. whether the model can reconstruct that prior belief after truth is revealed;
8. how one player should expect another player to interpret a strategic utterance.
```

## Benchmark Design

SocialOmni-Goose follows the Decrypto-style diagnostic-generation principle:

```text
6-POV aligned replay trajectory
-> oracle trajectory ledger
-> visibility / belief / memory projection
-> grouped A/B/C/D ToM probes
-> leaderboard trials
-> controlled scoring
```

The storage unit is an aligned segment or phase video, but the benchmark unit is a trajectory node:

```text
cutoff_abs_sec + target_player + query_variable + probe_type
```

Each probe group is generated around an information gap, a private observation, a verifiable claim, a delayed public reveal, or a strategic communication event.

## Probe Types

| probe type | purpose |
|---|---|
| `A_pre_reveal_belief` | Ask what the target player can believe before truth reveal, using only target-available evidence. |
| `B_post_reveal_reconstruct_previous_belief` | Reveal oracle truth, then ask the model to reconstruct the earlier belief of the target player without hindsight leakage. |
| `C_other_agent_false_belief` | Ask what another player would believe before reveal, given the limited information available to that player. |
| `D_perspective_taking_prediction` | Ask how a speaker should expect a listener to interpret a strategic claim or accusation. |

## Metrics

Main metrics include:

| metric | meaning |
|---|---|
| `RC_weak` / `RC_strong` | Representational-change consistency after reveal. |
| `FB_weak` / `FB_strong` | False-belief recognition and reconstruction. |
| `PT_weak` / `PT_strong` | Perspective-taking and strategic-communication prediction. |
| `claim_verification_global` | Whether a claim is globally supported, contradicted, or unresolved. |
| `claim_verification_local` | Whether the target player can know the truth status of the claim. |
| `perspective_leakage_rate` | Whether a response cites information unavailable to the simulated player. |
| `forbidden_evidence_usage_rate` | Whether hidden scorer-only evidence is used in the answer. |
| `death_skill_overclaim_rate` | Whether visible death/body/blood evidence is overclaimed as killer, role, alignment, skill, or mechanism. |
| `evidence_support_rate` | Whether cited evidence is compatible with the allowed input condition. |
| `json_parse_success` / `schema_validation_success` | Output reliability and schema compliance. |

The aggregate score is:

```text
SoG-ToM-Core =
  0.20 * RC_strong
+ 0.20 * FB_strong
+ 0.20 * PT_strong
+ 0.15 * claim_verification_local
+ 0.10 * evidence_support_rate
+ 0.10 * (1 - perspective_leakage_rate)
+ 0.05 * (1 - death_skill_overclaim_rate)
```

## Clean Benchmark Release

The current clean benchmark package is:

```text
release/social_omni_goose_clean_benchmark_pass248/
```

Layout:

```text
public/                 # model-facing benchmark inputs, no scorer-private answers
scorer_private/         # gold and scorer-private answer files for local scoring only
tables/                 # Markdown and CSV summary tables
manifests/              # file list, hashes, and public-file leak scan
```

Important files:

| file | content |
|---|---|
| `public/leaderboard_core/trials.jsonl` | Structured leaderboard core, 889 public trials. |
| `public/leaderboard_core/probe_groups_public.jsonl` | Public probe-group metadata with internal review fields removed. |
| `public/raw_video_smoke/raw_video_smoke.jsonl` | 48 public raw-video smoke trials. |
| `scorer_private/leaderboard_core/gold.jsonl` | Public gold metadata for scoring. |
| `scorer_private/leaderboard_core/hidden_gold.jsonl` | Scorer-only hidden answers. Do not place into model prompts. |
| `tables/README.md` | Main score table and stratified result tables. |
| `manifests/clean_benchmark_manifest.json` | File sizes, SHA-256 hashes, and public-file leak scan. |

Current package checks:

```text
public_file_leak_hits = []
nonprivate_internal_name_hits = []
leaderboard_core trials = 889
raw-video smoke trials = 48
```

## Evaluation Runner

The evaluation workspace and outputs are local-only and ignored by Git. The reusable runner scripts are tracked:

```text
scripts/prepare_social_omni_goose_qwen_eval_workspace.py
scripts/run_social_omni_goose_qwen_eval.py
scripts/score_social_omni_goose_qwen_eval.py
slurm/qwen3_omni_social_omni_eval.slurm
```

Quality-first Qwen3-Omni defaults used by the Slurm runner:

```text
QWEN3_OMNI_MAX_TOKENS=16384
QWEN3_OMNI_TEXT_MERGE_MAX_TOKENS=32768
QWEN3_OMNI_VIDEO_FPS=1.0
QWEN3_OMNI_VIDEO_MAX_FRAMES=128
QWEN3_OMNI_VIDEO_MAX_PIXELS=401408
OMNI_HTTP_TIMEOUT_SEC=1200
```

## Local-Only Artifacts

The following are intentionally ignored and should not be pushed:

```text
runs/
annotations_qwen/
benchmark/
goose_data/
results/*
tmp_pass*.py
work/
.codex_tmp_sync/
```

Model responses and raw run scores are not part of the clean benchmark package. A local result summary may exist under the ignored evaluation workspace as `LOCAL_RESULTS.md`.

## Validation

Recommended pre-push checks:

```bash
python -m py_compile scripts/*.py socialomni_annotation/**/*.py models/utils/omni_http_client.py
bash -n slurm/*.slurm
python -m pytest tests/test_decrypto_diagnostics.py tests/test_leaderboard_core_validation.py
```

## Citation

This repository is being reorganized as the SocialOmni-Goose benchmark foundation. Add the paper citation here once the benchmark paper metadata is finalized.
