# SocialOmni-Goose

SocialOmni-Goose is a trajectory-derived Theory-of-Mind benchmark built from a six-POV aligned Goose Goose Duck replay. It evaluates whether an omni model can separate oracle truth from each player perspective, reconstruct pre-reveal false beliefs, and predict strategic communication effects.

## Benchmark Question

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

## Project Layout

```text
benchmark/social_omni_goose_v1/   Official benchmark data and scorer-private gold
src/socialomni_goose/             Decrypto-style ToM schemas, builders, validators, scorers
src/socialomni_annotation/        Legacy aligned-video annotation utilities used by the build chain
tools/                            Build, annotation, evaluation, packaging, and validation CLIs
configs/slurm/                    HPC job templates and runtime examples
integrations/models/              Local model server and client integrations
docs/                             Benchmark design and annotation documentation
tests/                            Unit and release validation tests
```

The repository root is intentionally small. Generated annotations, local runs, raw videos, and model responses stay outside Git.

## Benchmark Data

Current benchmark version:

```text
benchmark/social_omni_goose_v1/
```

Important files:

| file | content |
|---|---|
| `public/leaderboard_core/trials.jsonl` | Structured leaderboard core, 889 public trials. |
| `public/leaderboard_core/interactive_prompts.jsonl` | A/B/C/D public prompts for interactive diagnostics. |
| `public/leaderboard_core/probe_groups_public.jsonl` | Public probe-group metadata with internal review fields removed. |
| `public/raw_video_smoke/raw_video_smoke.jsonl` | 48 raw-video smoke trials. |
| `private/leaderboard_core/gold.jsonl` | Scoring metadata. |
| `private/leaderboard_core/hidden_gold.jsonl` | Scorer-only answers. Never put this file into model prompts. |
| `tables/README.md` | Human-readable metric and stratified tables. |
| `manifest.json` | File list, SHA-256 hashes, line counts, and public-file leak scan. |

Core counts:

```text
leaderboard_core trials          889
leaderboard_core hidden gold     889
raw-video smoke trials            48
public probe groups              276
public_file_leak_hits             []
```

## Diagnostic Design

SocialOmni-Goose follows a Decrypto-style diagnostic-generation pipeline:

```text
6-POV aligned replay trajectory
-> oracle trajectory ledger
-> visibility / belief / memory projection
-> grouped A/B/C/D ToM probes
-> leaderboard trials
-> controlled scoring
```

The storage unit is an aligned video segment or phase clip. The benchmark unit is a trajectory node:

```text
cutoff_abs_sec + target_player + query_variable + probe_type
```

Probe groups are generated around information gaps, private observations, verifiable claims, delayed public reveals, and strategic communication events.

## Probe Types

| probe type | purpose |
|---|---|
| `A_pre_reveal_belief` | Ask what the target player can believe before truth reveal, using only target-available evidence. |
| `B_post_reveal_reconstruct_previous_belief` | Reveal oracle truth, then ask the model to reconstruct the target player earlier belief without hindsight leakage. |
| `C_other_agent_false_belief` | Ask what another player would believe before reveal, given that player limited information. |
| `D_perspective_taking_prediction` | Ask how a speaker should expect a listener to interpret a strategic claim or accusation. |

## Metrics

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

Aggregate score:

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

## Evaluation Tools

```text
tools/build/prepare_social_omni_goose_qwen_eval_workspace.py
tools/eval/run_social_omni_goose_qwen_eval.py
tools/eval/score_social_omni_goose_qwen_eval.py
configs/slurm/qwen3_omni_social_omni_eval.slurm
```

Use `PYTHONPATH=src:.` when running tools directly from a checkout:

```bash
PYTHONPATH=src:. .venv/bin/python tools/eval/run_social_omni_goose_qwen_eval.py --help
PYTHONPATH=src:. .venv/bin/python tools/eval/score_social_omni_goose_qwen_eval.py --help
```

The Qwen3-Omni Slurm runner uses quality-first defaults, including `QWEN3_OMNI_MAX_TOKENS=16384`, `QWEN3_OMNI_TEXT_MERGE_MAX_TOKENS=32768`, `QWEN3_OMNI_VIDEO_FPS=1.0`, `QWEN3_OMNI_VIDEO_MAX_FRAMES=128`, and `OMNI_HTTP_TIMEOUT_SEC=1200`.

## Local-Only Artifacts

The following paths are intentionally ignored and should not be pushed:

```text
runs/
annotations_qwen/
goose_data/
results/*
tmp_pass*.py
work/
.codex_tmp_sync/
benchmark/* except benchmark/social_omni_goose_v1/
```

Model responses and raw run scores are not part of the clean benchmark. Local evaluation summaries may exist under ignored workspaces such as `runs/social_omni_goose_eval_qwen3omni_pass001/LOCAL_RESULTS.md`.

## Validation

Recommended pre-push checks:

```bash
PYTHONPATH=src:. .venv/bin/python -m py_compile $(git ls-files "*.py")
bash -n configs/slurm/*.slurm configs/slurm/*.sh
PYTHONPATH=src:. .venv/bin/python -m pytest tests/test_decrypto_diagnostics.py tests/test_leaderboard_core_validation.py
```

## Citation

This repository is being organized as the SocialOmni-Goose benchmark foundation. Add the paper citation here once the benchmark paper metadata is finalized.
