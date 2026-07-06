# SocialOmni-Goose Clean Benchmark pass248

这是从 `pass248_death_skill_scope_audit` 整理出的干净 benchmark 目录。

## 目录结构

```text
public/                 # 给被测模型和公开评测使用，不含 hidden gold
scorer_private/         # scorer-only gold / scorer_private_gold，本地评分使用
tables/                 # 汇总表格: Markdown + CSV
manifests/              # 文件清单、大小、sha256、public JSON leak scan
```

## 公开输入

- `public/leaderboard_core/trials.jsonl`: Structured Leaderboard Core，889 条。
- `public/leaderboard_core/probe_groups_public.jsonl`: 已移除内部复核字段的 probe group。
- `public/static_trials/trials.jsonl`: static trials。
- `public/interactive_diagnostics/prompts.jsonl`: interactive diagnostic prompts。
- `public/raw_video_smoke/raw_video_smoke.jsonl`: raw-video smoke 48 条公共输入。

## Scorer-only

`scorer_private/` 包含 gold 与 scorer_private_gold。不要把这些内容放进被测模型 prompt。

## 表格

- `tables/README.md`: 主指标和分层表。
- `tables/metric_summary.csv`
- `tables/by_probe_type.csv`
- `tables/by_input_condition.csv`
- `tables/by_template.csv`
- `tables/dataset_composition.json`

## 当前模型 pass001 参考结果

参考结果不放在本目录，只保留在本地 ignored 的评测工作区 `LOCAL_RESULTS.md` 中。

完整模型 responses 和 scores 不包含在 clean benchmark 目录中。