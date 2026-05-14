# Deep Research Archive

AI-powered deep research reports with full source materials and citations.

## Structure

```
research/
├── YYYY-MM-DD-topic-slug/
│   ├── report.md           # Final research report
│   ├── sources/            # Raw extracted source materials
│   │   ├── 01-source-title.md
│   │   ├── 02-source-title.md
│   │   └── ...
│   ├── plan.md             # Research plan & sub-questions
│   └── metadata.json       # Search queries, timestamps, stats
└── ...
```

## Levels

| Level | Depth | Time |
|-------|-------|------|
| L1 Quick | 3-5 searches | 1-3 min |
| L2 Standard | 5-10 searches | 5-10 min |
| L3 Deep | 15-30 searches, parallel agents | 10-20 min |
| L4 Exhaustive | 30-50+ searches, multi-pass | 20-45 min |

## Usage

Research is conducted by [Hermes Agent](https://github.com/NousResearch/hermes-agent) using the `deep-research` skill.

Each research session creates a dated folder with the full report, all raw source materials, research plan, and metadata for reproducibility.
