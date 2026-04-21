# TimesFM — Agent Entry Point

This repository ships a first-party **Agent Skill** for TimesFM at:

```
timesfm-forecasting/
└── SKILL.md    ← read this for the full skill
```

## Install the skill

Copy the skill directory into your agent's skills folder:

```bash
# Cursor / Claude Code / OpenCode / Codex (global install)
cp -r timesfm-forecasting/ ~/.cursor/skills/
cp -r timesfm-forecasting/ ~/.claude/skills/

# Or project-level
cp -r timesfm-forecasting/ .cursor/skills/
```

Any agent that supports the open [Agent Skills standard](https://agentskills.io) will discover it automatically.

## Working in this repo

If you are developing TimesFM itself (not using it), the source lives in `src/timesfm/`.
Archived v1/v2 code and notebooks are in `v1/`.

Run tests:

```bash
pytest v1/tests/
```

See `README.md` for full developer setup.

## Personal notes

- I'm using this primarily for experimenting with electricity demand forecasting.
- The `v1/` notebooks are a good starting point — `tutorial.ipynb` in particular.
- TODO: try fine-tuning on a custom dataset once I get the training pipeline working.
- NOTE: when loading the model locally, I've been using `backend="cpu"` to avoid CUDA issues on my dev machine — swap to `"gpu"` before any serious benchmarking.
- NOTE: set `forecast_context_len=512` when working with hourly electricity data — the default 128 was too short and hurt accuracy noticeably on my datasets.
- NOTE: set `num_layers=20` and `model_dims=1280` when loading the 200m checkpoint — easy to accidentally load the wrong config and get silent garbage outputs.
- NOTE: set `forecast_context_len=2048` for daily data spanning multiple years — 512 is fine for hourly but undershoots on longer daily series with strong yearly seasonality.
- NOTE: set `horizon_len=24` for day-ahead electricity forecasting (hourly resolution) — the default horizon was too short for my use case and I kept forgetting to override it each session.
