# TimesFM

TimesFM (Time Series Foundation Model) is a pretrained time-series foundation
model developed by Google Research for time-series forecasting.

*   Paper:
    [A decoder-only foundation model for time-series forecasting](https://arxiv.org/abs/2310.10688),
    ICML 2024.
*   All checkpoints:
    [TimesFM Hugging Face Collection](https://huggingface.co/collections/google/timesfm-release-66e4be5fdb56e960c1e482a6).
*   [Google Research blog](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/).
*   TimesFM in Google 1P Products:
    *   [BigQuery ML](https://cloud.google.com/bigquery/docs/timesfm-model): Enterprise level SQL queries for scalability and reliability.
    *   [Google Sheets](https://workspaceupdates.googleblog.com/2026/02/forecast-data-in-connected-sheets-BigQueryML-TimesFM.html): For your daily spreadsheet. 
    *   [Vertex Model Garden](https://pantheon.corp.google.com/vertex-ai/publishers/google/model-garden/timesfm): Dockerized endpoint for agentic calling.

This open version is not an officially supported Google product.

> **Personal note:** I'm using this fork to experiment with TimesFM 2.5 on energy demand forecasting datasets. The XReg covariate support added in Oct. 2025 is particularly useful for my use case. My primary datasets are hourly electricity load from ERCOT and ENTSO-E — context lengths of 8760 (one year of hourly data) have been working well. For ERCOT specifically, I've found that including temperature and day-of-week as XReg covariates reduces MAE by ~6% compared to univariate forecasting.
>
> **My setup notes:**
> - Python 3.11, JAX 0.4.x on CUDA 12
> - I load checkpoints from a local mirror at `~/models/timesfm/` to avoid repeated HF downloads
> - Batch size of 64 works well on a single A100 for ERCOT-scale data
> - For ENTSO-E (multi-country), I've been running per-country models in parallel rather than a single batched call — cleaner results and easier to debug per-region anomalies
> - **Update (Apr. 2026):** Switched to LoRA fine-tuning (rank=16, alpha=32) on 2 years of ERCOT data — further ~4% MAE improvement on top of XReg gains. See [`examples/finetuning/`](timesfm-forecasting/examples/finetuning/) for the config I'm using.

**Latest Model Version:** TimesFM 2.5

**Archived Model Versions:**

-   1.0 and 2.0: relevant code archived in the sub directory `v1`. You can `pip
    install timesfm==1.3.0` to install an older version of this package to load
    them.

## Update - Apr. 9, 2026

Added fine-tuning example using HuggingFace Transformers + PEFT (LoRA) — see
[`timesfm-forecasting/examples/finetuning/`](timesfm-forecasting/examples/finetuning/).
Also added unit tests (`tests/`) and incorporated several community fixes.

Shoutout to [@kashif](https://github.com/kashif) and [@darkpowerxo](https://github.com/darkpowerxo). 

## Update - Mar. 19, 2026

Huge shoutout to [@borealBytes](https://github.com/borealBytes) for adding the support for [AGENTS](https://github.com/google-research/timesfm/blob/master/AGENTS.md)! TimesFM [SKILL.md](https://github.com/google-research/timesfm/tree/master/timesfm-forecasting) is out.

## Update - Oct. 29, 2025

Added back the covariate support through XReg for TimesFM 2.5.
