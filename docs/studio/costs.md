# Infrastructure and cost estimates

Nothing paid was bought or deployed. GPU prices are from third-party comparison pages (checked 2026-11-14); **verify them on the providers' own pages before committing**:
[Modal per-second pricing summary](https://spheron.network/blog/modal-gpu-pricing-2026-per-second-billing/),
[Modal vs RunPod serverless](https://lyceum.technology/magazine/modal-vs-runpod-for-serverless-gpu-inference/),
[serverless GPU comparison, Jan 2026](https://www.koyeb.com/blog/best-serverless-gpu-platforms-for-ai-apps-and-inference-in-2026),
[gpuperhour serverless pricing](https://gpuperhour.com/blog/serverless-gpu-pricing).

Approximate serverless rates: L4 about $0.70–0.80/h, A10 about $1.10/h, L40S about $1.75–1.95/h, A100 80 GB about $2.50–2.72/h, H100 about $3.95–4.79/h.

## Development (now)

| Item | Cost |
|---|---|
| Procedural engine, editor, DSP, Basic Pitch | $0. Runs on a laptop CPU |
| Your Mac (Apple Silicon) | Studio runs as is (`pip install -r requirements.txt`, then `SOUNDLENS_STUDIO_DEV_AUTH=1 uvicorn visual_app:app`). ACE-Step lists MPS/MLX support, but **no Mac timings are published**: test before relying on it. It needs ~5 GB of weights for the 2B model; 16 GB+ unified memory recommended |
| ACE-Step experiments | Rent a GPU by the hour: an L4/A10 (24 GB) fits 2B Turbo + Base. A day of testing is roughly $5–20 |
| Claude Max | Does **not** include inference for third-party models. Model hosting is a separate cost |

## Production estimate per song (assumptions stated; replace with measured numbers after the GPU test)

| Step | Where | Assumption | Cost |
|---|---|---|---|
| Structure + 8 stems, 3 min | CPU | ~50 s of 4 vCPU per song (measured scaling) | ≈ $0.001–0.003 at typical cloud vCPU prices |
| Edit (one track) | CPU | 0.2–2 s | ≈ $0 |
| ACE-Step full song (Turbo) | L4/A10 | vendor: <10 s on an RTX 3090 → assume 20–40 s on an L4 | ≈ $0.005–0.01 + cold start |
| ACE-Step vocal layer (`lego`, Base, 50 steps) | L4/A10 | ~6× Turbo → 2–4 min | ≈ $0.03–0.06 |
| Repaint a section (Turbo) | L4/A10 | 10–30 s | ≈ $0.003–0.008 |
| Cold start (load 5–10 GB of weights) | GPU | 30–90 s per new container | ≈ $0.01–0.02 each; keep one warm worker when there's traffic |
| Storage per project | volume / object store | rendered stem cache is ~63 MB per 3-min stem (float32), LRU-pruned (`SOUNDLENS_STUDIO_CACHE_GB`, default 5). Only `project.json` + uploads are permanent (≈ 50 KB + uploads) | small |

**Rough envelope.** An instrumental song with a few edits costs well under $0.01 of compute. A song with an AI vocal layer and two regenerations costs about $0.05–0.15, dominated by GPU time and cold starts. One always-warm L4 costs about $520–580/month. Serverless scale-to-zero is cheaper until steady traffic.

**Credits suggestion:** free plan = procedural songs (cheap) + N AI-vocal credits per month; Pro = more credits. `SOUNDLENS_STUDIO_FREE_DAILY` already caps free generations per day.

## Scaling notes

- Today's jobs run in-process (threads). For more than one web instance:
  - a queue (Redis/RQ, or the platform's queue);
  - a separate GPU worker service that runs `acestep-api` beside a small worker pulling jobs;
  - object storage for stems and exports.
- The stem cache is content-addressed, so CDN or object-store caching is trivial.
- The Railway volume `/data` is fine for the early beta. Watch volume size as the stem cache grows; the LRU cap protects it.
