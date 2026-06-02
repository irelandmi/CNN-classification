# Deployment Runbook: Inference + OOD Review Pipeline

## Overview

A local deployment pipeline that serves CIFAR-10 CNN inference via a Redis-backed queue, flags out-of-distribution inputs using energy scores, and provides a web UI for reviewing and exporting flagged items to lakeFS for retraining.

### Architecture

```
              ┌──────────────┐
              │  Client/UI   │
              └──────┬───────┘
                     │ HTTP :8080
              ┌──────▼───────┐
              │ inference-api│  FastAPI
              └──────┬───────┘
                     │ RPUSH / BLPOP
              ┌──────▼───────┐
              │    Redis     │  queue:inference, queue:ood_review
              └──────┬───────┘
                     │
              ┌──────▼───────┐
              │   worker     │  Cifar10CnnModel + energy OOD
              └──┬───┬───┬───┘
                 │   │   │
          ┌──────┘   │   └──────┐
          ▼          ▼          ▼
    ┌──────────┐ ┌──────────┐ ┌──────────┐
    │  MinIO   │ │ Postgres │ │  lakeFS  │
    │ ood-imgs │ │ ood table│ │ feedback │
    └──────────┘ └──────────┘ └──────────┘
```

### Services

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| redis | redis:7-alpine | 6379 | Message broker for inference + OOD queues |
| inference-api | Custom (Dockerfile) | 8080 | FastAPI — accepts images, serves results, hosts UI |
| inference-worker | Custom (Dockerfile) | — | Consumes queue, runs inference, routes OOD items |
| minio | minio/minio | 9000/9001 | Object storage for MLflow artifacts + OOD images |
| postgres | postgres:16 | 5432 | MLflow metadata + `ood_detections` table |
| mlflow | mlflow:v2.22.0 | 5050 | Experiment tracking, model artifact registry |
| lakefs | treeverse/lakefs:1.48 | 8000 | Data versioning, feedback export target |

## Prerequisites

- **Podman** with at least **6GB VM memory** (see [VM Setup](#vm-setup))
- **uv** (Python package manager)
- A trained model in MLflow — fetch data with `uv run python training/fetch_data.py` then train with `uv run python training/train.py`

## Quick Start

```bash
# From ml-infra/
podman compose up -d

# Wait ~3 minutes for worker to download model from MLflow
# Then verify
curl http://localhost:8080/health
```

Open http://localhost:8080 for the dashboard.

## VM Setup

The inference worker loads PyTorch + the CNN model into memory. The default 2GB Podman VM is not enough.

```bash
podman machine stop
podman machine set --memory 6144
podman machine start
```

Verify: `podman info | grep memTotal` should show ~6GB.

## Building

Both `inference-api` and `inference-worker` share one Dockerfile at `deploy/Dockerfile`.

```bash
# From ml-infra/
podman compose build inference-api inference-worker
```

The `.dockerignore` excludes `.venv/`, `datasets/`, `docs/`, `tests/`, `*.ipynb`, `__pycache__/`, and `.git/` — build context is ~150KB.

## Starting Services

```bash
# All services
podman compose up -d

# Just the inference stack (assumes infra already running)
podman compose up -d redis inference-api inference-worker
```

The worker takes 2-3 minutes on first start to download the model artifact from MLflow/MinIO.

## Web UI

| Page | URL | Purpose |
|------|-----|---------|
| Dashboard | http://localhost:8080/ | Live stats, queue depth, reset button |
| OOD Review | http://localhost:8080/ood/review | Paginated labeling UI for flagged images, export to lakeFS |
| OOD Records | http://localhost:8080/ood/records | Paginated table of all OOD detections |

## API Reference

### Inference

```bash
# Submit image (multipart)
curl -F file=@image.png http://localhost:8080/predict
# → {"job_id": "uuid", "status": "queued"}

# Submit image (base64 JSON)
curl -X POST http://localhost:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{"image_base64": "<base64>"}'

# Poll result
curl http://localhost:8080/result/{job_id}
# → {"job_id": "...", "status": "complete", "class_name": "cat",
#    "confidence": 0.92, "energy_score": -4.21, "is_ood": false}
```

### OOD Management

```bash
# List pending OOD items
curl http://localhost:8080/ood/pending

# Label an OOD item
curl -X POST http://localhost:8080/ood/{job_id}/label \
  -H 'Content-Type: application/json' \
  -d '{"true_label": "flower", "action": "new_class"}'
# Actions: "correct", "discard", "new_class"

# View OOD image
curl http://localhost:8080/ood/{job_id}/image > image.png

# Export labeled items to lakeFS
curl -X POST http://localhost:8080/ood/export
# → {"branch": "ood-feedback-2026-06-02", "items_exported": 5}
```

### System

```bash
# Health check
curl http://localhost:8080/health

# Stats
curl http://localhost:8080/stats
# → {"total_inferences": 150, "ood_count": 12, "queue_depth": 0, "ood_pending_review": 8}

# Reset all data (queues, DB, OOD images)
curl -X POST http://localhost:8080/reset
```

## Batch Processing

Submit all 10,000 CIFAR-100 test images:

```bash
cd CNN-classification
uv run python deploy/batch_submit.py           # all 10,000
uv run python deploy/batch_submit.py 100       # first 100
uv run python deploy/batch_submit.py 1000 32   # 1000 images, 32 threads
```

The script submits concurrently then polls `/stats` until the queue drains.

## OOD Detection

### How It Works

1. Worker computes logits from the CNN
2. Energy score: `energy = -log(Σ exp(logit_i))`
3. If `energy > threshold` → flagged as OOD
4. OOD images are stored in MinIO (`ood-images` bucket) and metadata in Postgres (`ood_detections` table)

### Threshold

The default threshold is **-2.36** (95th percentile of CIFAR-10 energy scores). Configure via `OOD_ENERGY_THRESHOLD` in `.env`.

At this threshold on CIFAR-100 test set: ~26% of images are flagged as OOD.

### Feedback Loop

1. OOD items appear in the review UI
2. A human labels them (true class + action)
3. "Export to lakeFS" uploads labeled images to branch `ood-feedback-{date}` in the `cifar10` repo
4. Images are organized as `ood-feedback/{label}/{job_id}.png`
5. The existing `fetch_data.py` can pull this data for retraining

## Verification

```bash
# From ml-infra/
bash ../jobs/CNN-classification/deploy/verify.sh
```

Runs 11 sections (~18 checks): health, inference (multipart + base64), OOD detection, labeling, image serving, review page, stats, container health. Includes a warmup step that waits up to 300s for the worker to load the model.

## Troubleshooting

### Worker exits with code 137 (OOM)

Increase Podman VM memory to at least 6GB (see [VM Setup](#vm-setup)).

### Worker exits with code 1

Check logs: `podman logs ml-infra-inference-worker-1`

Common causes:
- MLflow not ready yet — the worker retries up to 10 recent runs to find one with a `cifar10-cnn.pth` artifact
- No trained model exists — run `uv run python training/train.py` first

### Results stay "pending"

The worker may still be downloading the model (2-3 min on first start). Check:
```bash
podman top ml-infra-inference-worker-1   # is it running?
podman logs ml-infra-inference-worker-1  # what's it doing?
```

### Build context is huge

Ensure `.dockerignore` exists in `CNN-classification/` — without it, the build sends ~741MB (datasets + venv).

## File Structure

```
deploy/
  Dockerfile           # Python 3.13 + uv, shared by api + worker
  __init__.py
  config.py            # Env var parsing
  schemas.py           # Pydantic request/response models
  db.py                # Postgres ood_detections CRUD
  model_loader.py      # MLflow model download + image preprocessing
  api.py               # FastAPI app (all endpoints)
  worker.py            # BLPOP inference loop
  batch_submit.py      # Batch submission script
  verify.sh            # End-to-end verification
  templates/
    dashboard.html     # Stats dashboard with live refresh
    review.html        # OOD labeling UI
    records.html       # Paginated OOD records table
```

## Configuration

Credentials and shared settings are defined in `.env` (see `.env.example`). Service-specific config is set in `compose.yaml` using `${VAR}` substitution.

```bash
cp .env.example .env  # then edit as needed
```

### `.env` variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MINIO_ROOT_USER` | *(required)* | MinIO root username |
| `MINIO_ROOT_PASSWORD` | *(required)* | MinIO root password |
| `POSTGRES_USER` | *(required)* | Postgres username |
| `POSTGRES_PASSWORD` | *(required)* | Postgres password |
| `POSTGRES_DB` | *(required)* | Postgres database name |
| `LAKEFS_ENDPOINT` | `http://localhost:8000` | lakeFS endpoint (for host-side scripts) |
| `LAKEFS_ACCESS_KEY_ID` | *(required)* | lakeFS access key |
| `LAKEFS_SECRET_ACCESS_KEY` | *(required)* | lakeFS secret key |
| `LAKEFS_ENCRYPT_SECRET_KEY` | *(required)* | lakeFS auth encryption key |
| `CNN_CLASSIFICATION_PATH` | `../jobs/CNN-classification` | Path to CNN-classification repo (build context) |
| `OOD_ENERGY_THRESHOLD` | `-2.36` | Energy score above which inputs are flagged OOD |

### Application variables (set in `compose.yaml`)

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://redis:6379/0` | Redis connection |
| `DATABASE_URL` | `postgresql://...@postgres:5432/mlflow` | Postgres connection |
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | MLflow server |
| `MINIO_ENDPOINT` | `minio:9000` | MinIO for OOD image storage |
| `LAKEFS_ACCESS_KEY` | from `LAKEFS_ACCESS_KEY_ID` in `.env` | lakeFS access key (passed to app containers) |
| `LAKEFS_SECRET_KEY` | from `LAKEFS_SECRET_ACCESS_KEY` in `.env` | lakeFS secret key (passed to app containers) |
| `MAX_UPLOAD_BYTES` | `1048576` (1MB) | Max upload size per image |
| `RESULT_TTL_SECONDS` | `3600` | How long results stay in Redis |
