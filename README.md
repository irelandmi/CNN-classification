# CNN-Classification

CIFAR-10 image classifier with an end-to-end MLOps pipeline: training on a DGX node, out-of-distribution (OOD) detection via energy scoring, and a queue-based inference API with a human-in-the-loop feedback system.

## Architecture

```
training/        – Data fetching (lakeFS), model training, MLflow tracking
eval/            – OOD evaluation (Mahalanobis + energy), threshold calibration
deploy/          – FastAPI inference API + Redis queue + async worker
tests/           – Smoke tests and infra validation
```

## Stack

| Component | Role |
|-----------|------|
| PyTorch | CNN model (CIFAR-10, 512-dim feature embedding) |
| MLflow | Experiment tracking |
| lakeFS | Dataset versioning |
| MinIO | Object storage (model artifacts, OOD images) |
| Redis | Inference job queue |
| PostgreSQL | OOD review metadata |
| FastAPI | Serving API + dashboard |

## How It Works

1. **Training** – Fetches CIFAR-10 from lakeFS, trains the CNN on a DGX node, logs metrics to MLflow.
2. **OOD Detection** – Uses energy-based scoring on logits. Images above the calibrated threshold are flagged as out-of-distribution.
3. **Inference** – API accepts images (upload or base64), pushes to a Redis queue. A worker processes jobs, runs inference, and stores results.
4. **Feedback Loop** – OOD images are stored in MinIO. A review UI lets humans label them. Reviewed samples are exported to a lakeFS branch for retraining.

## Running

### Remote (DGX)

```bash
cp .env.example .env  # configure DGX_HOST, MLFLOW, lakeFS credentials
./run_remote.sh all    # smoke | train | eval | all
```

### Local (inference stack)

```bash
uv sync
uv run uvicorn deploy.api:app --host 0.0.0.0 --port 8000
uv run python deploy/worker.py  # in a separate terminal
```

Requires Redis, PostgreSQL, and MinIO running locally (see `.env.example` for connection details).
