# Development Process

How this project was built, from initial training script to full deployment pipeline.

This document has been generated based on my ai conversations for transparency.

## Phase 1: Infrastructure Foundation (May 31)

Started with a bare CIFAR-10 PyTorch training script (~75% accuracy) and an empty `ml-infra` repo. The goal was to build production-grade ML infrastructure for a demo.

**Decisions made:**
- **Podman Compose** over Docker — already installed, rootless, compatible with Docker Compose format
- **MLflow + MinIO + PostgreSQL** for experiment tracking and artifact storage
- **lakeFS** for dataset versioning — initially debated whether it was overkill vs DVC, decided it was worth it for demonstrating data lineage and the feedback loop
- Moved MLflow to **port 5050** to avoid macOS conflict on port 5000

**Problems solved:**
- MLflow container binding to `127.0.0.1` instead of `0.0.0.0` — caused by YAML multiline command quoting, fixed with list-style command format
- lakeFS pre-signed URLs unresolvable from the host (container hostnames) — fixed with `pre_sign=False`

## Phase 2: Data Pipeline (May 31)

Built scripts to version datasets in lakeFS:
- `load_data.py` — uploads CIFAR-10/CIFAR-100 batches to lakeFS repositories
- `fetch_data.py` — pulls versioned data to local disk for training

This established data lineage: every training run can trace back to a specific lakeFS commit.

## Phase 3: Remote Training on DGX Spark (May 31)

Developing on a MacBook but needed GPU training. Built `run_remote.sh` for SSH-based remote execution on an NVIDIA DGX Spark (GB10).

**How it works:** rsync project files to the remote, SSH in, run the command, results tracked in MLflow (accessible from both machines).

**Problems solved:**
- `uv: command not found` in non-interactive SSH — needed explicit PATH export
- Set up SSH keys to avoid repeated password prompts
- rsync directory path issues with trailing slashes

## Phase 4: MLflow Integration (May 31)

Wired MLflow tracking into `train.py`:
- Logged hyperparameters (batch size, learning rate, dropout, weight decay)
- Per-epoch metrics (train loss, val loss, val accuracy)
- Test set evaluation
- Model artifact (`cifar10-cnn.pth`)

Every training run is reproducible: parameters in, metrics out, artifact stored.

## Phase 5: Model Improvement (May 31 – June 1)

Iterated on the CNN architecture to improve accuracy:

| Change | Test Accuracy | Energy AUROC |
|--------|--------------|--------------|
| Baseline (no regularisation) | 75.0% | 0.756 |
| + BatchNorm + Dropout (0.3) + augmentation + weight decay | **85.2%** | **0.824** |

**Augmentation:** RandomCrop (32, padding=4) + RandomHorizontalFlip — standard for CIFAR-10.

All runs tracked in MLflow, making it easy to compare and select the best model.

## Phase 6: OOD Detection (June 1)

Implemented out-of-distribution detection to identify inputs the model wasn't trained on.

**Approach:** Energy-based scoring on logits — `energy = -log(Σ exp(logit_i))`. High energy = model is uncertain = likely OOD.

**Evaluation:**
- Used CIFAR-100 as the OOD dataset (100 classes, none overlapping with CIFAR-10)
- Energy scoring achieved **0.824 AUROC**
- Also tried Mahalanobis distance — performed poorly (0.43 AUROC) because BatchNorm compresses the feature space
- Built `calibrate_threshold.py` to select the energy threshold at 95% ID retention → **-2.36**

**Decision:** Stuck with energy scoring — simpler, better performing, no additional feature extraction needed.

## Phase 7: Deployment Pipeline (June 1 – June 2)

Built a full inference + OOD review system running locally via Podman Compose.

**Components built:**
- **FastAPI API** (`api.py`) — accepts images via upload or base64, returns async job IDs
- **Redis queue** — decouples API from inference, enables backpressure
- **Async worker** (`worker.py`) — BLPOP loop, runs inference, computes energy scores, routes OOD items
- **Postgres** — stores OOD detection metadata (energy score, predicted class, confidence, review status)
- **MinIO** — stores OOD images for review
- **Web UI** — dashboard with live stats, paginated OOD review page with labeling controls, records table
- **Dockerfile** — Python 3.13-slim + uv, shared by API and worker containers

**Model loading strategy:** Worker checks for a local cache first, then falls back to downloading from MLflow. First start takes 2-3 minutes; subsequent starts are instant.

## Phase 8: OOD Feedback Loop (June 2)

Closed the loop from detection to retraining data:

1. OOD images flagged during inference appear in the review UI
2. A human labels them (true class + action: correct/discard/new_class)
3. "Export to lakeFS" uploads labeled images to branch `ood-feedback-{date}` in the `cifar10` repo
4. Images organized as `ood-feedback/{label}/{job_id}.png`
5. `fetch_data.py` can pull this branch for retraining

**Batch test:** Submitted all 10,000 CIFAR-100 test images at ~1,200 img/s. 2,614 flagged as OOD (26.1%). Reviewed and exported samples successfully to lakeFS.

## Phase 9: Verification & Hardening (June 2)

- Built `verify.sh` — end-to-end test script (11 sections, ~18 checks) covering health, inference, OOD detection, labeling, export, and container health
- `smoke_test.py` — quick connectivity check for GPU, MLflow, lakeFS
- `test_infra.py` — round-trip lakeFS test (upload, commit, read back, verify bytes)

## Stack

```
PyTorch          Model training + inference
MLflow           Experiment tracking, artifact storage
lakeFS           Dataset versioning, feedback export
MinIO            S3-compatible object storage
PostgreSQL       MLflow metadata, OOD detection records
Redis            Inference job queue
FastAPI          Serving API + web dashboard
Podman Compose   Local orchestration (8 services)
uv               Python package management
```
