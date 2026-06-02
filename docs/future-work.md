# Future Work

Prior to building the deployment pipeline, speculative answers to follow-up questions were documented in [archive/further-questions.md](archive/further-questions.md). Many of those ideas have since been implemented. This document captures what's left to do.

## Implemented (no longer future work)

- **OOD detection** — Energy-based scoring on logits, calibrated threshold (-2.36), 0.824 AUROC on CIFAR-100
- **Scaling via queuing** — Redis queue decoupling API from inference, async job polling (`POST /predict` → `GET /result/{job_id}`)
- **Feedback loop** — Review UI for labeling OOD items, export to lakeFS branch for retraining

## Still To Do

### Model & Training

- **Adversarial robustness** — The model is not robust to adversarial perturbations. FGSM/PGD attacks would likely drop accuracy from 85.2% significantly. Adversarial training (PGD during training) is the most effective defence but 3-10x training cost.
- **Deeper architecture** — ResNet or similar would improve both accuracy and OOD separation. Current CNN tops out around 85%.
- **Outlier exposure** — Fine-tune with a small amount of OOD data using a uniform-distribution loss to explicitly teach low confidence on unknowns.
- **Temperature scaling** — Post-hoc calibration of logit magnitudes for better-calibrated confidence scores.

### Monitoring & Drift

- **Confidence distribution tracking** — Monitor prediction confidence over time; shift toward lower confidence signals drift.
- **Embedding drift** — Track penultimate-layer feature centroids via MMD against training reference.
- **Class distribution monitoring** — Alert on sudden changes in predicted class proportions.
- **Alerting stack** — Prometheus + Grafana tracking predictions/sec, confidence histograms, error rates. Alert on confidence below baseline, class distribution divergence, latency spikes.

### Security

This is a local proof-of-concept — security was not a primary concern. For a production deployment:

- **Secrets management** — Replace `.env` files with a proper secrets backend (Vault, AWS Secrets Manager, Kubernetes Secrets). No credentials in source or environment files.
- **Authentication** — All API endpoints are unauthenticated. Add API key or OAuth2 middleware. The `/reset` endpoint is especially dangerous without auth.
- **Network isolation** — Services like Redis, Postgres, and MinIO should not be exposed on host ports. Use internal compose networks only, with the API as the sole ingress point.
- **Image inputs** — The API accepts arbitrary image uploads. Add content-type validation, virus scanning, and stricter size limits for public-facing deployments.
- **Container hardening** — Run containers as non-root, use read-only filesystems where possible, pin image digests instead of tags.
- **TLS** — All inter-service communication is plaintext HTTP. Add TLS termination at the API layer and mTLS between services.

### Infrastructure

- **Worker scaling** — Add `deploy.replicas` or use `--scale inference-worker=N` for parallel processing.
- **Custom MLflow image** — Build with `psycopg2-binary` and `boto3` baked in instead of `pip install` at startup.
- **Dev dependencies** — Move `ipykernel` and `matplotlib` to optional deps to reduce production image size.
- **Input validation** — Cap `limit`/`offset` query params, paginate `/ood/pending` endpoint.
- **Auth** — The `/reset` endpoint (and all endpoints) have no authentication.
