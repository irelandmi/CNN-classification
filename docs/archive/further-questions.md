# Further Questions — CIFAR-10 CNN

## 1. Out-of-Distribution (OOD) Detection

**Current behaviour:** The model always outputs a softmax distribution over 10 classes. It will confidently assign some class even to completely unrelated images — there is no mechanism to say "I don't know."

**Improvements:**

- **Softmax thresholding (simplest):** Reject predictions where `max(softmax(logits)) < threshold` (e.g. 0.85). Label those as "unknown" in the API response.
- **Temperature scaling:** Learn a temperature parameter on a held-out set so softmax probabilities are better calibrated before thresholding.
- **Energy-based OOD detection:** Use log-sum-exp of logits as a scoring function — better at separating in-distribution from OOD than raw softmax (Liu et al., NeurIPS 2020).
- **MC Dropout:** Run multiple forward passes with dropout enabled at inference. High variance across passes signals OOD.
- **Explicit OOD detector:** Measure Mahalanobis distance in the penultimate-layer feature space against class centroids from training data.

## 2. Scaling the Endpoint — Batching & Queuing

- **Request batching:** Buffer incoming requests and run inference in batches. GPU throughput scales much better with batches than single images. Use an asyncio queue with max batch size / max wait time, or a dedicated server like NVIDIA Triton.
- **Message queue (Redis, RabbitMQ, SQS):** Decouple API from inference workers. API writes to queue, workers pull batches, results returned via callback or polling. Workers scale independently.
- **Horizontal scaling:** Multiple inference workers behind a load balancer. Scale with Kubernetes HPA keyed on queue depth or latency.
- **Model optimisation:** TorchScript (`torch.jit.trace`) or ONNX Runtime for faster inference. INT8 quantization for CPU deployments.
- **Async API design:** Return `202 Accepted` with a job ID; client polls `/results/{job_id}`. Prevents HTTP timeouts under load.
- **Caching:** Cache predictions keyed by image hash for repeated inputs.

## 3. Feeding "Unknown" Feedback Back Into the Model

- **Logging & labelling pipeline:** Log OOD-flagged inputs. Route to a labelling tool (e.g. Label Studio) for human annotation — is this a new class or noise?
- **Class expansion via fine-tuning:** Once enough labelled examples exist, expand the final linear layer (10 -> 11+ outputs) and fine-tune on combined data. Use a lower learning rate and freeze early conv layers to avoid catastrophic forgetting.
- **Continual learning:** Elastic Weight Consolidation (EWC) or replay buffers to add classes without forgetting.
- **Active learning:** Prioritise labelling the most uncertain OOD samples (highest entropy) for maximum annotation efficiency.
- **Safe rollout:** Maintain a "known good" model in production. Shadow-deploy the updated model and A/B test before promoting.

## 4. Detecting Performance Degradation, Drift, and Rising Error Rates

**Inference-time monitoring:**
- Prediction confidence distribution over time — shift toward lower confidence signals drift.
- Class distribution of predictions — sudden imbalance changes indicate input distribution shift.
- Latency tracking (p50, p95, p99).

**Data drift detection:**
- Compare input feature distributions against training reference using statistical tests (KS test, Population Stability Index).
- Embedding drift: extract penultimate-layer features, track centroid shift via Maximum Mean Discrepancy (MMD).

**Ground-truth feedback loop:**
- If labels are available (even partial), compute rolling accuracy/F1. Alert on drops.
- Deploy a canary/shadow model retrained on recent data; monitor disagreement rate with production model.

**Alerting:**
- Prometheus + Grafana (or equivalent) tracking predictions/sec, confidence histograms, class distributions, error rates.
- Alerts on: confidence below baseline, class distribution divergence (Jensen-Shannon), rising error rate, latency spikes.

## 5. Adversarial Robustness

**Current state:** The model is almost certainly not robust to adversarial inputs. Standard CNNs are trivially fooled by small perturbations.

**Demonstrating vulnerability:**
- **FGSM (Fast Gradient Sign Method):** Perturb input by `epsilon * sign(gradient of loss w.r.t. input)`. Even `epsilon=0.01` can flip predictions. ~10 lines of code.
- **PGD (Projected Gradient Descent):** Iterative FGSM, stronger attack. Available in `torchattacks` library.
- **Metric:** Report accuracy at various epsilon values, e.g. "Clean: 75%, FGSM eps=0.03: ~20%, PGD eps=0.03: ~8%."

**Improving robustness:**
- **Adversarial training:** Generate PGD adversarial examples during training, train on both clean and adversarial images. Most effective known defence, but 3-10x training cost.
- **Input preprocessing:** JPEG compression, spatial smoothing, feature squeezing — can neutralise some perturbations but are bypassable by adaptive attacks.
- **Certified defences:** Randomized smoothing provides provable robustness within a certified radius, at some accuracy cost.
