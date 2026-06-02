# CIFAR-10 CNN — Model Architecture & OOD Detection

## Model Architecture

A 6-layer CNN trained on CIFAR-10 (10 classes, 32x32 RGB images).

### Network Structure

```
Input (3x32x32)
  │
  ├─ Conv2d(3→32, 3x3) → BatchNorm → ReLU
  ├─ Conv2d(32→64, 3x3) → BatchNorm → ReLU
  ├─ MaxPool2d(2x2) → Dropout2d(0.3)          # 64x16x16
  │
  ├─ Conv2d(64→128, 3x3) → BatchNorm → ReLU
  ├─ Conv2d(128→128, 3x3) → BatchNorm → ReLU
  ├─ MaxPool2d(2x2) → Dropout2d(0.3)          # 128x8x8
  │
  ├─ Conv2d(128→256, 3x3) → BatchNorm → ReLU
  ├─ Conv2d(256→256, 3x3) → BatchNorm → ReLU
  ├─ MaxPool2d(2x2) → Dropout2d(0.3)          # 256x4x4
  │
  ├─ Flatten                                    # 4096
  ├─ Linear(4096→1024) → ReLU → Dropout(0.3)
  ├─ Linear(1024→512) → ReLU → Dropout(0.3)
  └─ Linear(512→10)                            # logits
```

### Regularisation

The initial model (no regularisation) overfit heavily — train loss of 0.19 vs val loss of 0.94 — and achieved 75% test accuracy. We added:

- **BatchNorm** after each conv layer — stabilises training and acts as a mild regulariser
- **Dropout (0.3)** after each pooling layer and each FC layer — prevents co-adaptation of features
- **Weight decay (1e-4)** — L2 penalty on the optimiser
- **Data augmentation** — random crop (32x32 with 4px padding) and horizontal flip

This closed the generalisation gap (train 0.51, val 0.42) and improved test accuracy to **85.2%**.

### Training Configuration

| Parameter     | Value                        |
|---------------|------------------------------|
| Optimiser     | Adam (lr=0.001, wd=1e-4)     |
| Batch size    | 128                          |
| Epochs        | 30                           |
| Val split     | 5,000 from training set      |
| Augmentation  | RandomCrop(32,pad=4) + HFlip |

## Out-of-Distribution Detection

### Problem

The model is trained on 10 CIFAR-10 classes. At inference, it may encounter images from classes it has never seen. A standard classifier will still assign one of the 10 known classes with high confidence — we need a way to flag these inputs as "unknown".

### Approach: Energy Score

We use the **energy score** derived from the model's logits:

```
energy(x) = -log(Σ exp(logit_i))
```

This is the negative log-sum-exp of the raw logits before softmax. The result is always negative — more negative values indicate the model is confident (in-distribution), while values closer to zero indicate uncertainty (out-of-distribution).

**Why energy over softmax confidence?** Softmax normalises logits to a probability distribution, discarding magnitude information. Two inputs can produce very different logit magnitudes but identical softmax distributions. The energy score preserves this magnitude — a model that "fires strongly" on a known class produces a very negative energy (e.g. -8.0), while uncertain/OOD inputs produce less negative energy closer to zero (e.g. -1.5).

### Evaluation

We evaluated using CIFAR-100 as the OOD dataset — same image dimensions (32x32 RGB) but 100 different fine-grained classes, some visually similar to CIFAR-10 (animals, vehicles).

| Metric      | Before regularisation | After regularisation |
|-------------|----------------------|---------------------|
| AUROC       | 0.756                | **0.824**           |
| FPR@95TPR   | 0.641                | **0.544**           |

**AUROC** — probability that a random OOD sample scores higher (more OOD) than a random ID sample. 0.824 means the energy score correctly ranks OOD above ID 82.4% of the time.

**FPR@95TPR** — at a threshold where 95% of known inputs are correctly kept, what fraction of OOD inputs slip through. 0.544 means 54.4% of unknowns are missed at this threshold.

### Score Distributions

```
ID  (CIFAR-10):  mean=-6.14, std=3.62
OOD (CIFAR-100): mean=-3.12, std=1.36
```

The regularised model produces lower-magnitude logits overall (mean -6.1 vs -9.5 before), but the separation between ID and OOD improved because the model is less overconfident on OOD inputs.

### Using the Threshold in Practice

From calibration on the test sets:

| Threshold | ID kept | OOD rejected |
|-----------|---------|--------------|
| 95% TPR   | 95.0%   | ~46%         |
| 90% TPR   | 90.0%   | ~58%         |
| 80% TPR   | 80.0%   | ~70%         |

At inference:
```python
logits = model(image)
energy = -torch.logsumexp(logits, dim=1)
if energy > threshold:
    prediction = "unknown"
else:
    prediction = classes[logits.argmax()]
```

### Other Methods Considered

**Max Softmax Probability (MSP)** — baseline method using the highest softmax probability as a confidence score. Simpler but consistently worse than energy (AUROC 0.799 vs 0.824) because softmax discards logit magnitude.

**Mahalanobis Distance** — measures distance from each input's 512-dim embedding to the nearest class-conditional Gaussian fitted on training data. Performed poorly on this model (AUROC 0.43, worse than random). An AUROC below 0.5 means the detector is inverted — OOD samples were actually *closer* to class centroids than ID samples. This is likely because BatchNorm compresses the feature space, collapsing the distance-based separation that Mahalanobis relies on.

### Future Improvements

- **Outlier exposure** — fine-tune with a small amount of OOD data using a uniform-distribution loss, explicitly teaching low confidence on unknowns
- **Temperature scaling** — post-hoc calibration of logit magnitudes
- **Deeper architecture** (ResNet, etc.) — better feature representations would improve both accuracy and OOD separation
