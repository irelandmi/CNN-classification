"""OOD detection using Mahalanobis distance in the 512-dim feature space."""
import os
import pickle
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
import mlflow

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
from train import Cifar10CnnModel, get_default_device, to_device

PROJECT_DIR = os.path.join(os.path.dirname(__file__), '..')
CIFAR10_DIR = os.path.join(PROJECT_DIR, 'datasets', 'cifar-10-batches-py')
CIFAR100_DIR = os.path.join(PROJECT_DIR, 'datasets', 'cifar-100')

CIFAR10_CLASSES = [
	'airplane', 'automobile', 'bird', 'cat', 'deer',
	'dog', 'frog', 'horse', 'ship', 'truck',
]


def load_cifar10_train():
	data, labels = [], []
	for i in range(1, 6):
		with open(os.path.join(CIFAR10_DIR, f'data_batch_{i}'), 'rb') as f:
			batch = pickle.load(f, encoding='bytes')
		data.append(batch[b'data'])
		labels.extend(batch[b'labels'])
	data = np.concatenate(data).reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
	return torch.from_numpy(data), np.array(labels)


def load_cifar10_test():
	with open(os.path.join(CIFAR10_DIR, 'test_batch'), 'rb') as f:
		batch = pickle.load(f, encoding='bytes')
	data = batch[b'data'].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
	return torch.from_numpy(data)


def load_cifar100_test():
	with open(os.path.join(CIFAR100_DIR, 'test'), 'rb') as f:
		batch = pickle.load(f, encoding='bytes')
	data = batch[b'data'].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
	return torch.from_numpy(data)


@torch.no_grad()
def extract_features(model, data, device, batch_size=256):
	model.eval()
	feats = []
	for i in range(0, len(data), batch_size):
		batch = data[i:i + batch_size].to(device)
		feats.append(model.features(batch).cpu())
	return torch.cat(feats).numpy()


def fit_class_stats(features, labels, num_classes=10):
	"""Compute per-class mean and shared covariance."""
	means = []
	for c in range(num_classes):
		class_feats = features[labels == c]
		means.append(class_feats.mean(axis=0))
	means = np.stack(means)

	# Shared covariance across all classes
	centered = features - means[labels]
	cov = (centered.T @ centered) / len(features)
	# Regularise for numerical stability
	cov += np.eye(cov.shape[0]) * 1e-6

	return means, cov


def mahalanobis_scores(features, means, cov_inv):
	"""Compute min Mahalanobis distance to any class (lower = more ID)."""
	scores = []
	for feat in features:
		dists = []
		for mean in means:
			diff = feat - mean
			dist = diff @ cov_inv @ diff
			dists.append(dist)
		scores.append(min(dists))
	return np.array(scores)


def compute_metrics(id_scores, ood_scores):
	labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
	scores = np.concatenate([id_scores, ood_scores])
	auroc = roc_auc_score(labels, scores)
	fpr, tpr, _ = roc_curve(labels, scores)
	idx = np.argmin(np.abs(tpr - 0.95))
	fpr_at_95 = fpr[idx]
	return auroc, fpr_at_95


def main():
	device = get_default_device()
	print(f"Using device: {device}")

	model = to_device(Cifar10CnnModel(), device)
	model_path = os.path.join(os.getcwd(), 'cifar10-cnn.pth')
	model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

	# 1. Fit class statistics from training data
	print("Extracting training features...")
	train_data, train_labels = load_cifar10_train()
	train_feats = extract_features(model, train_data, device)
	print(f"  Feature shape: {train_feats.shape}")

	print("Fitting class-conditional Gaussian...")
	means, cov = fit_class_stats(train_feats, train_labels)
	cov_inv = np.linalg.inv(cov)

	# 2. Score ID and OOD
	print("Scoring CIFAR-10 test set (ID)...")
	id_data = load_cifar10_test()
	id_feats = extract_features(model, id_data, device)
	id_scores = mahalanobis_scores(id_feats, means, cov_inv)

	print("Scoring CIFAR-100 test set (OOD)...")
	ood_data = load_cifar100_test()
	ood_feats = extract_features(model, ood_data, device)
	ood_scores = mahalanobis_scores(ood_feats, means, cov_inv)

	# 3. Metrics
	auroc, fpr95 = compute_metrics(id_scores, ood_scores)

	print(f"\nMahalanobis Distance OOD Detection")
	print(f"  AUROC:       {auroc:.4f}")
	print(f"  FPR@95TPR:   {fpr95:.4f}")

	print(f"\nScore distributions (higher = more likely OOD):")
	print(f"  ID:  min={id_scores.min():.1f}  median={np.median(id_scores):.1f}  p95={np.percentile(id_scores, 95):.1f}  max={id_scores.max():.1f}")
	print(f"  OOD: min={ood_scores.min():.1f}  median={np.median(ood_scores):.1f}  p95={np.percentile(ood_scores, 95):.1f}  max={ood_scores.max():.1f}")

	# Threshold analysis
	print(f"\nThreshold analysis:")
	print(f"  {'Threshold':>10}  {'ID kept':>8}  {'OOD rejected':>13}")
	print("  " + "-" * 35)
	for tpr_target in [0.99, 0.95, 0.90, 0.85, 0.80]:
		threshold = np.percentile(id_scores, tpr_target * 100)
		id_kept = (id_scores <= threshold).mean()
		ood_rejected = (ood_scores > threshold).mean()
		print(f"  {threshold:>10.1f}  {id_kept:>7.1%}  {ood_rejected:>12.1%}")

	# 4. Log to MLflow
	tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5050")
	mlflow.set_tracking_uri(tracking_uri)
	mlflow.set_experiment("cifar10-cnn")

	with mlflow.start_run(run_name="ood-mahalanobis"):
		mlflow.log_params({
			"ood_method": "mahalanobis",
			"feature_dim": train_feats.shape[1],
			"ood_dataset": "cifar-100",
			"id_dataset": "cifar-10",
		})
		mlflow.log_metrics({
			"mahalanobis_auroc": auroc,
			"mahalanobis_fpr_at_95tpr": fpr95,
			"id_mahalanobis_median": float(np.median(id_scores)),
			"ood_mahalanobis_median": float(np.median(ood_scores)),
		})
		print(f"\nLogged to MLflow: {tracking_uri}")


if __name__ == "__main__":
	main()
