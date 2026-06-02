"""Evaluate OOD detection using energy scores on CIFAR-10 (ID) vs CIFAR-100 (OOD)."""
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
import mlflow

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
from train import Cifar10CnnModel, get_default_device, to_device

PROJECT_DIR = os.path.join(os.path.dirname(__file__), '..')
CIFAR10_DIR = os.path.join(PROJECT_DIR, 'datasets', 'cifar-10-batches-py')
CIFAR100_DIR = os.path.join(PROJECT_DIR, 'datasets', 'cifar-100')


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


def energy_score(logits, temperature=1.0):
	"""Compute energy score: -T * log(sum(exp(logits/T)))"""
	return -temperature * torch.logsumexp(logits / temperature, dim=1)


def max_softmax_score(logits):
	"""Compute max softmax probability (baseline)."""
	probs = F.softmax(logits, dim=1)
	return -probs.max(dim=1).values  # negate so higher = more OOD


@torch.no_grad()
def compute_scores(model, data, device, batch_size=256):
	model.eval()
	energy_scores = []
	msp_scores = []

	for i in range(0, len(data), batch_size):
		batch = data[i:i + batch_size].to(device)
		logits = model(batch)
		energy_scores.append(energy_score(logits).cpu())
		msp_scores.append(max_softmax_score(logits).cpu())

	return torch.cat(energy_scores).numpy(), torch.cat(msp_scores).numpy()


def compute_metrics(id_scores, ood_scores):
	"""Compute AUROC and FPR@95TPR."""
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

	# Load model
	model = to_device(Cifar10CnnModel(), device)
	model_path = os.path.join(os.getcwd(), 'cifar10-cnn.pth')
	model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
	print(f"Loaded model from {model_path}")

	# Load datasets
	print("Loading CIFAR-10 test set (in-distribution)...")
	id_data = load_cifar10_test()
	print(f"  {len(id_data)} images")

	print("Loading CIFAR-100 test set (out-of-distribution)...")
	ood_data = load_cifar100_test()
	print(f"  {len(ood_data)} images")

	# Compute scores
	print("\nComputing scores...")
	id_energy, id_msp = compute_scores(model, id_data, device)
	ood_energy, ood_msp = compute_scores(model, ood_data, device)

	# Metrics
	energy_auroc, energy_fpr95 = compute_metrics(id_energy, ood_energy)
	msp_auroc, msp_fpr95 = compute_metrics(id_msp, ood_msp)

	print(f"\n{'Method':<25} {'AUROC':>8} {'FPR@95TPR':>10}")
	print("-" * 45)
	print(f"{'Energy Score':<25} {energy_auroc:>8.4f} {energy_fpr95:>10.4f}")
	print(f"{'Max Softmax Prob':<25} {msp_auroc:>8.4f} {msp_fpr95:>10.4f}")

	# Score distributions
	print(f"\nEnergy scores — ID: mean={id_energy.mean():.2f} std={id_energy.std():.2f}")
	print(f"Energy scores — OOD: mean={ood_energy.mean():.2f} std={ood_energy.std():.2f}")

	# Log to MLflow
	tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5050")
	mlflow.set_tracking_uri(tracking_uri)
	mlflow.set_experiment("cifar10-cnn")

	with mlflow.start_run(run_name="ood-evaluation"):
		mlflow.log_params({
			"ood_dataset": "cifar-100",
			"id_dataset": "cifar-10",
			"id_samples": len(id_data),
			"ood_samples": len(ood_data),
		})
		mlflow.log_metrics({
			"energy_auroc": energy_auroc,
			"energy_fpr_at_95tpr": energy_fpr95,
			"msp_auroc": msp_auroc,
			"msp_fpr_at_95tpr": msp_fpr95,
			"id_energy_mean": float(id_energy.mean()),
			"id_energy_std": float(id_energy.std()),
			"ood_energy_mean": float(ood_energy.mean()),
			"ood_energy_std": float(ood_energy.std()),
		})
		print(f"\nLogged to MLflow: {tracking_uri}")


if __name__ == "__main__":
	main()
