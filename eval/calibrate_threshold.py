"""Find the energy threshold for classifying inputs as 'unknown'."""
import os
import pickle
import sys

import numpy as np
import torch

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


@torch.no_grad()
def get_energy_scores(model, data, device, batch_size=256):
	model.eval()
	scores = []
	for i in range(0, len(data), batch_size):
		batch = data[i:i + batch_size].to(device)
		logits = model(batch)
		energy = -torch.logsumexp(logits, dim=1)
		scores.append(energy.cpu())
	return torch.cat(scores).numpy()


def main():
	device = get_default_device()
	model = to_device(Cifar10CnnModel(), device)
	model_path = os.path.join(os.getcwd(), 'cifar10-cnn.pth')
	model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

	id_data = load_cifar10_test()
	ood_data = load_cifar100_test()

	id_scores = get_energy_scores(model, id_data, device)
	ood_scores = get_energy_scores(model, ood_data, device)

	print("Energy score distributions (higher = more likely OOD)")
	print(f"  ID  (CIFAR-10):  min={id_scores.min():.2f}  p5={np.percentile(id_scores, 5):.2f}  median={np.median(id_scores):.2f}  p95={np.percentile(id_scores, 95):.2f}  max={id_scores.max():.2f}")
	print(f"  OOD (CIFAR-100): min={ood_scores.min():.2f}  p5={np.percentile(ood_scores, 5):.2f}  median={np.median(ood_scores):.2f}  p95={np.percentile(ood_scores, 95):.2f}  max={ood_scores.max():.2f}")

	print("\nThreshold analysis (threshold = max energy we consider 'known'):")
	print(f"  {'Threshold':>10}  {'ID kept':>8}  {'OOD rejected':>13}  {'ID miss rate':>13}")
	print("  " + "-" * 50)

	for tpr_target in [0.99, 0.95, 0.90, 0.85, 0.80]:
		threshold = np.percentile(id_scores, tpr_target * 100)
		id_kept = (id_scores <= threshold).mean()
		ood_rejected = (ood_scores > threshold).mean()
		print(f"  {threshold:>10.2f}  {id_kept:>7.1%}  {ood_rejected:>12.1%}  {1 - id_kept:>12.1%}")

	# Recommended threshold at 95% ID retention
	recommended = np.percentile(id_scores, 95)
	ood_caught = (ood_scores > recommended).mean()
	print(f"\nRecommended threshold: {recommended:.2f}")
	print(f"  Keeps {95.0:.0f}% of known (CIFAR-10) inputs")
	print(f"  Rejects {ood_caught:.1%} of unknown (CIFAR-100) inputs")
	print(f"\nUse this in inference: if energy > {recommended:.2f} → 'unknown'")


if __name__ == "__main__":
	main()
