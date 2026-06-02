"""Smoke test: verify GPU, MLflow, and lakeFS connectivity."""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / "ml-infra" / ".env", override=False)

import torch
import lakefs
import mlflow
from lakefs.client import Client

def check_gpu():
	if torch.cuda.is_available():
		device = torch.device("cuda")
		name = torch.cuda.get_device_name(0)
		# Quick matmul to confirm it actually works
		a = torch.randn(256, 256, device=device)
		b = torch.randn(256, 256, device=device)
		_ = a @ b
		torch.cuda.synchronize()
		print(f"  GPU: {name}")
		return True
	elif torch.backends.mps.is_available():
		print("  GPU: MPS (Apple Silicon)")
		return True
	else:
		print("  GPU: not available (CPU only)")
		return False


def check_mlflow():
	uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5050")
	mlflow.set_tracking_uri(uri)
	experiments = mlflow.search_experiments()
	print(f"  MLflow: {uri} ({len(experiments)} experiments)")
	return True


def check_lakefs():
	endpoint = os.environ.get("LAKEFS_ENDPOINT", "http://localhost:8000")
	client = Client(
		host=endpoint,
		username=os.environ.get("LAKEFS_ACCESS_KEY_ID", ""),
		password=os.environ.get("LAKEFS_SECRET_ACCESS_KEY", ""),
	)
	repos = client.sdk_client.repositories_api.list_repositories()
	names = [r.id for r in repos.results]
	print(f"  lakeFS: {endpoint} (repos: {', '.join(names) or 'none'})")
	return True


def main():
	print("Running smoke tests...\n")
	checks = {
		"GPU": check_gpu,
		"MLflow": check_mlflow,
		"lakeFS": check_lakefs,
	}

	failed = []
	for name, fn in checks.items():
		try:
			fn()
		except Exception as e:
			print(f"  {name}: FAILED — {e}")
			failed.append(name)

	print()
	if failed:
		print(f"FAILED: {', '.join(failed)}")
		sys.exit(1)
	else:
		print("All checks passed.")


if __name__ == "__main__":
	main()
