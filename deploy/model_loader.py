import io
import os
import tempfile

import boto3
import mlflow
import numpy as np
import torch
from PIL import Image

from deploy.config import (
	MINIO_ACCESS_KEY,
	MINIO_ENDPOINT,
	MINIO_SECRET_KEY,
	MLFLOW_S3_ENDPOINT_URL,
	MLFLOW_TRACKING_URI,
)

# Import model class from training code
# TODO: remove sys.path hack — use proper package import (PYTHONPATH=/app is already set in Dockerfile)
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
from train import Cifar10CnnModel


def get_minio_client():
	return boto3.client(
		"s3",
		endpoint_url=f"http://{MINIO_ENDPOINT}",
		aws_access_key_id=MINIO_ACCESS_KEY,
		aws_secret_access_key=MINIO_SECRET_KEY,
	)


MODEL_CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "/app/model-cache")
CACHED_MODEL_PATH = os.path.join(MODEL_CACHE_DIR, "cifar10-cnn.pth")


def load_model() -> Cifar10CnnModel:
	model = Cifar10CnnModel()

	if os.path.exists(CACHED_MODEL_PATH):
		print(f"Loading cached model from {CACHED_MODEL_PATH}")
		model.load_state_dict(torch.load(CACHED_MODEL_PATH, map_location="cpu", weights_only=True))
		model.eval()
		print("Model loaded from cache")
		return model

	os.environ["MLFLOW_S3_ENDPOINT_URL"] = MLFLOW_S3_ENDPOINT_URL
	os.environ["AWS_ACCESS_KEY_ID"] = MINIO_ACCESS_KEY
	os.environ["AWS_SECRET_ACCESS_KEY"] = MINIO_SECRET_KEY

	mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
	experiment = mlflow.get_experiment_by_name("cifar10-cnn")
	if experiment is None:
		raise RuntimeError("No 'cifar10-cnn' experiment found in MLflow")

	runs = mlflow.search_runs(
		experiment_ids=[experiment.experiment_id],
		order_by=["start_time DESC"],
		max_results=10,
	)
	if runs.empty:
		raise RuntimeError("No runs found in 'cifar10-cnn' experiment")

	for _, run in runs.iterrows():
		run_id = run.run_id
		print(f"Trying MLflow run: {run_id}")
		try:
			with tempfile.TemporaryDirectory() as tmpdir:
				artifact_path = mlflow.artifacts.download_artifacts(
					run_id=run_id,
					artifact_path="cifar10-cnn.pth",
					dst_path=tmpdir,
				)
				model.load_state_dict(torch.load(artifact_path, map_location="cpu", weights_only=True))
			print(f"Loaded model from MLflow run: {run_id}")
			model.eval()
			os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
			torch.save(model.state_dict(), CACHED_MODEL_PATH)
			print(f"Cached model to {CACHED_MODEL_PATH}")
			break
		except Exception as e:
			print(f"  Skipping run {run_id}: {e}")
	else:
		raise RuntimeError("No run with cifar10-cnn.pth artifact found")

	model.eval()
	print("Model loaded successfully")
	return model


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
	img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
	img = img.resize((32, 32))
	arr = np.array(img, dtype=np.float32) / 255.0
	# HWC -> CHW
	tensor = torch.from_numpy(arr.transpose(2, 0, 1))
	return tensor.unsqueeze(0)
