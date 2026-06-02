import os

REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://mlflow:mlflow@postgres:5432/mlflow")
OOD_ENERGY_THRESHOLD = float(os.environ.get("OOD_ENERGY_THRESHOLD", "-2.36"))
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_S3_ENDPOINT_URL = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
OOD_BUCKET = os.environ.get("OOD_BUCKET", "ood-images")
LAKEFS_ENDPOINT = os.environ.get("LAKEFS_ENDPOINT", "http://lakefs:8000")
LAKEFS_ACCESS_KEY = os.environ.get("LAKEFS_ACCESS_KEY", "AKIAIOSFODNN7EXAMPLE")
LAKEFS_SECRET_KEY = os.environ.get("LAKEFS_SECRET_KEY", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")
RESULT_TTL_SECONDS = int(os.environ.get("RESULT_TTL_SECONDS", "3600"))
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(1024 * 1024)))  # 1MB
CIFAR10_CLASSES = [
	"airplane", "automobile", "bird", "cat", "deer",
	"dog", "frog", "horse", "ship", "truck",
]
