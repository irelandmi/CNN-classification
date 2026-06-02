"""Quick smoke test: upload a CIFAR-10 batch to lakeFS, read it back, verify contents."""
import gzip
import os
import pickle
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / "ml-infra" / ".env", override=False)

import lakefs
from lakefs.client import Client
from lakefs.exceptions import BadRequestException

LAKEFS_ENDPOINT = os.environ.get("LAKEFS_ENDPOINT", "http://localhost:8000")
LAKEFS_ACCESS_KEY = os.environ["LAKEFS_ACCESS_KEY_ID"]
LAKEFS_SECRET_KEY = os.environ["LAKEFS_SECRET_ACCESS_KEY"]

BRANCH = "main"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "datasets", "cifar-10-batches-py")


def main():
	run_id = uuid.uuid4().hex[:8]
	repo_name = f"test-{run_id}"

	client = Client(
		host=LAKEFS_ENDPOINT,
		username=LAKEFS_ACCESS_KEY,
		password=LAKEFS_SECRET_KEY,
	)

	# 1. Create a fresh repo with unique name
	print("1. Creating lakeFS repository...")
	repo = lakefs.Repository(repo_name, client=client).create(
		storage_namespace=f"s3://lakefs/{repo_name}",
		default_branch=BRANCH,
	)
	print(f"   Created '{repo_name}'")

	try:
		branch = repo.branch(BRANCH)

		# 2. Upload a single batch file
		test_file = "data_batch_1"
		local_path = os.path.join(DATA_DIR, test_file)
		print(f"2. Uploading {test_file}...")
		with open(local_path, "rb") as f:
			local_data = f.read()
		branch.object(test_file).upload(data=local_data, pre_sign=False)
		print(f"   Uploaded {len(local_data)} bytes")

		# 3. Commit
		print("3. Committing...")
		commit = branch.commit(message="test: add data_batch_1")
		print(f"   Commit: {commit.id}")

		# 4. Read it back from lakeFS
		print("4. Reading back from lakeFS...")
		reader = branch.object(test_file).reader(pre_sign=False)
		remote_data = reader.read()
		reader.close()
		print(f"   Read {len(remote_data)} bytes")

		# 5. Decompress if gzipped, then verify
		print("5. Verifying data integrity...")
		if remote_data[:2] == b'\x1f\x8b':
			remote_data = gzip.decompress(remote_data)
		assert local_data == remote_data, "Data mismatch!"
		print("   Bytes match")

		# 6. Verify it's valid CIFAR-10 data
		batch = pickle.loads(remote_data, encoding='bytes')
		num_images = len(batch[b'labels'])
		print(f"   Valid CIFAR-10 batch: {num_images} images, keys={list(batch.keys())}")

		print("\nAll checks passed.")
	finally:
		# Cleanup
		print("Cleaning up...")
		repo.delete()
		print(f"   Deleted '{repo_name}'")


if __name__ == "__main__":
	main()
