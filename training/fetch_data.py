"""Fetch datasets from lakeFS to local disk."""
import gzip
import os

import lakefs
from lakefs.client import Client

LAKEFS_ENDPOINT = os.environ["LAKEFS_ENDPOINT"]
LAKEFS_ACCESS_KEY = os.environ["LAKEFS_ACCESS_KEY_ID"]
LAKEFS_SECRET_KEY = os.environ["LAKEFS_SECRET_ACCESS_KEY"]

BRANCH = os.environ.get("LAKEFS_BRANCH", "main")
SCRIPT_DIR = os.path.dirname(__file__)

DATASETS = {
	"cifar10": {
		"prefix": "cifar-10-batches-py",
		"out_dir": os.path.join(SCRIPT_DIR, "..", "datasets", "cifar-10-batches-py"),
	},
	"cifar100": {
		"prefix": "cifar-100",
		"out_dir": os.path.join(SCRIPT_DIR, "..", "datasets", "cifar-100"),
	},
}


def fetch_repo(client, repo_name, prefix, out_dir):
	os.makedirs(out_dir, exist_ok=True)

	repo = lakefs.Repository(repo_name, client=client)
	branch = repo.branch(BRANCH)

	objects = list(branch.objects(prefix=prefix))
	print(f"Fetching {len(objects)} files from lakefs://{repo_name}/{BRANCH}/{prefix}/")

	for obj in objects:
		filename = obj.path.split("/")[-1]
		local_path = os.path.join(out_dir, filename)

		if os.path.exists(local_path):
			print(f"  skipped {filename} (already exists)")
			continue

		reader = branch.object(obj.path).reader(pre_sign=False)
		data = reader.read()
		reader.close()

		if data[:2] == b'\x1f\x8b':
			data = gzip.decompress(data)

		with open(local_path, "wb") as f:
			f.write(data)
		size_mb = len(data) / (1024 * 1024)
		print(f"  fetched {filename} ({size_mb:.1f} MB)")


def main():
	client = Client(
		host=LAKEFS_ENDPOINT,
		username=LAKEFS_ACCESS_KEY,
		password=LAKEFS_SECRET_KEY,
	)

	for repo_name, ds in DATASETS.items():
		print(f"\n=== {repo_name} ===")
		fetch_repo(client, repo_name, ds["prefix"], ds["out_dir"])

	print("\nDone.")


if __name__ == "__main__":
	main()
