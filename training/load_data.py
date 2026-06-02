import argparse
import os
import lakefs
from lakefs.client import Client


LAKEFS_ENDPOINT = os.environ["LAKEFS_ENDPOINT"]
LAKEFS_ACCESS_KEY = os.environ["LAKEFS_ACCESS_KEY_ID"]
LAKEFS_SECRET_KEY = os.environ["LAKEFS_SECRET_ACCESS_KEY"]

BRANCH = "main"
SCRIPT_DIR = os.path.dirname(__file__)

DATASETS = {
	"cifar10": {
		"local_dir": os.path.join(SCRIPT_DIR, "..", "datasets", "cifar-10-batches-py"),
		"remote_prefix": "cifar-10-batches-py",
	},
	"cifar100": {
		"local_dir": os.path.join(SCRIPT_DIR, "..", "datasets", "cifar-100"),
		"remote_prefix": "cifar-100",
	},
}


def ensure_repo(client, repo_name):
	try:
		repo = lakefs.Repository(repo_name, client=client)
		repo.metadata
		print(f"Repository '{repo_name}' already exists")
		return repo
	except Exception:
		repo = lakefs.Repository(repo_name, client=client).create(
			storage_namespace=f"s3://lakefs/{repo_name}",
			default_branch=BRANCH,
		)
		print(f"Created repository '{repo_name}'")
		return repo


def upload_dataset(client, repo_name, local_dir, remote_prefix):
	repo = ensure_repo(client, repo_name)
	branch = repo.branch(BRANCH)

	data_dir = os.path.abspath(local_dir)
	files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]

	print(f"Uploading {len(files)} files from {data_dir}...")
	for filename in sorted(files):
		filepath = os.path.join(data_dir, filename)
		remote_path = f"{remote_prefix}/{filename}"
		with open(filepath, "rb") as f:
			branch.object(remote_path).upload(data=f.read(), pre_sign=False)
		size_mb = os.path.getsize(filepath) / (1024 * 1024)
		print(f"  uploaded {remote_path} ({size_mb:.1f} MB)")

	commit = branch.commit(message=f"Add {repo_name} dataset")
	print(f"Committed: {commit.id}")
	return commit


def main():
	parser = argparse.ArgumentParser(description="Load datasets into lakeFS")
	parser.add_argument("--endpoint", default=LAKEFS_ENDPOINT)
	parser.add_argument("--access-key", default=LAKEFS_ACCESS_KEY)
	parser.add_argument("--secret-key", default=LAKEFS_SECRET_KEY)
	parser.add_argument("--dataset", choices=list(DATASETS.keys()), nargs="+", default=list(DATASETS.keys()))
	args = parser.parse_args()

	client = Client(
		host=args.endpoint,
		username=args.access_key,
		password=args.secret_key,
	)

	for name in args.dataset:
		ds = DATASETS[name]
		print(f"\n=== {name} ===")
		upload_dataset(client, name, ds["local_dir"], ds["remote_prefix"])
		print(f"  lakeFS UI: {args.endpoint}/repositories/{name}")


if __name__ == "__main__":
	main()
