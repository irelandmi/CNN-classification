"""Batch-submit CIFAR-100 test images to the inference API."""
import io
import json
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests
from PIL import Image

API = "http://localhost:8080"


def load_cifar100_test(path):
	with open(path, "rb") as f:
		batch = pickle.load(f, encoding="bytes")
	data = batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
	labels = batch[b"fine_labels"]
	return data, labels


def image_to_png_bytes(arr):
	buf = io.BytesIO()
	Image.fromarray(arr.astype(np.uint8)).save(buf, format="PNG")
	return buf.getvalue()


def submit_image(png_bytes):
	resp = requests.post(
		f"{API}/predict",
		files={"file": ("image.png", png_bytes, "image/png")},
		timeout=10,
	)
	return resp.json()["job_id"]


def main():
	import os
	script_dir = os.path.dirname(__file__)
	cifar100_path = os.path.join(script_dir, "..", "datasets", "cifar-100", "test")

	n = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
	workers = int(sys.argv[2]) if len(sys.argv) > 2 else 16

	print(f"Loading CIFAR-100 test set...")
	data, labels = load_cifar100_test(cifar100_path)
	data = data[:n]
	print(f"Submitting {len(data)} images with {workers} threads...")

	job_ids = []
	submitted = 0
	t0 = time.time()

	with ThreadPoolExecutor(max_workers=workers) as pool:
		futures = {
			pool.submit(submit_image, image_to_png_bytes(data[i])): i
			for i in range(len(data))
		}
		for future in as_completed(futures):
			job_ids.append(future.result())  # TODO: handle exceptions from failed submissions (timeout, connection error)
			submitted += 1
			if submitted % 100 == 0:
				elapsed = time.time() - t0
				rate = submitted / elapsed
				print(f"  Submitted {submitted}/{len(data)} ({rate:.0f} img/s)")

	elapsed = time.time() - t0
	print(f"\nSubmitted {submitted} images in {elapsed:.1f}s ({submitted/elapsed:.0f} img/s)")
	print(f"Queue will drain as worker processes them.")

	# Poll stats until queue is empty
	print("\nWaiting for worker to finish...")
	while True:
		stats = requests.get(f"{API}/stats").json()
		depth = stats["queue_depth"]
		total = stats["total_inferences"]
		ood = stats["ood_count"]
		pending = stats["ood_pending_review"]
		print(f"  Queue: {depth} | Processed: {total} | OOD: {ood} | Pending review: {pending}", end="\r")
		if depth == 0:
			print()
			break
		time.sleep(2)

	print(f"\nDone. {ood} OOD detections from {submitted} CIFAR-100 images.")
	print(f"Review them at: {API}/ood/review")


if __name__ == "__main__":
	main()
