import json
import traceback

import redis
import torch

from deploy.config import (
	CIFAR10_CLASSES,
	MINIO_ACCESS_KEY,
	MINIO_ENDPOINT,
	MINIO_SECRET_KEY,
	OOD_BUCKET,
	OOD_ENERGY_THRESHOLD,
	REDIS_URL,
	RESULT_TTL_SECONDS,
)
from deploy.db import init_db, insert_ood
from deploy.model_loader import get_minio_client, load_model, preprocess_image


def ensure_ood_bucket(s3):
	try:
		s3.head_bucket(Bucket=OOD_BUCKET)
	except Exception:
		s3.create_bucket(Bucket=OOD_BUCKET)


def main():
	print("Initializing worker...")
	init_db()

	r = redis.from_url(REDIS_URL, socket_timeout=None)
	s3 = get_minio_client()
	ensure_ood_bucket(s3)
	model = load_model()

	print(f"OOD energy threshold: {OOD_ENERGY_THRESHOLD}")
	print("Worker ready, waiting for jobs...")

	while True:
		_, raw = r.blpop("queue:inference")
		job = json.loads(raw)
		job_id = job["job_id"]

		try:
			image_bytes = bytes.fromhex(job["image_hex"])
			tensor = preprocess_image(image_bytes)

			with torch.no_grad():
				logits = model(tensor)
				energy = -torch.logsumexp(logits, dim=1).item()
				probs = torch.softmax(logits, dim=1)
				confidence, predicted = probs.max(dim=1)
				confidence = confidence.item()
				predicted = predicted.item()

			class_name = CIFAR10_CLASSES[predicted]
			is_ood = energy > OOD_ENERGY_THRESHOLD

			result = {
				"status": "complete",
				"class_name": class_name,
				"class_id": predicted,
				"confidence": round(confidence, 4),
				"energy_score": round(energy, 4),
				"is_ood": is_ood,
			}

			if is_ood:
				s3.put_object(
					Bucket=OOD_BUCKET,
					Key=f"{job_id}.png",
					Body=image_bytes,
					ContentType="image/png",
				)
				insert_ood(job_id, energy, predicted, class_name, confidence)
				r.rpush("queue:ood_review", job_id)

			r.set(f"result:{job_id}", json.dumps(result), ex=RESULT_TTL_SECONDS)

			count = r.incr("stats:total_inferences")
			ood_label = " [OOD]" if is_ood else ""
			print(f"[{count}] {job_id}: {class_name} ({confidence:.2%}) energy={energy:.2f}{ood_label}")

		except Exception:
			traceback.print_exc()
			error_result = {"status": "error", "error": "Inference failed"}
			r.set(f"result:{job_id}", json.dumps(error_result), ex=RESULT_TTL_SECONDS)


if __name__ == "__main__":
	main()
