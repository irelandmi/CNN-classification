import base64
import json
import uuid
from contextlib import asynccontextmanager

import redis
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates

from deploy.config import (
	MINIO_ENDPOINT,
	MINIO_ACCESS_KEY,
	MINIO_SECRET_KEY,
	OOD_BUCKET,
	REDIS_URL,
	LAKEFS_ENDPOINT,
	LAKEFS_ACCESS_KEY,
	LAKEFS_SECRET_KEY,
	MAX_UPLOAD_BYTES,
)
from deploy.db import get_all_ood, get_pending_ood, get_pending_ood_count, get_reviewed_for_export, get_ood_stats, init_db, label_ood, truncate_ood
from deploy.model_loader import get_minio_client
from deploy.schemas import (
	ExportResponse,
	LabelRequest,
	PredictResponse,
	ResultResponse,
	StatsResponse,
)

import os
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

redis_pool = redis.ConnectionPool.from_url(REDIS_URL)


@asynccontextmanager
async def lifespan(app: FastAPI):
	init_db()
	yield
	redis_pool.disconnect()


app = FastAPI(title="CIFAR-10 Inference API", lifespan=lifespan)


def get_redis():
	return redis.Redis(connection_pool=redis_pool)


@app.get("/health")
def health():
	return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse, response_model_exclude_none=True)
async def predict(request: Request, file: UploadFile | None = File(None)):
	if file and file.size:
		if file.size > MAX_UPLOAD_BYTES:
			raise HTTPException(413, f"File too large (max {MAX_UPLOAD_BYTES} bytes)")
		image_bytes = await file.read()
	else:
		body = await request.json()
		if "image_base64" not in body:
			raise HTTPException(400, "Provide file upload or image_base64 JSON field")
		image_bytes = base64.b64decode(body["image_base64"])
		if len(image_bytes) > MAX_UPLOAD_BYTES:
			raise HTTPException(413, f"Image too large (max {MAX_UPLOAD_BYTES} bytes)")

	job_id = str(uuid.uuid4())
	r = get_redis()
	job = {"job_id": job_id, "image_hex": image_bytes.hex()}
	r.rpush("queue:inference", json.dumps(job))

	return PredictResponse(job_id=job_id)


@app.get("/result/{job_id}", response_model=ResultResponse)
def get_result(job_id: str):
	r = get_redis()
	raw = r.get(f"result:{job_id}")
	if raw is None:
		return ResultResponse(job_id=job_id, status="pending")
	result = json.loads(raw)
	return ResultResponse(job_id=job_id, **result)


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
	r = get_redis()
	total = int(r.get("stats:total_inferences") or 0)
	queue_depth = r.llen("queue:inference")
	ood_stats = get_ood_stats()
	return templates.TemplateResponse("dashboard.html", {
		"request": request,
		"total_inferences": total,
		"queue_depth": queue_depth,
		"ood_count": ood_stats["total"],
		"ood_pending": ood_stats["pending"],
		"ood_reviewed": ood_stats["total"] - ood_stats["pending"],
	})


@app.get("/ood/review", response_class=HTMLResponse)
def ood_review(request: Request, limit: int = 50, offset: int = 0):  # TODO: validate limit/offset (reject negative values, cap max limit)
	items = get_pending_ood(limit=limit, offset=offset)
	total = get_pending_ood_count()
	for item in items:
		item["image_url"] = f"/ood/{item['job_id']}/image"
	return templates.TemplateResponse("review.html", {
		"request": request,
		"items": items,
		"total": total,
		"limit": limit,
		"offset": offset,
	})


@app.get("/ood/records", response_class=HTMLResponse)
def ood_records(request: Request, limit: int = 100, offset: int = 0):  # TODO: validate limit/offset (reject negative values, cap max limit)
	items = get_all_ood(limit=limit, offset=offset)
	ood_stats = get_ood_stats()
	for item in items:
		item["image_url"] = f"/ood/{item['job_id']}/image"
	return templates.TemplateResponse("records.html", {
		"request": request,
		"items": items,
		"total": ood_stats["total"],
		"limit": limit,
		"offset": offset,
	})


# TODO: paginate this endpoint — currently returns all pending items unbounded
@app.get("/ood/pending")
def ood_pending():
	return get_pending_ood()


@app.get("/ood/{job_id}/image")
def ood_image(job_id: str):
	s3 = get_minio_client()
	try:
		obj = s3.get_object(Bucket=OOD_BUCKET, Key=f"{job_id}.png")
		return Response(content=obj["Body"].read(), media_type="image/png")
	except Exception:
		raise HTTPException(404, "Image not found")


@app.post("/ood/{job_id}/label")
def ood_label(job_id: str, req: LabelRequest):
	if req.action not in ("correct", "discard", "new_class"):
		raise HTTPException(400, "action must be correct, discard, or new_class")
	label_ood(job_id, req.true_label, req.action)
	return {"status": "labeled", "job_id": job_id}


@app.post("/ood/export", response_model=ExportResponse)
def ood_export():
	items = get_reviewed_for_export()
	if not items:
		raise HTTPException(404, "No reviewed items to export")

	import datetime
	import lakefs
	from lakefs.client import Client

	client = Client(
		host=LAKEFS_ENDPOINT,
		username=LAKEFS_ACCESS_KEY,
		password=LAKEFS_SECRET_KEY,
	)

	branch_name = f"ood-feedback-{datetime.date.today().isoformat()}"

	repo = lakefs.Repository("cifar10", client=client)
	try:
		branch = repo.branch(branch_name).create(source_reference="main")
	except Exception:
		branch = repo.branch(branch_name)

	s3 = get_minio_client()
	exported = 0
	for item in items:
		job_id = str(item["job_id"])
		label = item["true_label"]
		try:
			obj = s3.get_object(Bucket=OOD_BUCKET, Key=f"{job_id}.png")
			image_data = obj["Body"].read()
			path = f"ood-feedback/{label}/{job_id}.png"
			branch.object(path).upload(data=image_data, pre_sign=False)
			exported += 1
		except Exception as e:
			print(f"Failed to export {job_id}: {e}")

	return ExportResponse(branch=branch_name, items_exported=exported)


@app.get("/stats", response_model=StatsResponse)
def stats():
	r = get_redis()
	total = int(r.get("stats:total_inferences") or 0)
	queue_depth = r.llen("queue:inference")
	ood_stats = get_ood_stats()
	return StatsResponse(
		total_inferences=total,
		ood_count=ood_stats["total"],
		queue_depth=queue_depth,
		ood_pending_review=ood_stats["pending"],
	)


@app.post("/reset")
def reset():
	r = get_redis()
	r.delete("queue:inference", "queue:ood_review", "stats:total_inferences")
	# Clear result keys
	for key in r.scan_iter("result:*"):
		r.delete(key)
	truncate_ood()
	# Clear OOD images bucket
	s3 = get_minio_client()
	try:
		objects = s3.list_objects_v2(Bucket=OOD_BUCKET).get("Contents", [])
		if objects:
			s3.delete_objects(
				Bucket=OOD_BUCKET,
				Delete={"Objects": [{"Key": o["Key"]} for o in objects]},
			)
	except Exception:
		pass
	return {"status": "reset"}
