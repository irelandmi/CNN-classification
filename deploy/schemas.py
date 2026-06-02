from pydantic import BaseModel


class PredictResponse(BaseModel):
	job_id: str
	status: str = "queued"


class ResultResponse(BaseModel):
	job_id: str
	status: str
	class_name: str | None = None
	class_id: int | None = None
	confidence: float | None = None
	energy_score: float | None = None
	is_ood: bool | None = None


class LabelRequest(BaseModel):
	true_label: str
	action: str  # "correct", "discard", "new_class"


class ExportResponse(BaseModel):
	branch: str
	items_exported: int


class StatsResponse(BaseModel):
	total_inferences: int
	ood_count: int
	queue_depth: int
	ood_pending_review: int
