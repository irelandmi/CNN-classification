import psycopg2
from psycopg2.extras import RealDictCursor

from deploy.config import DATABASE_URL

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS ood_detections (
	id SERIAL PRIMARY KEY,
	job_id UUID UNIQUE NOT NULL,
	energy_score FLOAT NOT NULL,
	predicted_class INT,
	predicted_label VARCHAR(20),
	confidence FLOAT,
	true_label VARCHAR(50),
	action VARCHAR(20) DEFAULT 'pending',
	reviewed BOOLEAN DEFAULT FALSE,
	created_at TIMESTAMP DEFAULT NOW(),
	reviewed_at TIMESTAMP
);
"""


def get_connection():
	return psycopg2.connect(DATABASE_URL)


def init_db():
	conn = get_connection()
	try:
		with conn.cursor() as cur:
			cur.execute(CREATE_TABLE)
		conn.commit()
	finally:
		conn.close()


def insert_ood(job_id: str, energy_score: float, predicted_class: int, predicted_label: str, confidence: float):
	conn = get_connection()
	try:
		with conn.cursor() as cur:
			cur.execute(
				"""INSERT INTO ood_detections (job_id, energy_score, predicted_class, predicted_label, confidence)
				VALUES (%s, %s, %s, %s, %s)""",
				(job_id, energy_score, predicted_class, predicted_label, confidence),
			)
		conn.commit()
	finally:
		conn.close()


def get_pending_ood(limit=50, offset=0):
	conn = get_connection()
	try:
		with conn.cursor(cursor_factory=RealDictCursor) as cur:
			cur.execute(
				"SELECT * FROM ood_detections WHERE reviewed = FALSE ORDER BY created_at DESC LIMIT %s OFFSET %s",
				(limit, offset),
			)
			return cur.fetchall()
	finally:
		conn.close()


def get_pending_ood_count():
	conn = get_connection()
	try:
		with conn.cursor(cursor_factory=RealDictCursor) as cur:
			cur.execute("SELECT COUNT(*) as count FROM ood_detections WHERE reviewed = FALSE")
			return cur.fetchone()["count"]
	finally:
		conn.close()


def label_ood(job_id: str, true_label: str, action: str):
	conn = get_connection()
	try:
		with conn.cursor() as cur:
			cur.execute(
				"""UPDATE ood_detections
				SET true_label = %s, action = %s, reviewed = TRUE, reviewed_at = NOW()
				WHERE job_id = %s""",
				(true_label, action, job_id),
			)
		conn.commit()
	finally:
		conn.close()


def get_reviewed_for_export():
	conn = get_connection()
	try:
		with conn.cursor(cursor_factory=RealDictCursor) as cur:
			cur.execute(
				"SELECT * FROM ood_detections WHERE reviewed = TRUE AND action IN ('correct', 'new_class')"
			)
			return cur.fetchall()
	finally:
		conn.close()


def get_all_ood(limit=100, offset=0):
	conn = get_connection()
	try:
		with conn.cursor(cursor_factory=RealDictCursor) as cur:
			cur.execute(
				"SELECT * FROM ood_detections ORDER BY created_at DESC LIMIT %s OFFSET %s",
				(limit, offset),
			)
			return cur.fetchall()
	finally:
		conn.close()


def truncate_ood():
	conn = get_connection()
	try:
		with conn.cursor() as cur:
			cur.execute("TRUNCATE ood_detections RESTART IDENTITY")
		conn.commit()
	finally:
		conn.close()


def get_ood_stats():  # TODO: consolidate into single query using COUNT(*) FILTER (WHERE NOT reviewed)
	conn = get_connection()
	try:
		with conn.cursor(cursor_factory=RealDictCursor) as cur:
			cur.execute("SELECT COUNT(*) as total FROM ood_detections")
			total = cur.fetchone()["total"]
			cur.execute("SELECT COUNT(*) as pending FROM ood_detections WHERE reviewed = FALSE")
			pending = cur.fetchone()["pending"]
			return {"total": total, "pending": pending}
	finally:
		conn.close()
