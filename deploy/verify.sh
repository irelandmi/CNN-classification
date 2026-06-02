#!/usr/bin/env bash
# Verification script for the inference + OOD review deployment pipeline.
# Run from the ml-infra/ directory after `podman compose up -d`.
#
# Usage: bash ../jobs/CNN-classification/deploy/verify.sh
set -euo pipefail

API="http://localhost:8080"
PASS=0
FAIL=0
TOTAL=0

green() { printf "\033[32m%s\033[0m\n" "$1"; }
red()   { printf "\033[31m%s\033[0m\n" "$1"; }
bold()  { printf "\033[1m%s\033[0m\n" "$1"; }

check() {
	TOTAL=$((TOTAL + 1))
	local desc="$1" ok="$2"
	if [ "$ok" = "true" ]; then
		green "  PASS: $desc"
		PASS=$((PASS + 1))
	else
		red "  FAIL: $desc"
		FAIL=$((FAIL + 1))
	fi
}

wait_for_result() {
	local job_id="$1" max_wait="${2:-180}" elapsed=0
	while [ $elapsed -lt $max_wait ]; do
		local status
		status=$(curl -sf "$API/result/$job_id" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])" 2>/dev/null || echo "error")
		if [ "$status" = "complete" ] || [ "$status" = "error" ]; then
			echo "$status"
			return
		fi
		sleep 2
		elapsed=$((elapsed + 2))
	done
	echo "timeout"
}

# ── Generate test images ─────────────────────────────────────────────
bold "Generating test images..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_DIR"
uv run python3 -c "
import pickle, numpy as np
from PIL import Image

# CIFAR-10 test image (in-distribution)
with open('datasets/cifar-10-batches-py/test_batch', 'rb') as f:
    batch = pickle.load(f, encoding='bytes')
img = batch[b'data'][0].reshape(3, 32, 32).transpose(1, 2, 0)
Image.fromarray(img.astype(np.uint8)).save('/tmp/verify_cifar10.png')

# Random noise (out-of-distribution)
rng = np.random.RandomState(42)
noise = rng.randint(0, 256, (32, 32, 3), dtype=np.uint8)
Image.fromarray(noise).save('/tmp/verify_noise.png')

print('Test images generated')
" 2>/dev/null

# ── 0. Wait for worker to load model ─────────────────────────────────
bold "\n0. Waiting for worker readiness..."
echo "  Submitting warmup job and waiting up to 300s for worker to process it..."
WARMUP=$(curl -sf -F 'file=@/tmp/verify_cifar10.png' "$API/predict" 2>/dev/null || echo '{}')
WARMUP_ID=$(echo "$WARMUP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('job_id',''))" 2>/dev/null)
if [ -n "$WARMUP_ID" ]; then
	WARMUP_STATUS=$(wait_for_result "$WARMUP_ID" 300)
	if [ "$WARMUP_STATUS" = "complete" ]; then
		echo "  Worker is ready."
	else
		red "  Worker did not become ready within 300s (status: $WARMUP_STATUS)"
		red "  Check: podman logs ml-infra-inference-worker-1"
		exit 1
	fi
fi

# ── 1. Health check ──────────────────────────────────────────────────
bold "\n1. Health check"
HEALTH=$(curl -sf "$API/health" 2>/dev/null || echo '{}')
check "/health returns ok" "$(echo "$HEALTH" | python3 -c "import sys,json; print('true' if json.load(sys.stdin).get('status')=='ok' else 'false')" 2>/dev/null || echo false)"

# ── 2. Submit CIFAR-10 image (in-distribution) ──────────────────────
bold "\n2. CIFAR-10 inference (in-distribution)"
PRED=$(curl -sf -F 'file=@/tmp/verify_cifar10.png' "$API/predict" 2>/dev/null || echo '{}')
JOB_ID=$(echo "$PRED" | python3 -c "import sys,json; print(json.load(sys.stdin).get('job_id',''))" 2>/dev/null)
check "POST /predict returns job_id" "$([ -n "$JOB_ID" ] && echo true || echo false)"

echo "  Waiting for result (job: ${JOB_ID:0:12}...)..."
STATUS=$(wait_for_result "$JOB_ID")
check "Result completes" "$([ "$STATUS" = "complete" ] && echo true || echo false)"

if [ "$STATUS" = "complete" ]; then
	RESULT=$(curl -sf "$API/result/$JOB_ID")
	CLASS=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['class_name'])" 2>/dev/null)
	IS_OOD=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['is_ood'])" 2>/dev/null)
	ENERGY=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['energy_score'])" 2>/dev/null)
	echo "  Class: $CLASS, Energy: $ENERGY, OOD: $IS_OOD"
	check "Returns a CIFAR-10 class" "$(python3 -c "print('true' if '$CLASS' in ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'] else 'false')")"
	check "Has energy score" "$([ -n "$ENERGY" ] && [ "$ENERGY" != "None" ] && echo true || echo false)"
fi

# ── 3. Submit noise image (likely OOD) ──────────────────────────────
bold "\n3. Noise image inference"
PRED2=$(curl -sf -F 'file=@/tmp/verify_noise.png' "$API/predict" 2>/dev/null || echo '{}')
JOB_ID2=$(echo "$PRED2" | python3 -c "import sys,json; print(json.load(sys.stdin).get('job_id',''))" 2>/dev/null)
check "POST /predict returns job_id" "$([ -n "$JOB_ID2" ] && echo true || echo false)"

echo "  Waiting for result (job: ${JOB_ID2:0:12}...)..."
STATUS2=$(wait_for_result "$JOB_ID2")
check "Result completes" "$([ "$STATUS2" = "complete" ] && echo true || echo false)"

if [ "$STATUS2" = "complete" ]; then
	RESULT2=$(curl -sf "$API/result/$JOB_ID2")
	ENERGY2=$(echo "$RESULT2" | python3 -c "import sys,json; print(json.load(sys.stdin)['energy_score'])" 2>/dev/null)
	IS_OOD2=$(echo "$RESULT2" | python3 -c "import sys,json; print(json.load(sys.stdin)['is_ood'])" 2>/dev/null)
	echo "  Energy: $ENERGY2, OOD: $IS_OOD2"
	check "Has energy score" "$([ -n "$ENERGY2" ] && [ "$ENERGY2" != "None" ] && echo true || echo false)"
	# Note: whether noise is OOD depends on the threshold. We just verify the field exists.
	check "is_ood field is boolean" "$(python3 -c "print('true' if '$IS_OOD2' in ['True','False'] else 'false')")"
fi

# ── 4. Stats endpoint ───────────────────────────────────────────────
bold "\n4. Stats endpoint"
STATS=$(curl -sf "$API/stats" 2>/dev/null || echo '{}')
TOTAL_INF=$(echo "$STATS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('total_inferences', -1))" 2>/dev/null)
check "GET /stats returns total_inferences >= 2" "$(python3 -c "print('true' if int('$TOTAL_INF') >= 2 else 'false')")"
echo "  Stats: $STATS"

# ── 5. OOD pending endpoint ─────────────────────────────────────────
bold "\n5. OOD pending endpoint"
PENDING=$(curl -sf "$API/ood/pending" 2>/dev/null || echo '[]')
PENDING_COUNT=$(echo "$PENDING" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null)
check "GET /ood/pending returns valid JSON array" "$([ "$PENDING_COUNT" != "" ] && echo true || echo false)"
echo "  Pending OOD items: $PENDING_COUNT"

# ── 6. OOD review page ──────────────────────────────────────────────
bold "\n6. OOD review page"
REVIEW_STATUS=$(curl -sf -o /dev/null -w '%{http_code}' "$API/ood/review" 2>/dev/null)
check "GET /ood/review returns 200" "$([ "$REVIEW_STATUS" = "200" ] && echo true || echo false)"
REVIEW_BODY=$(curl -sf "$API/ood/review" 2>/dev/null)
check "Review page contains HTML" "$(echo "$REVIEW_BODY" | python3 -c "import sys; print('true' if 'OOD Review' in sys.stdin.read() else 'false')")"

# ── 7. OOD labeling (if items exist) ────────────────────────────────
bold "\n7. OOD labeling"
if [ "$PENDING_COUNT" -gt "0" ]; then
	FIRST_OOD_ID=$(echo "$PENDING" | python3 -c "import sys,json; print(json.load(sys.stdin)[0]['job_id'])" 2>/dev/null)
	LABEL_RESP=$(curl -sf -X POST "$API/ood/$FIRST_OOD_ID/label" \
		-H 'Content-Type: application/json' \
		-d '{"true_label": "unknown", "action": "correct"}' 2>/dev/null || echo '{}')
	LABEL_STATUS=$(echo "$LABEL_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null)
	check "POST /ood/{id}/label returns labeled" "$([ "$LABEL_STATUS" = "labeled" ] && echo true || echo false)"
else
	echo "  Skipped (no OOD items to label — threshold may not flag test images)"
	check "OOD labeling (skipped, no items)" "true"
fi

# ── 8. OOD image endpoint ───────────────────────────────────────────
bold "\n8. OOD image endpoint"
if [ "$PENDING_COUNT" -gt "0" ]; then
	IMG_ID=$(echo "$PENDING" | python3 -c "import sys,json; print(json.load(sys.stdin)[0]['job_id'])" 2>/dev/null)
	IMG_STATUS=$(curl -sf -o /dev/null -w '%{http_code}' "$API/ood/$IMG_ID/image" 2>/dev/null)
	check "GET /ood/{id}/image returns 200" "$([ "$IMG_STATUS" = "200" ] && echo true || echo false)"
else
	echo "  Skipped (no OOD images stored)"
	check "OOD image endpoint (skipped, no items)" "true"
fi

# ── 9. Base64 JSON prediction ───────────────────────────────────────
bold "\n9. Base64 JSON prediction"
B64=$(python3 -c "import base64; print(base64.b64encode(open('/tmp/verify_cifar10.png','rb').read()).decode())")
PRED3=$(curl -sf -X POST "$API/predict" \
	-H 'Content-Type: application/json' \
	-d "{\"image_base64\": \"$B64\"}" 2>/dev/null || echo '{}')
JOB_ID3=$(echo "$PRED3" | python3 -c "import sys,json; print(json.load(sys.stdin).get('job_id',''))" 2>/dev/null)
check "POST /predict with base64 JSON returns job_id" "$([ -n "$JOB_ID3" ] && echo true || echo false)"

if [ -n "$JOB_ID3" ]; then
	echo "  Waiting for result..."
	STATUS3=$(wait_for_result "$JOB_ID3")
	check "Base64 result completes" "$([ "$STATUS3" = "complete" ] && echo true || echo false)"
fi

# ── 10. Container health ────────────────────────────────────────────
bold "\n10. Container health"
API_UP=$(podman ps --format '{{.Names}} {{.Status}}' 2>/dev/null | grep inference-api | grep -c "Up" || echo 0)
WORKER_UP=$(podman ps --format '{{.Names}} {{.Status}}' 2>/dev/null | grep inference-worker | grep -c "Up" || echo 0)
REDIS_UP=$(podman ps --format '{{.Names}} {{.Status}}' 2>/dev/null | grep redis | grep -c "Up" || echo 0)
check "inference-api container is running" "$([ "$API_UP" -ge 1 ] && echo true || echo false)"
check "inference-worker container is running" "$([ "$WORKER_UP" -ge 1 ] && echo true || echo false)"
check "redis container is running" "$([ "$REDIS_UP" -ge 1 ] && echo true || echo false)"

# ── Summary ──────────────────────────────────────────────────────────
bold "\n════════════════════════════════"
bold "Results: $PASS/$TOTAL passed, $FAIL failed"
bold "════════════════════════════════"

if [ "$FAIL" -gt 0 ]; then
	exit 1
fi
