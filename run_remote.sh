#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ ! -f "$SCRIPT_DIR/.env" ]; then
	echo "Error: .env file not found. Copy .env.example to .env and configure it."
	exit 1
fi

source "$SCRIPT_DIR/.env"

echo "==> Syncing project to DGX..."
rsync -az --delete \
	--exclude .venv \
	--exclude .git \
	--exclude datasets \
	--exclude __pycache__ \
	--exclude '*.pth' \
	. "${DGX_HOST}:~/${REMOTE_DIR}/"

SETUP="export PATH=\$HOME/.local/bin:\$PATH && cd ~/${REMOTE_DIR} && set -a && source .env && set +a && uv sync"

case "${1:-all}" in
	smoke)
		echo "==> Running smoke test..."
		ssh -t "${DGX_HOST}" "${SETUP} && uv run python tests/smoke_test.py"
		;;
	train)
		echo "==> Running training pipeline..."
		ssh -t "${DGX_HOST}" "${SETUP} && uv run python training/fetch_data.py && uv run python training/train.py"
		;;
	eval)
		echo "==> Running evaluation..."
		ssh -t "${DGX_HOST}" "${SETUP} && uv run python eval/eval_ood.py && uv run python eval/calibrate_threshold.py"
		;;
	all)
		echo "==> Running full pipeline..."
		ssh -t "${DGX_HOST}" "${SETUP} && uv run python tests/smoke_test.py && uv run python training/fetch_data.py && uv run python training/train.py && uv run python eval/eval_ood.py && uv run python eval/calibrate_threshold.py"
		;;
	*)
		echo "Usage: $0 {smoke|train|eval|all}"
		exit 1
		;;
esac
