.PHONY: dev dev-api dev-web test build train export audit clean

# Start both backend and frontend development servers
dev:
	@echo "Starting AeroPulse API (port 8008) and Frontend (port 5173)..."
	@trap 'kill 0' EXIT; \
	PYTHONPATH=. .venv/bin/uvicorn backend.api:app --host 127.0.0.1 --port 8008 & \
	(cd frontend && npm run preview -- --port 5173 --host 127.0.0.1) & \
	wait

# Start backend FastAPI only
dev-api:
	PYTHONPATH=. .venv/bin/uvicorn backend.api:app --host 127.0.0.1 --port 8008 --reload

# Start frontend dev server only
dev-web:
	cd frontend && npm run dev

# Run all tests
test:
	PYTHONPATH=. .venv/bin/python -m pytest -v tests/

# Typecheck and build frontend for production
build:
	cd frontend && npm run typecheck && npm run build

# Train PyTorch LSTM, GRU, and baselines on NASA C-MAPSS data
train:
	PYTHONPATH=. .venv/bin/python -m backend.train --epochs 35 --batch-size 128 --seed 42

# Export precomputed JSON artifacts for client consumption
export:
	PYTHONPATH=. .venv/bin/python -m backend.export

# Run automated browser screenshot visual audit
audit:
	.venv/bin/python scripts/capture_audit.py

# Clean cache directories
clean:
	rm -rf .pytest_cache frontend/dist artifacts/metrics/*.tmp
