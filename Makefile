.PHONY: dev dev-api dev-web test build train refit benchmark figures audit clean

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

# Run all backend unit and integration tests
test:
	PYTHONPATH=. .venv/bin/python -m pytest -v tests/

# Typecheck and build frontend for production
build:
	cd frontend && npm run typecheck && npm run build

# Train PyTorch LSTM, GRU, and baselines with 80/20 engine group validation
train:
	PYTHONPATH=. .venv/bin/python -m backend.train --epochs 35 --batch-size 128 --seed 42

# Run optional final refit on all 100 development engines using validation-selected epochs
refit:
	PYTHONPATH=. .venv/bin/python -m backend.refit --epochs 6 --seed 42

# Run multi-seed reproducibility benchmark (5 seeds)
benchmark:
	PYTHONPATH=. .venv/bin/python backend/experiments.py --epochs 10

# Generate publication diagnostic figures
figures:
	PYTHONPATH=. .venv/bin/python scripts/generate_research_figures.py

# Export precomputed JSON artifacts for client consumption
export:
	PYTHONPATH=. .venv/bin/python -m backend.export

# Run automated browser screenshot visual audit
audit:
	.venv/bin/python scripts/capture_audit.py

# Clean cache directories
clean:
	rm -rf .pytest_cache frontend/dist artifacts/metrics/*.tmp artifacts/experiments/*.tmp
