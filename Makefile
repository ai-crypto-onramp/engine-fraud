.PHONY: build test run lint typecheck migrate up down docker-build docker-run train clean

build:
	pip install -r requirements-dev.txt
	pip install -e .

test:
	pytest -q --cov=fraud_engine --cov-report=xml:coverage.xml

lint:
	ruff check src tests feature_repo

typecheck:
	mypy src/fraud_engine

run:
	uvicorn fraud_engine.app:app --host 0.0.0.0 --port 8080

migrate:
	@echo "Apply migrations/001_init.sql against DB_URL (idempotent, tracked via schema_migrations)"
	python -c "from fraud_engine.db import PostgresStore; from fraud_engine.config import get_settings; s=PostgresStore(get_settings().db_url); s.apply_migrations(); print('migrated')"

train:
	python -m fraud_engine.training.train_chargeback --dataset data/chargeback_dataset.json
	python -m fraud_engine.training.train_velocity --dataset data/velocity_dataset.json

up:
	docker compose up -d

down:
	docker compose down

docker-build:
	docker build -t ai-crypto-onramp/fraud-engine .

docker-run:
	docker run --rm -p 8080:8080 ai-crypto-onramp/fraud-engine

clean:
	rm -rf dist build *.egg-info .pytest_cache coverage.xml .coverage