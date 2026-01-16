#!/usr/bin/env bash
set -e

echo "Running database migrations..."
uv run alembic upgrade head

set -a
source .env
set +a

echo "Starting FastAPI server..."
exec uv run fastapi dev --host 0.0.0.0 app/main.py
