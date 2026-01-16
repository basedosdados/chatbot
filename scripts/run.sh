#!/usr/bin/env bash
set -e

echo "Running database migrations..."
alembic upgrade head

echo "Starting FastAPI server..."
exec fastapi run --workers 3 app/main.py
