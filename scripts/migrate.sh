#!/usr/bin/env bash
set -euo pipefail

printf "Running Alembic migrations...\n"
alembic upgrade head

printf "Setting up LangGraph checkpointer...\n"
python -m scripts.setup_checkpointer

printf "Migration completed successfully.\n"
