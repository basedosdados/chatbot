#!/usr/bin/env bash
set -e

if [[ -t 1 ]]; then
    BOLD="\e[1m"
    CYAN="\e[36m"
    RESET="\e[0m"
else
    BOLD=""
    CYAN=""
    RESET=""
fi

printf "${BOLD}${CYAN}> Running database migrations...${RESET}\n"
alembic upgrade head

printf "\n${BOLD}${CYAN}> Starting FastAPI server...${RESET}"
exec fastapi run --workers 2 app/main.py
