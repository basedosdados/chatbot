.PHONY: up up-dev down

up: # Start all services (with website api network)
	docker compose up --build --watch

up-dev: # Start in dev mode (no website api network)
	docker compose -f compose.yaml up --build --watch

down: # Stop all services
	docker compose down
