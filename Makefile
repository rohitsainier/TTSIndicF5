# IndicF5 TTS Docker Management

.PHONY: build run stop clean logs shell test help version-patch version-minor version-major version-show

# Default target
help:
	@echo "Available commands:"
	@echo "  build         - Build the Docker image"
	@echo "  run           - Run the application with docker-compose"
	@echo "  run-prod      - Run with production profile (nginx)"
	@echo "  stop          - Stop the application"
	@echo "  clean         - Clean up containers and images"
	@echo "  logs          - View application logs"
	@echo "  shell         - Open shell in running container"
	@echo "  test          - Test the application"
	@echo "  version-patch - Increment patch version and push tag"
	@echo "  version-minor - Increment minor version and push tag"
	@echo "  version-major - Increment major version and push tag"
	@echo "  version-show  - Show current version"
	@echo "  help          - Show this help message"

# Build the Docker image
build:
	docker-compose build

# Run the application
run:
	docker-compose up -d
	@echo "Application running at http://localhost:8000"

# Run with production profile
run-prod:
	docker-compose --profile production up -d
	@echo "Application running at http://localhost"

# Stop the application
stop:
	docker-compose down

# Clean up containers and images
clean:
	docker-compose down -v --rmi all
	docker system prune -f

# View logs
logs:
	docker-compose logs -f indicf5-tts

# Open shell in running container
shell:
	docker-compose exec indicf5-tts bash

# Test the application
test:
	@echo "Testing health endpoint..."
	@curl -f http://localhost:8000/health || echo "Application not running or unhealthy"

# Quick development cycle
dev: build run logs

# Version management
version-patch:
	@./version.sh patch

version-minor:
	@./version.sh minor

version-major:
	@./version.sh major

version-show:
	@echo "Current version information:"
	@if [ -f .version ]; then \
		echo "File version: $$(cat .version)"; \
	else \
		echo "No .version file found"; \
	fi
	@echo "Latest git tag: $$(git tag --sort=-version:refname | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$$' | head -n1 || echo 'No version tags found')"
