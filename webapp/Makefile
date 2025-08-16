# TAROT CKD Risk Prediction - Docker Deployment Makefile

.PHONY: help build up down logs health test clean deploy stop restart

# Default target
help:
	@echo "🩺 TAROT CKD Risk Prediction - Docker Deployment"
	@echo "================================================"
	@echo ""
	@echo "Available commands:"
	@echo "  make deploy     - Build and deploy all services (frontend on :3000, backend on :8000)"
	@echo "  make up         - Start services (without rebuild)"
	@echo "  make down       - Stop and remove containers"
	@echo "  make restart    - Restart all services"
	@echo "  make logs       - Show logs from all services"
	@echo "  make health     - Check service health"
	@echo "  make test       - Run deployment tests"
	@echo "  make clean      - Clean up Docker resources"
	@echo "  make stop       - Stop services without removing containers"
	@echo ""
	@echo "Quick start: make deploy && make health"

# Build and deploy all services
deploy:
	@echo "🚀 Building and deploying TAROT CKD Risk Prediction..."
	docker-compose up --build -d
	@echo "✅ Deployment complete!"
	@echo "   Frontend: http://localhost:3000"
	@echo "   Backend:  http://localhost:8000"
	@echo "   API Docs: http://localhost:8000/docs"

# Start services without rebuild
up:
	@echo "⬆️  Starting services..."
	docker-compose up -d

# Stop and remove containers
down:
	@echo "⬇️  Stopping and removing containers..."
	docker-compose down

# Stop services without removing containers
stop:
	@echo "⏹️  Stopping services..."
	docker-compose stop

# Restart all services
restart:
	@echo "🔄 Restarting services..."
	docker-compose restart

# Show logs from all services
logs:
	@echo "📋 Service logs:"
	docker-compose logs -f --tail=50

# Check service health
health:
	@echo "🏥 Checking service health..."
	@echo "Frontend health:"
	@curl -s http://localhost:3000/health || echo "❌ Frontend not responding"
	@echo ""
	@echo "Backend health:"
	@curl -s http://localhost:8000/health | python -m json.tool || echo "❌ Backend not responding"
	@echo ""
	@echo "Service status:"
	docker-compose ps

# Run deployment readiness tests
test:
	@echo "🧪 Running deployment tests..."
	python ../test_docker_deployment.py 2>/dev/null || python /tmp/test_docker_deployment.py

# Clean up Docker resources
clean:
	@echo "🧹 Cleaning up Docker resources..."
	docker-compose down -v
	docker system prune -f
	@echo "✅ Cleanup complete"

# Build only (no deploy)
build:
	@echo "🔨 Building Docker images..."
	docker-compose build --no-cache

# Production deployment with nginx
production:
	@echo "🏭 Starting production deployment with nginx..."
	docker-compose --profile production up --build -d
	@echo "✅ Production deployment complete!"
	@echo "   Access via: http://localhost/"

# Development mode (with file watching)
dev:
	@echo "👨‍💻 Starting development mode..."
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up --build

# Show container resource usage
stats:
	@echo "📊 Container resource usage:"
	docker stats --no-stream

# Backup models directory
backup-models:
	@echo "💾 Creating models backup..."
	tar -czf models-backup-$(shell date +%Y%m%d-%H%M%S).tar.gz ../foundation_models/
	@echo "✅ Backup created"

# Quick smoke test
smoke-test:
	@echo "🚨 Running smoke test..."
	@echo "Testing frontend..."
	@curl -s http://localhost:3000 > /dev/null && echo "✅ Frontend OK" || echo "❌ Frontend failed"
	@echo "Testing backend..."
	@curl -s http://localhost:8000/health > /dev/null && echo "✅ Backend OK" || echo "❌ Backend failed"
	@echo "Testing API..."
	@curl -s http://localhost:8000/api/v1/health > /dev/null && echo "✅ API OK" || echo "❌ API failed"