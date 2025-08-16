# TAROT CKD Risk Prediction - Docker Deployment Guide

This guide helps you deploy the complete TAROT CKD Risk Prediction system using Docker containers.

## 🚀 Quick Start

### Prerequisites

- **Docker** (version 20.0+)
- **Docker Compose** (version 2.0+)
- **Foundation Models** directory with pre-trained models
- **Git** (to clone the repository)

### 1. Clone Repository and Setup Models

```bash
# Clone the repository
git clone <repository-url>
cd tarot2/webapp

# Ensure you have the foundation models directory
# The models should be located at: ../foundation_models/
ls -la ../foundation_models/
```

### 2. Deploy with Docker Compose

```bash
# Build and start all services
docker-compose up --build -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f
```

### 3. Access the Application

- **Frontend**: http://localhost:3000 (React Web App)
- **Backend API**: http://localhost:8000 (FastAPI with 36 models)
- **API Documentation**: http://localhost:8000/docs

### 4. Stop Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (optional)
docker-compose down -v
```

## 📋 Service Details

### Frontend Service
- **Container**: `tarot-frontend`
- **Technology**: React 18 + TypeScript + Material-UI
- **Port Mapping**: Host 3000 → Container 80
- **Features**: 11 features × 10 timepoints input form, real-time validation

### Backend Service
- **Container**: `tarot-backend`
- **Technology**: FastAPI + PyTorch + PyCox
- **Port Mapping**: Host 8000 → Container 8000
- **Models**: 36 ensemble models (24 DeepSurv + 12 DeepHit)
- **API**: RESTful with temporal prediction support

## 🔧 Configuration Options

### Environment Variables

Create a `.env` file in the webapp directory:

```env
# Backend Configuration
LOG_LEVEL=INFO
MODEL_PATH=/app/models
ENVIRONMENT=production

# Frontend Configuration
REACT_APP_API_URL=http://localhost:8000/api/v1
NODE_ENV=production

# CORS Settings
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

### Custom Model Path

If your models are in a different location, update the docker-compose.yml:

```yaml
volumes:
  - /your/custom/path/to/models:/app/models:ro
```

## 📊 Health Monitoring

### Health Check Endpoints

```bash
# Frontend health
curl http://localhost:3000/health

# Backend health with model count
curl http://localhost:8000/health

# Detailed API health
curl http://localhost:8000/api/v1/health
```

### Expected Responses

```json
# Backend Health Response
{
  "status": "healthy",
  "timestamp": 1755314659.4272811,
  "version": "1.0.0",
  "models_loaded": 36
}
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Models Not Found
```bash
# Check if models directory exists
ls -la ../foundation_models/

# Check container model mounting
docker exec tarot-backend ls -la /app/models/
```

#### 2. Frontend Connection Issues
```bash
# Check if backend is running
curl http://localhost:8000/health

# Check container networking
docker network ls
docker network inspect webapp_tarot-network
```

#### 3. CORS Issues
- Ensure `CORS_ORIGINS` includes your frontend URL
- Check browser console for CORS errors
- Verify nginx proxy configuration

### Log Analysis

```bash
# View specific service logs
docker-compose logs backend
docker-compose logs frontend

# Follow logs in real-time
docker-compose logs -f --tail=100

# Check container resource usage
docker stats
```

### Container Debugging

```bash
# Access backend container
docker exec -it tarot-backend bash

# Access frontend container
docker exec -it tarot-frontend sh

# Inspect container configuration
docker inspect tarot-backend
```

## 🚀 Production Deployment

### With Reverse Proxy (Nginx)

For production, enable the nginx service:

```bash
# Start with production profile
docker-compose --profile production up -d

# Access via nginx (port 80/443)
curl http://localhost/
```

### SSL/HTTPS Setup

1. Place SSL certificates in `./ssl/` directory
2. Update `nginx.conf` with SSL configuration
3. Restart the nginx service

### Scaling

```bash
# Scale backend service for high load
docker-compose up -d --scale backend=3

# Use load balancer configuration in nginx.conf
```

## 📈 Performance Tuning

### Backend Optimization

```yaml
# In docker-compose.yml
services:
  backend:
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 8G
        reservations:
          cpus: '2.0'
          memory: 4G
```

### Frontend Optimization

- Static assets are cached for 1 year
- Gzip compression enabled
- React app pre-compiled for production

## 🔒 Security Considerations

### Container Security
- Non-root users in all containers
- Read-only model volumes
- Security headers in nginx
- No sensitive data in environment variables

### Network Security
- Internal Docker network for service communication
- Firewall rules for external access
- CORS restrictions properly configured

## 📚 API Usage Examples

### Test Temporal Prediction API

```bash
curl -X POST http://localhost:8000/api/v1/predict/temporal \
  -H "Content-Type: application/json" \
  -d '{
    "feature_matrix": {
      "age_at_obs": [68, null, null, null, null, null, null, null, null, null],
      "albumin": [3.2, null, null, null, null, null, null, null, null, null],
      "uacr": [150.0, null, null, null, null, null, null, null, null, null],
      "bicarbonate": [22.0, null, null, null, null, null, null, null, null, null],
      "cci_score_total": [2, null, null, null, null, null, null, null, null, null],
      "creatinine": [280.0, null, null, null, null, null, null, null, null, null],
      "gender": [0, null, null, null, null, null, null, null, null, null],
      "hemoglobin": [8.5, null, null, null, null, null, null, null, null, null],
      "ht": [1, null, null, null, null, null, null, null, null, null],
      "observation_period": [365, null, null, null, null, null, null, null, null, null],
      "phosphate": [1.8, null, null, null, null, null, null, null, null, null]
    },
    "timepoint_dates": ["2024-01-15", "", "", "", "", "", "", "", "", ""],
    "patient_info": {
      "age_at_obs": 68,
      "gender": "female",
      "observation_period": 365
    }
  }'
```

## 🆘 Support

### Getting Help

1. Check logs: `docker-compose logs -f`
2. Verify health endpoints
3. Review this documentation
4. Check GitHub issues/documentation

### System Requirements

- **Minimum**: 8GB RAM, 4 CPU cores, 10GB disk space
- **Recommended**: 16GB RAM, 8 CPU cores, 20GB disk space
- **Network**: Internet access for initial Docker image downloads

### Updates

```bash
# Pull latest images
docker-compose pull

# Rebuild and restart
docker-compose up --build -d

# Clean up old images
docker image prune -a
```

---

## 🎯 Quick Verification Checklist

- [ ] Docker and Docker Compose installed
- [ ] Foundation models directory available
- [ ] `docker-compose up -d` completes successfully
- [ ] Frontend accessible at http://localhost:3000
- [ ] Backend health shows 36 models loaded
- [ ] Can complete a test risk prediction
- [ ] All services have healthy status

🎉 **Deployment Complete!** Your TAROT CKD Risk Prediction system is ready to use.