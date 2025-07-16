# QueryLex V4 - Deployment Guide 🚀

This document provides comprehensive instructions for deploying the QueryLex V4 Legal RAG application in various environments.

---

## Prerequisites

- **Python 3.8+** (tested with Python 3.12)
- **Docker and Docker Compose** (for containerized deployment)
- **Supabase Project** with pgvector extension enabled
- **System Dependencies**: Tesseract OCR, Poppler
- **Environment Variables**: OpenRouter API key, Supabase credentials

---

## Quick Start with Docker

### 1. Environment Setup

1. **Create environment file**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

2. **Required environment variables**:
   ```bash
   # API Keys
   OPENROUTER_API_KEY=your_openrouter_key
   OPENROUTER_API_BASE=https://openrouter.ai/api/v1
   
   # Security Keys
   SECRET_KEY=your_flask_secret_key
   JWT_SECRET_KEY=your_jwt_secret_key
   DOCUMENT_ENCRYPTION_KEY=your_32_byte_encryption_key
   
   # Database (Supabase)
   SUPABASE_URL=your_supabase_url
   SUPABASE_ANON_KEY=your_supabase_anon_key
   SUPABASE_SERVICE_KEY=your_supabase_service_key
   
   # Model Configuration
   MODEL_PROVIDER=openrouter
   MODEL_NAME=meta-llama/llama-3.3-70b-instruct
   EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
   ```

### 2. Docker Deployment

1. **Build and run with Docker Compose**:
   ```bash
   docker-compose up -d
   ```

2. **Check container status**:
   ```bash
   docker-compose ps
   docker-compose logs -f
   ```

3. **Access the application**:
   - Web Interface: http://localhost:5000
   - Health Check: http://localhost:5000/api/health

---

## Production Deployment

### 1. Manual Deployment

#### System Dependencies Installation

**Ubuntu/Debian**:
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install python3 python3-pip python3-venv -y

# Install system dependencies
sudo apt install tesseract-ocr poppler-utils -y

# Install build tools
sudo apt install build-essential -y
```

**CentOS/RHEL**:
```bash
# Install Python and dependencies
sudo yum install python3 python3-pip -y

# Install system dependencies
sudo yum install tesseract poppler-utils -y

# Install build tools
sudo yum groupinstall "Development Tools" -y
```

#### Application Setup

1. **Clone and setup**:
   ```bash
   git clone https://github.com/your-username/QueryLexV4.git
   cd QueryLexV4
   
   # Create virtual environment
   python3 -m venv venv
   source venv/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Generate security keys**:
   ```bash
   python generate_keys.py
   ```

3. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Run with Gunicorn**:
   ```bash
   gunicorn --bind 0.0.0.0:5000 --workers 4 --timeout 300 app.main:app
   ```

### 2. Docker Production Deployment

#### Build Optimized Image

1. **Build production image**:
   ```bash
   docker build -t querylex-v4:latest .
   ```

2. **Run production container**:
   ```bash
   docker run -d \
     --name querylex-v4 \
     -p 5000:5000 \
     --env-file .env \
     -v $(pwd)/data:/app/data \
     --restart unless-stopped \
     querylex-v4:latest
   ```

#### Docker Compose Production

```yaml
version: '3.8'

services:
  querylex-v4:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
      - WORKERS=4
    env_file:
      - .env
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - querylex-v4
    restart: unless-stopped
```

---

## Cloud Deployment

### 1. Google Cloud Run

#### Prerequisites
- Google Cloud SDK installed
- Docker image built and pushed to GCR

#### Deployment Steps

1. **Build and push image**:
   ```bash
   # Configure Docker for GCR
   gcloud auth configure-docker
   
   # Build and tag image
   docker build -t gcr.io/YOUR_PROJECT_ID/querylex-v4:latest .
   docker push gcr.io/YOUR_PROJECT_ID/querylex-v4:latest
   ```

2. **Deploy to Cloud Run**:
   ```bash
   gcloud run deploy querylex-v4 \
     --image gcr.io/YOUR_PROJECT_ID/querylex-v4:latest \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --memory 2Gi \
     --cpu 2 \
     --timeout 300 \
     --max-instances 10 \
     --set-env-vars OPENROUTER_API_KEY="$OPENROUTER_API_KEY" \
     --set-env-vars SUPABASE_URL="$SUPABASE_URL" \
     --set-env-vars SUPABASE_ANON_KEY="$SUPABASE_ANON_KEY" \
     --set-env-vars SUPABASE_SERVICE_KEY="$SUPABASE_SERVICE_KEY"
   ```

3. **Configure custom domain** (optional):
   ```bash
   gcloud run domain-mappings create \
     --service querylex-v4 \
     --domain your-domain.com \
     --region us-central1
   ```

### 2. AWS ECS/Fargate

#### Prerequisites
- AWS CLI configured
- ECR repository created

#### Deployment Steps

1. **Create ECR repository**:
   ```bash
   aws ecr create-repository --repository-name querylex-v4
   ```

2. **Build and push image**:
   ```bash
   # Get login token
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com
   
   # Build and push
   docker build -t querylex-v4 .
   docker tag querylex-v4:latest YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/querylex-v4:latest
   docker push YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/querylex-v4:latest
   ```

3. **Create ECS task definition**:
   ```json
   {
     "family": "querylex-v4",
     "networkMode": "awsvpc",
     "requiresCompatibilities": ["FARGATE"],
     "cpu": "1024",
     "memory": "2048",
     "executionRoleArn": "arn:aws:iam::YOUR_ACCOUNT_ID:role/ecsTaskExecutionRole",
     "containerDefinitions": [
       {
         "name": "querylex-v4",
         "image": "YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/querylex-v4:latest",
         "portMappings": [
           {
             "containerPort": 5000,
             "protocol": "tcp"
           }
         ],
         "environment": [
           {
             "name": "OPENROUTER_API_KEY",
             "value": "your_key"
           }
         ],
         "logConfiguration": {
           "logDriver": "awslogs",
           "options": {
             "awslogs-group": "/ecs/querylex-v4",
             "awslogs-region": "us-east-1",
             "awslogs-stream-prefix": "ecs"
           }
         }
       }
     ]
   }
   ```

### 3. Railway

1. **Connect GitHub repository**:
   - Login to Railway
   - Connect your GitHub repository
   - Select the QueryLexV4 repository

2. **Configure environment variables**:
   ```bash
   # Add in Railway dashboard
   OPENROUTER_API_KEY=your_key
   SUPABASE_URL=your_url
   SUPABASE_ANON_KEY=your_key
   SUPABASE_SERVICE_KEY=your_key
   SECRET_KEY=your_key
   JWT_SECRET_KEY=your_key
   DOCUMENT_ENCRYPTION_KEY=your_key
   ```

3. **Deploy automatically**:
   - Railway will automatically build and deploy on git push

---

## Performance Optimization

### 1. Application Optimization

**Gunicorn Configuration**:
```bash
# Production settings
gunicorn --bind 0.0.0.0:5000 \
  --workers 4 \
  --worker-class gevent \
  --worker-connections 1000 \
  --timeout 300 \
  --keep-alive 2 \
  --max-requests 1000 \
  --max-requests-jitter 100 \
  --preload \
  app.main:app
```

**Environment Variables**:
```bash
# Performance settings
WORKERS=4
WORKER_CLASS=gevent
WORKER_CONNECTIONS=1000
TIMEOUT=300
```

### 2. Database Optimization

**Supabase Configuration**:
- Enable connection pooling
- Use appropriate index strategies
- Monitor query performance
- Consider read replicas for high traffic

**Vector Search Optimization**:
```sql
-- Optimize vector search performance
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_embedding_hnsw 
ON documents USING hnsw (embedding vector_cosine_ops);

-- Optimize metadata searches
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_metadata_gin 
ON documents USING gin (metadata);
```

### 3. Caching Strategy

**Redis Configuration** (optional):
```bash
# Install Redis
sudo apt install redis-server

# Configure in .env
REDIS_URL=redis://localhost:6379
CACHE_TYPE=redis
CACHE_DEFAULT_TIMEOUT=300
```

**Application Caching**:
```python
# Cache embeddings for frequently accessed documents
# Cache user sessions
# Cache dataset metadata
```

### 4. Load Balancing

**Nginx Configuration**:
```nginx
upstream querylex_backend {
    server 127.0.0.1:5000;
    server 127.0.0.1:5001;
    server 127.0.0.1:5002;
    server 127.0.0.1:5003;
}

server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://querylex_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
}
```

---

## Monitoring and Maintenance

### 1. Health Checks

**Application Health Endpoints**:
```bash
# General health check
curl http://localhost:5000/api/health

# Document processing health
curl http://localhost:5000/api/health/document-processing

# Database connection health
curl http://localhost:5000/api/health/database
```

**Docker Health Checks**:
```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5000/api/health || exit 1
```

### 2. Logging

**Application Logging**:
```python
# Configure structured logging
import logging
import json
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
```

**Log Aggregation**:
```bash
# Using Fluentd/Fluent Bit for log collection
# Configure to send to ElasticSearch, CloudWatch, etc.
```

### 3. Monitoring

**Application Metrics**:
```python
# Monitor key metrics
- Request count and latency
- Document processing times
- Database query performance
- Memory and CPU usage
- Error rates
```

**Alerting**:
```yaml
# Example Prometheus alerts
groups:
  - name: querylex-v4
    rules:
      - alert: HighErrorRate
        expr: rate(flask_request_exceptions_total[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: High error rate detected
```

### 4. Backup and Recovery

**Database Backup**:
```bash
# Supabase automatic backups are enabled
# Additional backup strategy:
pg_dump -h your-supabase-host -U postgres -d your-database > backup.sql
```

**Document Backup**:
```bash
# Backup encrypted documents
rsync -av data/ backup-location/
```

**Recovery Procedures**:
```bash
# Database recovery
psql -h your-supabase-host -U postgres -d your-database < backup.sql

# Document recovery
rsync -av backup-location/ data/
```

---

## Security Considerations

### 1. Network Security

**HTTPS Configuration**:
```nginx
server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/private.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
}
```

**Firewall Configuration**:
```bash
# UFW configuration
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw --force enable
```

### 2. Application Security

**Environment Variables**:
```bash
# Store sensitive data in secure vaults
# Use different keys for different environments
# Rotate keys regularly
```

**Rate Limiting**:
```python
# Implement rate limiting
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)
```

### 3. Data Protection

**Encryption at Rest**:
```bash
# Use encrypted storage volumes
# Encrypt database connections
# Encrypt document storage
```

**Access Control**:
```python
# Implement proper RBAC
# Use JWT tokens with short expiration
# Implement API key rotation
```

---

## Troubleshooting

### Common Issues

**1. Tesseract not found**:
```bash
# Install Tesseract
sudo apt install tesseract-ocr
# or
winget install UB-Mannheim.TesseractOCR
```

**2. Memory issues**:
```bash
# Increase container memory
docker run -m 4g querylex-v4
# or adjust worker count
```

**3. Database connection errors**:
```bash
# Check Supabase credentials
# Verify network connectivity
# Check connection limits
```

**4. Model loading failures**:
```bash
# Check internet connectivity
# Verify HuggingFace access
# Clear model cache
```

### Debugging

**Enable Debug Mode**:
```bash
export FLASK_ENV=development
export DEBUG=true
python app/main.py
```

**Check Logs**:
```bash
# Application logs
tail -f app.log

# Container logs
docker logs -f querylex-v4

# System logs
journalctl -u querylex-v4
```

---

## Scaling Strategies

### Horizontal Scaling

**Multiple Instances**:
```yaml
# Docker Compose scaling
version: '3.8'
services:
  querylex-v4:
    build: .
    deploy:
      replicas: 3
    ports:
      - "5000-5002:5000"
```

**Load Balancer**:
```nginx
upstream querylex_cluster {
    least_conn;
    server querylex-v4-1:5000;
    server querylex-v4-2:5000;
    server querylex-v4-3:5000;
}
```

### Vertical Scaling

**Resource Allocation**:
```yaml
# Docker Compose resources
services:
  querylex-v4:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

---

## Support

For deployment issues:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review application logs
3. Verify system dependencies
4. Check environment variables
5. Create an issue on GitHub

**Production Support Checklist**:
- [ ] Environment variables configured
- [ ] System dependencies installed
- [ ] Database schema applied
- [ ] SSL certificates configured
- [ ] Health checks passing
- [ ] Monitoring configured
- [ ] Backup strategy implemented
- [ ] Security hardening applied

---

**Happy Deploying! 🚀**