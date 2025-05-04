# Deployment Guide

## Overview

This document provides comprehensive instructions for deploying the Demand Forecasting System in various environments, including development, staging, and production.

## Deployment Architecture

### 1. Components

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│             │     │             │     │             │
│  Load       │────▶│  API        │────▶│  Model      │
│  Balancer   │     │  Servers    │     │  Servers    │
│             │     │             │     │             │
└─────────────┘     └─────────────┘     └─────────────┘
        │                   │                   │
        │                   │                   │
        ▼                   ▼                   ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│             │     │             │     │             │
│  Cache      │     │  Database   │     │  Monitoring │
│  Cluster    │     │  Cluster    │     │  Stack      │
│             │     │             │     │             │
└─────────────┘     └─────────────┘     └─────────────┘
```

### 2. Infrastructure Requirements

- **Compute Resources**
  - API Servers: 4+ vCPUs, 8GB+ RAM
  - Model Servers: 8+ vCPUs, 16GB+ RAM
  - Database Servers: 4+ vCPUs, 8GB+ RAM
  - Cache Servers: 2+ vCPUs, 4GB+ RAM

- **Storage Requirements**
  - Database: 100GB+ SSD
  - Model Storage: 50GB+ SSD
  - Log Storage: 100GB+ HDD
  - Backup Storage: 200GB+ HDD

- **Network Requirements**
  - High-speed internet connection
  - Load balancer
  - Firewall
  - VPN access

## Deployment Methods

### 1. Docker Deployment

#### Docker Compose

```yaml
version: '3.8'

services:
  api:
    build: 
      context: .
      dockerfile: Dockerfile.api
    ports:
      - "8000:8000"
    environment:
      - DB_HOST=db
      - REDIS_HOST=redis
    depends_on:
      - db
      - redis

  model:
    build:
      context: .
      dockerfile: Dockerfile.model
    environment:
      - DB_HOST=db
      - REDIS_HOST=redis
    depends_on:
      - db
      - redis

  db:
    image: postgres:13
    environment:
      - POSTGRES_DB=demand_forecasting
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:6
    volumes:
      - redis_data:/data

  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    depends_on:
      - prometheus

volumes:
  postgres_data:
  redis_data:
```

#### Dockerfile

```dockerfile
# API Service
FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Model Service
FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-m", "model.service"]
```

### 2. Kubernetes Deployment

#### Deployment Configuration

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: api-service
  template:
    metadata:
      labels:
        app: api-service
    spec:
      containers:
      - name: api
        image: demand-forecasting/api:latest
        ports:
        - containerPort: 8000
        env:
        - name: DB_HOST
          valueFrom:
            configMapKeyRef:
              name: app-config
              key: db_host
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
```

#### Service Configuration

```yaml
apiVersion: v1
kind: Service
metadata:
  name: api-service
spec:
  selector:
    app: api-service
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

#### Ingress Configuration

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: api-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  rules:
  - host: api.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: api-service
            port:
              number: 80
```

### 3. Cloud Deployment

#### AWS Deployment

```yaml
# AWS CloudFormation Template
AWSTemplateFormatVersion: '2010-09-09'
Resources:
  ApiServer:
    Type: AWS::EC2::Instance
    Properties:
      ImageId: ami-0c55b159cbfafe1f0
      InstanceType: t3.large
      SecurityGroups:
        - !Ref ApiSecurityGroup
      UserData:
        Fn::Base64: !Sub |
          #!/bin/bash
          yum update -y
          yum install -y docker
          service docker start
          docker run -d -p 8000:8000 demand-forecasting/api:latest

  ApiSecurityGroup:
    Type: AWS::SecurityGroup
    Properties:
      GroupDescription: API Server Security Group
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 80
          ToPort: 80
          CidrIp: 0.0.0.0/0
```

#### Azure Deployment

```yaml
# Azure Resource Manager Template
{
  "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
  "contentVersion": "1.0.0.0",
  "resources": [
    {
      "type": "Microsoft.Compute/virtualMachines",
      "name": "api-server",
      "properties": {
        "hardwareProfile": {
          "vmSize": "Standard_D2s_v3"
        },
        "osProfile": {
          "computerName": "api-server",
          "adminUsername": "azureuser",
          "adminPassword": "password"
        },
        "storageProfile": {
          "imageReference": {
            "publisher": "Canonical",
            "offer": "UbuntuServer",
            "sku": "18.04-LTS",
            "version": "latest"
          }
        }
      }
    }
  ]
}
```

## Configuration Management

### 1. Environment Configuration

```python
# config.py
import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4
    API_TIMEOUT: int = 30

    # Database Settings
    DB_HOST: str
    DB_PORT: int
    DB_NAME: str
    DB_USER: str
    DB_PASSWORD: str

    # Redis Settings
    REDIS_HOST: str
    REDIS_PORT: int
    REDIS_DB: int

    # Model Settings
    MODEL_PATH: str
    MODEL_TYPE: str
    MODEL_PARAMS: dict

    class Config:
        env_file = ".env"
```

### 2. Secret Management

```python
# secrets.py
from kubernetes import client, config
from google.cloud import secretmanager

class SecretManager:
    def __init__(self, provider="kubernetes"):
        self.provider = provider
        if provider == "kubernetes":
            config.load_kube_config()
            self.client = client.CoreV1Api()
        elif provider == "gcp":
            self.client = secretmanager.SecretManagerServiceClient()

    def get_secret(self, name: str) -> str:
        if self.provider == "kubernetes":
            secret = self.client.read_namespaced_secret(
                name=name,
                namespace="default"
            )
            return secret.data
        elif self.provider == "gcp":
            name = f"projects/{project_id}/secrets/{name}/versions/latest"
            response = self.client.access_secret_version(request={"name": name})
            return response.payload.data.decode("UTF-8")
```

## Deployment Process

### 1. Pre-deployment Checklist

- [ ] Code review completed
- [ ] Tests passed
- [ ] Documentation updated
- [ ] Security scan completed
- [ ] Performance testing done
- [ ] Backup created
- [ ] Rollback plan prepared

### 2. Deployment Steps

```bash
# 1. Build Docker images
docker build -t demand-forecasting/api:latest -f Dockerfile.api .
docker build -t demand-forecasting/model:latest -f Dockerfile.model .

# 2. Push images to registry
docker push demand-forecasting/api:latest
docker push demand-forecasting/model:latest

# 3. Update Kubernetes deployments
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml

# 4. Verify deployment
kubectl get pods
kubectl get services
kubectl get ingress
```

### 3. Post-deployment Verification

```python
def verify_deployment():
    checks = [
        check_api_health(),
        check_model_health(),
        check_database_connection(),
        check_redis_connection(),
        check_monitoring()
    ]
    
    return all(checks)

def check_api_health():
    response = requests.get("http://api.example.com/health")
    return response.status_code == 200

def check_model_health():
    response = requests.get("http://api.example.com/model/health")
    return response.status_code == 200
```

## Monitoring and Logging

### 1. Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'api'
    static_configs:
      - targets: ['api-service:8000']
  
  - job_name: 'model'
    static_configs:
      - targets: ['model-service:8000']
```

### 2. Grafana Dashboards

```json
{
  "dashboard": {
    "id": null,
    "title": "API Metrics",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "datasource": "Prometheus",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])"
          }
        ]
      }
    ]
  }
}
```

### 3. Log Aggregation

```python
# logging.py
import logging
from logging.handlers import RotatingFileHandler

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            RotatingFileHandler(
                'app.log',
                maxBytes=10485760,
                backupCount=5
            ),
            logging.StreamHandler()
        ]
    )
```

## Backup and Recovery

### 1. Database Backup

```bash
# Backup script
#!/bin/bash
BACKUP_DIR="/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/backup_$TIMESTAMP.sql"

# Create backup
pg_dump -h $DB_HOST -U $DB_USER $DB_NAME > $BACKUP_FILE

# Compress backup
gzip $BACKUP_FILE

# Upload to S3
aws s3 cp "$BACKUP_FILE.gz" "s3://backups/database/"
```

### 2. Model Backup

```python
def backup_model(model_path: str, backup_path: str):
    # Create backup directory
    os.makedirs(backup_path, exist_ok=True)
    
    # Copy model files
    shutil.copytree(model_path, backup_path)
    
    # Upload to cloud storage
    upload_to_cloud(backup_path)
```

### 3. Recovery Process

```python
def recover_from_backup(backup_id: str):
    # Download backup
    backup_path = download_backup(backup_id)
    
    # Restore database
    restore_database(backup_path)
    
    # Restore model
    restore_model(backup_path)
    
    # Verify recovery
    verify_recovery()
```

## Scaling

### 1. Horizontal Scaling

```yaml
# Horizontal Pod Autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-service
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api-service
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### 2. Vertical Scaling

```yaml
# Vertical Pod Autoscaler
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: api-service
spec:
  targetRef:
    apiVersion: "apps/v1"
    kind: Deployment
    name: api-service
  updatePolicy:
    updateMode: "Auto"
```

### 3. Database Scaling

```yaml
# Database Replication
apiVersion: v1
kind: Service
metadata:
  name: postgres-primary
spec:
  selector:
    app: postgres
    role: primary
  ports:
  - port: 5432
---
apiVersion: v1
kind: Service
metadata:
  name: postgres-replica
spec:
  selector:
    app: postgres
    role: replica
  ports:
  - port: 5432
```

## Maintenance

### 1. Regular Maintenance Tasks

```python
def perform_maintenance():
    tasks = [
        cleanup_old_logs(),
        optimize_database(),
        update_dependencies(),
        rotate_certificates(),
        check_disk_space()
    ]
    
    for task in tasks:
        try:
            task()
        except Exception as e:
            log_error(f"Maintenance task failed: {e}")
```

### 2. Update Process

```python
def update_system():
    # 1. Backup current state
    backup_system()
    
    # 2. Update code
    update_code()
    
    # 3. Update dependencies
    update_dependencies()
    
    # 4. Run migrations
    run_migrations()
    
    # 5. Verify update
    verify_update()
    
    # 6. Rollback if needed
    if not verify_update():
        rollback_update()
```

### 3. Health Checks

```python
def health_check():
    checks = {
        'api': check_api_health(),
        'model': check_model_health(),
        'database': check_database_health(),
        'redis': check_redis_health(),
        'monitoring': check_monitoring_health()
    }
    
    return all(checks.values()), checks
```

## Troubleshooting

### 1. Common Issues

- API service not responding
- Model service errors
- Database connection issues
- Redis connection issues
- Monitoring system failures

### 2. Debugging Tools

```python
def debug_system():
    # Collect system information
    system_info = collect_system_info()
    
    # Check logs
    log_analysis = analyze_logs()
    
    # Check metrics
    metrics_analysis = analyze_metrics()
    
    # Generate report
    generate_debug_report(system_info, log_analysis, metrics_analysis)
```

### 3. Recovery Procedures

```python
def recover_system():
    # 1. Identify issue
    issue = identify_issue()
    
    # 2. Stop affected services
    stop_services(issue.affected_services)
    
    # 3. Restore from backup
    restore_from_backup(issue.backup_id)
    
    # 4. Start services
    start_services(issue.affected_services)
    
    # 5. Verify recovery
    verify_recovery()
```

## Future Improvements

### 1. Deployment Improvements

- Automated deployment pipeline
- Blue-green deployment
- Canary releases
- Automated rollback

### 2. Monitoring Improvements

- Advanced metrics
- Predictive monitoring
- Automated alerting
- Custom dashboards

### 3. Scaling Improvements

- Auto-scaling optimization
- Load balancing improvements
- Database sharding
- Cache optimization

### 4. Security Improvements

- Automated security scanning
- Compliance monitoring
- Access control improvements
- Security automation