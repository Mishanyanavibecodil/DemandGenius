# Installation and Setup Guide

## System Requirements

- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- 10GB free disk space
- Operating System: Linux, macOS, or Windows 10/11

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/example/demand-forecasting.git
cd demand-forecasting
```

### 2. Create Virtual Environment

#### Using venv (recommended)

```bash
# Linux/macOS
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
.\venv\Scripts\activate
```

#### Using conda

```bash
conda create -n demand-forecasting python=3.8
conda activate demand-forecasting
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Development Dependencies (Optional)

```bash
pip install -r requirements-dev.txt
```

## Configuration

### 1. Basic Configuration

Create a configuration file at `config/config.yaml`:

```yaml
# Model Configuration
model:
  type: random_forest
  params:
    n_estimators: 100
    max_depth: 10
    random_state: 42

# Data Processing
data:
  date_column: date
  item_column: item_id
  target_column: quantity
  train_test_split: 0.8
  validation_split: 0.1

# Feature Engineering
features:
  time_features:
    - day_of_week
    - month
    - quarter
    - year
  lag_features:
    - 7
    - 14
    - 30
  rolling_features:
    - mean_7
    - std_7
    - min_7
    - max_7

# API Configuration
api:
  host: localhost
  port: 8000
  debug: false
  workers: 4
  timeout: 30

# Database Configuration
database:
  type: postgresql
  host: localhost
  port: 5432
  database: demand_forecasting
  user: your_username
  password: your_password

# Monitoring
monitoring:
  metrics_port: 9090
  log_level: INFO
  alert_thresholds:
    mae: 20
    rmse: 30
    mape: 25
```

### 2. Environment Variables

Create a `.env` file in the project root:

```bash
# API Configuration
API_HOST=localhost
API_PORT=8000
API_DEBUG=false
API_WORKERS=4

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=demand_forecasting
DB_USER=your_username
DB_PASSWORD=your_password

# Security
API_KEY=your-secret-api-key
JWT_SECRET=your-jwt-secret

# Monitoring
METRICS_PORT=9090
LOG_LEVEL=INFO
```

### 3. Database Setup

#### PostgreSQL

1. Install PostgreSQL
2. Create database and user:

```sql
CREATE DATABASE demand_forecasting;
CREATE USER your_username WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE demand_forecasting TO your_username;
```

3. Run migrations:

```bash
python scripts/migrate.py
```

## Running the Application

### 1. Start the API Server

```bash
# Development mode
python -m demand_forecasting.api

# Production mode
gunicorn demand_forecasting.api:app --workers 4 --bind 0.0.0.0:8000
```

### 2. Start the Monitoring Server

```bash
python -m demand_forecasting.monitoring
```

### 3. Run Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_forecaster.py

# Run with coverage
pytest --cov=demand_forecasting
```

## Docker Deployment

### 1. Build the Docker Image

```bash
docker build -t demand-forecasting .
```

### 2. Run the Container

```bash
docker run -d \
  --name demand-forecasting \
  -p 8000:8000 \
  -p 9090:9090 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  --env-file .env \
  demand-forecasting
```

## Troubleshooting

### Common Issues

1. **Database Connection Error**
   - Check if database server is running
   - Verify credentials in config file
   - Ensure database exists

2. **API Server Not Starting**
   - Check if port 8000 is available
   - Verify environment variables
   - Check logs for errors

3. **Model Training Issues**
   - Ensure sufficient memory
   - Check data format
   - Verify feature configuration

### Logs

Logs are stored in the `logs` directory:
- `api.log`: API server logs
- `model.log`: Model training and prediction logs
- `monitoring.log`: Monitoring system logs

## Updating

### 1. Update Code

```bash
git pull origin main
```

### 2. Update Dependencies

```bash
pip install -r requirements.txt --upgrade
```

### 3. Run Migrations

```bash
python scripts/migrate.py
```

## Security Considerations

1. **API Security**
   - Use HTTPS in production
   - Rotate API keys regularly
   - Implement rate limiting
   - Use secure headers

2. **Data Security**
   - Encrypt sensitive data
   - Regular backups
   - Access control
   - Audit logging

3. **System Security**
   - Regular updates
   - Firewall configuration
   - Secure configuration
   - Monitoring and alerts

## Support

For additional support:
- Documentation: https://docs.example.com
- GitHub Issues: https://github.com/example/demand-forecasting/issues
- Email: support@example.com 