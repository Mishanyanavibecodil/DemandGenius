# Testing Guide

## Overview

This document outlines the testing strategy and procedures for the Demand Forecasting System, including unit tests, integration tests, performance tests, and security tests.

## Testing Architecture

### 1. Components

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│             │     │             │     │             │
│  Unit       │────▶│  Integration│────▶│  System     │
│  Tests      │     │  Tests      │     │  Tests      │
│             │     │             │     │             │
└─────────────┘     └─────────────┘     └─────────────┘
        │                   │                   │
        │                   │                   │
        ▼                   ▼                   ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│             │     │             │     │             │
│  Performance│     │  Security   │     │  Load       │
│  Tests      │     │  Tests      │     │  Tests      │
│             │     │             │     │             │
└─────────────┘     └─────────────┘     └─────────────┘
```

### 2. Test Categories

1. **Unit Tests**
   - Individual component testing
   - Function-level testing
   - Mock dependencies
   - Fast execution

2. **Integration Tests**
   - Component interaction testing
   - API endpoint testing
   - Database integration
   - External service integration

3. **System Tests**
   - End-to-end testing
   - User flow testing
   - System integration
   - Environment testing

4. **Performance Tests**
   - Load testing
   - Stress testing
   - Scalability testing
   - Resource monitoring

5. **Security Tests**
   - Vulnerability scanning
   - Penetration testing
   - Security compliance
   - Access control testing

## Unit Testing

### 1. Test Structure

```python
# test_forecaster.py
import unittest
from demand_forecasting.forecaster import DemandForecaster

class TestDemandForecaster(unittest.TestCase):
    def setUp(self):
        self.forecaster = DemandForecaster()
        self.test_data = self._generate_test_data()

    def test_initialization(self):
        self.assertIsNotNone(self.forecaster)
        self.assertIsNone(self.forecaster.model)

    def test_fit(self):
        self.forecaster.fit(self.test_data)
        self.assertIsNotNone(self.forecaster.model)

    def test_predict(self):
        self.forecaster.fit(self.test_data)
        predictions = self.forecaster.predict(days=7)
        self.assertEqual(len(predictions), 7)

    def _generate_test_data(self):
        # Generate synthetic test data
        return pd.DataFrame({
            'date': pd.date_range(start='2024-01-01', periods=100),
            'item_id': ['A'] * 100,
            'quantity': np.random.randint(1, 100, 100)
        })
```

### 2. Test Fixtures

```python
# conftest.py
import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'date': pd.date_range(start='2024-01-01', periods=100),
        'item_id': ['A'] * 100,
        'quantity': np.random.randint(1, 100, 100)
    })

@pytest.fixture
def forecaster():
    from demand_forecasting.forecaster import DemandForecaster
    return DemandForecaster()

@pytest.fixture
def trained_forecaster(forecaster, sample_data):
    forecaster.fit(sample_data)
    return forecaster
```

### 3. Mocking

```python
# test_api.py
from unittest.mock import Mock, patch

def test_predict_endpoint():
    with patch('demand_forecasting.api.Forecaster') as mock_forecaster:
        # Setup mock
        mock_forecaster.return_value.predict.return_value = [100, 120, 140]
        
        # Test endpoint
        response = client.post("/predict", json={"days": 3})
        
        # Verify response
        assert response.status_code == 200
        assert response.json() == {"predictions": [100, 120, 140]}
```

## Integration Testing

### 1. API Testing

```python
# test_api_integration.py
import requests

def test_api_integration():
    # Test API endpoints
    base_url = "http://localhost:8000"
    
    # Test training endpoint
    train_data = {
        "data": {
            "date": ["2024-01-01", "2024-01-02"],
            "item_id": ["A", "A"],
            "quantity": [100, 120]
        }
    }
    response = requests.post(f"{base_url}/train", json=train_data)
    assert response.status_code == 200
    
    # Test prediction endpoint
    predict_data = {"days": 7}
    response = requests.post(f"{base_url}/predict", json=predict_data)
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 7
```

### 2. Database Testing

```python
# test_database.py
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

@pytest.fixture
def db_session():
    engine = create_engine('sqlite:///:memory:')
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()

def test_database_operations(db_session):
    # Test database operations
    from demand_forecasting.models import Forecast
    
    # Create forecast
    forecast = Forecast(
        item_id="A",
        date="2024-01-01",
        quantity=100
    )
    db_session.add(forecast)
    db_session.commit()
    
    # Verify forecast
    result = db_session.query(Forecast).first()
    assert result.item_id == "A"
    assert result.quantity == 100
```

### 3. External Service Testing

```python
# test_external_services.py
import pytest
from unittest.mock import patch

def test_external_api_integration():
    with patch('requests.post') as mock_post:
        # Setup mock response
        mock_post.return_value.json.return_value = {
            "status": "success",
            "data": [100, 120, 140]
        }
        
        # Test integration
        from demand_forecasting.external import ExternalAPI
        api = ExternalAPI()
        result = api.get_forecast()
        
        # Verify result
        assert result == [100, 120, 140]
```

## Performance Testing

### 1. Load Testing

```python
# test_performance.py
import locust
from locust import HttpUser, task, between

class ForecastUser(HttpUser):
    wait_time = between(1, 5)
    
    @task
    def test_prediction(self):
        self.client.post("/predict", json={
            "item_id": "A",
            "days": 7
        })
    
    @task
    def test_training(self):
        self.client.post("/train", json={
            "data": {
                "date": ["2024-01-01", "2024-01-02"],
                "item_id": ["A", "A"],
                "quantity": [100, 120]
            }
        })
```

### 2. Stress Testing

```python
# test_stress.py
import pytest
import time

def test_stress_prediction():
    start_time = time.time()
    requests = 1000
    
    for _ in range(requests):
        response = client.post("/predict", json={
            "item_id": "A",
            "days": 7
        })
        assert response.status_code == 200
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Verify performance
    assert duration < 60  # Should complete within 60 seconds
    assert requests / duration > 10  # Should handle at least 10 requests per second
```

### 3. Resource Monitoring

```python
# test_resources.py
import psutil
import pytest

def test_memory_usage():
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    
    # Perform memory-intensive operation
    forecaster.fit(large_dataset)
    
    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory
    
    # Verify memory usage
    assert memory_increase < 1024 * 1024 * 100  # Less than 100MB increase
```

## Security Testing

### 1. Vulnerability Scanning

```python
# test_security.py
import pytest
from bandit.core import manager

def test_code_security():
    # Run Bandit security scanner
    b_mgr = manager.BanditManager()
    b_mgr.discover_files(['demand_forecasting'])
    b_mgr.run_tests()
    
    # Verify no high severity issues
    assert len(b_mgr.get_issue_list(severity='HIGH')) == 0
```

### 2. Penetration Testing

```python
# test_penetration.py
import pytest
import requests

def test_api_security():
    # Test authentication
    response = requests.get("/api/data", headers={})
    assert response.status_code == 401
    
    # Test authorization
    response = requests.get("/api/admin", headers={"Authorization": "Bearer user_token"})
    assert response.status_code == 403
    
    # Test input validation
    response = requests.post("/api/predict", json={"days": -1})
    assert response.status_code == 400
```

### 3. Access Control Testing

```python
# test_access.py
import pytest

def test_role_based_access():
    # Test admin access
    admin_response = client.get("/admin", headers={"Role": "admin"})
    assert admin_response.status_code == 200
    
    # Test user access
    user_response = client.get("/admin", headers={"Role": "user"})
    assert user_response.status_code == 403
    
    # Test viewer access
    viewer_response = client.get("/admin", headers={"Role": "viewer"})
    assert viewer_response.status_code == 403
```

## Test Automation

### 1. CI/CD Pipeline

```yaml
# .github/workflows/tests.yml
name: Tests

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run tests
      run: |
        pytest tests/
```

### 2. Test Reports

```python
# test_reporting.py
import pytest
import json

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    # Generate test report
    report = {
        'total': terminalreporter.stats.get('total', 0),
        'passed': len(terminalreporter.stats.get('passed', [])),
        'failed': len(terminalreporter.stats.get('failed', [])),
        'skipped': len(terminalreporter.stats.get('skipped', [])),
        'duration': terminalreporter.duration
    }
    
    # Save report
    with open('test_report.json', 'w') as f:
        json.dump(report, f)
```

### 3. Test Coverage

```python
# pytest.ini
[pytest]
addopts = --cov=demand_forecasting --cov-report=html --cov-report=term-missing
testpaths = tests
python_files = test_*.py
```

## Best Practices

### 1. Test Organization

- Group tests by functionality
- Use descriptive test names
- Follow AAA pattern (Arrange, Act, Assert)
- Keep tests independent
- Use appropriate fixtures

### 2. Test Data Management

- Use synthetic test data
- Clean up test data
- Use data factories
- Version control test data
- Document test data

### 3. Test Maintenance

- Regular test updates
- Remove obsolete tests
- Update test documentation
- Monitor test performance
- Review test coverage

### 4. Test Documentation

- Document test purpose
- Document test setup
- Document test data
- Document test results
- Document test maintenance

## Future Improvements

### 1. Test Automation Improvements

- Automated test generation
- Test case optimization
- Test data generation
- Test environment management
- Test reporting improvements

### 2. Test Coverage Improvements

- Increase unit test coverage
- Add integration tests
- Add performance tests
- Add security tests
- Add accessibility tests

### 3. Test Quality Improvements

- Test code quality
- Test performance
- Test reliability
- Test maintainability
- Test documentation

### 4. Test Infrastructure Improvements

- Test environment automation
- Test data management
- Test reporting
- Test monitoring
- Test analytics