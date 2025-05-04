# Models and Algorithms

## Overview

The Demand Forecasting System implements various machine learning models and algorithms for time series forecasting. This document describes the available models, their implementations, and usage guidelines.

## Available Models

### 1. Random Forest

A tree-based ensemble model that combines multiple decision trees for improved prediction accuracy.

**Implementation:**
```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
```

**Key Parameters:**
- `n_estimators`: Number of trees in the forest
- `max_depth`: Maximum depth of trees
- `min_samples_split`: Minimum samples required to split a node
- `min_samples_leaf`: Minimum samples required at leaf node

**Use Cases:**
- Complex non-linear relationships
- Feature importance analysis
- Robust to outliers
- Handles mixed data types

### 2. XGBoost

A gradient boosting framework that uses tree-based models.

**Implementation:**
```python
import xgboost as xgb

model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1
)
```

**Key Parameters:**
- `n_estimators`: Number of boosting rounds
- `max_depth`: Maximum tree depth
- `learning_rate`: Step size shrinkage
- `subsample`: Subsample ratio of training instances

**Use Cases:**
- High-dimensional data
- Complex patterns
- Fast training
- Good with missing values

### 3. LightGBM

A gradient boosting framework that uses tree-based learning algorithms.

**Implementation:**
```python
import lightgbm as lgb

model = lgb.LGBMRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1
)
```

**Key Parameters:**
- `n_estimators`: Number of boosting iterations
- `max_depth`: Maximum tree depth
- `learning_rate`: Step size shrinkage
- `num_leaves`: Maximum number of leaves

**Use Cases:**
- Large datasets
- Fast training
- Memory efficient
- Good with categorical features

### 4. Prophet

A procedure for forecasting time series data based on an additive model.

**Implementation:**
```python
from prophet import Prophet

model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=True
)
```

**Key Parameters:**
- `yearly_seasonality`: Fit yearly seasonality
- `weekly_seasonality`: Fit weekly seasonality
- `daily_seasonality`: Fit daily seasonality
- `changepoint_prior_scale`: Flexibility of trend

**Use Cases:**
- Strong seasonality
- Holiday effects
- Missing data
- Trend changes

### 5. ARIMA

AutoRegressive Integrated Moving Average model for time series forecasting.

**Implementation:**
```python
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(
    data,
    order=(p, d, q)
)
```

**Key Parameters:**
- `p`: Number of autoregressive terms
- `d`: Number of differences
- `q`: Number of moving average terms

**Use Cases:**
- Stationary time series
- Linear relationships
- Short-term forecasting
- Traditional time series

## Feature Engineering

### 1. Time Features

```python
def create_time_features(df):
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    return df
```

### 2. Lag Features

```python
def create_lag_features(df, lags=[1, 7, 14, 30]):
    for lag in lags:
        df[f'lag_{lag}'] = df['target'].shift(lag)
    return df
```

### 3. Rolling Features

```python
def create_rolling_features(df, windows=[7, 14, 30]):
    for window in windows:
        df[f'rolling_mean_{window}'] = df['target'].rolling(window=window).mean()
        df[f'rolling_std_{window}'] = df['target'].rolling(window=window).std()
    return df
```

## Model Training

### 1. Data Preparation

```python
def prepare_data(data, target_col, date_col):
    # Sort by date
    data = data.sort_values(date_col)
    
    # Create features
    data = create_time_features(data)
    data = create_lag_features(data)
    data = create_rolling_features(data)
    
    # Split features and target
    X = data.drop(columns=[target_col, date_col])
    y = data[target_col]
    
    return X, y
```

### 2. Model Training

```python
def train_model(X, y, model_type='random_forest'):
    if model_type == 'random_forest':
        model = RandomForestRegressor()
    elif model_type == 'xgboost':
        model = XGBRegressor()
    elif model_type == 'lightgbm':
        model = LGBMRegressor()
    
    model.fit(X, y)
    return model
```

### 3. Model Evaluation

```python
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    
    metrics = {
        'mae': mean_absolute_error(y_test, predictions),
        'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
        'mape': mean_absolute_percentage_error(y_test, predictions)
    }
    
    return metrics
```

## Hyperparameter Tuning

### 1. Grid Search

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestRegressor(),
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error'
)
```

### 2. Random Search

```python
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(5, 20),
    'min_samples_split': randint(2, 20)
}

random_search = RandomizedSearchCV(
    RandomForestRegressor(),
    param_dist,
    n_iter=100,
    cv=5,
    scoring='neg_mean_squared_error'
)
```

### 3. Bayesian Optimization

```python
from skopt import BayesSearchCV

param_space = {
    'n_estimators': (100, 500),
    'max_depth': (5, 20),
    'min_samples_split': (2, 20)
}

bayes_search = BayesSearchCV(
    RandomForestRegressor(),
    param_space,
    n_iter=100,
    cv=5,
    scoring='neg_mean_squared_error'
)
```

## Model Persistence

### 1. Save Model

```python
import joblib

def save_model(model, path):
    joblib.dump(model, path)
```

### 2. Load Model

```python
def load_model(path):
    return joblib.load(path)
```

## Model Monitoring

### 1. Performance Metrics

```python
def calculate_metrics(y_true, y_pred):
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mape': mean_absolute_percentage_error(y_true, y_pred)
    }
```

### 2. Feature Importance

```python
def get_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    else:
        importance = model.coef_
    
    return pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
```

## Best Practices

### 1. Model Selection

- Consider data characteristics
- Evaluate model complexity
- Balance accuracy and interpretability
- Consider computational resources

### 2. Feature Engineering

- Create domain-specific features
- Handle missing values appropriately
- Normalize/scale features
- Remove redundant features

### 3. Model Training

- Use cross-validation
- Implement early stopping
- Monitor training progress
- Save checkpoints

### 4. Model Evaluation

- Use multiple metrics
- Compare with baseline
- Analyze error patterns
- Validate assumptions

### 5. Model Deployment

- Version control models
- Monitor model drift
- Implement fallback strategies
- Regular retraining

## Future Improvements

### 1. Model Enhancements

- Deep learning models
- Ensemble methods
- Transfer learning
- Online learning

### 2. Feature Engineering

- Automated feature selection
- Feature interaction detection
- Advanced time features
- External data integration

### 3. Training Process

- Distributed training
- Automated hyperparameter tuning
- Model compression
- Incremental learning

### 4. Evaluation Methods

- Advanced metrics
- Uncertainty quantification
- Explainable AI
- Fairness metrics 