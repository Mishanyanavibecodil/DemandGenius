# Примеры использования DemandGenius

## Базовые примеры

### 1. Простой прогноз

```python
from src.core.forecaster import DemandForecaster
from src.core.models import RandomForestModel
import pandas as pd

# Создание прогнозировщика
forecaster = DemandForecaster(
    model=RandomForestModel(),
    config_path='config/local.yaml'
)

# Загрузка данных
data = pd.read_csv('data/raw/historical_data.csv')
data['date'] = pd.to_datetime(data['date'])

# Обучение модели
forecaster.fit(
    data=data,
    target_column='quantity',
    feature_columns=['price', 'promotion']
)

# Создание прогноза
forecast = forecaster.predict(days_ahead=30)

# Визуализация результатов
forecaster.plot_forecast(forecast)
```

### 2. Прогноз с внешними факторами

```python
# Загрузка внешних факторов
external_factors = pd.read_csv('data/raw/external_factors.csv')

# Создание прогноза с учетом внешних факторов
forecast = forecaster.predict(
    days_ahead=30,
    external_factors=external_factors
)

# Анализ влияния факторов
importance = forecaster.get_feature_importance()
forecaster.plot_feature_importance(importance)
```

### 3. Пакетный прогноз

```python
# Создание прогнозов для нескольких товаров
items = ['item1', 'item2', 'item3']
forecasts = forecaster.predict_batch(
    items=items,
    days_ahead=30
)

# Экспорт результатов
forecaster.export_forecasts(forecasts, 'output/forecasts.csv')
```

## Продвинутые примеры

### 1. Оптимизация гиперпараметров

```python
from src.core.optimizer import HyperparameterOptimizer

# Создание оптимизатора
optimizer = HyperparameterOptimizer(
    model=RandomForestModel(),
    param_grid={
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5, 10]
    }
)

# Поиск оптимальных параметров
best_params = optimizer.optimize(
    data=data,
    target_column='quantity',
    cv=5,
    scoring='neg_mean_absolute_error'
)

# Создание модели с оптимальными параметрами
forecaster = DemandForecaster(
    model=RandomForestModel(**best_params),
    config_path='config/local.yaml'
)
```

### 2. Работа с API

```python
from src.api.client import APIClient

# Создание клиента
client = APIClient(
    base_url="http://localhost:8000",
    api_key="your_api_key"
)

# Обучение модели через API
client.train_model(
    data=data,
    model_config={
        "model_type": "random_forest",
        "params": {
            "n_estimators": 100,
            "max_depth": 10
        }
    }
)

# Получение прогноза через API
forecast = client.get_forecast(
    item_id="item1",
    days_ahead=30
)
```

### 3. Мониторинг и логирование

```python
from src.monitoring.logger import setup_logger
from src.monitoring.metrics import PerformanceMonitor

# Настройка логгера
logger = setup_logger(
    name='forecaster',
    log_file='logs/app/forecaster.log'
)

# Создание монитора
monitor = PerformanceMonitor()

# Отслеживание метрик
with monitor.track_prediction_time():
    forecast = forecaster.predict(days_ahead=30)

# Логирование результатов
logger.info(f"Прогноз создан: {forecast}")
logger.info(f"Время выполнения: {monitor.get_last_prediction_time():.2f} сек")
```

## Примеры использования в продакшене

### 1. Настройка безопасности

```python
from src.security.auth import APIAuthenticator
from src.security.encryption import DataEncryptor

# Настройка аутентификации
auth = APIAuthenticator(
    api_key='your_api_key',
    jwt_secret='your_jwt_secret'
)

# Настройка шифрования
encryptor = DataEncryptor(
    key='your_encryption_key'
)

# Шифрование чувствительных данных
encrypted_data = encryptor.encrypt(sensitive_data)

# Проверка токена
is_valid = auth.validate_token(token)
```

### 2. Распределенные вычисления

```python
from src.distributed.dask_utils import DaskForecaster

# Создание распределенного прогнозировщика
dask_forecaster = DaskForecaster(
    n_workers=4,
    memory_limit='4GB'
)

# Параллельное создание прогнозов
forecasts = dask_forecaster.predict_batch(
    items=items,
    days_ahead=30
)

# Ожидание завершения вычислений
results = forecasts.compute()
```

### 3. Кэширование результатов

```python
from src.core.cache import ForecastCache

# Создание кэша
cache = ForecastCache(
    cache_dir='cache/forecasts',
    ttl=3600  # время жизни кэша в секундах
)

# Получение прогноза из кэша
forecast = cache.get_forecast(
    item_id='item1',
    days_ahead=30
)

# Если прогноза нет в кэше, создаем новый
if forecast is None:
    forecast = forecaster.predict(
        item_id='item1',
        days_ahead=30
    )
    cache.save_forecast(forecast)
```

## Примеры визуализации

### 1. График прогноза

```python
import matplotlib.pyplot as plt

# Создание прогноза
forecast = forecaster.predict(days_ahead=30)

# Визуализация
plt.figure(figsize=(12, 6))
plt.plot(forecast.index, forecast.values, label='Прогноз')
plt.fill_between(
    forecast.index,
    forecast['lower_bound'],
    forecast['upper_bound'],
    alpha=0.2,
    label='Доверительный интервал'
)
plt.title('Прогноз спроса')
plt.xlabel('Дата')
plt.ylabel('Количество')
plt.legend()
plt.grid(True)
plt.show()
```

### 2. Анализ трендов

```python
# Анализ трендов
trends = forecaster.analyze_trends(data)

# Визуализация трендов
plt.figure(figsize=(12, 6))
plt.plot(trends.index, trends['trend'], label='Тренд')
plt.plot(trends.index, trends['seasonal'], label='Сезонность')
plt.title('Анализ трендов')
plt.xlabel('Дата')
plt.ylabel('Значение')
plt.legend()
plt.grid(True)
plt.show()
```

### 3. Важность признаков

```python
# Получение важности признаков
importance = forecaster.get_feature_importance()

# Визуализация
plt.figure(figsize=(10, 6))
plt.bar(importance.index, importance.values)
plt.title('Важность признаков')
plt.xlabel('Признак')
plt.ylabel('Важность')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

## Примеры обработки данных

### 1. Предобработка данных

```python
from src.data.processors import DataProcessor

# Создание процессора данных
processor = DataProcessor()

# Обработка данных
processed_data = processor.process(
    data=data,
    target_column='quantity',
    feature_columns=['price', 'promotion'],
    handle_missing='interpolate',
    normalize=True
)
```

### 2. Валидация данных

```python
from src.data.validators import DataValidator

# Создание валидатора
validator = DataValidator()

# Проверка данных
is_valid, errors = validator.validate(
    data=data,
    schema={
        'date': 'datetime',
        'item_id': 'string',
        'quantity': 'numeric',
        'price': 'numeric'
    }
)

# Обработка ошибок
if not is_valid:
    print("Ошибки в данных:")
    for error in errors:
        print(f"- {error}")
```

### 3. Агрегация данных

```python
# Агрегация по дням
daily_data = data.groupby('date').agg({
    'quantity': 'sum',
    'price': 'mean',
    'promotion': 'max'
}).reset_index()

# Агрегация по товарам
item_data = data.groupby('item_id').agg({
    'quantity': ['sum', 'mean', 'std'],
    'price': ['mean', 'min', 'max']
}).reset_index()
```

## Примеры тестирования

### 1. Модульные тесты

```python
import pytest
from src.core.forecaster import DemandForecaster

def test_forecaster_initialization():
    forecaster = DemandForecaster(
        model=RandomForestModel(),
        config_path='config/local.yaml'
    )
    assert forecaster is not None
    assert forecaster.model is not None

def test_forecaster_prediction():
    forecaster = DemandForecaster(
        model=RandomForestModel(),
        config_path='config/local.yaml'
    )
    forecast = forecaster.predict(days_ahead=30)
    assert len(forecast) == 30
    assert all(forecast >= 0)
```

### 2. Интеграционные тесты

```python
def test_api_integration():
    client = APIClient(
        base_url="http://localhost:8000",
        api_key="test_api_key"
    )
    
    # Тест обучения модели
    response = client.train_model(
        data=test_data,
        model_config=test_config
    )
    assert response.status_code == 200
    
    # Тест получения прогноза
    forecast = client.get_forecast(
        item_id="test_item",
        days_ahead=30
    )
    assert forecast is not None
    assert len(forecast) == 30
```

### 3. Тесты производительности

```python
def test_prediction_performance():
    forecaster = DemandForecaster(
        model=RandomForestModel(),
        config_path='config/local.yaml'
    )
    
    # Измерение времени выполнения
    start_time = time.time()
    forecast = forecaster.predict(days_ahead=30)
    end_time = time.time()
    
    # Проверка времени выполнения
    execution_time = end_time - start_time
    assert execution_time < 1.0  # должно выполняться менее 1 секунды
```

## Поддержка

Если у вас возникли вопросы по примерам или нужна дополнительная помощь:
1. Ознакомьтесь с [документацией API](api/README.md)
2. Проверьте [руководство по использованию](usage.md)
3. Создайте issue на [GitHub](https://github.com/your-username/demand-genius/issues) 