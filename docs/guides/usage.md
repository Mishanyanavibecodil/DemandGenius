# Руководство по использованию DemandGenius

## Начало работы

### 1. Импорт и инициализация

```python
from src.core.forecaster import DemandForecaster
from src.core.models import RandomForestModel

# Создание экземпляра прогнозировщика
forecaster = DemandForecaster(
    model=RandomForestModel(),
    config_path='config/local.yaml'
)
```

### 2. Загрузка данных

```python
import pandas as pd

# Загрузка исторических данных
data = pd.read_csv('data/raw/historical_data.csv')

# Предобработка данных
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values('date')
```

### 3. Обучение модели

```python
# Обучение модели
forecaster.fit(
    data=data,
    target_column='quantity',
    feature_columns=['price', 'promotion', 'season']
)

# Сохранение модели
forecaster.save_model('models/forecast_model.pkl')
```

## Создание прогнозов

### 1. Базовый прогноз

```python
# Создание прогноза на 30 дней вперед
forecast = forecaster.predict(
    days_ahead=30,
    confidence_interval=0.95
)

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
```

### 3. Пакетный прогноз

```python
# Создание прогнозов для нескольких товаров
items = ['item1', 'item2', 'item3']
forecasts = forecaster.predict_batch(
    items=items,
    days_ahead=30
)
```

## Оценка модели

### 1. Метрики производительности

```python
# Оценка на тестовых данных
metrics = forecaster.evaluate(
    test_data=test_data,
    metrics=['mae', 'rmse', 'mape']
)

print(f"MAE: {metrics['mae']:.2f}")
print(f"RMSE: {metrics['rmse']:.2f}")
print(f"MAPE: {metrics['mape']:.2f}%")
```

### 2. Валидация прогнозов

```python
# Кросс-валидация
cv_results = forecaster.cross_validate(
    data=data,
    n_splits=5,
    metrics=['mae', 'rmse']
)

# Анализ результатов
forecaster.plot_cv_results(cv_results)
```

## Работа с API

### 1. Обучение модели через API

```python
import requests

# Настройка API
API_URL = "http://localhost:8000"
API_KEY = "your_api_key"

headers = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

# Отправка данных для обучения
response = requests.post(
    f"{API_URL}/train",
    headers=headers,
    json={
        "data": data.to_dict(),
        "config": {
            "model_type": "random_forest",
            "params": {
                "n_estimators": 100,
                "max_depth": 10
            }
        }
    }
)
```

### 2. Получение прогнозов через API

```python
# Запрос прогноза
response = requests.post(
    f"{API_URL}/predict",
    headers=headers,
    json={
        "item_id": "item1",
        "days_ahead": 30
    }
)

# Обработка ответа
forecast = response.json()['forecast']
```

## Мониторинг и логирование

### 1. Настройка логирования

```python
from src.monitoring.logger import setup_logger

# Настройка логгера
logger = setup_logger(
    name='forecaster',
    log_file='logs/app/forecaster.log'
)

# Использование логгера
logger.info("Начало обучения модели")
logger.error("Ошибка при создании прогноза")
```

### 2. Мониторинг производительности

```python
from src.monitoring.metrics import PerformanceMonitor

# Создание монитора
monitor = PerformanceMonitor()

# Отслеживание метрик
monitor.track_prediction_time(start_time, end_time)
monitor.track_model_accuracy(metrics)
```

## Оптимизация

### 1. Настройка гиперпараметров

```python
from src.core.optimizer import HyperparameterOptimizer

# Создание оптимизатора
optimizer = HyperparameterOptimizer(
    model=RandomForestModel(),
    param_grid={
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15]
    }
)

# Поиск оптимальных параметров
best_params = optimizer.optimize(
    data=data,
    target_column='quantity',
    cv=5
)
```

### 2. Кэширование результатов

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
```

## Безопасность

### 1. Аутентификация

```python
from src.security.auth import APIAuthenticator

# Создание аутентификатора
auth = APIAuthenticator(
    api_key='your_api_key',
    jwt_secret='your_jwt_secret'
)

# Проверка токена
is_valid = auth.validate_token(token)
```

### 2. Шифрование данных

```python
from src.security.encryption import DataEncryptor

# Создание шифровальщика
encryptor = DataEncryptor(
    key='your_encryption_key'
)

# Шифрование данных
encrypted_data = encryptor.encrypt(sensitive_data)
```

## Рекомендации по использованию

1. **Подготовка данных**
   - Очищайте данные от выбросов
   - Обрабатывайте пропущенные значения
   - Нормализуйте числовые признаки

2. **Выбор модели**
   - Используйте RandomForest для базовых прогнозов
   - Применяйте LSTM для временных рядов
   - Экспериментируйте с различными моделями

3. **Оптимизация производительности**
   - Используйте кэширование для частых запросов
   - Применяйте пакетную обработку
   - Оптимизируйте гиперпараметры

4. **Мониторинг**
   - Отслеживайте точность прогнозов
   - Мониторьте время ответа
   - Анализируйте ошибки

## Примеры использования

### 1. Полный цикл прогнозирования

```python
# Инициализация
forecaster = DemandForecaster(
    model=RandomForestModel(),
    config_path='config/local.yaml'
)

# Загрузка и подготовка данных
data = pd.read_csv('data/raw/historical_data.csv')
data['date'] = pd.to_datetime(data['date'])

# Обучение модели
forecaster.fit(
    data=data,
    target_column='quantity',
    feature_columns=['price', 'promotion']
)

# Создание прогноза
forecast = forecaster.predict(
    days_ahead=30,
    confidence_interval=0.95
)

# Визуализация результатов
forecaster.plot_forecast(forecast)

# Оценка модели
metrics = forecaster.evaluate(
    test_data=test_data,
    metrics=['mae', 'rmse', 'mape']
)
```

### 2. Работа с API

```python
# Настройка клиента
client = APIClient(
    base_url="http://localhost:8000",
    api_key="your_api_key"
)

# Обучение модели
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

# Получение прогноза
forecast = client.get_forecast(
    item_id="item1",
    days_ahead=30
)
```

## Поддержка

Если у вас возникли вопросы или проблемы:
1. Ознакомьтесь с [документацией API](api/README.md)
2. Проверьте [руководство по установке](installation.md)
3. Создайте issue на [GitHub](https://github.com/your-username/demand-genius/issues) 