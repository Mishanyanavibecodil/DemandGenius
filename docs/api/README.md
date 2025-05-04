# Документация API DemandGenius

## Обзор

DemandGenius предоставляет REST API для взаимодействия с моделями прогнозирования. API построен с использованием FastAPI и предоставляет эндпоинты для обучения моделей, создания прогнозов и оценки производительности моделей.

## Базовый URL

```
http://localhost:8000
```

## Аутентификация

Все эндпоинты API требуют аутентификации с использованием API-ключей. Включите API-ключ в заголовок запроса:

```
X-API-Key: ваш-api-ключ
```

## Эндпоинты

### 1. Обучение модели

Обучение новой модели прогнозирования с использованием исторических данных.

**Эндпоинт:** `POST /train`

**Тело запроса:**
```json
{
    "data": {
        "date": ["2024-01-01", "2024-01-02", ...],
        "item_id": ["A", "A", ...],
        "quantity": [100, 120, ...]
    },
    "config": {
        "model_type": "random_forest",
        "params": {
            "n_estimators": 100,
            "max_depth": 10
        }
    }
}
```

**Ответ:**
```json
{
    "success": true,
    "message": "Модель успешно обучена",
    "model_id": "model_123",
    "metrics": {
        "mae": 10.5,
        "rmse": 15.2,
        "mape": 12.3
    }
}
```

### 2. Создание прогноза

Получение прогнозов спроса для конкретного товара.

**Эндпоинт:** `POST /predict`

**Тело запроса:**
```json
{
    "item_id": "A",
    "days_ahead": 30,
    "confidence_interval": 0.95
}
```

**Ответ:**
```json
{
    "success": true,
    "forecast": [
        {
            "date": "2024-02-01",
            "forecast": 150,
            "lower_bound": 120,
            "upper_bound": 180
        },
        ...
    ]
}
```

### 3. Оценка модели

Оценка производительности модели с использованием тестовых данных.

**Эндпоинт:** `POST /evaluate`

**Тело запроса:**
```json
{
    "data": {
        "date": ["2024-02-01", "2024-02-02", ...],
        "item_id": ["A", "A", ...],
        "quantity": [140, 160, ...]
    }
}
```

**Ответ:**
```json
{
    "success": true,
    "metrics": {
        "mae": 11.2,
        "rmse": 16.5,
        "mape": 13.1
    }
}
```

### 4. Получение информации о модели

Получение информации о текущей модели.

**Эндпоинт:** `GET /model`

**Ответ:**
```json
{
    "success": true,
    "model_info": {
        "model_type": "random_forest",
        "features": ["date", "item_id", "quantity"],
        "last_training_date": "2024-01-31T12:00:00",
        "performance_metrics": {
            "mae": 10.5,
            "rmse": 15.2,
            "mape": 12.3
        }
    }
}
```

### 5. Обновление модели

Обновление конфигурации модели.

**Эндпоинт:** `PUT /model`

**Тело запроса:**
```json
{
    "config": {
        "model_type": "random_forest",
        "params": {
            "n_estimators": 200,
            "max_depth": 15
        }
    }
}
```

**Ответ:**
```json
{
    "success": true,
    "message": "Конфигурация модели успешно обновлена"
}
```

## Обработка ошибок

Все эндпоинты возвращают ответы об ошибках в следующем формате:

```json
{
    "success": false,
    "error": {
        "code": "КОД_ОШИБКИ",
        "message": "Описание ошибки"
    }
}
```

Распространенные коды ошибок:
- `INVALID_REQUEST`: Неверные параметры запроса
- `AUTHENTICATION_ERROR`: Ошибка аутентификации
- `MODEL_NOT_TRAINED`: Модель требует обучения
- `DATA_VALIDATION_ERROR`: Неверный формат данных
- `INTERNAL_ERROR`: Внутренняя ошибка сервера

## Ограничение запросов

API запросы ограничены:
- 100 запросов в минуту для аутентифицированных пользователей
- 10 запросов в минуту для неаутентифицированных пользователей

## Примеры использования

### Python

```python
import requests
import json

API_KEY = "ваш-api-ключ"
BASE_URL = "http://localhost:8000"

headers = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

# Обучение модели
train_data = {
    "data": {
        "date": ["2024-01-01", "2024-01-02"],
        "item_id": ["A", "A"],
        "quantity": [100, 120]
    },
    "config": {
        "model_type": "random_forest",
        "params": {
            "n_estimators": 100,
            "max_depth": 10
        }
    }
}

response = requests.post(
    f"{BASE_URL}/train",
    headers=headers,
    json=train_data
)

# Создание прогноза
prediction_data = {
    "item_id": "A",
    "days_ahead": 30
}

response = requests.post(
    f"{BASE_URL}/predict",
    headers=headers,
    json=prediction_data
)
```

### cURL

```bash
# Обучение модели
curl -X POST http://localhost:8000/train \
    -H "X-API-Key: ваш-api-ключ" \
    -H "Content-Type: application/json" \
    -d '{
        "data": {
            "date": ["2024-01-01", "2024-01-02"],
            "item_id": ["A", "A"],
            "quantity": [100, 120]
        },
        "config": {
            "model_type": "random_forest",
            "params": {
                "n_estimators": 100,
                "max_depth": 10
            }
        }
    }'

# Создание прогноза
curl -X POST http://localhost:8000/predict \
    -H "X-API-Key: ваш-api-ключ" \
    -H "Content-Type: application/json" \
    -d '{
        "item_id": "A",
        "days_ahead": 30
    }'
```

## Рекомендации по использованию

1. **Обработка ошибок**
   - Всегда проверяйте поле `success` в ответах
   - Обрабатывайте ошибки ограничения запросов
   - Реализуйте логику повторных попыток для временных ошибок

2. **Валидация данных**
   - Проверяйте данные перед отправкой в API
   - Убедитесь, что даты в формате ISO (YYYY-MM-DD)
   - Проверяйте отсутствующие или неверные значения

3. **Производительность**
   - Используйте пакетные эндпоинты для нескольких товаров
   - Кэшируйте результаты моделей при необходимости
   - Отслеживайте время ответа API

4. **Безопасность**
   - Храните API-ключи в безопасности
   - Используйте HTTPS в продакшене
   - Регулярно обновляйте API-ключи

## Поддержка

Для получения поддержки по API, пожалуйста, обращайтесь:
- Email: support@example.com
- Документация: https://docs.example.com
- GitHub Issues: https://github.com/your-username/demand-genius/issues 