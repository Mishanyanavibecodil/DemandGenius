# DemandGenius

Система прогнозирования спроса - это мощный инструмент для анализа и прогнозирования спроса на товары и услуги. Система использует современные методы машинного обучения для создания точных прогнозов с учетом различных факторов.

## Основные возможности

- 📊 Прогнозирование спроса с использованием различных моделей машинного обучения
- 🔄 Поддержка временных рядов и регрессионного анализа
- 📈 Учет сезонности и трендов
- 🔍 Анализ влияния внешних факторов
- 📱 REST API для интеграции с другими системами
- 🔒 Безопасность и аутентификация
- 📝 Подробное логирование и мониторинг

## Установка

```bash
# Клонирование репозитория
git clone https://github.com/your-username/demand-genius.git
cd demand-genius

# Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # Linux/MacOS
.\venv\Scripts\activate   # Windows

# Установка зависимостей
pip install -r requirements.txt
```

Подробные инструкции по установке можно найти в [руководстве по установке](docs/guides/installation.md).

## Быстрый старт

```python
from src.core.forecaster import DemandForecaster
from src.core.models import RandomForestModel

# Создание прогнозировщика
forecaster = DemandForecaster(
    model=RandomForestModel(),
    config_path='config/local.yaml'
)

# Загрузка данных
data = pd.read_csv('data/raw/historical_data.csv')

# Обучение модели
forecaster.fit(
    data=data,
    target_column='quantity',
    feature_columns=['price', 'promotion']
)

# Создание прогноза
forecast = forecaster.predict(days_ahead=30)
```

Более подробные примеры использования можно найти в [руководстве по использованию](docs/guides/usage.md).

## Документация

- [Руководство по установке](docs/guides/installation.md)
- [Руководство по использованию](docs/guides/usage.md)
- [Документация API](docs/api/README.md)
- [Примеры использования](docs/examples/README.md)

## Структура проекта

```
demand-genius/
├── src/
│   ├── core/           # Основные компоненты системы
│   ├── api/            # API endpoints
│   ├── monitoring/     # Мониторинг и логирование
│   └── security/       # Безопасность и аутентификация
├── tests/              # Тесты
├── docs/               # Документация
├── config/             # Конфигурационные файлы
├── data/               # Данные
│   ├── raw/           # Исходные данные
│   └── processed/     # Обработанные данные
├── notebooks/          # Jupyter notebooks
└── scripts/            # Вспомогательные скрипты
```

## Требования

- Python 3.8+
- pandas
- numpy
- scikit-learn
- fastapi
- uvicorn
- pydantic

Полный список зависимостей можно найти в [requirements.txt](requirements.txt).

## Разработка

### Установка зависимостей для разработки

```bash
pip install -r requirements-dev.txt
```

### Запуск тестов

```bash
pytest tests/
```

### Проверка стиля кода

```bash
flake8 src/
black src/
```

## Вклад в проект

Мы приветствуем вклад в развитие проекта! Пожалуйста, ознакомьтесь с [руководством по внесению изменений](CONTRIBUTING.md) для получения дополнительной информации.

## Лицензия

Этот проект распространяется под лицензией MIT. Подробности можно найти в файле [LICENSE](LICENSE).

## Поддержка

Если у вас возникли вопросы или проблемы:

1. Ознакомьтесь с [документацией](docs/)
2. Создайте issue на [GitHub](https://github.com/your-username/demand-genius/issues)
3. Напишите нам на support@example.com

## Авторы

- Ваше Имя - [GitHub](https://github.com/your-username)
- Другие участники - [GitHub](https://github.com/your-username/demand-genius/graphs/contributors)

## Благодарности

- [scikit-learn](https://scikit-learn.org/)
- [pandas](https://pandas.pydata.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [pydantic](https://pydantic-docs.helpmanual.io/) 