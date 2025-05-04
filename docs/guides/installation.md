# Руководство по установке DemandGenius

## Требования

- Python 3.8 или выше
- pip (менеджер пакетов Python)
- Git (опционально, для клонирования репозитория)

## Установка

### 1. Клонирование репозитория (опционально)

```bash
git clone https://github.com/your-username/demand-genius.git
cd demand-genius
```

### 2. Создание виртуального окружения

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/MacOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Установка зависимостей

```bash
# Установка основных зависимостей
pip install -r requirements.txt

# Установка зависимостей для разработки (опционально)
pip install -r requirements-dev.txt
```

### 4. Настройка конфигурации

1. Скопируйте файл конфигурации по умолчанию:
```bash
cp config/default.yaml config/local.yaml
```

2. Отредактируйте `config/local.yaml` в соответствии с вашими настройками:
```yaml
database:
  host: localhost
  port: 5432
  name: demand_forecasting
  user: your_username
  password: your_password

api:
  host: 0.0.0.0
  port: 8000
  debug: false

security:
  api_key: your_api_key
  jwt_secret: your_jwt_secret
```

### 5. Инициализация базы данных

```bash
python scripts/init_db.py
```

## Проверка установки

### 1. Запуск тестов

```bash
pytest tests/
```

### 2. Запуск сервера разработки

```bash
uvicorn src.api.main:app --reload
```

### 3. Проверка API

Откройте в браузере:
```
http://localhost:8000/docs
```

## Установка в продакшене

### 1. Настройка переменных окружения

```bash
# Windows
set DATABASE_URL=postgresql://user:password@localhost:5432/dbname
set API_KEY=your_api_key
set JWT_SECRET=your_jwt_secret

# Linux/MacOS
export DATABASE_URL=postgresql://user:password@localhost:5432/dbname
export API_KEY=your_api_key
export JWT_SECRET=your_jwt_secret
```

### 2. Запуск с использованием Gunicorn

```bash
gunicorn src.api.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

### 3. Настройка Nginx (опционально)

```nginx
server {
    listen 80;
    server_name your_domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Устранение неполадок

### Общие проблемы

1. **Ошибка подключения к базе данных**
   - Проверьте настройки подключения в `config/local.yaml`
   - Убедитесь, что база данных запущена
   - Проверьте права доступа пользователя

2. **Ошибки зависимостей**
   - Обновите pip: `pip install --upgrade pip`
   - Переустановите зависимости: `pip install -r requirements.txt --force-reinstall`

3. **Проблемы с правами доступа**
   - Проверьте права доступа к директориям
   - Убедитесь, что у пользователя есть необходимые разрешения

### Логи

- Логи приложения: `logs/app/app.log`
- Логи ошибок: `logs/errors/error.log`
- Логи безопасности: `logs/security/security.log`

## Обновление

### 1. Получение обновлений

```bash
git pull origin main
```

### 2. Обновление зависимостей

```bash
pip install -r requirements.txt --upgrade
```

### 3. Применение миграций базы данных

```bash
python scripts/migrate_db.py
```

## Удаление

### 1. Деактивация виртуального окружения

```bash
deactivate
```

### 2. Удаление виртуального окружения

```bash
# Windows
rmdir /s /q venv

# Linux/MacOS
rm -rf venv
```

### 3. Удаление базы данных (опционально)

```bash
python scripts/drop_db.py
```

## Поддержка

Если у вас возникли проблемы с установкой, пожалуйста:
1. Проверьте [раздел устранения неполадок](#устранение-неполадок)
2. Ознакомьтесь с [документацией](https://docs.example.com)
3. Создайте issue на [GitHub](https://github.com/your-username/demand-genius/issues) 