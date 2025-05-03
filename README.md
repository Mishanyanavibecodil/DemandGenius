# 🛍️ Система прогнозирования спроса на товары

## Что это такое? 🤔

Это программа, которая помогает магазинам понять, сколько товаров им нужно заказать. Она смотрит на прошлые продажи и говорит: "Эй, похоже, вам понадобится больше этого товара!"

## Как это работает? 🎯

1. **Сбор информации** 📊
   - Программа смотрит, сколько товаров продавалось раньше
   - Запоминает, когда были заказы
   - Учитывает, сколько товаров сейчас на складе

2. **Умные расчеты** 🧮
   - Считает, как менялись продажи со временем
   - Учитывает сезонность (например, больше продаж перед праздниками)
   - Помнит, когда товаров не хватало

3. **Советы по заказам** 📝
   - Говорит, сколько товаров нужно заказать
   - Учитывает время доставки
   - Следит, чтобы не заказать слишком много или слишком мало

## Установка и настройка 🛠️

1. **Установка Python** 🐍
   ```bash
   # Скачайте Python с официального сайта
   # https://www.python.org/downloads/
   ```

2. **Установка зависимостей** 📦
   ```bash
   pip install pandas numpy matplotlib google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client python-dotenv
   ```

3. **Настройка Google Sheets API** 🔑
   1. Перейдите в [Google Cloud Console](https://console.cloud.google.com/)
   2. Создайте новый проект
   3. Включите Google Sheets API
   4. Создайте учетные данные (OAuth 2.0)
   5. Скачайте JSON-файл с учетными данными

4. **Настройка файлов конфигурации** ⚙️
   
   Создайте файл `.env`:
   ```env
   GOOGLE_SHEETS_CREDENTIALS=path/to/your/credentials.json
   SPREADSHEET_ID=your_spreadsheet_id
   SALES_SHEET_NAME=Sales
   ORDERS_SHEET_NAME=Orders
   ```

   Создайте файл `config.json`:
   ```json
   {
       "lead_time": 7,
       "order_period": 10,
       "alpha": 0.3,
       "lot_size": 10,
       "max_growth_multiplier": 3.0,
       "check_trend_days": 30,
       "growth_factor": 1.2,
       "lost_demand_factor": {
           "food": 0.1,
           "drinks": 0.15,
           "other": 0.05
       },
       "seasonality_coeffs": {
           "1": 0.8,  // Январь
           "2": 0.9,  // Февраль
           "3": 1.0,  // Март
           "4": 1.1,  // Апрель
           "5": 1.2,  // Май
           "6": 1.3,  // Июнь
           "7": 1.2,  // Июль
           "8": 1.1,  // Август
           "9": 1.0,  // Сентябрь
           "10": 1.1, // Октябрь
           "11": 1.2, // Ноябрь
           "12": 1.3  // Декабрь
       }
   }
   ```

## Структура Google Sheets 📊

1. **Лист "Sales" (Продажи)**
   ```
   | date       | item_id | quantity |
   |------------|---------|----------|
   | 2024-01-01 | milk_1  | 10       |
   | 2024-01-02 | milk_1  | 15       |
   ```

2. **Лист "Orders" (Заказы)**
   ```
   | order_id | item_id | ordered_qty | sold_qty |
   |----------|---------|-------------|----------|
   | 1        | milk_1  | 100         | 80       |
   | 2        | milk_1  | 50          | 45       |
   ```

## Как использовать? 🚀

1. **Подготовка данных** 📋
   ```python
   from google.oauth2.credentials import Credentials
   from google_auth_oauthlib.flow import InstalledAppFlow
   from google.auth.transport.requests import Request
   import pandas as pd
   import json
   from dotenv import load_dotenv
   import os

   # Загрузка переменных окружения
   load_dotenv()

   # Загрузка конфигурации
   with open('config.json', 'r') as f:
       config = json.load(f)

   # Настройка Google Sheets API
   SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
   creds = None
   if os.path.exists('token.json'):
       creds = Credentials.from_authorized_user_file('token.json', SCOPES)
   if not creds or not creds.valid:
       if creds and creds.expired and creds.refresh_token:
           creds.refresh(Request())
       else:
           flow = InstalledAppFlow.from_client_secrets_file(
               os.getenv('GOOGLE_SHEETS_CREDENTIALS'), SCOPES)
           creds = flow.run_local_server(port=0)
       with open('token.json', 'w') as token:
           token.write(creds.to_json())
   ```

2. **Загрузка данных** 📥
   ```python
   from googleapiclient.discovery import build
   
   # Подключение к Google Sheets
   service = build('sheets', 'v4', credentials=creds)
   sheet = service.spreadsheets()
   
   # Получение данных о продажах
   sales_range = f"{os.getenv('SALES_SHEET_NAME')}!A:C"
   sales_result = sheet.values().get(
       spreadsheetId=os.getenv('SPREADSHEET_ID'),
       range=sales_range
   ).execute()
   sales_values = sales_result.get('values', [])
   
   # Преобразование в DataFrame
   sales_history = pd.DataFrame(sales_values[1:], columns=sales_values[0])
   ```

3. **Запуск прогноза** 🎯
   ```python
   # Создаем прогнозировщик
   forecaster = DemandForecaster(config)
   
   # Загружаем данные
   forecaster.fit(sales_history, past_orders)
   
   # Получаем прогноз
   forecast = forecaster.predict('молоко', days_ahead=30)
   ```

## Что умеет программа? ✨

- 📈 Показывает график продаж и прогноз
- 🎯 Считает, сколько товаров нужно заказать
- 📅 Учитывает сезонность продаж
- ⚠️ Помнит, когда товаров не хватало
- 🔄 Учитывает тренды роста или падения продаж
- 🔄 Автоматически обновляет данные из Google Sheets
- 📊 Экспортирует результаты в Google Sheets

## Примеры использования 📝

### 1. Простой прогноз
```python
# Сколько товаров продастся в следующие 30 дней?
forecast = forecaster.predict('молоко', days_ahead=30)
```

### 2. Расчет заказа
```python
# Сколько нужно заказать, если на складе 100 штук?
order = forecaster.calculate_order('молоко', current_stock=100, incoming_stock=0)
```

### 3. Визуализация
```python
# Показать график продаж и прогноза
forecaster.visualize_forecast('молоко', days_ahead=30)
```

### 4. Экспорт в Google Sheets
```python
# Экспорт прогноза в Google Sheets
def export_to_sheets(forecast, sheet_name):
    values = [[date.strftime('%Y-%m-%d'), value] 
              for date, value in forecast.items()]
    body = {'values': values}
    sheet.values().update(
        spreadsheetId=os.getenv('SPREADSHEET_ID'),
        range=f"{sheet_name}!A:B",
        valueInputOption='RAW',
        body=body
    ).execute()
```

## Важные моменты ⚠️

1. **Данные должны быть качественными** 📊
   - Все даты должны быть правильными
   - Количество товаров должно быть точным
   - Не должно быть пропусков в данных
   - Формат дат должен быть YYYY-MM-DD

2. **Настройки важны** ⚙️
   - Время доставки должно быть реальным
   - Размер партии должен быть разумным
   - Коэффициенты роста не должны быть слишком большими
   - Сезонные коэффициенты должны отражать реальность

3. **Регулярное обновление** 🔄
   - Данные нужно обновлять регулярно
   - Настройки можно менять под ситуацию
   - Прогнозы лучше проверять на практике
   - Токены Google API нужно обновлять при истечении срока

4. **Безопасность** 🔒
   - Храните credentials.json в безопасном месте
   - Не публикуйте токены в публичном доступе
   - Используйте .env для хранения чувствительных данных
   - Регулярно обновляйте зависимости

## Если что-то пошло не так 🤔

1. **Проверьте данные** 📋
   - Все ли даты правильные?
   - Нет ли пропусков?
   - Правильные ли форматы?
   - Есть ли доступ к Google Sheets?

2. **Посмотрите настройки** ⚙️
   - Не слишком ли большие коэффициенты?
   - Правильное ли время доставки?
   - Реальный ли размер партии?
   - Правильные ли ID таблиц?

3. **Проверьте логи** 📝
   - Программа пишет, что не так
   - Можно понять, где ошибка
   - Есть подсказки по исправлению
   - Проверьте ошибки Google API

4. **Проверьте доступ** 🔑
   - Работает ли Google Sheets API?
   - Не истек ли срок токена?
   - Правильные ли права доступа?
   - Есть ли доступ к интернету?

## Нужна помощь? 🆘

Если что-то непонятно или не работает:
1. Посмотрите на сообщения об ошибках
2. Проверьте, все ли данные правильные
3. Попробуйте изменить настройки
4. Проверьте доступ к Google Sheets
5. Если не помогло - обратитесь к разработчикам

## Дополнительные ресурсы 📚

1. [Google Sheets API Documentation](https://developers.google.com/sheets/api)
2. [Python Google API Client Library](https://github.com/googleapis/google-api-python-client)
3. [Pandas Documentation](https://pandas.pydata.org/docs/)
4. [Matplotlib Documentation](https://matplotlib.org/stable/contents.html) 