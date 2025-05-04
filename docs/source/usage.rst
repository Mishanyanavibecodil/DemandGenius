Usage Guide
===========

Basic Usage
----------

1. Инициализация прогнозировщика:

   .. code-block:: python

      from demand_forecasting import DemandForecaster
      
      config = {
          'lead_time': 7,
          'order_period': 30,
          'alpha': 0.3,
          'cache_size': 1000,
          'cache_ttl': 3600,
          'n_processes': 4
      }
      
      forecaster = DemandForecaster(config)

2. Загрузка данных:

   .. code-block:: python

      import pandas as pd
      
      # Загрузка исторических данных
      sales_data = pd.read_csv('sales_history.csv')
      
      # Оптимизация хранения данных
      forecaster.optimize_data_storage(sales_data)

3. Прогнозирование спроса:

   .. code-block:: python

      # Прогноз для одного товара
      forecast = forecaster.predict(
          item_id='A',
          days_ahead=30
      )
      
      # Пакетное прогнозирование
      item_ids = ['A', 'B', 'C', 'D']
      forecasts = forecaster.predict_batch(
          item_ids=item_ids,
          days_ahead=30
      )

4. Расчет заказов:

   .. code-block:: python

      # Расчет для одного товара
      order = forecaster.calculate_order(
          item_id='A',
          current_stock=100
      )
      
      # Пакетный расчет
      items_data = [
          {'item_id': 'A', 'current_stock': 100},
          {'item_id': 'B', 'current_stock': 150}
      ]
      orders = forecaster.calculate_orders_batch(items_data)

Advanced Usage
-------------

1. Использование различных моделей:

   .. code-block:: python

      from models import (
          ExponentialSmoothingModel,
          LinearRegressionModel,
          RandomForestModel
      )
      
      # Создание моделей
      exp_model = ExponentialSmoothingModel()
      lin_model = LinearRegressionModel()
      rf_model = RandomForestModel()
      
      # Обучение моделей
      exp_model.fit(sales_data)
      lin_model.fit(sales_data)
      rf_model.fit(sales_data)
      
      # Прогнозирование
      exp_forecast = exp_model.predict(days_ahead=30)
      lin_forecast = lin_model.predict(days_ahead=30)
      rf_forecast = rf_model.predict(days_ahead=30)

2. Анализ точности прогнозов:

   .. code-block:: python

      from metrics import ForecastMetrics
      
      # Расчет метрик
      metrics = ForecastMetrics.calculate_metrics(
          y_true=actual_values,
          y_pred=forecast
      )
      
      # Оценка моделей
      model_metrics = ForecastMetrics.evaluate_models(
          models={
              'exponential': exp_forecast,
              'linear': lin_forecast,
              'random_forest': rf_forecast
          },
          y_true=actual_values
      )

3. Оптимизация параметров:

   .. code-block:: python

      from optimizer import HyperparameterOptimizer
      
      # Оптимизация параметров
      optimizer = HyperparameterOptimizer(
          model_class=RandomForestModel,
          training_data=sales_data
      )
      
      best_params = optimizer.optimize(
          n_trials=100,
          param_ranges={
              'n_estimators': (50, 200),
              'max_depth': (3, 10)
          }
      )

4. Учет внешних факторов:

   .. code-block:: python

      from external_factors import ExternalFactorsProcessor
      
      # Инициализация процессора внешних факторов
      processor = ExternalFactorsProcessor(
          config={
              'weather_api_key': 'your_api_key',
              'holiday_impact': {
                  'new_year': 1.5,
                  'womens_day': 1.2
              }
          }
      )
      
      # Получение данных о погоде
      weather_data = processor.get_weather_data(
          location='Moscow',
          start_date='2024-01-01',
          end_date='2024-01-31'
      )
      
      # Корректировка прогноза
      adjusted_forecast = processor.adjust_forecast(
          forecast=forecast,
          weather_data=weather_data
      )

5. Мониторинг производительности:

   .. code-block:: python

      from monitoring import MonitoringSystem, ModelPerformanceMonitor
      
      # Инициализация системы мониторинга
      monitoring = MonitoringSystem(port=8000)
      
      # Создание монитора производительности
      performance_monitor = ModelPerformanceMonitor(monitoring)
      
      # Мониторинг прогнозирования
      forecast = performance_monitor.monitor_prediction(
          model='random_forest',
          func=rf_model.predict,
          days_ahead=30
      )
      
      # Мониторинг точности
      performance_monitor.monitor_accuracy(
          model='random_forest',
          y_true=actual_values,
          y_pred=forecast
      )

Best Practices
-------------

1. Подготовка данных:
   * Проверяйте данные на наличие пропусков и выбросов
   * Нормализуйте данные при необходимости
   * Используйте оптимизацию хранения для больших наборов данных

2. Выбор модели:
   * Начните с простых моделей (экспоненциальное сглаживание)
   * Используйте перекрестную валидацию для оценки моделей
   * Учитывайте специфику данных при выборе модели

3. Оптимизация производительности:
   * Используйте кэширование для часто запрашиваемых прогнозов
   * Применяйте параллельные вычисления для больших наборов данных
   * Регулярно очищайте кэш

4. Мониторинг:
   * Регулярно проверяйте метрики производительности
   * Настройте оповещения о проблемах
   * Ведите журнал ошибок и предупреждений

5. Обновление моделей:
   * Регулярно переобучайте модели на новых данных
   * Следите за дрейфом данных
   * Обновляйте параметры моделей при необходимости 