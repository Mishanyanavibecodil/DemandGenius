Monitoring Guide
==============

Overview
--------

Система мониторинга предоставляет инструменты для отслеживания производительности и состояния системы прогнозирования спроса. 
Она включает в себя:

* Мониторинг метрик производительности моделей
* Отслеживание системных ресурсов
* Мониторинг кэша
* Сбор метрик ошибок
* Генерацию отчетов

Setup
-----

1. Установка зависимостей:

   .. code-block:: bash

      pip install prometheus-client psutil

2. Инициализация системы мониторинга:

   .. code-block:: python

      from monitoring import MonitoringSystem
      
      monitoring = MonitoringSystem(port=8000)

Available Metrics
---------------

1. Метрики прогнозирования:

   * ``forecast_accuracy``: Точность прогноза по различным метрикам
   * ``prediction_latency_seconds``: Время выполнения прогноза
   * ``model_errors_total``: Количество ошибок моделей

2. Метрики кэша:

   * ``cache_hits_total``: Количество попаданий в кэш
   * ``cache_misses_total``: Количество промахов кэша

3. Системные метрики:

   * ``cpu_usage_percent``: Использование CPU
   * ``memory_usage_bytes``: Использование памяти
   * ``disk_usage_percent``: Использование диска

Usage
-----

1. Мониторинг прогнозирования:

   .. code-block:: python

      from monitoring import ModelPerformanceMonitor
      
      # Создание монитора производительности
      performance_monitor = ModelPerformanceMonitor(monitoring)
      
      # Мониторинг прогнозирования
      forecast = performance_monitor.monitor_prediction(
          model='random_forest',
          func=model.predict,
          days_ahead=30
      )
      
      # Мониторинг точности
      performance_monitor.monitor_accuracy(
          model='random_forest',
          y_true=actual_values,
          y_pred=forecast
      )

2. Мониторинг кэша:

   .. code-block:: python

      # Запись события кэша
      monitoring.record_cache_event(is_hit=True)  # для попадания
      monitoring.record_cache_event(is_hit=False)  # для промаха

3. Получение системных метрик:

   .. code-block:: python

      # Получение текущих метрик
      metrics = monitoring.get_system_metrics()
      
      # Генерация отчета
      report = monitoring.generate_report()

Prometheus Integration
--------------------

1. Настройка Prometheus:

   .. code-block:: yaml

      scrape_configs:
        - job_name: 'demand_forecasting'
          static_configs:
            - targets: ['localhost:8000']

2. Доступ к метрикам:

   * Метрики доступны по адресу: http://localhost:8000/metrics
   * Формат метрик соответствует стандарту Prometheus

3. Примеры запросов PromQL:

   * Средняя точность прогноза:
     .. code-block::

        avg(forecast_accuracy)

   * 95-й перцентиль времени выполнения:
     .. code-block::

        histogram_quantile(0.95, sum(rate(prediction_latency_seconds_bucket[5m])) by (le))

   * Количество ошибок по типам:
     .. code-block::

        sum(model_errors_total) by (error_type)

Grafana Dashboard
---------------

1. Импорт дашборда:

   * Создайте новый дашборд в Grafana
   * Импортируйте JSON-конфигурацию дашборда
   * Настройте источник данных Prometheus

2. Доступные панели:

   * Обзор системы:
     * Использование CPU, памяти и диска
     * Количество активных моделей
     * Общая статистика кэша

   * Метрики прогнозирования:
     * Точность прогнозов по моделям
     * Время выполнения прогнозов
     * Распределение ошибок

   * Кэш:
     * Hit/miss ratio
     * Размер кэша
     * Время жизни записей

Alerting
--------

1. Настройка алертов в Prometheus:

   .. code-block:: yaml

      groups:
        - name: demand_forecasting
          rules:
            - alert: HighErrorRate
              expr: rate(model_errors_total[5m]) > 0.1
              for: 5m
              labels:
                severity: warning
              annotations:
                summary: High error rate detected
                description: Error rate is {{ $value }} per second

            - alert: HighLatency
              expr: histogram_quantile(0.95, sum(rate(prediction_latency_seconds_bucket[5m])) by (le)) > 2
              for: 5m
              labels:
                severity: warning
              annotations:
                summary: High prediction latency
                description: 95th percentile latency is {{ $value }} seconds

2. Настройка уведомлений:

   * Email
   * Slack
   * Telegram
   * Webhook

Best Practices
-------------

1. Мониторинг:

   * Регулярно проверяйте метрики
   * Настройте алерты для критических показателей
   * Ведите историю метрик для анализа трендов

2. Оптимизация:

   * Используйте метрики для выявления узких мест
   * Оптимизируйте параметры кэша на основе статистики
   * Настройте автоматическое масштабирование при необходимости

3. Безопасность:

   * Ограничьте доступ к метрикам
   * Используйте HTTPS для API
   * Регулярно обновляйте зависимости

4. Документация:

   * Документируйте все метрики
   * Ведите журнал изменений
   * Обновляйте дашборды при изменении системы 