Welcome to Demand Forecasting System's documentation!
=================================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   api
   monitoring
   contributing

Introduction
------------

Система прогнозирования спроса - это комплексное решение для предсказания будущего спроса на товары и услуги. 
Система использует различные модели машинного обучения и учитывает внешние факторы для повышения точности прогнозов.

Основные возможности:

* Поддержка различных моделей прогнозирования (экспоненциальное сглаживание, линейная регрессия, случайный лес)
* Анализ точности прогнозов с использованием различных метрик
* Автоматическая настройка параметров моделей
* Учет внешних факторов (погода, праздники)
* Система мониторинга производительности
* Кэширование результатов для оптимизации производительности

Installation
-----------

Для установки системы выполните:

.. code-block:: bash

   pip install -r requirements.txt

Quick Start
----------

Пример использования системы:

.. code-block:: python

   from demand_forecasting import DemandForecaster
   
   # Инициализация конфигурации
   config = {
       'lead_time': 7,
       'order_period': 30,
       'alpha': 0.3,
       'cache_size': 1000,
       'cache_ttl': 3600,
       'n_processes': 4
   }
   
   # Создание экземпляра прогнозировщика
   forecaster = DemandForecaster(config)
   
   # Прогнозирование спроса
   forecast = forecaster.predict(item_id='A', days_ahead=30)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 