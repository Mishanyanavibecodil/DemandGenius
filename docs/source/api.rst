API Reference
============

DemandForecaster
--------------

.. autoclass:: demand_forecasting.DemandForecaster
   :members:
   :undoc-members:
   :show-inheritance:

Models
------

BaseForecastingModel
~~~~~~~~~~~~~~~~~~

.. autoclass:: models.BaseForecastingModel
   :members:
   :undoc-members:
   :show-inheritance:

ExponentialSmoothingModel
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: models.ExponentialSmoothingModel
   :members:
   :undoc-members:
   :show-inheritance:

LinearRegressionModel
~~~~~~~~~~~~~~~~~~~

.. autoclass:: models.LinearRegressionModel
   :members:
   :undoc-members:
   :show-inheritance:

RandomForestModel
~~~~~~~~~~~~~~~

.. autoclass:: models.RandomForestModel
   :members:
   :undoc-members:
   :show-inheritance:

Metrics
-------

ForecastMetrics
~~~~~~~~~~~~~

.. autoclass:: metrics.ForecastMetrics
   :members:
   :undoc-members:
   :show-inheritance:

Optimizer
--------

HyperparameterOptimizer
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: optimizer.HyperparameterOptimizer
   :members:
   :undoc-members:
   :show-inheritance:

ModelSelector
~~~~~~~~~~~

.. autoclass:: optimizer.ModelSelector
   :members:
   :undoc-members:
   :show-inheritance:

External Factors
--------------

ExternalFactorsProcessor
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: external_factors.ExternalFactorsProcessor
   :members:
   :undoc-members:
   :show-inheritance:

Monitoring
---------

MonitoringSystem
~~~~~~~~~~~~~

.. autoclass:: monitoring.MonitoringSystem
   :members:
   :undoc-members:
   :show-inheritance:

ModelPerformanceMonitor
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: monitoring.ModelPerformanceMonitor
   :members:
   :undoc-members:
   :show-inheritance:

Exceptions
---------

.. automodule:: exceptions
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-----------

The configuration dictionary for DemandForecaster should contain the following keys:

* ``lead_time`` (int): Время выполнения заказа в днях
* ``order_period`` (int): Период заказа в днях
* ``alpha`` (float): Параметр сглаживания (0 < alpha < 1)
* ``cache_size`` (int): Максимальный размер кэша
* ``cache_ttl`` (int): Время жизни кэша в секундах
* ``n_processes`` (int): Количество процессов для параллельных вычислений

Example:

.. code-block:: python

   config = {
       'lead_time': 7,
       'order_period': 30,
       'alpha': 0.3,
       'cache_size': 1000,
       'cache_ttl': 3600,
       'n_processes': 4
   }

Data Format
----------

Input Data
~~~~~~~~~

The input data should be a pandas DataFrame with the following columns:

* ``item_id``: Идентификатор товара
* ``date``: Дата продажи
* ``quantity``: Количество проданных единиц
* ``price``: Цена за единицу (опционально)

Example:

.. code-block:: python

   import pandas as pd
   
   data = pd.DataFrame({
       'item_id': ['A', 'A', 'B', 'B'],
       'date': ['2024-01-01', '2024-01-02', '2024-01-01', '2024-01-02'],
       'quantity': [10, 15, 20, 25],
       'price': [100, 100, 200, 200]
   })

Output Data
~~~~~~~~~

The output data format depends on the method:

1. predict() returns a pandas Series with dates as index and predicted quantities as values:

.. code-block:: python

   forecast = forecaster.predict(item_id='A', days_ahead=30)
   # Returns: pd.Series with dates and quantities

2. predict_batch() returns a dictionary with item_ids as keys and forecasts as values:

.. code-block:: python

   forecasts = forecaster.predict_batch(item_ids=['A', 'B'], days_ahead=30)
   # Returns: {'A': pd.Series, 'B': pd.Series}

3. calculate_order() returns a dictionary with order details:

.. code-block:: python

   order = forecaster.calculate_order(item_id='A', current_stock=100)
   # Returns: {
   #     'item_id': 'A',
   #     'order_quantity': 50,
   #     'order_date': '2024-01-01',
   #     'delivery_date': '2024-01-08'
   # }

4. calculate_orders_batch() returns a list of order dictionaries:

.. code-block:: python

   orders = forecaster.calculate_orders_batch([
       {'item_id': 'A', 'current_stock': 100},
       {'item_id': 'B', 'current_stock': 150}
   ])
   # Returns: [order_dict1, order_dict2] 