Installation Guide
=================

Requirements
-----------

* Python 3.8 или выше
* pip (менеджер пакетов Python)
* Git (опционально, для клонирования репозитория)

Installation Steps
----------------

1. Клонируйте репозиторий (опционально):

   .. code-block:: bash

      git clone https://github.com/your-username/demand-forecasting.git
      cd demand-forecasting

2. Создайте виртуальное окружение:

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # для Linux/Mac
      venv\Scripts\activate     # для Windows

3. Установите зависимости:

   .. code-block:: bash

      pip install -r requirements.txt

4. Установите дополнительные зависимости для разработки (опционально):

   .. code-block:: bash

      pip install -r requirements-dev.txt

Configuration
------------

1. Создайте файл конфигурации:

   .. code-block:: python

      config = {
          'lead_time': 7,          # время выполнения заказа в днях
          'order_period': 30,      # период заказа в днях
          'alpha': 0.3,            # параметр сглаживания
          'cache_size': 1000,      # размер кэша
          'cache_ttl': 3600,       # время жизни кэша в секундах
          'n_processes': 4         # количество процессов для параллельных вычислений
      }

2. Настройте логирование:

   .. code-block:: python

      import logging

      logging.basicConfig(
          level=logging.INFO,
          format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
          handlers=[
              logging.FileHandler('forecasting.log'),
              logging.StreamHandler()
          ]
      )

3. Настройте мониторинг:

   .. code-block:: python

      from monitoring import MonitoringSystem

      monitoring = MonitoringSystem(port=8000)  # порт для метрик Prometheus

Verification
-----------

Для проверки установки выполните:

.. code-block:: bash

   python -m pytest tests/

Для проверки документации:

.. code-block:: bash

   cd docs
   make html

Troubleshooting
--------------

1. Проблемы с зависимостями:

   * Убедитесь, что у вас установлена последняя версия pip:
     .. code-block:: bash

        pip install --upgrade pip

   * Попробуйте установить зависимости по одной:
     .. code-block:: bash

        pip install pandas numpy scikit-learn

2. Проблемы с виртуальным окружением:

   * Удалите и пересоздайте виртуальное окружение
   * Убедитесь, что вы активировали виртуальное окружение

3. Проблемы с правами доступа:

   * Используйте sudo для Linux/Mac
   * Запустите командную строку от имени администратора в Windows 