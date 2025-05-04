import logging
import time
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import prometheus_client as prom
from prometheus_client import start_http_server
import psutil
import threading

class MonitoringSystem:
    """Система мониторинга для прогнозирования спроса"""
    
    def __init__(self, port: int = 8000):
        """
        Инициализация системы мониторинга
        
        Args:
            port: порт для метрик Prometheus
        """
        self.logger = logging.getLogger(__name__)
        
        # Метрики Prometheus
        self.forecast_accuracy = prom.Gauge(
            'forecast_accuracy',
            'Точность прогноза',
            ['model', 'metric']
        )
        
        self.prediction_latency = prom.Histogram(
            'prediction_latency_seconds',
            'Время выполнения прогноза',
            ['model'],
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0)
        )
        
        self.model_errors = prom.Counter(
            'model_errors_total',
            'Количество ошибок модели',
            ['model', 'error_type']
        )
        
        self.cache_hits = prom.Counter(
            'cache_hits_total',
            'Количество попаданий в кэш'
        )
        
        self.cache_misses = prom.Counter(
            'cache_misses_total',
            'Количество промахов кэша'
        )
        
        self.system_metrics = {
            'cpu_usage': prom.Gauge('cpu_usage_percent', 'CPU usage'),
            'memory_usage': prom.Gauge('memory_usage_bytes', 'Memory usage'),
            'disk_usage': prom.Gauge('disk_usage_percent', 'Disk usage')
        }
        
        # Запуск сервера метрик
        start_http_server(port)
        self.logger.info(f"Мониторинг запущен на порту {port}")
        
        # Запуск мониторинга системных метрик
        self._start_system_monitoring()
        
    def _start_system_monitoring(self):
        """Запуск мониторинга системных метрик"""
        def monitor_system():
            while True:
                try:
                    self.system_metrics['cpu_usage'].set(psutil.cpu_percent())
                    self.system_metrics['memory_usage'].set(psutil.virtual_memory().used)
                    self.system_metrics['disk_usage'].set(psutil.disk_usage('/').percent)
                    time.sleep(60)  # Обновление каждую минуту
                except Exception as e:
                    self.logger.error(f"Ошибка мониторинга системы: {e}")
                    
        thread = threading.Thread(target=monitor_system, daemon=True)
        thread.start()
        
    def record_forecast_accuracy(
        self,
        model: str,
        metrics: Dict[str, float]
    ) -> None:
        """
        Запись метрик точности прогноза
        
        Args:
            model: название модели
            metrics: словарь с метриками
        """
        for metric_name, value in metrics.items():
            self.forecast_accuracy.labels(model=model, metric=metric_name).set(value)
            
    def record_prediction_latency(
        self,
        model: str,
        start_time: float,
        end_time: float
    ) -> None:
        """
        Запись времени выполнения прогноза
        
        Args:
            model: название модели
            start_time: время начала
            end_time: время окончания
        """
        latency = end_time - start_time
        self.prediction_latency.labels(model=model).observe(latency)
        
    def record_model_error(
        self,
        model: str,
        error_type: str
    ) -> None:
        """
        Запись ошибки модели
        
        Args:
            model: название модели
            error_type: тип ошибки
        """
        self.model_errors.labels(model=model, error_type=error_type).inc()
        
    def record_cache_event(self, is_hit: bool) -> None:
        """
        Запись события кэша
        
        Args:
            is_hit: True для попадания, False для промаха
        """
        if is_hit:
            self.cache_hits.inc()
        else:
            self.cache_misses.inc()
            
    def get_system_metrics(self) -> Dict[str, float]:
        """
        Получение текущих системных метрик
        
        Returns:
            Dict[str, float]: словарь с метриками
        """
        return {
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().used,
            'disk_usage': psutil.disk_usage('/').percent
        }
        
    def generate_report(self) -> Dict[str, Any]:
        """
        Генерация отчета о состоянии системы
        
        Returns:
            Dict[str, Any]: словарь с отчетом
        """
        return {
            'system_metrics': self.get_system_metrics(),
            'cache_stats': {
                'hits': self.cache_hits._value.get(),
                'misses': self.cache_misses._value.get()
            },
            'model_errors': {
                model: self.model_errors.labels(model=model)._value.get()
                for model in ['exponential', 'linear', 'random_forest']
            }
        }
        
class ModelPerformanceMonitor:
    """Мониторинг производительности моделей"""
    
    def __init__(self, monitoring_system: MonitoringSystem):
        """
        Инициализация монитора производительности
        
        Args:
            monitoring_system: система мониторинга
        """
        self.monitoring = monitoring_system
        self.logger = logging.getLogger(__name__)
        
    def monitor_prediction(
        self,
        model: str,
        func: callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Мониторинг выполнения прогноза
        
        Args:
            model: название модели
            func: функция прогнозирования
            *args: аргументы функции
            **kwargs: именованные аргументы функции
            
        Returns:
            Any: результат функции
        """
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            
            self.monitoring.record_prediction_latency(
                model,
                start_time,
                end_time
            )
            
            return result
            
        except Exception as e:
            self.monitoring.record_model_error(model, type(e).__name__)
            self.logger.error(f"Ошибка при прогнозировании: {e}")
            raise
            
    def monitor_accuracy(
        self,
        model: str,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> None:
        """
        Мониторинг точности прогноза
        
        Args:
            model: название модели
            y_true: реальные значения
            y_pred: прогнозные значения
        """
        from metrics import ForecastMetrics
        
        metrics = ForecastMetrics.calculate_metrics(y_true, y_pred)
        self.monitoring.record_forecast_accuracy(model, metrics) 