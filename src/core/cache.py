from typing import Any, Dict, Optional
from datetime import datetime, timedelta
import threading
from functools import lru_cache
import numpy as np
import pandas as pd

class ForecastCache:
    """Класс для кэширования результатов прогнозов"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """
        Инициализация кэша
        
        Args:
            max_size: максимальное количество элементов в кэше
            ttl_seconds: время жизни кэша в секундах
        """
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        
    def _generate_key(self, item_id: str, days_ahead: int) -> str:
        """Генерация ключа кэша"""
        return f"{item_id}_{days_ahead}"
        
    def _is_expired(self, timestamp: datetime) -> bool:
        """Проверка срока действия кэша"""
        return (datetime.now() - timestamp).total_seconds() > self._ttl_seconds
        
    def get(self, item_id: str, days_ahead: int) -> Optional[pd.Series]:
        """
        Получение значения из кэша
        
        Args:
            item_id: идентификатор товара
            days_ahead: горизонт прогноза
            
        Returns:
            Optional[pd.Series]: закэшированное значение или None
        """
        key = self._generate_key(item_id, days_ahead)
        
        with self._lock:
            if key in self._cache:
                cache_entry = self._cache[key]
                if not self._is_expired(cache_entry['timestamp']):
                    return cache_entry['value']
                else:
                    del self._cache[key]
        return None
        
    def set(self, item_id: str, days_ahead: int, value: pd.Series) -> None:
        """
        Сохранение значения в кэш
        
        Args:
            item_id: идентификатор товара
            days_ahead: горизонт прогноза
            value: значение для кэширования
        """
        key = self._generate_key(item_id, days_ahead)
        
        with self._lock:
            # Очистка устаревших записей
            self._cleanup()
            
            # Проверка размера кэша
            if len(self._cache) >= self._max_size:
                # Удаление самой старой записи
                oldest_key = min(self._cache.keys(), 
                               key=lambda k: self._cache[k]['timestamp'])
                del self._cache[oldest_key]
            
            # Сохранение нового значения
            self._cache[key] = {
                'value': value,
                'timestamp': datetime.now()
            }
            
    def _cleanup(self) -> None:
        """Очистка устаревших записей"""
        current_time = datetime.now()
        expired_keys = [
            key for key, entry in self._cache.items()
            if self._is_expired(entry['timestamp'])
        ]
        for key in expired_keys:
            del self._cache[key]
            
    def clear(self) -> None:
        """Очистка всего кэша"""
        with self._lock:
            self._cache.clear()

# Декоратор для кэширования результатов функций
def cached(maxsize: int = 128):
    """
    Декоратор для кэширования результатов функций
    
    Args:
        maxsize: максимальный размер кэша
    """
    def decorator(func):
        cache = {}
        lock = threading.Lock()
        
        def wrapper(*args, **kwargs):
            # Создание ключа кэша
            key = str(args) + str(kwargs)
            
            with lock:
                if key in cache:
                    return cache[key]
                    
                result = func(*args, **kwargs)
                
                # Проверка размера кэша
                if len(cache) >= maxsize:
                    # Удаление самой старой записи
                    oldest_key = next(iter(cache))
                    del cache[oldest_key]
                    
                cache[key] = result
                return result
                
        return wrapper
    return decorator 