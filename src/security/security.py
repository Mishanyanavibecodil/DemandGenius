import logging
import re
import time
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import hashlib
import hmac
import secrets
from functools import wraps
import json
import pandas as pd
import numpy as np

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('security.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SecurityManager:
    """Менеджер безопасности для системы прогнозирования"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация менеджера безопасности
        
        Args:
            config: Конфигурация безопасности
        """
        self.config = config
        self.rate_limits = {}  # Для отслеживания ограничений запросов
        self.blocked_ips = set()  # Для блокировки подозрительных IP
        self.suspicious_patterns = [
            r'(\b)(on\S+)(\s*)=|javascript:|(<\s*)(\/*)script',
            r'exec\s*\(|eval\s*\(|system\s*\(',
            r'(\b)(select\s*.+from\s*.+)|(insert\s*.+into\s*.+)|(update\s*.+set\s*.+)',
            r'(\b)(drop\s*.+)|(delete\s*.+from\s*.+)|(truncate\s*.+)',
            r'(\b)(union\s*.+select\s*.+)',
            r'(\b)(outfile\s*.+)|(dumpfile\s*.+)',
            r'(\b)(load_file\s*\(.+\))',
            r'(\b)(benchmark\s*\(.+\))',
            r'(\b)(sleep\s*\(.+\))',
            r'(\b)(waitfor\s*.+delay\s*.+)'
        ]
        
    def validate_input(self, data: Any, input_type: str) -> bool:
        """
        Валидация входных данных
        
        Args:
            data: Данные для валидации
            input_type: Тип входных данных
            
        Returns:
            bool: True если данные валидны, False иначе
        """
        try:
            if input_type == 'item_id':
                return bool(re.match(r'^[a-zA-Z0-9_-]{1,50}$', str(data)))
            elif input_type == 'date':
                return bool(re.match(r'^\d{4}-\d{2}-\d{2}$', str(data)))
            elif input_type == 'quantity':
                return isinstance(data, (int, float)) and data >= 0
            elif input_type == 'config':
                return self._validate_config(data)
            elif input_type == 'forecast_params':
                return self._validate_forecast_params(data)
            else:
                logger.warning(f"Неизвестный тип входных данных: {input_type}")
                return False
                
        except Exception as e:
            logger.error(f"Ошибка при валидации данных: {e}")
            return False
            
    def _validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Валидация конфигурации
        
        Args:
            config: Конфигурация для проверки
            
        Returns:
            bool: True если конфигурация валидна, False иначе
        """
        required_keys = {'lead_time', 'order_period', 'alpha', 'cache_size', 'cache_ttl'}
        if not all(key in config for key in required_keys):
            logger.error("Отсутствуют обязательные параметры конфигурации")
            return False
            
        if not all(isinstance(config[key], (int, float)) for key in required_keys):
            logger.error("Некорректные типы данных в конфигурации")
            return False
            
        if not (0 < config['alpha'] < 1):
            logger.error("Некорректное значение alpha")
            return False
            
        return True
        
    def _validate_forecast_params(self, params: Dict[str, Any]) -> bool:
        """
        Валидация параметров прогноза
        
        Args:
            params: Параметры прогноза
            
        Returns:
            bool: True если параметры валидны, False иначе
        """
        if 'days_ahead' not in params:
            logger.error("Отсутствует параметр days_ahead")
            return False
            
        if not isinstance(params['days_ahead'], int) or params['days_ahead'] <= 0:
            logger.error("Некорректное значение days_ahead")
            return False
            
        return True
        
    def check_rate_limit(self, ip: str) -> bool:
        """
        Проверка ограничения запросов
        
        Args:
            ip: IP-адрес клиента
            
        Returns:
            bool: True если запрос разрешен, False если превышен лимит
        """
        current_time = time.time()
        if ip in self.rate_limits:
            if current_time - self.rate_limits[ip]['last_request'] < 1:  # 1 запрос в секунду
                self.rate_limits[ip]['count'] += 1
                if self.rate_limits[ip]['count'] > 100:  # Максимум 100 запросов в секунду
                    logger.warning(f"Превышен лимит запросов для IP: {ip}")
                    return False
            else:
                self.rate_limits[ip] = {'count': 1, 'last_request': current_time}
        else:
            self.rate_limits[ip] = {'count': 1, 'last_request': current_time}
        return True
        
    def check_sql_injection(self, query: str) -> bool:
        """
        Проверка на SQL-инъекции
        
        Args:
            query: SQL-запрос для проверки
            
        Returns:
            bool: True если запрос безопасен, False если обнаружена инъекция
        """
        query = query.lower()
        for pattern in self.suspicious_patterns:
            if re.search(pattern, query):
                logger.warning(f"Обнаружена возможная SQL-инъекция: {query}")
                return False
        return True
        
    def sanitize_input(self, data: str) -> str:
        """
        Очистка входных данных
        
        Args:
            data: Данные для очистки
            
        Returns:
            str: Очищенные данные
        """
        # Удаление потенциально опасных символов
        data = re.sub(r'[<>"\']', '', data)
        # Экранирование специальных символов
        data = data.replace('&', '&amp;')
        data = data.replace('<', '&lt;')
        data = data.replace('>', '&gt;')
        return data
        
    def generate_token(self, data: str) -> str:
        """
        Генерация токена безопасности
        
        Args:
            data: Данные для токена
            
        Returns:
            str: Сгенерированный токен
        """
        salt = secrets.token_hex(16)
        return hmac.new(
            salt.encode(),
            data.encode(),
            hashlib.sha256
        ).hexdigest()
        
    def log_operation(self, operation: str, details: Dict[str, Any]) -> None:
        """
        Логирование операций
        
        Args:
            operation: Название операции
            details: Детали операции
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'details': details
        }
        logger.info(json.dumps(log_entry))
        
    def validate_dataframe(self, df: pd.DataFrame) -> bool:
        """
        Валидация DataFrame
        
        Args:
            df: DataFrame для проверки
            
        Returns:
            bool: True если DataFrame валиден, False иначе
        """
        try:
            # Проверка на наличие обязательных колонок
            required_columns = {'date', 'item_id', 'quantity'}
            if not all(col in df.columns for col in required_columns):
                logger.error("Отсутствуют обязательные колонки в DataFrame")
                return False
                
            # Проверка типов данных
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                logger.error("Некорректный тип данных для колонки date")
                return False
                
            if not pd.api.types.is_numeric_dtype(df['quantity']):
                logger.error("Некорректный тип данных для колонки quantity")
                return False
                
            # Проверка на отрицательные значения
            if (df['quantity'] < 0).any():
                logger.error("Обнаружены отрицательные значения в колонке quantity")
                return False
                
            # Проверка на пропуски
            if df.isnull().any().any():
                logger.error("Обнаружены пропуски в данных")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при валидации DataFrame: {e}")
            return False
            
def security_decorator(func):
    """
    Декоратор для добавления проверок безопасности
    
    Args:
        func: Функция для декорирования
        
    Returns:
        callable: Декорированная функция
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        security = SecurityManager(self.config)
        
        # Логирование вызова
        security.log_operation(
            func.__name__,
            {'args': args, 'kwargs': kwargs}
        )
        
        # Валидация входных данных
        if 'item_id' in kwargs:
            if not security.validate_input(kwargs['item_id'], 'item_id'):
                raise ValueError("Некорректный item_id")
                
        if 'days_ahead' in kwargs:
            if not security.validate_input(kwargs['days_ahead'], 'quantity'):
                raise ValueError("Некорректное значение days_ahead")
                
        # Проверка ограничения запросов
        if not security.check_rate_limit(kwargs.get('ip', 'unknown')):
            raise Exception("Превышен лимит запросов")
            
        # Выполнение функции
        try:
            result = func(self, *args, **kwargs)
            
            # Валидация результата
            if isinstance(result, pd.DataFrame):
                if not security.validate_dataframe(result):
                    raise ValueError("Некорректный результат")
                    
            return result
            
        except Exception as e:
            security.log_operation(
                'error',
                {
                    'function': func.__name__,
                    'error': str(e)
                }
            )
            raise
            
    return wrapper 