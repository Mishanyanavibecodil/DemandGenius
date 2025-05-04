"""
Модуль с вспомогательными функциями.
"""
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

def load_config(config_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Загрузка конфигурации из файла.
    
    Args:
        config_path: Путь к файлу конфигурации
        
    Returns:
        Dict[str, Any]: Конфигурация или None в случае ошибки
    """
    try:
        config_path = Path(config_path)
        if config_path.suffix == '.json':
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif config_path.suffix in ['.yaml', '.yml']:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            logger.error(f"Неподдерживаемый формат файла: {config_path.suffix}")
            return None
    except Exception as e:
        logger.error(f"Ошибка при загрузке конфигурации: {str(e)}")
        return None

def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> bool:
    """
    Сохранение конфигурации в файл.
    
    Args:
        config: Конфигурация
        config_path: Путь к файлу конфигурации
        
    Returns:
        bool: True если сохранение успешно, False иначе
    """
    try:
        config_path = Path(config_path)
        if config_path.suffix == '.json':
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
        elif config_path.suffix in ['.yaml', '.yml']:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, allow_unicode=True)
        else:
            logger.error(f"Неподдерживаемый формат файла: {config_path.suffix}")
            return False
        return True
    except Exception as e:
        logger.error(f"Ошибка при сохранении конфигурации: {str(e)}")
        return False

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Расчет метрик качества прогноза.
    
    Args:
        y_true: Истинные значения
        y_pred: Прогнозные значения
        
    Returns:
        Dict[str, float]: Словарь с метриками
    """
    try:
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape)
        }
    except Exception as e:
        logger.error(f"Ошибка при расчете метрик: {str(e)}")
        return {}

def prepare_time_features(data: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """
    Подготовка временных признаков.
    
    Args:
        data: DataFrame с данными
        date_col: Название колонки с датой
        
    Returns:
        pd.DataFrame: DataFrame с временными признаками
    """
    try:
        df = data.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Добавление временных признаков
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        df['day'] = df[date_col].dt.day
        df['dayofweek'] = df[date_col].dt.dayofweek
        df['quarter'] = df[date_col].dt.quarter
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        
        return df
    except Exception as e:
        logger.error(f"Ошибка при подготовке временных признаков: {str(e)}")
        return pd.DataFrame()

def add_lag_features(data: pd.DataFrame, target_col: str, lag_periods: List[int]) -> pd.DataFrame:
    """
    Добавление лаговых признаков.
    
    Args:
        data: DataFrame с данными
        target_col: Название целевой колонки
        lag_periods: Список периодов для лагов
        
    Returns:
        pd.DataFrame: DataFrame с лаговыми признаками
    """
    try:
        df = data.copy()
        for lag in lag_periods:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        return df
    except Exception as e:
        logger.error(f"Ошибка при добавлении лаговых признаков: {str(e)}")
        return pd.DataFrame()

def add_rolling_features(data: pd.DataFrame, target_col: str, windows: List[int]) -> pd.DataFrame:
    """
    Добавление скользящих признаков.
    
    Args:
        data: DataFrame с данными
        target_col: Название целевой колонки
        windows: Список размеров окон
        
    Returns:
        pd.DataFrame: DataFrame со скользящими признаками
    """
    try:
        df = data.copy()
        for window in windows:
            df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
            df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window).std()
        return df
    except Exception as e:
        logger.error(f"Ошибка при добавлении скользящих признаков: {str(e)}")
        return pd.DataFrame()

def split_data(data: pd.DataFrame, date_col: str, test_size: float = 0.2) -> tuple:
    """
    Разделение данных на обучающую и тестовую выборки.
    
    Args:
        data: DataFrame с данными
        date_col: Название колонки с датой
        test_size: Доля тестовой выборки
        
    Returns:
        tuple: (train_data, test_data)
    """
    try:
        data = data.sort_values(date_col)
        split_idx = int(len(data) * (1 - test_size))
        return data.iloc[:split_idx], data.iloc[split_idx:]
    except Exception as e:
        logger.error(f"Ошибка при разделении данных: {str(e)}")
        return pd.DataFrame(), pd.DataFrame() 