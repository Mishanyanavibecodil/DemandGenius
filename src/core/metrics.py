import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

class ForecastMetrics:
    """Класс для расчета метрик точности прогнозов"""
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Расчет метрик точности прогноза
        
        Args:
            y_true: реальные значения
            y_pred: прогнозные значения
            
        Returns:
            Dict[str, float]: словарь с метриками
        """
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            'r2': r2_score(y_true, y_pred)
        }
        
    @staticmethod
    def calculate_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Расчет смещения прогноза"""
        return np.mean(y_pred - y_true)
        
    @staticmethod
    def calculate_tracking_signal(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Расчет сигнала отслеживания"""
        errors = y_true - y_pred
        return np.sum(errors) / np.sqrt(np.sum(errors ** 2))
        
    @staticmethod
    def evaluate_forecast(
        forecast: pd.Series,
        actual: pd.Series,
        horizon: int
    ) -> Dict[str, float]:
        """
        Оценка точности прогноза
        
        Args:
            forecast: прогнозные значения
            actual: реальные значения
            horizon: горизонт прогноза
            
        Returns:
            Dict[str, float]: словарь с метриками
        """
        metrics = ForecastMetrics.calculate_metrics(actual, forecast)
        metrics['bias'] = ForecastMetrics.calculate_bias(actual, forecast)
        metrics['tracking_signal'] = ForecastMetrics.calculate_tracking_signal(
            actual, forecast
        )
        return metrics
        
    @staticmethod
    def evaluate_models(
        models_predictions: Dict[str, pd.Series],
        actual: pd.Series
    ) -> pd.DataFrame:
        """
        Сравнение точности различных моделей
        
        Args:
            models_predictions: словарь с прогнозами моделей
            actual: реальные значения
            
        Returns:
            pd.DataFrame: таблица с метриками для каждой модели
        """
        results = []
        for model_name, predictions in models_predictions.items():
            metrics = ForecastMetrics.calculate_metrics(actual, predictions)
            metrics['model'] = model_name
            results.append(metrics)
        return pd.DataFrame(results)
        
    @staticmethod
    def calculate_confidence_intervals(
        predictions: pd.Series,
        std_dev: float,
        confidence_level: float = 0.95
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Расчет доверительных интервалов
        
        Args:
            predictions: прогнозные значения
            std_dev: стандартное отклонение
            confidence_level: уровень доверия
            
        Returns:
            Tuple[pd.Series, pd.Series]: нижняя и верхняя границы интервала
        """
        z_score = 1.96  # для 95% доверительного интервала
        margin = z_score * std_dev
        lower_bound = predictions - margin
        upper_bound = predictions + margin
        return lower_bound, upper_bound 