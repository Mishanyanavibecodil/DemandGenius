from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import optuna
from optuna.trial import Trial
import logging

class HyperparameterOptimizer:
    """Класс для оптимизации гиперпараметров моделей"""
    
    def __init__(self, model_class: Any, data: pd.DataFrame, metric: str = 'rmse'):
        """
        Инициализация оптимизатора
        
        Args:
            model_class: класс модели для оптимизации
            data: данные для обучения
            metric: метрика для оптимизации
        """
        self.model_class = model_class
        self.data = data
        self.metric = metric
        self.logger = logging.getLogger(__name__)
        
    def optimize(
        self,
        n_trials: int = 100,
        param_ranges: Dict[str, Tuple[Any, Any]] = None
    ) -> Dict[str, Any]:
        """
        Оптимизация гиперпараметров
        
        Args:
            n_trials: количество попыток
            param_ranges: диапазоны параметров для оптимизации
            
        Returns:
            Dict[str, Any]: оптимальные параметры
        """
        def objective(trial: Trial) -> float:
            params = self._suggest_params(trial, param_ranges)
            model = self.model_class(params)
            
            # Кросс-валидация по времени
            tscv = TimeSeriesSplit(n_splits=5)
            scores = []
            
            for train_idx, val_idx in tscv.split(self.data):
                train_data = self.data.iloc[train_idx]
                val_data = self.data.iloc[val_idx]
                
                model.fit(train_data)
                predictions = model.predict(len(val_data))
                
                score = mean_squared_error(
                    val_data['quantity'],
                    predictions,
                    squared=(self.metric == 'rmse')
                )
                scores.append(score)
                
            return np.mean(scores)
            
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        return study.best_params
        
    def _suggest_params(self, trial: Trial, param_ranges: Dict[str, Tuple[Any, Any]]) -> Dict[str, Any]:
        """Предложение параметров для оптимизации"""
        params = {}
        
        if param_ranges is None:
            # Параметры по умолчанию
            param_ranges = {
                'alpha': (0.1, 0.9),
                'beta': (0.1, 0.9),
                'gamma': (0.1, 0.9),
                'seasonal_periods': (7, 30)
            }
            
        for param, (low, high) in param_ranges.items():
            if isinstance(low, int) and isinstance(high, int):
                params[param] = trial.suggest_int(param, low, high)
            else:
                params[param] = trial.suggest_float(param, low, high)
                
        return params
        
class ModelSelector:
    """Класс для выбора лучшей модели"""
    
    def __init__(self, models: Dict[str, Any], data: pd.DataFrame):
        """
        Инициализация селектора моделей
        
        Args:
            models: словарь моделей для сравнения
            data: данные для обучения
        """
        self.models = models
        self.data = data
        self.logger = logging.getLogger(__name__)
        
    def select_best_model(
        self,
        metric: str = 'rmse',
        cv_splits: int = 5
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Выбор лучшей модели
        
        Args:
            metric: метрика для сравнения
            cv_splits: количество разбиений для кросс-валидации
            
        Returns:
            Tuple[str, Dict[str, Any]]: имя лучшей модели и её параметры
        """
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        model_scores = {}
        
        for model_name, model_class in self.models.items():
            scores = []
            for train_idx, val_idx in tscv.split(self.data):
                train_data = self.data.iloc[train_idx]
                val_data = self.data.iloc[val_idx]
                
                model = model_class({})  # Используем параметры по умолчанию
                model.fit(train_data)
                predictions = model.predict(len(val_data))
                
                score = mean_squared_error(
                    val_data['quantity'],
                    predictions,
                    squared=(metric == 'rmse')
                )
                scores.append(score)
                
            model_scores[model_name] = np.mean(scores)
            
        best_model = min(model_scores.items(), key=lambda x: x[1])[0]
        return best_model, self.models[best_model] 