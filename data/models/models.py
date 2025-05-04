from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from typing import Dict, Any, Optional
import logging

class BaseForecastingModel(ABC):
    """Базовый класс для моделей прогнозирования"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    @abstractmethod
    def fit(self, data: pd.DataFrame) -> None:
        """Обучение модели"""
        pass
        
    @abstractmethod
    def predict(self, horizon: int) -> pd.Series:
        """Прогнозирование"""
        pass
        
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Получение параметров модели"""
        pass

class ExponentialSmoothingModel(BaseForecastingModel):
    """Модель экспоненциального сглаживания"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = None
        self.data = None
        
    def fit(self, data: pd.DataFrame) -> None:
        self.data = data
        self.model = ExponentialSmoothing(
            data['quantity'],
            trend=self.config.get('trend', 'add'),
            seasonal=self.config.get('seasonal', 'add'),
            seasonal_periods=self.config.get('seasonal_periods', 7)
        ).fit()
        
    def predict(self, horizon: int) -> pd.Series:
        if self.model is None:
            raise ValueError("Модель не обучена")
        return self.model.forecast(horizon)
        
    def get_params(self) -> Dict[str, Any]:
        if self.model is None:
            return {}
        return {
            'alpha': self.model.params['smoothing_level'],
            'beta': self.model.params['smoothing_trend'],
            'gamma': self.model.params['smoothing_seasonal']
        }

class LinearRegressionModel(BaseForecastingModel):
    """Модель линейной регрессии"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = LinearRegression()
        self.data = None
        
    def fit(self, data: pd.DataFrame) -> None:
        self.data = data
        X = np.arange(len(data)).reshape(-1, 1)
        y = data['quantity'].values
        self.model.fit(X, y)
        
    def predict(self, horizon: int) -> pd.Series:
        if self.model is None:
            raise ValueError("Модель не обучена")
        X_pred = np.arange(len(self.data), len(self.data) + horizon).reshape(-1, 1)
        predictions = self.model.predict(X_pred)
        return pd.Series(predictions, index=pd.date_range(
            start=self.data.index[-1] + pd.Timedelta(days=1),
            periods=horizon
        ))
        
    def get_params(self) -> Dict[str, Any]:
        if self.model is None:
            return {}
        return {
            'coefficient': self.model.coef_[0],
            'intercept': self.model.intercept_
        }

class RandomForestModel(BaseForecastingModel):
    """Модель случайного леса"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = RandomForestRegressor(
            n_estimators=config.get('n_estimators', 100),
            max_depth=config.get('max_depth', None)
        )
        self.data = None
        
    def fit(self, data: pd.DataFrame) -> None:
        self.data = data
        X = self._prepare_features(data)
        y = data['quantity'].values
        self.model.fit(X, y)
        
    def predict(self, horizon: int) -> pd.Series:
        if self.model is None:
            raise ValueError("Модель не обучена")
        future_data = self._generate_future_features(horizon)
        predictions = self.model.predict(future_data)
        return pd.Series(predictions, index=pd.date_range(
            start=self.data.index[-1] + pd.Timedelta(days=1),
            periods=horizon
        ))
        
    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Подготовка признаков для модели"""
        features = []
        for i in range(len(data)):
            if i < 7:  # Используем историю за 7 дней
                features.append([0] * 7)
            else:
                features.append(data['quantity'].iloc[i-7:i].values)
        return np.array(features)
        
    def _generate_future_features(self, horizon: int) -> np.ndarray:
        """Генерация признаков для будущих прогнозов"""
        last_7_days = self.data['quantity'].iloc[-7:].values
        features = []
        for _ in range(horizon):
            features.append(last_7_days)
            last_7_days = np.roll(last_7_days, -1)
            last_7_days[-1] = features[-1][-1]
        return np.array(features)
        
    def get_params(self) -> Dict[str, Any]:
        if self.model is None:
            return {}
        return {
            'n_estimators': self.model.n_estimators,
            'max_depth': self.model.max_depth,
            'feature_importances': dict(zip(
                [f'lag_{i}' for i in range(7)],
                self.model.feature_importances_
            ))
        } 