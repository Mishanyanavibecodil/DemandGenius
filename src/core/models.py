"""
Модуль с реализациями различных моделей прогнозирования.
"""
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import logging

logger = logging.getLogger(__name__)

class BaseModel:
    """Базовый класс для моделей прогнозирования."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация модели.
        
        Args:
            config: Словарь с конфигурацией модели
        """
        self.config = config
        self.model = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """
        Обучение модели.
        
        Args:
            X: Признаки
            y: Целевая переменная
            
        Returns:
            bool: True если обучение успешно, False иначе
        """
        raise NotImplementedError
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Получение прогнозов.
        
        Args:
            X: Признаки для прогноза
            
        Returns:
            np.ndarray: Прогнозы
        """
        raise NotImplementedError
        
class LinearRegressionModel(BaseModel):
    """Модель линейной регрессии."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = LinearRegression(**config.get('model_params', {}))
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> bool:
        try:
            self.model.fit(X, y)
            return True
        except Exception as e:
            logger.error(f"Ошибка при обучении линейной регрессии: {str(e)}")
            return False
            
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        try:
            return self.model.predict(X)
        except Exception as e:
            logger.error(f"Ошибка при получении прогнозов: {str(e)}")
            return np.array([])
            
class RandomForestModel(BaseModel):
    """Модель случайного леса."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = RandomForestRegressor(**config.get('model_params', {}))
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> bool:
        try:
            self.model.fit(X, y)
            return True
        except Exception as e:
            logger.error(f"Ошибка при обучении случайного леса: {str(e)}")
            return False
            
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        try:
            return self.model.predict(X)
        except Exception as e:
            logger.error(f"Ошибка при получении прогнозов: {str(e)}")
            return np.array([])
            
class ExponentialSmoothingModel(BaseModel):
    """Модель экспоненциального сглаживания."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> bool:
        try:
            self.model = ExponentialSmoothing(
                y,
                trend=self.config.get('trend', 'add'),
                seasonal=self.config.get('seasonal', 'add'),
                seasonal_periods=self.config.get('seasonal_periods', 12)
            ).fit()
            return True
        except Exception as e:
            logger.error(f"Ошибка при обучении экспоненциального сглаживания: {str(e)}")
            return False
            
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        try:
            steps = len(X)
            return self.model.forecast(steps)
        except Exception as e:
            logger.error(f"Ошибка при получении прогнозов: {str(e)}")
            return np.array([])
            
def create_model(model_type: str, config: Dict[str, Any]) -> Optional[BaseModel]:
    """
    Фабрика для создания моделей.
    
    Args:
        model_type: Тип модели ('linear', 'random_forest', 'exponential')
        config: Конфигурация модели
        
    Returns:
        BaseModel: Экземпляр модели или None в случае ошибки
    """
    try:
        if model_type == 'linear':
            return LinearRegressionModel(config)
        elif model_type == 'random_forest':
            return RandomForestModel(config)
        elif model_type == 'exponential':
            return ExponentialSmoothingModel(config)
        else:
            logger.error(f"Неизвестный тип модели: {model_type}")
            return None
    except Exception as e:
        logger.error(f"Ошибка при создании модели: {str(e)}")
        return None 