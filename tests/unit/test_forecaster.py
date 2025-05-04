"""
Модульные тесты для прогнозировщика.
"""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.core.forecaster import DemandForecaster
from src.core.utils import load_config

class TestDemandForecaster(unittest.TestCase):
    """Тесты для класса DemandForecaster."""
    
    @classmethod
    def setUpClass(cls):
        """Подготовка данных для тестов."""
        # Загрузка конфигурации
        cls.config = load_config('../../config/default.yaml')
        
        # Создание тестовых данных
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        items = ['A', 'B', 'C']
        
        data = []
        for date in dates:
            for item in items:
                # Создаем синтетические данные с трендом и сезонностью
                base = 100
                trend = (date - dates[0]).days * 0.1
                season = 20 * np.sin(2 * np.pi * (date.month - 1) / 12)
                noise = np.random.normal(0, 10)
                quantity = max(0, int(base + trend + season + noise))
                
                data.append({
                    'date': date,
                    'item_id': item,
                    'quantity': quantity
                })
        
        cls.data = pd.DataFrame(data)
        
    def setUp(self):
        """Подготовка перед каждым тестом."""
        self.forecaster = DemandForecaster(self.config)
        
    def test_initialization(self):
        """Тест инициализации прогнозировщика."""
        self.assertIsNotNone(self.forecaster)
        self.assertEqual(self.forecaster.config, self.config)
        self.assertIsNone(self.forecaster.model)
        self.assertIsNone(self.forecaster.features)
        self.assertIsNone(self.forecaster.last_training_date)
        self.assertEqual(self.forecaster.performance_metrics, {})
        
    def test_fit(self):
        """Тест обучения модели."""
        success = self.forecaster.fit(self.data)
        self.assertTrue(success)
        self.assertIsNotNone(self.forecaster.model)
        self.assertIsNotNone(self.forecaster.features)
        self.assertIsNotNone(self.forecaster.last_training_date)
        self.assertNotEqual(self.forecaster.performance_metrics, {})
        
    def test_predict(self):
        """Тест создания прогноза."""
        # Сначала обучаем модель
        self.forecaster.fit(self.data)
        
        # Тестируем прогноз
        item_id = 'A'
        days_ahead = 30
        forecast = self.forecaster.predict(item_id, days_ahead)
        
        self.assertIsNotNone(forecast)
        self.assertIsInstance(forecast, pd.DataFrame)
        self.assertEqual(len(forecast), days_ahead)
        self.assertTrue(all(col in forecast.columns for col in ['date', 'forecast', 'lower_bound', 'upper_bound']))
        
    def test_evaluate(self):
        """Тест оценки модели."""
        # Сначала обучаем модель
        self.forecaster.fit(self.data)
        
        # Создаем тестовые данные
        test_dates = pd.date_range(start='2025-01-01', end='2025-01-31', freq='D')
        test_data = []
        for date in test_dates:
            for item in ['A', 'B', 'C']:
                quantity = np.random.randint(50, 150)
                test_data.append({
                    'date': date,
                    'item_id': item,
                    'quantity': quantity
                })
        test_df = pd.DataFrame(test_data)
        
        # Тестируем оценку
        metrics = self.forecaster.evaluate(test_df)
        
        self.assertIsNotNone(metrics)
        self.assertIsInstance(metrics, dict)
        self.assertTrue(all(metric in metrics for metric in ['mae', 'rmse', 'mape']))
        
    def test_validate_data(self):
        """Тест валидации данных."""
        # Тест с правильными данными
        self.assertTrue(self.forecaster._validate_data(self.data))
        
        # Тест с пустыми данными
        empty_df = pd.DataFrame()
        self.assertFalse(self.forecaster._validate_data(empty_df))
        
        # Тест с отсутствующими колонками
        invalid_df = self.data.drop('quantity', axis=1)
        self.assertFalse(self.forecaster._validate_data(invalid_df))
        
    def test_feature_importance(self):
        """Тест получения важности признаков."""
        # Сначала обучаем модель
        self.forecaster.fit(self.data)
        
        # Получаем важность признаков
        importance = self.forecaster.get_feature_importance()
        
        self.assertIsNotNone(importance)
        self.assertIsInstance(importance, pd.DataFrame)
        self.assertTrue(all(col in importance.columns for col in ['feature', 'importance']))
        
if __name__ == '__main__':
    unittest.main() 