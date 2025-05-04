import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from Script import DemandForecaster
from exceptions import *

class TestDemandForecaster(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Подготовка тестовых данных и конфигурации"""
        # Базовая конфигурация
        cls.config = {
            'lead_time': 7,
            'order_period': 14,
            'alpha': 0.3,
            'check_trend_days': 30,
            'growth_factor': 1.2,
            'lot_size': 10,
            'lost_demand_factor': {'A': 1.1, 'B': 1.2},
            'seasonality_coeffs': {1: 1.2, 2: 1.1, 3: 1.0},
            'max_growth_multiplier': 3.0
        }
        
        # Создание тестовых данных
        cls.sales_history = cls._create_test_sales_history()
        cls.past_orders = cls._create_test_past_orders()
        
        # Инициализация тестируемого объекта
        cls.forecaster = DemandForecaster(cls.config)
        cls.forecaster.fit(cls.sales_history, cls.past_orders)

    @staticmethod
    def _create_test_sales_history():
        """Создание тестовой истории продаж"""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        items = ['A', 'B']
        data = []
        
        for date in dates:
            for item in items:
                # Создаем реалистичные данные с трендом и сезонностью
                base = 100 if item == 'A' else 50
                trend = (date - dates[0]).days * 0.1
                season = np.sin((date.month - 1) * np.pi / 6) * 20
                noise = np.random.normal(0, 10)
                quantity = max(0, int(base + trend + season + noise))
                data.append({'date': date, 'item_id': item, 'quantity': quantity})
        
        return pd.DataFrame(data)

    @staticmethod
    def _create_test_past_orders():
        """Создание тестовой истории заказов"""
        data = []
        for i in range(100):
            data.append({
                'order_id': f'ORD{i}',
                'item_id': 'A' if i % 2 == 0 else 'B',
                'ordered_qty': np.random.randint(50, 200),
                'sold_qty': np.random.randint(40, 180)
            })
        return pd.DataFrame(data)

    # Тесты инициализации и конфигурации
    def test_initialization(self):
        """Тест инициализации объекта"""
        self.assertEqual(self.forecaster.lead_time, 7)
        self.assertEqual(self.forecaster.order_period, 14)
        self.assertEqual(self.forecaster.alpha, 0.3)
        self.assertEqual(self.forecaster.lot_size, 10)

    def test_invalid_config(self):
        """Тест обработки некорректной конфигурации"""
        invalid_config = self.config.copy()
        invalid_config['alpha'] = 2.0  # Некорректное значение alpha
        
        with self.assertRaises(ConfigurationError):
            DemandForecaster(invalid_config)

    def test_missing_required_params(self):
        """Тест отсутствия обязательных параметров"""
        invalid_config = self.config.copy()
        del invalid_config['lead_time']
        
        with self.assertRaises(ConfigurationError):
            DemandForecaster(invalid_config)

    # Тесты валидации данных
    def test_invalid_sales_data(self):
        """Тест валидации некорректных данных продаж"""
        invalid_sales = self.sales_history.copy()
        invalid_sales['quantity'] = -1  # Отрицательные значения
        
        with self.assertRaises(DataValidationError):
            self.forecaster.fit(invalid_sales, self.past_orders)

    def test_missing_sales_columns(self):
        """Тест отсутствия обязательных колонок в данных продаж"""
        invalid_sales = self.sales_history.drop('quantity', axis=1)
        
        with self.assertRaises(DataValidationError):
            self.forecaster.fit(invalid_sales, self.past_orders)

    def test_invalid_orders_data(self):
        """Тест валидации некорректных данных заказов"""
        invalid_orders = self.past_orders.copy()
        invalid_orders['ordered_qty'] = -1  # Отрицательные значения
        
        with self.assertRaises(DataValidationError):
            self.forecaster.fit(self.sales_history, invalid_orders)

    # Тесты обнаружения выбросов
    def test_outlier_detection(self):
        """Тест обнаружения выбросов в данных"""
        sales_with_outliers = self.sales_history.copy()
        # Добавляем выброс
        sales_with_outliers.loc[0, 'quantity'] = 1000
        
        with self.assertRaises(OutlierError):
            self.forecaster.fit(sales_with_outliers, self.past_orders)

    # Тесты прогнозирования
    def test_predict_basic(self):
        """Тест базового прогнозирования"""
        forecast = self.forecaster.predict('A', 30)
        self.assertEqual(len(forecast), 30)
        self.assertTrue(all(forecast >= 0))

    def test_predict_unfitted_model(self):
        """Тест прогнозирования без обучения модели"""
        unfitted_forecaster = DemandForecaster(self.config)
        
        with self.assertRaises(PredictionError):
            unfitted_forecaster.predict('A', 30)

    def test_predict_invalid_item(self):
        """Тест прогнозирования для несуществующего товара"""
        forecast = self.forecaster.predict('NONEXISTENT', 30)
        self.assertEqual(len(forecast), 30)
        self.assertTrue(all(forecast == 0))

    # Тесты расчета заказа
    def test_calculate_order(self):
        """Тест расчета заказа"""
        order = self.forecaster.calculate_order('A', 100, 50)
        self.assertIsInstance(order, int)
        self.assertTrue(order >= 0)
        self.assertTrue(order % self.forecaster.lot_size == 0)

    def test_calculate_order_negative_stock(self):
        """Тест расчета заказа с отрицательным запасом"""
        with self.assertRaises(ValueError):
            self.forecaster.calculate_order('A', -100, 50)

    def test_calculate_order_unfitted_model(self):
        """Тест расчета заказа без обучения модели"""
        unfitted_forecaster = DemandForecaster(self.config)
        
        with self.assertRaises(OrderCalculationError):
            unfitted_forecaster.calculate_order('A', 100, 50)

    # Интеграционные тесты
    def test_full_forecasting_cycle(self):
        """Тест полного цикла прогнозирования"""
        # Прогноз
        forecast = self.forecaster.predict('A', 30)
        self.assertIsNotNone(forecast)
        
        # Расчет заказа
        order = self.forecaster.calculate_order('A', 100, 50)
        self.assertIsNotNone(order)
        
        # Визуализация
        try:
            self.forecaster.visualize_forecast('A', 30)
        except Exception as e:
            self.fail(f"Визуализация вызвала исключение: {e}")

    def test_data_consistency(self):
        """Тест согласованности данных"""
        # Проверка соответствия типов данных
        self.assertIsInstance(self.forecaster.sales_history, pd.DataFrame)
        self.assertIsInstance(self.forecaster.past_orders, pd.DataFrame)
        
        # Проверка наличия необходимых колонок
        required_columns = ['date', 'item_id', 'quantity']
        for col in required_columns:
            self.assertIn(col, self.forecaster.sales_history.columns)

    # Тесты производительности
    def test_performance_predict(self):
        """Тест производительности метода predict"""
        start_time = time.time()
        self.forecaster.predict('A', 30)
        end_time = time.time()
        
        execution_time = end_time - start_time
        self.assertLess(execution_time, 1.0)

    def test_performance_calculate_order(self):
        """Тест производительности метода calculate_order"""
        start_time = time.time()
        self.forecaster.calculate_order('A', 100, 50)
        end_time = time.time()
        
        execution_time = end_time - start_time
        self.assertLess(execution_time, 0.5)

    def test_large_dataset_performance(self):
        """Тест производительности на большом наборе данных"""
        # Создание большого набора данных
        large_sales = pd.concat([self.sales_history] * 10)
        large_orders = pd.concat([self.past_orders] * 10)
        
        start_time = time.time()
        self.forecaster.fit(large_sales, large_orders)
        self.forecaster.predict('A', 30)
        end_time = time.time()
        
        execution_time = end_time - start_time
        self.assertLess(execution_time, 5.0)

if __name__ == '__main__':
    unittest.main() 