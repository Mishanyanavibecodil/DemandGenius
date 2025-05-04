import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from exceptions import *
from cache import ForecastCache, cached
import concurrent.futures
from functools import partial
import multiprocessing

class DemandForecaster:
    """
    Класс для прогнозирования спроса на товары.
    
    Основные возможности:
    - Прогнозирование спроса с учетом тренда и сезонности
    - Расчет оптимального размера заказа
    - Визуализация прогнозов
    - Учет потерянного спроса
    - Ограничение роста прогноза
    """
    
    def __init__(self, config: Dict):
        """
        Инициализация системы прогнозирования спроса
        
        Args:
            config: словарь конфигурации с параметрами:
                lead_time: int - время поставки (в днях)
                order_period: int - период заказа (в днях)
                alpha: float - коэффициент сглаживания (0-1)
                check_trend_days: int - количество дней для проверки тренда
                growth_factor: float - фактор роста
                lot_size: int - размер партии
                lost_demand_factor: Dict[str, float] - коэффициенты потерянного спроса по категориям
                seasonality_coeffs: Dict[int, float] - сезонные коэффициенты по месяцам
                max_growth_multiplier: float - максимальный множитель роста (по умолчанию 3.0)
        
        Raises:
            ConfigurationError: при некорректных значениях параметров конфигурации
            KeyError: при отсутствии обязательных параметров
        """
        self._setup_logging()
        self.logger.info("Инициализация DemandForecaster")
        
        try:
            self._validate_config_structure(config)
            self._init_parameters(config)
            self._validate_config_values()
            
            # Инициализация кэша
            self.cache = ForecastCache(
                max_size=config.get('cache_size', 1000),
                ttl_seconds=config.get('cache_ttl', 3600)
            )
            
            # Определение количества процессов для параллельных вычислений
            self.n_processes = config.get('n_processes', 
                                        max(1, multiprocessing.cpu_count() - 1))
            
        except Exception as e:
            self.logger.error(f"Ошибка инициализации: {str(e)}")
            raise ConfigurationError(f"Ошибка инициализации: {str(e)}")

    def _setup_logging(self) -> None:
        """
        Настройка системы логирования.
        Устанавливает формат и уровень логирования.
        """
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _validate_config_structure(self, config: Dict) -> None:
        """Проверка структуры конфигурации"""
        required_params = ['lead_time', 'order_period', 'alpha']
        missing_params = [param for param in required_params if param not in config]
        if missing_params:
            raise ConfigurationError(f"Отсутствуют обязательные параметры: {missing_params}")

    def _init_parameters(self, config: Dict) -> None:
        """Инициализация параметров из конфигурации"""
        self.lead_time = config['lead_time']
        self.order_period = config['order_period']
        self.alpha = config['alpha']
        self.check_trend_days = config.get('check_trend_days', 30)
        self.growth_factor = config.get('growth_factor', 1.2)
        self.lot_size = config.get('lot_size', 10)
        self.max_growth_multiplier = config.get('max_growth_multiplier', 3.0)
        self.lost_demand_factor = config.get('lost_demand_factor', {})
        self.seasonality_coeffs = config.get('seasonality_coeffs', {})
        self.sales_history = None
        self.past_orders = None

    def _validate_config_values(self) -> None:
        """Проверка значений параметров конфигурации"""
        if not isinstance(self.alpha, (int, float)) or not 0 <= self.alpha <= 1:
            raise ConfigurationError("Alpha должен быть между 0 и 1")
            
        if not isinstance(self.lot_size, int) or self.lot_size <= 0:
            raise ConfigurationError("Размер партии должен быть положительным числом")
            
        if not isinstance(self.lead_time, int) or self.lead_time <= 0:
            raise ConfigurationError("Время поставки должно быть положительным числом")
            
        if not isinstance(self.order_period, int) or self.order_period <= 0:
            raise ConfigurationError("Период заказа должен быть положительным числом")
            
        if not isinstance(self.max_growth_multiplier, (int, float)) or self.max_growth_multiplier <= 1:
            raise ConfigurationError("Максимальный множитель роста должен быть больше 1")

    def _validate_sales_data(self, sales_data: pd.DataFrame) -> None:
        """Валидация данных продаж"""
        if not isinstance(sales_data, pd.DataFrame):
            raise DataValidationError("Данные продаж должны быть DataFrame")
            
        required_columns = ['date', 'item_id', 'quantity']
        missing_columns = [col for col in required_columns if col not in sales_data.columns]
        if missing_columns:
            raise DataValidationError(f"Отсутствуют обязательные колонки: {missing_columns}")
            
        if sales_data['quantity'].isnull().any():
            raise DataValidationError("Обнаружены пропущенные значения в колонке quantity")
            
        if (sales_data['quantity'] < 0).any():
            raise DataValidationError("Обнаружены отрицательные значения в колонке quantity")

    def _validate_orders_data(self, orders_data: pd.DataFrame) -> None:
        """Валидация данных заказов"""
        if not isinstance(orders_data, pd.DataFrame):
            raise DataValidationError("Данные заказов должны быть DataFrame")
            
        required_columns = ['order_id', 'item_id', 'ordered_qty', 'sold_qty']
        missing_columns = [col for col in required_columns if col not in orders_data.columns]
        if missing_columns:
            raise DataValidationError(f"Отсутствуют обязательные колонки: {missing_columns}")
            
        if orders_data[['ordered_qty', 'sold_qty']].isnull().any().any():
            raise DataValidationError("Обнаружены пропущенные значения в данных заказов")
            
        if (orders_data['ordered_qty'] < 0).any() or (orders_data['sold_qty'] < 0).any():
            raise DataValidationError("Обнаружены отрицательные значения в данных заказов")

    def _detect_outliers(self, data: pd.Series, threshold: float = 3.0) -> List[int]:
        """
        Обнаружение выбросов с помощью z-score
        
        Args:
            data: Series с данными
            threshold: пороговое значение для определения выброса
            
        Returns:
            List[int]: индексы выбросов
        """
        z_scores = np.abs((data - data.mean()) / data.std())
        return list(np.where(z_scores > threshold)[0])

    def fit(self, sales_history: pd.DataFrame, past_orders: pd.DataFrame) -> None:
        """
        Обучение модели на исторических данных
        
        Args:
            sales_history: DataFrame с историей продаж
            past_orders: DataFrame с историей заказов
            
        Raises:
            DataValidationError: при ошибках в данных
            OutlierError: при обнаружении выбросов
        """
        try:
            self.logger.info("Начало обучения модели")
            
            # Валидация входных данных
            self._validate_sales_data(sales_history)
            self._validate_orders_data(past_orders)
            
            # Проверка на выбросы
            outliers = self._detect_outliers(sales_history['quantity'])
            if outliers:
                self.logger.warning(f"Обнаружены выбросы в данных продаж: {len(outliers)} точек")
                raise OutlierError(f"Обнаружены выбросы в данных продаж: {len(outliers)} точек")
            
            # Подготовка данных
            sales_history['date'] = pd.to_datetime(sales_history['date'])
            sales_history.sort_values(['item_id', 'date'], inplace=True)
            
            self.sales_history = sales_history.copy()
            self.past_orders = past_orders.copy()
            
            self.logger.info(f"Загружено {len(sales_history)} записей продаж")
            self.logger.info(f"Загружено {len(past_orders)} записей заказов")
            
        except Exception as e:
            self.logger.error(f"Ошибка при обучении модели: {str(e)}")
            raise

    @cached(maxsize=128)
    def _exp_smoothing(self, sales_data: pd.DataFrame) -> pd.Series:
        """
        Экспоненциальное сглаживание с кэшированием
        
        Args:
            sales_data: DataFrame с данными продаж
            
        Returns:
            Series со сглаженными значениями
        """
        return pd.Series(
            np.convolve(sales_data['quantity'], 
                       [self.alpha, 1-self.alpha], 
                       mode='full')[:len(sales_data)],
            index=sales_data['date']
        )

    def predict(self, item_id: str, days_ahead: int) -> pd.Series:
        """
        Прогнозирование спроса для товара с использованием кэша
        
        Args:
            item_id: идентификатор товара
            days_ahead: количество дней для прогноза
            
        Returns:
            Series с прогнозируемыми значениями
        """
        try:
            # Проверка кэша
            cached_forecast = self.cache.get(item_id, days_ahead)
            if cached_forecast is not None:
                self.logger.info(f"Использован кэшированный прогноз для {item_id}")
                return cached_forecast
                
            if not isinstance(item_id, str):
                raise ValueError("item_id должен быть строкой")
            if not isinstance(days_ahead, int) or days_ahead <= 0:
                raise ValueError("days_ahead должен быть положительным целым числом")
                
            self.logger.info(f"Прогнозирование для товара {item_id}, горизонт {days_ahead} дней")
            
            if self.sales_history is None:
                raise PredictionError("Модель не обучена")
                
            item_sales = self.sales_history[self.sales_history['item_id'] == item_id].copy()
            
            if len(item_sales) == 0:
                self.logger.warning(f"Нет истории продаж для товара {item_id}")
                return pd.Series(index=pd.date_range(start=datetime.now(), periods=days_ahead))
                
            forecast = self._exp_smoothing(item_sales)
            trend = self._calculate_trend(item_sales)
            seasonal_forecast = self._apply_seasonality(forecast)
            limited_forecast = self._adjust_for_growth_limit(seasonal_forecast, item_sales['quantity'])
            final_forecast = self._handle_lost_demand(limited_forecast, item_id)
            
            final_forecast = final_forecast.clip(lower=0)
            
            # Сохранение в кэш
            self.cache.set(item_id, days_ahead, final_forecast)
            
            return final_forecast
            
        except Exception as e:
            self.logger.error(f"Ошибка при прогнозировании: {str(e)}")
            raise PredictionError(f"Ошибка при прогнозировании: {str(e)}")

    def predict_batch(self, item_ids: List[str], days_ahead: int) -> Dict[str, pd.Series]:
        """
        Параллельное прогнозирование для нескольких товаров
        
        Args:
            item_ids: список идентификаторов товаров
            days_ahead: количество дней для прогноза
            
        Returns:
            Dict[str, pd.Series]: словарь с прогнозами для каждого товара
        """
        results = {}
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_processes) as executor:
            # Создание частичной функции с фиксированным параметром days_ahead
            predict_func = partial(self.predict, days_ahead=days_ahead)
            
            # Параллельное выполнение прогнозов
            future_to_item = {
                executor.submit(predict_func, item_id): item_id 
                for item_id in item_ids
            }
            
            # Сбор результатов
            for future in concurrent.futures.as_completed(future_to_item):
                item_id = future_to_item[future]
                try:
                    results[item_id] = future.result()
                except Exception as e:
                    self.logger.error(f"Ошибка при прогнозировании для {item_id}: {str(e)}")
                    results[item_id] = pd.Series(index=pd.date_range(start=datetime.now(), periods=days_ahead))
                    
        return results

    def calculate_order(self, item_id: str, current_stock: int, incoming_stock: int) -> int:
        """
        Расчет рекомендуемого заказа
        
        Args:
            item_id: идентификатор товара
            current_stock: текущий запас
            incoming_stock: поступающий запас
            
        Returns:
            рекомендованное количество для заказа
            
        Raises:
            OrderCalculationError: при ошибках расчета заказа
            ValueError: при некорректных значениях запасов
        """
        try:
            if not isinstance(current_stock, int) or current_stock < 0:
                raise ValueError("current_stock должен быть неотрицательным целым числом")
            if not isinstance(incoming_stock, int) or incoming_stock < 0:
                raise ValueError("incoming_stock должен быть неотрицательным целым числом")
                
            self.logger.info(f"Расчет заказа для товара {item_id}")
            
            forecast = self.predict(item_id, self.lead_time + self.order_period)
            deficit = self._calculate_deficit(current_stock, incoming_stock, forecast[:self.lead_time])
            
            order_quantity = max(0, deficit)
            order_quantity = ((order_quantity + self.lot_size - 1) // self.lot_size) * self.lot_size
            
            self.logger.info(f"Рекомендуемый заказ для {item_id}: {order_quantity}")
            return order_quantity
            
        except Exception as e:
            self.logger.error(f"Ошибка при расчете заказа: {str(e)}")
            raise OrderCalculationError(f"Ошибка при расчете заказа: {str(e)}")

    def calculate_orders_batch(self, items_data: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Параллельный расчет заказов для нескольких товаров
        
        Args:
            items_data: список словарей с данными товаров
                      [{'item_id': str, 'current_stock': int, 'incoming_stock': int}, ...]
                      
        Returns:
            Dict[str, int]: словарь с рекомендуемыми заказами
        """
        results = {}
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_processes) as executor:
            # Параллельное выполнение расчетов
            future_to_item = {
                executor.submit(
                    self.calculate_order,
                    item['item_id'],
                    item['current_stock'],
                    item['incoming_stock']
                ): item['item_id'] 
                for item in items_data
            }
            
            # Сбор результатов
            for future in concurrent.futures.as_completed(future_to_item):
                item_id = future_to_item[future]
                try:
                    results[item_id] = future.result()
                except Exception as e:
                    self.logger.error(f"Ошибка при расчете заказа для {item_id}: {str(e)}")
                    results[item_id] = 0
                    
        return results

    def _calculate_trend(self, sales_data: pd.DataFrame) -> float:
        """
        Расчет тренда как скользящее среднее за check_trend_days дней.
        Учитывает только положительный тренд.
        
        Args:
            sales_data: DataFrame с данными продаж
            
        Returns:
            float: значение тренда
        """
        if len(sales_data) < self.check_trend_days:
            return 0.0
            
        recent_sales = sales_data['quantity'].tail(self.check_trend_days)
        if recent_sales.sum() == 0:
            return 0.0
            
        trend = (recent_sales.iloc[-1] - recent_sales.iloc[0]) / self.check_trend_days
        return max(0.0, trend)

    def _apply_seasonality(self, forecast: pd.Series) -> pd.Series:
        """
        Применение сезонных коэффициентов.
        Умножает прогноз на соответствующие коэффициенты по месяцам.
        
        Args:
            forecast: Series с прогнозом
            
        Returns:
            Series с учетом сезонности
        """
        seasonal_forecast = forecast.copy()
        for month, coeff in self.seasonality_coeffs.items():
            mask = seasonal_forecast.index.month == month
            seasonal_forecast[mask] *= coeff
        return seasonal_forecast

    def _adjust_for_growth_limit(self, forecast: pd.Series, history: pd.Series) -> pd.Series:
        """
        Ограничение роста прогноза.
        Не позволяет прогнозу превышать среднее историческое значение более чем в max_growth_multiplier раз.
        
        Args:
            forecast: Series с прогнозом
            history: Series с историческими данными
            
        Returns:
            Series с ограниченным ростом
        """
        if len(history) == 0:
            return forecast
            
        avg_history = history.mean()
        max_allowed = avg_history * self.max_growth_multiplier
        return forecast.apply(lambda x: min(x, max_allowed))

    def _handle_lost_demand(self, forecast: pd.Series, item_id: str) -> pd.Series:
        """
        Учет потерянного спроса.
        Увеличивает прогноз на коэффициент потерянного спроса для соответствующей категории.
        
        Args:
            forecast: Series с прогнозом
            item_id: идентификатор товара
            
        Returns:
            Series с учетом потерянного спроса
            
        Raises:
            ValueError: при некорректном формате item_id
        """
        try:
            category = item_id.split('_')[0]  # извлечение категории из item_id
        except IndexError:
            self.logger.error(f"Некорректный формат item_id: {item_id}")
            raise ValueError("item_id должен быть в формате 'category_item'")
            
        factor = self.lost_demand_factor.get(category, 0)  # получение коэффициента для категории
        return forecast * (1 + factor)

    def _calculate_deficit(self, current_stock: int, incoming_stock: int, 
                          forecast: pd.Series) -> int:
        """
        Расчет дефицита между текущим запасом и прогнозируемым спросом.
        
        Args:
            current_stock: текущий запас
            incoming_stock: поступающий запас
            forecast: Series с прогнозом
            
        Returns:
            int: значение дефицита
        """
        total_available = current_stock + incoming_stock
        total_demand = forecast.sum()
        return max(0, int(total_demand - total_available))

    def visualize_forecast(self, item_id: str, days_ahead: int) -> None:
        """
        Визуализация прогноза и исторических данных.
        Создает график с прогнозом и историей продаж.
        
        Args:
            item_id: идентификатор товара
            days_ahead: количество дней для прогноза
            
        Raises:
            ValueError: при некорректных входных параметрах
        """
        if not isinstance(days_ahead, int) or days_ahead <= 0:
            raise ValueError("days_ahead должен быть положительным целым числом")
            
        # Получение прогноза
        forecast = self.predict(item_id, days_ahead)
        
        # Создание графика
        plt.figure(figsize=(12, 6))
        plt.plot(forecast.index, forecast.values, label='Прогноз', color='blue')
        
        # Добавление исторических данных
        if self.sales_history is not None:
            item_sales = self.sales_history[self.sales_history['item_id'] == item_id]
            if len(item_sales) > 0:
                plt.plot(item_sales['date'], item_sales['quantity'], 
                        label='Исторические данные', color='gray', alpha=0.5)
        
        # Настройка внешнего вида графика
        plt.title(f'Прогноз спроса для товара {item_id}')
        plt.xlabel('Дата')
        plt.ylabel('Количество')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def optimize_data_storage(self) -> None:
        """
        Оптимизация хранения данных для больших наборов
        """
        if self.sales_history is not None:
            # Оптимизация типов данных
            self.sales_history['quantity'] = pd.to_numeric(
                self.sales_history['quantity'], 
                downcast='integer'
            )
            
            # Создание индекса для ускорения поиска
            self.sales_history.set_index(['item_id', 'date'], inplace=True)
            self.sales_history.sort_index(inplace=True)
            
        if self.past_orders is not None:
            # Оптимизация типов данных
            for col in ['ordered_qty', 'sold_qty']:
                self.past_orders[col] = pd.to_numeric(
                    self.past_orders[col], 
                    downcast='integer'
                )
                
            # Создание индекса для ускорения поиска
            self.past_orders.set_index(['item_id', 'order_id'], inplace=True)
            self.past_orders.sort_index(inplace=True)

    def clear_cache(self) -> None:
        """Очистка кэша прогнозов"""
        self.cache.clear()
        self.logger.info("Кэш прогнозов очищен")