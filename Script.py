import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

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
            ValueError: при некорректных значениях параметров конфигурации
            KeyError: при отсутствии обязательных параметров
        """
        # Настройка системы логирования
        self._setup_logging()
        self.logger.info("Инициализация DemandForecaster")
        
        # Проверка наличия обязательных параметров
        required_params = ['lead_time', 'order_period', 'alpha']
        for param in required_params:
            if param not in config:
                raise KeyError(f"Отсутствует обязательный параметр: {param}")
        
        # Инициализация основных параметров
        self.lead_time = config['lead_time']  # время поставки
        self.order_period = config['order_period']  # период заказа
        self.alpha = config['alpha']  # коэффициент сглаживания
        self.check_trend_days = config.get('check_trend_days', 30)  # период для анализа тренда
        self.growth_factor = config.get('growth_factor', 1.2)  # фактор роста
        self.lot_size = config.get('lot_size', 10)  # размер партии
        self.max_growth_multiplier = config.get('max_growth_multiplier', 3.0)  # максимальный рост
        
        # Инициализация категорийных параметров
        self.lost_demand_factor = config.get('lost_demand_factor', {})  # коэффициенты потерянного спроса
        self.seasonality_coeffs = config.get('seasonality_coeffs', {})  # сезонные коэффициенты
        
        # Инициализация хранилищ данных
        self.sales_history: Optional[pd.DataFrame] = None  # история продаж
        self.past_orders: Optional[pd.DataFrame] = None    # история заказов
        
        # Валидация конфигурации
        self._validate_config()

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

    def _validate_config(self) -> None:
        """
        Проверка корректности конфигурации.
        Проверяет типы и диапазоны значений параметров.
        
        Raises:
            ValueError: при некорректных значениях параметров
        """
        # Проверка коэффициента сглаживания
        if not isinstance(self.alpha, (int, float)) or not 0 <= self.alpha <= 1:
            raise ValueError("Alpha должен быть между 0 и 1")
            
        # Проверка размера партии
        if not isinstance(self.lot_size, int) or self.lot_size <= 0:
            raise ValueError("Размер партии должен быть положительным числом")
            
        # Проверка времени поставки
        if not isinstance(self.lead_time, int) or self.lead_time <= 0:
            raise ValueError("Время поставки должно быть положительным числом")
            
        # Проверка периода заказа
        if not isinstance(self.order_period, int) or self.order_period <= 0:
            raise ValueError("Период заказа должен быть положительным числом")
            
        # Проверка максимального множителя роста
        if not isinstance(self.max_growth_multiplier, (int, float)) or self.max_growth_multiplier <= 1:
            raise ValueError("Максимальный множитель роста должен быть больше 1")

    def fit(self, sales_history: pd.DataFrame, past_orders: pd.DataFrame) -> None:
        """
        Обучение модели на исторических данных.
        Загружает и подготавливает данные для прогнозирования.
        
        Args:
            sales_history: DataFrame с колонками ['date', 'item_id', 'quantity']
            past_orders: DataFrame с колонками ['order_id', 'item_id', 'ordered_qty', 'sold_qty']
            
        Raises:
            ValueError: при отсутствии необходимых колонок в данных
        """
        self.logger.info("Начало обучения модели")
        
        # Проверка наличия необходимых колонок
        required_columns = {
            'sales_history': ['date', 'item_id', 'quantity'],
            'past_orders': ['order_id', 'item_id', 'ordered_qty', 'sold_qty']
        }
        
        # Проверка каждой таблицы на наличие необходимых колонок
        for df_name, columns in required_columns.items():
            df = sales_history if df_name == 'sales_history' else past_orders
            missing_columns = [col for col in columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"В {df_name} отсутствуют колонки: {missing_columns}")
        
        # Подготовка данных
        sales_history['date'] = pd.to_datetime(sales_history['date'])  # преобразование дат
        sales_history.sort_values(['item_id', 'date'], inplace=True)   # сортировка
        
        # Сохранение подготовленных данных
        self.sales_history = sales_history.copy()
        self.past_orders = past_orders.copy()
        
        self.logger.info(f"Загружено {len(sales_history)} записей продаж")
        self.logger.info(f"Загружено {len(past_orders)} записей заказов")

    def predict(self, item_id: str, days_ahead: int) -> pd.Series:
        """
        Прогнозирование спроса для товара.
        Выполняет многоэтапный прогноз с учетом различных факторов.
        
        Args:
            item_id: идентификатор товара
            days_ahead: количество дней для прогноза
            
        Returns:
            Series с прогнозируемыми значениями
            
        Raises:
            ValueError: при некорректных входных параметрах
            KeyError: при отсутствии товара в истории продаж
        """
        # Валидация входных параметров
        if not isinstance(item_id, str):
            raise ValueError("item_id должен быть строкой")
        if not isinstance(days_ahead, int) or days_ahead <= 0:
            raise ValueError("days_ahead должен быть положительным целым числом")
            
        self.logger.info(f"Прогнозирование для товара {item_id}, горизонт {days_ahead} дней")
        
        # Получение истории продаж для товара
        item_sales = self.sales_history[self.sales_history['item_id'] == item_id].copy()
        
        # Обработка случая отсутствия истории
        if len(item_sales) == 0:
            self.logger.warning(f"Нет истории продаж для товара {item_id}")
            return pd.Series(index=pd.date_range(start=datetime.now(), periods=days_ahead))
            
        # Многоэтапный прогноз
        forecast = self._exp_smoothing(item_sales)  # базовый прогноз
        trend = self._calculate_trend(item_sales)   # расчет тренда
        seasonal_forecast = self._apply_seasonality(forecast)  # учет сезонности
        limited_forecast = self._adjust_for_growth_limit(seasonal_forecast, item_sales['quantity'])  # ограничение роста
        final_forecast = self._handle_lost_demand(limited_forecast, item_id)  # учет потерянного спроса
        
        # Обеспечение неотрицательности прогноза
        final_forecast = final_forecast.clip(lower=0)
        
        return final_forecast

    def calculate_order(self, item_id: str, current_stock: int, incoming_stock: int) -> int:
        """
        Расчет рекомендуемого заказа.
        Учитывает текущий запас, поступающий запас и прогноз спроса.
        
        Args:
            item_id: идентификатор товара
            current_stock: текущий запас
            incoming_stock: поступающий запас
            
        Returns:
            рекомендованное количество для заказа
            
        Raises:
            ValueError: при некорректных значениях запасов
        """
        # Валидация входных параметров
        if not isinstance(current_stock, int) or current_stock < 0:
            raise ValueError("current_stock должен быть неотрицательным целым числом")
        if not isinstance(incoming_stock, int) or incoming_stock < 0:
            raise ValueError("incoming_stock должен быть неотрицательным целым числом")
            
        self.logger.info(f"Расчет заказа для товара {item_id}")
        
        # Прогноз на период поставки и заказа
        forecast = self.predict(item_id, self.lead_time + self.order_period)
        
        # Расчет дефицита
        deficit = self._calculate_deficit(current_stock, incoming_stock, forecast[:self.lead_time])
        
        # Округление до размера партии
        order_quantity = max(0, deficit)
        order_quantity = ((order_quantity + self.lot_size - 1) // self.lot_size) * self.lot_size
        
        self.logger.info(f"Рекомендуемый заказ для {item_id}: {order_quantity}")
        return order_quantity

    def _exp_smoothing(self, sales_data: pd.DataFrame) -> pd.Series:
        """
        Экспоненциальное сглаживание.
        Использует свертку для эффективного расчета.
        
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