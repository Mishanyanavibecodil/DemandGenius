from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import logging
from dataclasses import dataclass
from enum import Enum

class WeatherCondition(Enum):
    """Типы погодных условий"""
    SUNNY = "sunny"
    CLOUDY = "cloudy"
    RAINY = "rainy"
    SNOWY = "snowy"
    STORMY = "stormy"

@dataclass
class Holiday:
    """Информация о празднике"""
    name: str
    date: datetime
    type: str  # national, regional, etc.
    impact: float  # влияние на продажи (множитель)

class ExternalFactorsProcessor:
    """Класс для обработки внешних факторов"""
    
    def __init__(self, config: Dict):
        """
        Инициализация процессора внешних факторов
        
        Args:
            config: конфигурация с API ключами и настройками
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.weather_api_key = config.get('weather_api_key')
        self.holidays: List[Holiday] = []
        
    def get_weather_data(
        self,
        location: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Получение данных о погоде
        
        Args:
            location: местоположение
            start_date: начальная дата
            end_date: конечная дата
            
        Returns:
            pd.DataFrame: данные о погоде
        """
        if not self.weather_api_key:
            self.logger.warning("API ключ для погоды не указан")
            return pd.DataFrame()
            
        try:
            # Здесь должен быть реальный API запрос
            # Это пример структуры данных
            dates = pd.date_range(start_date, end_date)
            weather_data = pd.DataFrame({
                'date': dates,
                'temperature': np.random.normal(20, 5, len(dates)),
                'precipitation': np.random.uniform(0, 10, len(dates)),
                'condition': np.random.choice(list(WeatherCondition), len(dates))
            })
            return weather_data
        except Exception as e:
            self.logger.error(f"Ошибка при получении данных о погоде: {e}")
            return pd.DataFrame()
            
    def load_holidays(self, year: int) -> None:
        """
        Загрузка информации о праздниках
        
        Args:
            year: год для загрузки праздников
        """
        # Здесь должна быть реальная загрузка праздников
        # Это пример данных
        self.holidays = [
            Holiday("Новый год", datetime(year, 1, 1), "national", 1.5),
            Holiday("8 марта", datetime(year, 3, 8), "national", 1.3),
            Holiday("День Победы", datetime(year, 5, 9), "national", 1.2)
        ]
        
    def get_holiday_impact(self, date: datetime) -> float:
        """
        Получение влияния праздника на продажи
        
        Args:
            date: дата для проверки
            
        Returns:
            float: множитель влияния праздника
        """
        for holiday in self.holidays:
            if holiday.date.date() == date.date():
                return holiday.impact
        return 1.0
        
    def process_external_factors(
        self,
        data: pd.DataFrame,
        location: str
    ) -> pd.DataFrame:
        """
        Обработка внешних факторов для данных
        
        Args:
            data: исходные данные
            location: местоположение
            
        Returns:
            pd.DataFrame: данные с учетом внешних факторов
        """
        # Получение данных о погоде
        weather_data = self.get_weather_data(
            location,
            data.index.min(),
            data.index.max()
        )
        
        if not weather_data.empty:
            # Объединение с данными о погоде
            data = data.join(weather_data.set_index('date'))
            
            # Добавление признаков погоды
            data['is_rainy'] = (data['condition'] == WeatherCondition.RAINY).astype(int)
            data['is_snowy'] = (data['condition'] == WeatherCondition.SNOWY).astype(int)
            data['is_stormy'] = (data['condition'] == WeatherCondition.STORMY).astype(int)
            
        # Добавление влияния праздников
        data['holiday_impact'] = data.index.map(self.get_holiday_impact)
        
        # Корректировка продаж с учетом внешних факторов
        data['adjusted_quantity'] = data['quantity'] * data['holiday_impact']
        
        return data
        
    def get_seasonal_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Расчет сезонных факторов
        
        Args:
            data: данные для анализа
            
        Returns:
            pd.DataFrame: сезонные факторы
        """
        # Расчет средних продаж по дням недели
        daily_avg = data.groupby(data.index.dayofweek)['quantity'].mean()
        overall_avg = data['quantity'].mean()
        
        # Расчет сезонных индексов
        seasonal_factors = daily_avg / overall_avg
        
        return pd.DataFrame({
            'day_of_week': range(7),
            'seasonal_factor': seasonal_factors
        })
        
    def adjust_forecast(
        self,
        forecast: pd.Series,
        location: str,
        weather_data: Optional[pd.DataFrame] = None
    ) -> pd.Series:
        """
        Корректировка прогноза с учетом внешних факторов
        
        Args:
            forecast: исходный прогноз
            location: местоположение
            weather_data: данные о погоде
            
        Returns:
            pd.Series: скорректированный прогноз
        """
        adjusted_forecast = forecast.copy()
        
        # Корректировка по погоде
        if weather_data is not None:
            for date in forecast.index:
                if date in weather_data.index:
                    weather = weather_data.loc[date]
                    if weather['condition'] in [WeatherCondition.RAINY, WeatherCondition.SNOWY]:
                        adjusted_forecast[date] *= 0.8
                    elif weather['condition'] == WeatherCondition.STORMY:
                        adjusted_forecast[date] *= 0.6
                        
        # Корректировка по праздникам
        for date in forecast.index:
            holiday_impact = self.get_holiday_impact(date)
            adjusted_forecast[date] *= holiday_impact
            
        return adjusted_forecast 