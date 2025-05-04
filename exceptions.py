class DemandForecastingError(Exception):
    """Базовый класс для всех исключений системы прогнозирования"""
    pass

class ConfigurationError(DemandForecastingError):
    """Ошибка в конфигурации"""
    pass

class DataValidationError(DemandForecastingError):
    """Ошибка валидации данных"""
    pass

class OutlierError(DemandForecastingError):
    """Ошибка, связанная с выбросами в данных"""
    pass

class PredictionError(DemandForecastingError):
    """Ошибка при выполнении прогноза"""
    pass

class OrderCalculationError(DemandForecastingError):
    """Ошибка при расчете заказа"""
    pass 