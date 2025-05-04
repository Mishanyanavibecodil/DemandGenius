from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
import yaml
import xml.etree.ElementTree as ET
import csv
import pickle
import h5py
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.csv as csv
import pyarrow.json as json
import pyarrow.feather as feather
from abc import ABC, abstractmethod

class DataFormatHandler(ABC):
    """Абстрактный класс для обработки форматов данных"""
    
    @abstractmethod
    def read(self, path: str) -> pd.DataFrame:
        """
        Чтение данных
        
        Args:
            path: Путь к файлу
            
        Returns:
            pd.DataFrame: Прочитанные данные
        """
        pass
        
    @abstractmethod
    def write(self, data: pd.DataFrame, path: str):
        """
        Запись данных
        
        Args:
            data: Данные для записи
            path: Путь к файлу
        """
        pass

class CSVHandler(DataFormatHandler):
    """Обработчик CSV формата"""
    
    def __init__(self, **kwargs):
        """
        Инициализация обработчика
        
        Args:
            **kwargs: Параметры чтения/записи
        """
        self.kwargs = kwargs
        self.logger = logging.getLogger(__name__)
        
    def read(self, path: str) -> pd.DataFrame:
        """
        Чтение CSV файла
        
        Args:
            path: Путь к файлу
            
        Returns:
            pd.DataFrame: Прочитанные данные
        """
        try:
            return pd.read_csv(path, **self.kwargs)
        except Exception as e:
            self.logger.error(f"Ошибка при чтении CSV файла: {e}")
            raise
            
    def write(self, data: pd.DataFrame, path: str):
        """
        Запись в CSV файл
        
        Args:
            data: Данные для записи
            path: Путь к файлу
        """
        try:
            data.to_csv(path, **self.kwargs)
        except Exception as e:
            self.logger.error(f"Ошибка при записи в CSV файл: {e}")
            raise

class JSONHandler(DataFormatHandler):
    """Обработчик JSON формата"""
    
    def __init__(self, **kwargs):
        """
        Инициализация обработчика
        
        Args:
            **kwargs: Параметры чтения/записи
        """
        self.kwargs = kwargs
        self.logger = logging.getLogger(__name__)
        
    def read(self, path: str) -> pd.DataFrame:
        """
        Чтение JSON файла
        
        Args:
            path: Путь к файлу
            
        Returns:
            pd.DataFrame: Прочитанные данные
        """
        try:
            return pd.read_json(path, **self.kwargs)
        except Exception as e:
            self.logger.error(f"Ошибка при чтении JSON файла: {e}")
            raise
            
    def write(self, data: pd.DataFrame, path: str):
        """
        Запись в JSON файл
        
        Args:
            data: Данные для записи
            path: Путь к файлу
        """
        try:
            data.to_json(path, **self.kwargs)
        except Exception as e:
            self.logger.error(f"Ошибка при записи в JSON файл: {e}")
            raise

class ParquetHandler(DataFormatHandler):
    """Обработчик Parquet формата"""
    
    def __init__(self, **kwargs):
        """
        Инициализация обработчика
        
        Args:
            **kwargs: Параметры чтения/записи
        """
        self.kwargs = kwargs
        self.logger = logging.getLogger(__name__)
        
    def read(self, path: str) -> pd.DataFrame:
        """
        Чтение Parquet файла
        
        Args:
            path: Путь к файлу
            
        Returns:
            pd.DataFrame: Прочитанные данные
        """
        try:
            return pd.read_parquet(path, **self.kwargs)
        except Exception as e:
            self.logger.error(f"Ошибка при чтении Parquet файла: {e}")
            raise
            
    def write(self, data: pd.DataFrame, path: str):
        """
        Запись в Parquet файл
        
        Args:
            data: Данные для записи
            path: Путь к файлу
        """
        try:
            data.to_parquet(path, **self.kwargs)
        except Exception as e:
            self.logger.error(f"Ошибка при записи в Parquet файл: {e}")
            raise

class HDF5Handler(DataFormatHandler):
    """Обработчик HDF5 формата"""
    
    def __init__(self, **kwargs):
        """
        Инициализация обработчика
        
        Args:
            **kwargs: Параметры чтения/записи
        """
        self.kwargs = kwargs
        self.logger = logging.getLogger(__name__)
        
    def read(self, path: str) -> pd.DataFrame:
        """
        Чтение HDF5 файла
        
        Args:
            path: Путь к файлу
            
        Returns:
            pd.DataFrame: Прочитанные данные
        """
        try:
            return pd.read_hdf(path, **self.kwargs)
        except Exception as e:
            self.logger.error(f"Ошибка при чтении HDF5 файла: {e}")
            raise
            
    def write(self, data: pd.DataFrame, path: str):
        """
        Запись в HDF5 файл
        
        Args:
            data: Данные для записи
            path: Путь к файлу
        """
        try:
            data.to_hdf(path, **self.kwargs)
        except Exception as e:
            self.logger.error(f"Ошибка при записи в HDF5 файл: {e}")
            raise

class XMLHandler(DataFormatHandler):
    """Обработчик XML формата"""
    
    def __init__(self, root_tag: str = 'data', row_tag: str = 'row', **kwargs):
        """
        Инициализация обработчика
        
        Args:
            root_tag: Тег корневого элемента
            row_tag: Тег элемента строки
            **kwargs: Дополнительные параметры
        """
        self.root_tag = root_tag
        self.row_tag = row_tag
        self.kwargs = kwargs
        self.logger = logging.getLogger(__name__)
        
    def read(self, path: str) -> pd.DataFrame:
        """
        Чтение XML файла
        
        Args:
            path: Путь к файлу
            
        Returns:
            pd.DataFrame: Прочитанные данные
        """
        try:
            tree = ET.parse(path)
            root = tree.getroot()
            
            data = []
            for row in root.findall(self.row_tag):
                row_data = {}
                for child in row:
                    row_data[child.tag] = child.text
                data.append(row_data)
                
            return pd.DataFrame(data)
        except Exception as e:
            self.logger.error(f"Ошибка при чтении XML файла: {e}")
            raise
            
    def write(self, data: pd.DataFrame, path: str):
        """
        Запись в XML файл
        
        Args:
            data: Данные для записи
            path: Путь к файлу
        """
        try:
            root = ET.Element(self.root_tag)
            
            for _, row in data.iterrows():
                row_elem = ET.SubElement(root, self.row_tag)
                for col, val in row.items():
                    col_elem = ET.SubElement(row_elem, col)
                    col_elem.text = str(val)
                    
            tree = ET.ElementTree(root)
            tree.write(path)
        except Exception as e:
            self.logger.error(f"Ошибка при записи в XML файл: {e}")
            raise

class DataFormatFactory:
    """Фабрика для создания обработчиков форматов данных"""
    
    @staticmethod
    def create_handler(format_type: str, **kwargs) -> DataFormatHandler:
        """
        Создание обработчика формата
        
        Args:
            format_type: Тип формата
            **kwargs: Параметры обработчика
            
        Returns:
            DataFormatHandler: Обработчик формата
        """
        if format_type == 'csv':
            return CSVHandler(**kwargs)
        elif format_type == 'json':
            return JSONHandler(**kwargs)
        elif format_type == 'parquet':
            return ParquetHandler(**kwargs)
        elif format_type == 'hdf5':
            return HDF5Handler(**kwargs)
        elif format_type == 'xml':
            return XMLHandler(**kwargs)
        else:
            raise ValueError(f"Неподдерживаемый формат данных: {format_type}")
            
class DataConverter:
    """Конвертер между различными форматами данных"""
    
    def __init__(self):
        """Инициализация конвертера"""
        self.logger = logging.getLogger(__name__)
        
    def convert(self, input_path: str, output_path: str, input_format: str, output_format: str, **kwargs):
        """
        Конвертация данных между форматами
        
        Args:
            input_path: Путь к входному файлу
            output_path: Путь к выходному файлу
            input_format: Формат входного файла
            output_format: Формат выходного файла
            **kwargs: Параметры чтения/записи
        """
        try:
            # Создание обработчиков
            input_handler = DataFormatFactory.create_handler(input_format, **kwargs)
            output_handler = DataFormatFactory.create_handler(output_format, **kwargs)
            
            # Чтение данных
            data = input_handler.read(input_path)
            
            # Запись данных
            output_handler.write(data, output_path)
            
        except Exception as e:
            self.logger.error(f"Ошибка при конвертации данных: {e}")
            raise