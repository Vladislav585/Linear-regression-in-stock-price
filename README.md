# Stock Price Prediction System (VTBR)
## Description
This project implements a modular software system for analyzing and forecasting the stock value of VTB Bank (ticker: VTBR) using Deep Learning algorithms.

## Key Features
* Automatic data ingestion: Direct integration with the Moscow Exchange API (MOEX ISS API).
* Intelligent preprocessing: Built in logic for automatic correction of the 5000:1 stock split from July 2024.
* Feature engineering: Application of auto regressive lags and Simple Moving Averages.
* Deep Learning architecture: Multi layer Perceptron (MLP) built with TensorFlow and Keras.
* Training stability: Usage of Dropout layers and EarlyStopping callback to prevent model overfitting.
* Trend prediction: Recursive forecasting method to generate a smooth 22 day price trend.

## Technologies
* Python 3.10+
* TensorFlow / Keras
* Pandas / NumPy
* Scikit learn
* Matplotlib
* Requests

## Project Architecture
* main.py: Entry point for the full execution pipeline.
* Loading_and_Processing.py: Scripts for API interaction and data cleaning.
* Neuron_network.py: Neural network model definition and forecasting logic.

## Installation and Setup
```bash
git clone https://github.com/yourusername/vtb-stock-prediction.git
pip install tensorflow pandas numpy scikit-learn matplotlib requests
python main.py
```bash

--------------------------------------------------------------------------------------------------------
Система прогнозирования стоимости акций (VTBR)

## Описание
Данный проект представляет собой модульную программную систему для анализа и
прогнозирования курсовой стоимости акций ПАО ВТБ (тикер: VTBR) с
использованием алгоритмов глубокого обучения.

## Ключевые особенности:
  - Автоматическая загрузка данных: Прямая интеграция с API Московской Биржи
    (MOEX ISS API).
  - Интеллектуальная предобработка: Встроенная логика корректировки консолидации
    акций 5000:1 (июль 2024 г.).
  - Генерация признаков: Применение авторегрессионных лагов и скользящих
    средних.
  - Архитектура Deep Learning: Многослойный перцептрон (MLP) на базе TensorFlow
    и Keras.
  - Стабильность обучения: Использование слоев Dropout и функции EarlyStopping
    для предотвращения переобучения.
  - Прогнозирование тренда: Рекурсивный метод для генерации плавного прогноза
    цены на 22 торговых дня.

## Технологии:
  - Python 3.10+
  - TensorFlow / Keras
  - Pandas / NumPy
  - Scikit learn
  - Matplotlib
  - Requests

## Структура проекта:
  - main.py: Точка входа для запуска полного цикла обработки и прогноза.
  - Loading_and_Processing.py: Модуль для взаимодействия с API и очистки данных.
  - Neuron_network.py: Описание архитектуры нейросети и функции прогнозирования.

## Установка и запуск:
```bash
git clone https://github.com/yourusername/vtb-stock-prediction.git
pip install tensorflow pandas numpy scikit-learn matplotlib requests
python main.py
```bash

