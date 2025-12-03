# Stock_Price_Prediction
# Система прогнозирования акций и фьючерсов

> Выпускная квалификационная работа по направлению "Искусственный интеллект"

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18+-blue.svg)](https://react.dev/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Предварительное Описание проекта

Гибридная система прогнозирования стоимости акций компаний и фьючерсов на фондовые индексы, объединяющая:
- Статистические методы (ARIMA, GARCH)
- Глубокое обучение (LSTM, Transformer)
- Анализ новостей (NLP, FinBERT)
- Обучение с подкреплением (PPO)
- Мультимодальный AI

**Цель**: Создать точную систему прогнозирования с веб-интерфейсом (или telegramm bot) для демонстрации.

---

## Ключевые задачи

1. Сбор и обработка мультимодальных данных (цены, новости, фундаментальные показатели)
2. Разработка и обучение комплекса ML/DL моделей
3. Создание ансамбля моделей
4. Разработка REST API и веб-интерфейса
5. Бэктестинг и валидация на исторических данных
6. Сравнительный анализ подходов

---

## Архитектура системы (в проработке)
```
┌─────────────────────────────────────┐
│      ИСТОЧНИКИ ДАННЫХ               │
│  Yahoo Finance │ News API │ FRED    |
|  MOEX │ 
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│     ОБРАБОТКА И ХРАНЕНИЕ            │
│  TimescaleDB │ Redis │ Features     │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│      МОДЕЛИ ПРОГНОЗИРОВАНИЯ         │
│  ┌────────┐  ┌────────┐  ┌───────┐ │
│  │ ARIMA  │  │  LSTM  │  │FinBERT│ │
│  │ GARCH  │  │  TFT   │  │  NLP  │ │
│  │  VAR   │  │XGBoost │  │Sentim.│ │
│  └────────┘  └────────┘  └───────┘ │
│          ↓         ↓         ↓      │
│     ┌─────────────────────────┐    │
│     │  Meta-Learner (Ensemble)│    │
│     │       + RL Agent        │    │
│     └─────────────────────────┘    │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│        СЕРВИСНЫЙ СЛОЙ               │
│   FastAPI (Backend) + React (UI)    |
|    Или телеграм бот или Streamlite  │
└─────────────────────────────────────┘
```

---

## План работ

### Этап 1: Подготовка данных
- [ ] Настройка окружения (Python, PyTorch, CUDA)
- [ ] Регистрация API (MOEX, BCS, Yahoo Finance, Alpha Vantage, NewsAPI)
- [ ] Сбор исторических данных (3-10 лет, 10+ акций)
- [ ] Настройка TimescaleDB + PostgreSQL
- [ ] Exploratory Data Analysis (EDA)
- [ ] Feature Engineering (технические индикаторы, lag features)

**Результаты:**
- Датасет с ценами, объёмами, новостями  (частично собраны)
- EDA отчёт с визуализациями
- Feature store (10+ признаков)

---

### Этап 2: Базовые модели
- [ ] **Статистические модели**
  - ARIMA для прогноза цены
  - GARCH для волатильности
- [ ] **XGBoost модель**
  - Обучение на табличных данных
  - Feature importance анализ
- [ ] **Простая LSTM**
  - 2-3 слойная архитектура
  - Обучение на последовательностях
- [ ] **Базовый ансамбль** (simple averaging)

**Результаты:**
- 3 обученные модели
- Baseline метрики (MAE, RMSE, Directional Accuracy)
- Jupyter notebooks с экспериментами

---

###  Этап 3: Продвинутые модели
- [ ] **Temporal Fusion Transformer (TFT)**
  - Использование PyTorch Forecasting
  - Обучение на GPU (Paperspace/Colab)
  - Интерпретация attention weights
- [ ] **NLP модуль**
  - FinBERT для sentiment analysis
  - Сбор и обработка новостей
  - Агрегация sentiment scores
- [ ] **Мультимодальная модель**
  - Fusion layer (цены + новости + фундаментальные данные)
  - End-to-end обучение
- [ ] **Продвинутый ансамбль**
  - Stacking с meta-learner
  - Динамическое взвешивание

**Результаты:**
- TFT модель (SOTA для временных рядов)
- NLP pipeline для sentiment
- Мультимодальная fusion модель
- Улучшение точности на 15-20%

---

### Этап 4: RL агент (опционально)
- [ ] **Trading Environment**
  - State: цены, индикаторы, прогнозы моделей
  - Action: Buy/Hold/Sell
  - Reward: прибыль - риск - комиссии
- [ ] **PPO Agent**
  - Actor-Critic архитектура
  - Обучение на исторических данных
- [ ] **Бэктестинг**
  - Walk-forward backtesting
  - Финансовые метрики (Sharpe Ratio, Max Drawdown)

**Результаты:**
- RL-агент для торговых решений
- Backtesting framework
- Сравнение: RL vs статистические модели

---

### Этап 5: Backend разработка
- [ ] **FastAPI сервер**
  - REST API endpoints:
    - `POST /api/predict` - получить прогноз
    - `GET /api/historical/{ticker}` - исторические данные
    - `GET /api/models/performance` - метрики моделей
    - `POST /api/backtest` - бэктестинг стратегии
  - WebSocket для real-time обновлений
- [ ] **Model Serving**
  - Загрузка всех обученных моделей
  - Inference pipeline
  - Кэширование (Redis)
- [ ] **Документация**
  - Swagger/OpenAPI автогенерация

**Результаты:**
- REST API с документацией
- WebSocket для real-time данных
- Docker образ backend

---

### Этап 6: Frontend разработка
- [ ] **React приложение**
  - Dashboard с графиками (Plotly.js/Recharts)
  - Панель прогнозов с confidence intervals
  - Сравнение моделей
  - Технический и фундаментальный анализ
  - Новостная лента с sentiment
- [ ] **Responsive дизайн** (TailwindCSS)
- [ ] **Real-time обновления** (WebSocket)

**Результаты:**
- Полнофункциональный веб-интерфейс
- Интерактивные визуализации
- Docker образ frontend

---

### Этап 7: Финализация
- [ ] **Интеграция**
  - Docker Compose (Backend + Frontend + DB + Redis)
  - CI/CD pipeline (GitHub Actions)
- [ ] **Тестирование**
  - Unit tests (coverage > 80%)
  - Integration tests
  - Валидация моделей
- [ ] **Документация**
  - Текст ВКР (40-60 страниц)
  - Презентация (20-30 слайдов)
  - README, API docs
  - Демо видео (5-10 минут)

**Результаты:**
- Полностью работающая система
- Защищённая ВКР

---

## Технологический стек

### Backend
```
Python 3.10+
├── FastAPI              # REST API
├── PyTorch 2.0+         # Deep Learning
├── Statsmodels          # Statistical models
├── XGBoost/LightGBM     # Gradient Boosting
├── Transformers         # FinBERT, NLP
├── TA-Lib               # Technical indicators
├── Pandas/NumPy         # Data processing
└── MLflow               # Experiment tracking
```

### Frontend
```
TypeScript + React 18+
├── TanStack Query       # Data fetching
├── Recharts/Plotly      # Visualization
├── TailwindCSS          # Styling
├── Zustand              # State management
└── Vite                 # Build tool
```

### Database & Infrastructure
```
├── TimescaleDB          # Time-series data
├── PostgreSQL           # Relational data
├── Redis                # Caching
├── Docker               # Containerization
└── Nginx                # Reverse proxy
```

---

## Модели и методы

### Статистические методы
- **ARIMA** - прогноз цены
- **GARCH** - моделирование волатильности
- **VAR** - взаимосвязи между активами

### Machine Learning
- **XGBoost/LightGBM** - табличные данные
- **LSTM** - последовательности цен
- **Temporal Fusion Transformer** - SOTA для временных рядов

### Natural Language Processing
- **FinBERT** - sentiment analysis новостей
- **Named Entity Recognition** - извлечение упоминаний компаний
- **Event Detection** - классификация событий

### Reinforcement Learning
- **PPO (Proximal Policy Optimization)** - торговые решения
- **Actor-Critic** архитектура
- **Custom Trading Environment** с учётом комиссий и рисков

### Ensemble
- **Stacking** - meta-learner на базе XGBoost
- **Dynamic Weighting** - адаптивные веса моделей
- **RL-based Ensemble** - оптимизация через обучение с подкреплением

---

##  Ожидаемые результаты

| Метрика | Baseline (ARIMA) | Ensemble | Улучшение |
|---------|------------------|----------|-----------|
| MAE     | 2.5              | **1.2**  | -52% |
| RMSE    | 3.2              | **1.7** | -47% |
| MAPE    | 4.1%             | **2.1%** | -49% |
| Dir. Accuracy | 52%        | **64%** | +23% |
| Sharpe Ratio | 0.3         | **0.9** | +200% |

---

---

## Структура проекта
```
stock-prediction-system/
│
├── data/                    # Данные
│   ├── raw/                 # Сырые данные
│   ├── processed/           # Обработанные
│   └── collectors/          # Скрипты сбора
│
├── notebooks/               # Jupyter notebooks
│   ├── 01_eda.ipynb
│   ├── 02_baseline_models.ipynb
│   ├── 03_deep_learning.ipynb
│   └── 04_ensemble.ipynb
│
├── models/                  # ML модели
│   ├── statistical/         # ARIMA, GARCH
│   ├── ml/                  # LSTM, TFT, XGBoost
│   ├── nlp/                 # FinBERT
│   ├── rl/                  # PPO Agent
│   └── ensemble/            # Meta-learner
│
├── backend/                 # FastAPI backend
│   ├── api/                 # API endpoints
│   ├── services/            # Business logic
│   └── tests/               # Tests
│
├── frontend/                # React frontend
│   └── src/
│       ├── components/      # React компоненты
│       ├── hooks/           # Custom hooks
│       └── utils/           # Utilities
│
├── docs/                    # Документация
│   ├── thesis/              # Текст ВКР
│   └── presentation/        # Презентация
│
├── scripts/                 # Utility scripts
│   ├── train_models.py
│   └── backtest.py
│
├── docker-compose.yml       # Docker setup
├── requirements.txt         # Python deps
└── README.md               # Этот файл
```

---

##  Тестирование
```bash
# Unit tests
pytest tests/unit -v

# Integration tests
pytest tests/integration -v

# Coverage
pytest --cov=models --cov=backend
```

---

##  Источники данных

- **Цены**: [Yahoo Finance API](https://finance.yahoo.com/)
- **Фундаментальные данные**: [Alpha Vantage](https://www.alphavantage.co/)
- **Новости**: [NewsAPI](https://newsapi.org/), [GDELT](https://www.gdeltproject.org/)
- **Макроэкономика**: [FRED](https://fred.stlouisfed.org/)
- **Sentiment**: Twitter API (опционально)

---

## Литература
- собраны 39 научных работ для анализа.
- 


### Книги
1. **"В зеркале супермоделей"** - Кирилл Ильинский, Максим Буев
2. **"Управление портфелем ценных бумаг"** - Буренин
3. **"Advances in Financial Machine Learning"** - Marcos López de Prado
4. **"Machine Learning for Asset Managers"** - Marcos López de Prado
5. **"Algorithmic Trading"** - Ernest Chan

### Papers
1. [Temporal Fusion Transformers for Time Series Forecasting](https://arxiv.org/abs/1912.09363) (Google, 2019)
2. [FinBERT: Financial Sentiment Analysis](https://arxiv.org/abs/1908.10063) (Prosus AI, 2020)
3. [Deep Reinforcement Learning for Trading](https://arxiv.org/abs/1911.10107) (2019)

---

---

## Метрики качества

### Для регрессии
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **MAPE** (Mean Absolute Percentage Error)
- **Directional Accuracy** - точность предсказания направления

### Финансовые метрики
- **Sharpe Ratio** - соотношение доходность/риск
- **Maximum Drawdown** - максимальная просадка
- **Sortino Ratio** - downside risk
- **Win Rate** - процент прибыльных сделок

---
