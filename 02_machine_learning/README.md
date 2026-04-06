# Part II — Machine Learning

ML-based demand forecasting, from decision trees to XGBoost with external drivers.
Based on Vandeput (2021), Chapters 12–24.

## Notebooks

| # | Notebook | Concepts |
|---|----------|----------|
| 01 | `01_decision_trees.ipynb` | Tree models, feature splits, overfitting |
| 02 | `02_random_forests.ipynb` | Ensemble learning, bagging, feature importance |
| 03 | `03_xgboost_forecasting.ipynb` | XGBoost, feature engineering, time-series CV |
| 04 | `04_external_demand_drivers.ipynb` | Weather API, events, leading indicators |
| 05 | `05_neural_networks.ipynb` | LSTM basics for sequence forecasting |

## The Key Insight

The jump from statistical models to ML is not just about accuracy.
It is about the ability to incorporate **external signals** that pure time series
models cannot use.

```
Statistical model: f(demand history) → forecast
ML model:          f(demand history + weather + events + ...) → forecast
```

Notebook `04` demonstrates this gap and how to close it.

## Data Requirements

- Notebooks 01–03: self-contained synthetic data
- Notebook 04: fetches real weather data from Open-Meteo (free, no key)
- Notebook 05: self-contained synthetic sequences
