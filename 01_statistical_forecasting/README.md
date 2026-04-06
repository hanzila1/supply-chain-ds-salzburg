# Part I — Statistical Forecasting

Classical time series forecasting models, implemented from scratch in Python.
Based on Vandeput (2021), Chapters 1–11.

## Notebooks

| # | Notebook | Chapter | Concepts |
|---|----------|---------|----------|
| 01 | `01_moving_average.ipynb` | Ch 1 | MA model, naïve forecast, reactivity vs smoothness |
| 02 | `02_exponential_smoothing.ipynb` | Ch 3 | SES, alpha optimisation |
| 03 | `03_double_exponential_smoothing.ipynb` | Ch 5, 7 | Holt's method, damped trend |
| 04 | `04_triple_exponential_smoothing.ipynb` | Ch 9, 11 | Holt-Winters multiplicative & additive |
| 05 | `05_outlier_handling.ipynb` | Ch 10 | Winsorization, std dev method |
| 06 | `06_forecast_kpis.ipynb` | Ch 2 | MAE, RMSE, MAPE, Bias, model comparison |

## Key Concepts

**The forecasting model progression:**
```
Moving Average → Exponential Smoothing → Holt's (+ trend) → Holt-Winters (+ seasonality)
```

Each step adds one capability while inheriting the previous model's strengths.

## Running the Notebooks

```bash
cd 01_statistical_forecasting
jupyter notebook
```

All notebooks are self-contained — they generate their own synthetic demand data.
No external data download required for Part I.
