"""
┌─────────────────────────────────────────────────────────────────────┐
│  DEMAND FORECASTING INTELLIGENCE PLATFORM                           │
│  Supply Chain Analytics Suite — Statistical Engine v2.0             │
│                                                                     │
│  Architecture: Modular, ML-ready                                    │
│  Models: Naïve · MA · SES · Holt's Linear · Damped Trend           │
│  Pending: XGBoost · LightGBM · LSTM (ML Studio)                    │
└─────────────────────────────────────────────────────────────────────┘
Run:  streamlit run app.py
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import io
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Demand Intelligence Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM
# ══════════════════════════════════════════════════════════════════════
C = {
    # Backgrounds — warm off-white, not stark white
    "bg":       "#f8f9fb",
    "surface":  "#ffffff",
    "surface2": "#f1f3f6",
    "border":   "#e2e6ec",
    "border2":  "#c8cdd6",
    # Text — near-black for strong contrast
    "text":     "#0f1923",
    "dim":      "#5a6475",
    # Accents — vivid but professional
    "teal":     "#0077b6",
    "blue":     "#0096c7",
    "amber":    "#e07b00",
    "red":      "#c0392b",
    "green":    "#1a7f5a",
    "purple":   "#6246ea",
    "pink":     "#d63384",
    # Chart series — distinct, colourblind-friendly
    "demand":   "#1a2332",
    "naive":    "#94a3b8",
    "ma":       "#e07b00",
    "ses":      "#0077b6",
    "holt":     "#6246ea",
    "damped":   "#c0392b",
    # Chart regions
    "train_bg": "rgba(0,119,182,0.04)",
    "test_bg":  "rgba(98,70,234,0.05)",
    "grid":     "#edf0f4",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor=C["surface"],
    plot_bgcolor=C["surface"],
    font=dict(color=C["dim"], family="'Inter', sans-serif", size=11),
    xaxis=dict(gridcolor=C["grid"], linecolor=C["border"], showgrid=True, zeroline=False),
    yaxis=dict(gridcolor=C["grid"], linecolor=C["border"], showgrid=True, zeroline=False),
    legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor=C["border"], borderwidth=1,
                font=dict(size=11, color=C["dim"])),
    margin=dict(l=50, r=20, t=40, b=40),
    hovermode="x unified",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, .stApp {
    background: #f8f9fb !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, .stDeployButton { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stHeader"] { background: transparent !important; }

/* ── Sidebar toggle — always visible ── */
[data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    color: #0f1923 !important;
    background: #ffffff !important;
    border: 1px solid #e2e6ec !important;
    border-radius: 0 6px 6px 0 !important;
    box-shadow: 2px 0 6px rgba(0,0,0,0.06) !important;
}

/* ── Layout ── */
.block-container { padding: 0.75rem 1.5rem 1rem 1.5rem !important; max-width: 100% !important; }
section.main > div { padding-top: 0 !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e2e6ec !important;
    box-shadow: 2px 0 8px rgba(0,0,0,0.04) !important;
}
[data-testid="stSidebar"] .block-container { padding: 0 !important; }

/* ── Nav radio ── */
[data-testid="stSidebar"] .stRadio > div { gap: 1px !important; }
[data-testid="stSidebar"] .stRadio label {
    background: transparent !important;
    border-radius: 0 !important;
    padding: 9px 18px !important;
    font-size: 13px !important;
    color: #5a6475 !important;
    cursor: pointer;
    transition: all 0.1s ease;
    width: 100%;
    border-left: 3px solid transparent !important;
    margin: 0 !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: #f1f3f6 !important;
    color: #0f1923 !important;
}
[data-testid="stSidebar"] .stRadio [aria-checked="true"] ~ div label,
[data-testid="stSidebar"] .stRadio input:checked ~ label {
    background: #eef3fb !important;
    color: #0077b6 !important;
    border-left: 3px solid #0077b6 !important;
    font-weight: 500 !important;
}

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: #ffffff !important;
    border: 1px solid #e2e6ec !important;
    border-radius: 8px !important;
    padding: 10px 14px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    color: #0f1923 !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
}
[data-testid="stMetricLabel"] {
    color: #5a6475 !important;
    font-size: 10.5px !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    font-weight: 500 !important;
}
[data-testid="stMetricDelta"] {
    font-size: 11px !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #ffffff !important;
    border: 1px solid #e2e6ec !important;
    border-radius: 8px !important;
    padding: 3px !important;
    gap: 2px !important;
}
.stTabs [data-baseweb="tab"] {
    color: #5a6475 !important;
    font-size: 12px !important;
    border-radius: 6px !important;
    padding: 5px 14px !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: #0077b6 !important;
    background: #eef3fb !important;
    font-weight: 500 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: #ffffff !important;
    border: 1px solid #c8cdd6 !important;
    color: #0f1923 !important;
    border-radius: 7px !important;
    font-size: 12px !important;
    padding: 6px 18px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    transition: all 0.1s ease !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
}
.stButton > button:hover {
    border-color: #0077b6 !important;
    color: #0077b6 !important;
    background: #eef3fb !important;
}

/* ── Sliders ── */
.stSlider [data-baseweb="slider"] { padding: 2px 0 !important; }
.stSlider p { font-size: 11.5px !important; color: #5a6475 !important; }

/* ── DataFrames ── */
.stDataFrame {
    border: 1px solid #e2e6ec !important;
    border-radius: 8px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
}
iframe { border-radius: 6px; }

/* ── Expanders ── */
.stExpander {
    border: 1px solid #e2e6ec !important;
    border-radius: 8px !important;
    background: #ffffff !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
}
details summary { color: #0f1923 !important; font-size: 13px !important; }

/* ── Upload ── */
[data-testid="stFileUploader"] {
    background: #ffffff !important;
    border: 1.5px dashed #c8cdd6 !important;
    border-radius: 8px !important;
    padding: 12px !important;
}

/* ── Select/input ── */
[data-testid="stSelectbox"] > div,
[data-testid="stNumberInput"] > div { font-size: 12px !important; }

/* ── Typography ── */
h1, h2, h3, h4 {
    color: #0f1923 !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
}
p, li { color: #5a6475 !important; font-size: 13px !important; }
hr { border-color: #e2e6ec !important; margin: 8px 0 !important; }
.stMarkdown code {
    background: #eef3fb !important;
    color: #0077b6 !important;
    border-radius: 4px !important;
    font-size: 11px !important;
    padding: 2px 5px !important;
}

/* ── Custom classes ── */
.page-header {
    display: flex; align-items: center; gap: 10px;
    padding: 0 0 10px 0; margin-bottom: 4px;
    border-bottom: 1px solid #e2e6ec;
}
.page-header h2 { margin: 0 !important; padding: 0 !important; font-size: 17px !important; }
.page-header .subtitle { color: #5a6475; font-size: 12px; margin-top: 1px; }

.section-label {
    font-size: 10.5px; font-weight: 600; color: #5a6475;
    text-transform: uppercase; letter-spacing: 0.1em;
    margin: 12px 0 6px 0; padding-bottom: 4px;
    border-bottom: 1px solid #e2e6ec;
}

.kb-box {
    background: #f1f3f6; border-left: 2px solid #0077b6;
    border-radius: 0 5px 5px 0; padding: 9px 13px;
    margin: 5px 0; font-size: 12px; color: #5a6475; line-height: 1.55;
}
.warn-box {
    background: #fdf5ec; border-left: 2px solid #e07b00;
    border-radius: 0 5px 5px 0; padding: 9px 13px;
    margin: 5px 0; font-size: 12px; color: #5a6475; line-height: 1.55;
}
.info-box {
    background: #eef3fb; border-left: 2px solid #0096c7;
    border-radius: 0 5px 5px 0; padding: 9px 13px;
    margin: 5px 0; font-size: 12px; color: #5a6475; line-height: 1.55;
}

.badge {
    display: inline-block; padding: 2px 9px; border-radius: 20px;
    font-size: 10.5px; font-weight: 600; font-family: 'JetBrains Mono', monospace;
}
.badge-green  { background: rgba(26,127,90,0.10);  color: #1a7f5a; border: 1px solid rgba(26,127,90,0.20);  }
.badge-red    { background: rgba(192,57,43,0.10);   color: #c0392b; border: 1px solid rgba(192,57,43,0.20);  }
.badge-amber  { background: rgba(224,123,0,0.10);   color: #e07b00; border: 1px solid rgba(224,123,0,0.20);  }
.badge-teal   { background: rgba(0,119,182,0.10);   color: #0077b6; border: 1px solid rgba(0,119,182,0.20);  }
.badge-blue   { background: rgba(0,150,199,0.10);   color: #0096c7; border: 1px solid rgba(0,150,199,0.20);  }
.badge-purple { background: rgba(98,70,234,0.10);   color: #6246ea; border: 1px solid rgba(98,70,234,0.20);  }

.stat-row { display: flex; gap: 10px; flex-wrap: wrap; margin: 6px 0; }
.stat-pill {
    background: #ffffff; border: 1px solid #e2e6ec; border-radius: 20px;
    padding: 3px 12px; font-size: 11.5px; color: #5a6475;
    font-family: 'JetBrains Mono', monospace;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}
.stat-pill span { color: #0f1923; font-weight: 500; }

.formula-card {
    background: #f8f9fb; border: 1px solid #e2e6ec; border-radius: 8px;
    padding: 14px 18px; margin: 8px 0;
}
.formula-title { font-size: 11px; color: #0077b6; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px; font-weight: 600; }

.sc-risk-low  { color: #1a7f5a; }
.sc-risk-med  { color: #e07b00; }
.sc-risk-high { color: #c0392b; }

.model-color-naive  { color: #94a3b8; }
.model-color-ma     { color: #e07b00; }
.model-color-ses    { color: #0077b6; }
.model-color-holt   { color: #6246ea; }
.model-color-damped { color: #c0392b; }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# ── MODULE 1: DATA ENGINE ─────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def generate_demand(
    n_periods: int = 60,
    base_level: float = 1000.0,
    trend_slope: float = 0.0,
    seasonality_amp: float = 0.0,
    season_length: int = 12,
    noise_std: float = 50.0,
    n_outliers: int = 0,
    outlier_magnitude: float = 3.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Synthetic demand = Level + Trend + Seasonality + Noise + Outliers
    """
    np.random.seed(seed)
    t = np.arange(n_periods)
    level  = np.full(n_periods, base_level)
    trend  = trend_slope * t
    season = (seasonality_amp * np.sin(2 * np.pi * t / season_length)
              if seasonality_amp > 0 else np.zeros(n_periods))
    noise  = np.random.normal(0, noise_std, n_periods)
    demand = level + trend + season + noise

    if n_outliers > 0:
        outlier_idx = np.random.choice(n_periods, size=min(n_outliers, n_periods), replace=False)
        for idx in outlier_idx:
            direction = np.random.choice([-1, 1])
            demand[idx] += direction * outlier_magnitude * noise_std

    return np.clip(demand, 0, None)


def decompose_demand(demand, trend_slope, base_level, seasonality_amp, season_length):
    """Return trend / seasonal / residual components."""
    n = len(demand)
    t = np.arange(n)
    trend_comp  = base_level + trend_slope * t
    season_comp = (seasonality_amp * np.sin(2 * np.pi * t / season_length)
                   if seasonality_amp > 0 else np.zeros(n))
    residual = demand - trend_comp - season_comp
    return trend_comp, season_comp, residual


def load_user_data(uploaded_file) -> pd.DataFrame | None:
    """Parse CSV or Excel upload."""
    try:
        name = uploaded_file.name.lower()
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        else:
            return None
        return df
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════
# ── MODULE 2: FORECASTING MODELS ─────────────────────────────────────
# Pure Python implementations — no external forecast libraries.
# ML_HOOK: Add ML models to MODEL_REGISTRY below.
# ══════════════════════════════════════════════════════════════════════

def naive_forecast(d: np.ndarray, extra_periods: int = 12) -> np.ndarray:
    """Naïve — last observed value carried forward. Benchmark baseline."""
    cols = len(d)
    f = np.full(cols + extra_periods, np.nan)
    for t in range(1, cols):
        f[t] = d[t - 1]
    f[cols:] = d[cols - 1]
    return f


def moving_average(d: np.ndarray, n: int = 3, extra_periods: int = 12) -> np.ndarray:
    """Moving Average — average of last n observed periods."""
    cols = len(d)
    f = np.full(cols + extra_periods, np.nan)
    for t in range(n, cols):
        f[t] = np.mean(d[t - n:t])
    f[cols:] = np.mean(d[cols - n:cols])
    return f


def simple_exp_smoothing(d: np.ndarray, alpha: float = 0.3,
                          extra_periods: int = 12) -> tuple:
    """
    Simple Exponential Smoothing (SES) — level only.
    F_{t+1} = α·D_t + (1-α)·F_t
    Returns: (forecast_array, level_array)
    """
    cols = len(d)
    f = np.full(cols + extra_periods, np.nan)
    a = np.full(cols + extra_periods, np.nan)
    a[0] = d[0]
    f[1] = a[0]
    for t in range(1, cols):
        a[t] = alpha * d[t] + (1 - alpha) * a[t - 1]
        if t + 1 < cols + extra_periods:
            f[t + 1] = a[t]
    f[cols:] = a[cols - 1]
    return f, a


def holts_linear(d: np.ndarray, alpha: float = 0.3, beta: float = 0.2,
                 extra_periods: int = 12) -> tuple:
    """
    Holt's Linear Trend — double exponential smoothing.
    L_t = α·D_t + (1-α)·(L_{t-1} + T_{t-1})
    T_t = β·(L_t - L_{t-1}) + (1-β)·T_{t-1}
    F_{t+m} = L_t + m·T_t
    Returns: (forecast_array, level_array, trend_array)
    """
    cols = len(d)
    f = np.full(cols + extra_periods, np.nan)
    a = np.full(cols + extra_periods, np.nan)
    b = np.full(cols + extra_periods, np.nan)
    a[0] = d[0]
    b[0] = d[1] - d[0] if len(d) > 1 else 0
    for t in range(1, cols):
        f[t] = a[t - 1] + b[t - 1]
        a[t] = alpha * d[t] + (1 - alpha) * (a[t - 1] + b[t - 1])
        b[t] = beta * (a[t] - a[t - 1]) + (1 - beta) * b[t - 1]
    for m in range(1, extra_periods + 1):
        f[cols + m - 1] = a[cols - 1] + m * b[cols - 1]
    return f, a, b


def damped_trend(d: np.ndarray, alpha: float = 0.3, beta: float = 0.2,
                 phi: float = 0.9, extra_periods: int = 12) -> tuple:
    """
    Damped Trend (Gardner & McKenzie, 1985).
    L_t = α·D_t + (1-α)·(L_{t-1} + φ·T_{t-1})
    T_t = β·(L_t - L_{t-1}) + (1-β)·φ·T_{t-1}
    F_{t+m} = L_t + (φ + φ² + … + φ^m)·T_t
    Returns: (forecast_array, level_array, trend_array)
    """
    cols = len(d)
    f = np.full(cols + extra_periods, np.nan)
    a = np.full(cols + extra_periods, np.nan)
    b = np.full(cols + extra_periods, np.nan)
    a[0] = d[0]
    b[0] = d[1] - d[0] if len(d) > 1 else 0
    for t in range(1, cols):
        f[t] = a[t - 1] + phi * b[t - 1]
        a[t] = alpha * d[t] + (1 - alpha) * (a[t - 1] + phi * b[t - 1])
        b[t] = beta * (a[t] - a[t - 1]) + (1 - beta) * phi * b[t - 1]
    for m in range(1, extra_periods + 1):
        damping_sum = sum(phi ** i for i in range(1, m + 1))
        f[cols + m - 1] = a[cols - 1] + damping_sum * b[cols - 1]
    return f, a, b


# ── ML_HOOK: Future model registry ────────────────────────────────────
# To add ML models, register them here following the same interface:
#   input:  np.ndarray (demand), **kwargs
#   output: np.ndarray (forecast array, same length convention)
#
# MODEL_REGISTRY = {
#     "statistical": {
#         "Naïve":          naive_forecast,
#         "Moving Average": moving_average,
#         "SES":            simple_exp_smoothing,
#         "Holt's Linear":  holts_linear,
#         "Damped Trend":   damped_trend,
#     },
#     "ml": {
#         "XGBoost":        xgb_forecast,      # future
#         "LightGBM":       lgbm_forecast,     # future
#         "LSTM":           lstm_forecast,     # future
#     }
# }
# ─────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════
# ── MODULE 3: KPI ENGINE ─────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def compute_kpis(actual: np.ndarray, forecast: np.ndarray,
                 naive_forecast_arr: np.ndarray) -> dict:
    """Full KPI suite. Returns dict with all accuracy metrics."""
    n  = min(len(actual), len(forecast))
    a  = actual[:n];  f = forecast[:n];  fn = naive_forecast_arr[:n]
    mask  = ~(np.isnan(a) | np.isnan(f))
    maskn = ~(np.isnan(a) | np.isnan(fn))
    a_v, f_v   = a[mask],   f[mask]
    a_n, fn_v  = a[maskn],  fn[maskn]

    if len(a_v) == 0:
        return {k: np.nan for k in
                ["bias_abs","bias_pct","mae","mae_pct","rmse","rmse_pct","mape","fva","n"]}

    errors   = f_v  - a_v
    errors_n = fn_v - a_n
    dem_avg  = a_v.mean()

    bias_abs = errors.mean()
    bias_pct = bias_abs / dem_avg * 100 if dem_avg != 0 else np.nan
    mae      = np.abs(errors).mean()
    mae_pct  = mae / dem_avg * 100 if dem_avg != 0 else np.nan
    rmse     = np.sqrt((errors ** 2).mean())
    rmse_pct = rmse / dem_avg * 100 if dem_avg != 0 else np.nan
    nonzero  = a_v != 0
    mape     = np.mean(np.abs(errors[nonzero] / a_v[nonzero])) * 100 if nonzero.any() else np.nan
    mae_naive = np.abs(errors_n).mean() if len(errors_n) > 0 else np.nan
    fva = ((mae_naive - mae) / mae_naive * 100) if (mae_naive and mae_naive != 0) else np.nan

    return {
        "bias_abs":  round(bias_abs, 2),
        "bias_pct":  round(bias_pct, 2),
        "mae":       round(mae, 2),
        "mae_pct":   round(mae_pct, 2),
        "rmse":      round(rmse, 2),
        "rmse_pct":  round(rmse_pct, 2),
        "mape":      round(mape, 2),
        "fva":       round(fva, 2),
        "n":         int(mask.sum()),
    }


def forecast_risk_level(mae_pct: float) -> tuple:
    """Map MAE% to supply chain risk level + color."""
    if np.isnan(mae_pct):
        return "Unknown", C["dim"]
    if mae_pct < 10:
        return "Low", C["green"]
    elif mae_pct < 25:
        return "Medium", C["amber"]
    else:
        return "High", C["red"]


# ══════════════════════════════════════════════════════════════════════
# ── MODULE 4: PARAMETER OPTIMISER ────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def optimise_ses(d_train: np.ndarray) -> dict:
    best_mae, best_alpha = np.inf, 0.3
    for alpha in np.arange(0.05, 0.96, 0.05):
        f, _ = simple_exp_smoothing(d_train, alpha=round(alpha, 2), extra_periods=0)
        mask  = ~np.isnan(f[:len(d_train)])
        if mask.sum() > 0:
            mae = np.abs((f[:len(d_train)] - d_train)[mask]).mean()
            if mae < best_mae:
                best_mae, best_alpha = mae, round(alpha, 2)
    return {"alpha": best_alpha, "mae": round(best_mae, 2)}


def optimise_holts(d_train: np.ndarray) -> dict:
    best_mae, best_alpha, best_beta = np.inf, 0.3, 0.2
    for alpha in np.arange(0.05, 0.61, 0.05):
        for beta in np.arange(0.05, 0.61, 0.05):
            f, _, _ = holts_linear(d_train, alpha=round(alpha, 2),
                                    beta=round(beta, 2), extra_periods=0)
            mask = ~np.isnan(f[:len(d_train)])
            if mask.sum() > 0:
                mae = np.abs((f[:len(d_train)] - d_train)[mask]).mean()
                if mae < best_mae:
                    best_mae, best_alpha, best_beta = mae, round(alpha, 2), round(beta, 2)
    return {"alpha": best_alpha, "beta": best_beta, "mae": round(best_mae, 2)}


def optimise_damped(d_train: np.ndarray) -> dict:
    best_mae = np.inf
    best = {"alpha": 0.3, "beta": 0.2, "phi": 0.9}
    for alpha in np.arange(0.05, 0.61, 0.1):
        for beta in np.arange(0.05, 0.41, 0.1):
            for phi in np.arange(0.70, 1.01, 0.05):
                f, _, _ = damped_trend(d_train, alpha=round(alpha, 2),
                                        beta=round(beta, 2), phi=round(phi, 2),
                                        extra_periods=0)
                mask = ~np.isnan(f[:len(d_train)])
                if mask.sum() > 0:
                    mae = np.abs((f[:len(d_train)] - d_train)[mask]).mean()
                    if mae < best_mae:
                        best_mae = mae
                        best = {"alpha": round(alpha, 2),
                                "beta":  round(beta, 2),
                                "phi":   round(phi, 2)}
    best["mae"] = round(best_mae, 2)
    return best


# ══════════════════════════════════════════════════════════════════════
# ── MODULE 5: CHART FACTORY ───────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def make_model_chart(periods, demand, train_end, forecasts_dict, extra_periods):
    """Primary chart: History | Train region | Test region | Forecast."""
    fig = go.Figure()
    n = len(demand)

    # Region shading
    fig.add_vrect(x0=0, x1=train_end, fillcolor=C["train_bg"], opacity=1,
                  layer="below", line_width=0,
                  annotation_text="TRAIN", annotation_position="top left",
                  annotation_font=dict(color=C["teal"], size=9))
    fig.add_vrect(x0=train_end, x1=n - 1, fillcolor=C["test_bg"], opacity=1,
                  layer="below", line_width=0,
                  annotation_text="TEST", annotation_position="top left",
                  annotation_font=dict(color=C["blue"], size=9))

    # Actual demand
    fig.add_trace(go.Scatter(
        x=list(range(n)), y=demand, mode="lines",
        name="Actual Demand",
        line=dict(color=C["demand"], width=2),
        hovertemplate="%{y:.0f}<extra>Actual</extra>"
    ))

    # Forecast lines
    all_x = list(range(n + extra_periods))
    for label, (color, f_arr) in forecasts_dict.items():
        fig.add_trace(go.Scatter(
            x=all_x[:len(f_arr)], y=f_arr,
            mode="lines", name=label,
            line=dict(color=color, width=1.6, dash="solid"),
            hovertemplate=f"%{{y:.0f}}<extra>{label}</extra>",
            opacity=0.85,
        ))

    # Train/test divider
    fig.add_vline(x=train_end, line_dash="dot", line_color=C["teal"],
                  line_width=1, opacity=0.4)

    fig.update_layout(**PLOTLY_LAYOUT,
                      xaxis_title="Period", yaxis_title="Demand (units)",
                      height=340)
    return fig


def make_residuals_chart(demand, forecast_arr, label, color):
    """Residuals over time + error distribution."""
    n      = min(len(demand), len(forecast_arr))
    errors = np.array(forecast_arr[:n]) - np.array(demand[:n])
    valid  = ~np.isnan(errors)
    e_v    = errors[valid]
    x_v    = np.where(valid)[0]

    fig = make_subplots(rows=1, cols=2, column_widths=[0.68, 0.32],
                        subplot_titles=["Residuals over Time", "Error Distribution"])

    fig.add_trace(go.Scatter(
        x=x_v, y=e_v, mode="lines+markers",
        line=dict(color=color, width=1.4),
        marker=dict(size=3, color=color),
        name="Residual",
    ), row=1, col=1)

    fig.add_hline(y=0,        line_dash="dash", line_color=C["dim"],   line_width=1, row=1, col=1)
    fig.add_hline(y=e_v.mean(), line_dash="dot", line_color=C["amber"], line_width=1.5,
                  annotation_text=f"Bias={e_v.mean():.1f}",
                  annotation_font=dict(color=C["amber"], size=10),
                  annotation_position="right", row=1, col=1)

    fig.add_trace(go.Histogram(
        x=e_v, nbinsx=15, marker_color=color, opacity=0.65,
    ), row=1, col=2)

    fig.update_layout(**PLOTLY_LAYOUT,
                      title=dict(text=f"Residual Analysis — {label}",
                                 font=dict(color=C["text"], size=13)),
                      height=280, showlegend=False)
    fig.update_xaxes(gridcolor=C["grid"], linecolor=C["border"])
    fig.update_yaxes(gridcolor=C["grid"], linecolor=C["border"])
    return fig


def make_decomposition_chart(demand, trend_comp, season_comp, residual):
    """4-panel demand decomposition."""
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        subplot_titles=["Demand", "Trend", "Seasonality", "Residual"],
                        vertical_spacing=0.07)
    data_panels = [
        (demand,      C["demand"], "Demand"),
        (trend_comp,  C["purple"], "Trend"),
        (season_comp, C["teal"],   "Seasonality"),
        (residual,    C["amber"],  "Residual"),
    ]
    for i, (y, color, name) in enumerate(data_panels, 1):
        fig.add_trace(go.Scatter(
            x=list(range(len(y))), y=y, mode="lines", name=name,
            line=dict(color=color, width=1.5),
        ), row=i, col=1)
        if name == "Residual":
            fig.add_hline(y=0, line_dash="dash", line_color=C["dim"],
                          line_width=1, row=i, col=1)

    fig.update_layout(**PLOTLY_LAYOUT,
                      height=560, showlegend=False,
                      title=dict(text="Demand Decomposition",
                                 font=dict(color=C["text"], size=13)))
    for i in range(1, 5):
        fig.update_xaxes(gridcolor=C["grid"], linecolor=C["border"], row=i, col=1)
        fig.update_yaxes(gridcolor=C["grid"], linecolor=C["border"], row=i, col=1)
    return fig


def make_weight_chart(alpha):
    """Exponential weighting — how much each past period is remembered."""
    n = 20
    weights = [(1 - alpha) ** i * alpha for i in range(n)]
    weights.reverse()
    opacities = [0.25 + 0.75 * (i / n) for i in range(n)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(-n + 1, 1)), y=weights,
        marker_color=C["teal"],
        marker_opacity=opacities,
        hovertemplate="t-%{x}: weight=%{y:.4f}<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT,
                      title=dict(text=f"Model Memory — Weight per Past Period  (α = {alpha})",
                                 font=dict(color=C["text"], size=12)),
                      xaxis_title="Periods Ago", yaxis_title="Weight",
                      height=240, showlegend=False)
    return fig


def make_sensitivity_heatmap(d_train):
    """Alpha × Beta MAE landscape for Holt's model."""
    alphas = np.arange(0.05, 0.65, 0.05)
    betas  = np.arange(0.05, 0.65, 0.05)
    Z = np.zeros((len(betas), len(alphas)))

    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            f, _, _ = holts_linear(d_train, alpha=round(alpha, 2),
                                    beta=round(beta, 2), extra_periods=0)
            mask = ~np.isnan(f[:len(d_train)])
            Z[j, i] = np.abs((f[:len(d_train)] - d_train)[mask]).mean() if mask.sum() > 0 else np.nan

    min_idx = np.unravel_index(np.nanargmin(Z), Z.shape)
    fig = go.Figure(go.Heatmap(
        z=Z,
        x=[f"{a:.2f}" for a in alphas],
        y=[f"{b:.2f}" for b in betas],
        colorscale="Plasma", reversescale=True,
        colorbar=dict(title="MAE", tickfont=dict(color=C["dim"])),
        hovertemplate="α=%{x}<br>β=%{y}<br>MAE=%{z:.1f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=[f"{alphas[min_idx[1]]:.2f}"],
        y=[f"{betas[min_idx[0]]:.2f}"],
        mode="markers",
        marker=dict(color=C["teal"], size=14, symbol="star"),
        name="Optimal",
    ))
    fig.update_layout(**PLOTLY_LAYOUT,
                      title=dict(text="Holt's — MAE Sensitivity: α × β",
                                 font=dict(color=C["text"], size=12)),
                      xaxis_title="Alpha (α) — level smoothing",
                      yaxis_title="Beta (β) — trend smoothing",
                      height=320)
    return fig


def make_phi_chart(demand, alpha, beta, train_end):
    """Damped vs undamped trend over long horizon."""
    n = len(demand)
    extra = 30
    f_damp,   _, _ = damped_trend(demand[:train_end], alpha=alpha, beta=beta,
                                    phi=0.85, extra_periods=n - train_end + extra)
    f_nodamp, _, _ = holts_linear(demand[:train_end], alpha=alpha, beta=beta,
                                    extra_periods=n - train_end + extra)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(n)), y=demand, mode="lines",
                             name="Actual", line=dict(color=C["demand"], width=1.8)))
    fig.add_trace(go.Scatter(x=list(range(len(f_nodamp))), y=f_nodamp, mode="lines",
                             name="Holt's (φ=1, undamped)",
                             line=dict(color=C["purple"], width=1.6, dash="dot")))
    fig.add_trace(go.Scatter(x=list(range(len(f_damp))), y=f_damp, mode="lines",
                             name="Damped (φ=0.85)",
                             line=dict(color=C["pink"], width=1.6, dash="dash")))
    fig.add_vline(x=train_end, line_dash="dash", line_color=C["teal"],
                  line_width=1, opacity=0.4)
    fig.update_layout(**PLOTLY_LAYOUT,
                      title=dict(text="Damped vs Undamped — Long-horizon Extrapolation",
                                 font=dict(color=C["text"], size=12)),
                      xaxis_title="Period", yaxis_title="Units", height=300)
    return fig


def make_ladder_chart(kpi_dict):
    """Horizontal bar chart: all models ranked by MAE%."""
    colors_map = {
        "Naïve": C["naive"], "Moving Average": C["ma"],
        "SES": C["ses"], "Holt's Linear": C["holt"], "Damped Trend": C["damped"]
    }
    models = [(m, v["mae_pct"]) for m, v in kpi_dict.items()
              if not np.isnan(v.get("mae_pct", np.nan))]
    models.sort(key=lambda x: x[1])
    labels = [m for m, _ in models]
    values = [v for _, v in models]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=[colors_map.get(l, C["dim"]) for l in labels],
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(color=C["text"]),
        hovertemplate="%{y}: MAE = %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT,
                      title=dict(text="Model Ladder — MAE% (lower is better)",
                                 font=dict(color=C["text"], size=12)),
                      xaxis_title="MAE % (relative to mean demand)",
                      height=260, showlegend=False)
    return fig


def make_fva_chart(kpis_all):
    """FVA bar chart."""
    fva_names  = ["Moving Average", "SES", "Holt's Linear", "Damped Trend"]
    fva_vals   = [kpis_all[m]["fva"] for m in fva_names]
    fva_colors = [C["teal"] if v > 0 else C["red"] for v in fva_vals]

    fig = go.Figure(go.Bar(
        x=fva_names, y=fva_vals,
        marker_color=fva_colors,
        text=[f"{v:+.1f}%" for v in fva_vals],
        textposition="outside",
        textfont=dict(color=C["text"]),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color=C["dim"], line_width=1)
    fig.update_layout(**PLOTLY_LAYOUT,
                      title=dict(text="Forecast Value Added (FVA) vs Naïve Benchmark",
                                 font=dict(color=C["text"], size=12)),
                      yaxis_title="FVA % (positive = beats Naïve)",
                      height=260)
    return fig


# ══════════════════════════════════════════════════════════════════════
# ── DIAGNOSTICS ───────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

CHALLENGES = [
    {"name": "The Stable Workhorse",
     "params": {"base_level": 800, "trend_slope": 0, "seasonality_amp": 0, "noise_std": 40},
     "best_model": "SES", "hint": "No trend, no seasonality — model the level, nothing else."},
    {"name": "The Rising Star",
     "params": {"base_level": 500, "trend_slope": 8, "seasonality_amp": 0, "noise_std": 50},
     "best_model": "Holt's Linear", "hint": "Steady linear growth. Moving Average lags badly."},
    {"name": "The Seasonal Surge",
     "params": {"base_level": 1000, "trend_slope": 2, "seasonality_amp": 300, "noise_std": 80},
     "best_model": "Damped Trend", "hint": "Trend + seasonality. Watch the long-horizon risk."},
    {"name": "The Noisy Flat",
     "params": {"base_level": 600, "trend_slope": 0, "seasonality_amp": 0, "noise_std": 200},
     "best_model": "Moving Average", "hint": "High noise, flat demand. High α chases noise."},
]


def generate_diagnostics(kpis_dict, params):
    flags = []
    for model, kpis in kpis_dict.items():
        if np.isnan(kpis.get("mae_pct", np.nan)):
            continue
        bias = kpis["bias_abs"]
        mae  = kpis["mae_pct"]
        fva  = kpis["fva"]

        if abs(bias / (kpis["mae"] + 1e-8)) > 0.5:
            direction = "over-forecasting" if bias > 0 else "under-forecasting"
            flags.append(("⚠️", model,
                f"Systematic {direction} detected (Bias/MAE > 0.5). "
                f"{'Excess inventory risk.' if bias > 0 else 'Stockout risk.'}"))
        if fva < 0:
            flags.append(("🔴", model,
                f"FVA = {fva:.1f}% — model is WORSE than Naïve. "
                f"Likely overfitting or wrong model class for this demand shape."))
        if mae > 30:
            flags.append(("⚠️", model,
                f"MAE% = {mae:.1f}% — high forecast error. "
                f"Safety stock requirements will be elevated."))
        if model == "SES" and params.get("trend_slope", 0) > 5:
            flags.append(("ℹ️", model,
                "Strong trend detected but SES only models level. "
                "Holt's Linear or Damped Trend will perform better."))
        if model in ("Holt's Linear", "Damped Trend") and params.get("extra_periods", 0) > 20:
            flags.append(("ℹ️", model,
                "Long forecast horizon with trend — verify φ damping "
                "to prevent unrealistic growth projections."))

    if not flags:
        flags.append(("✅", "All models",
            "No critical issues detected. Review KPIs above for fine-tuning."))
    return flags


# ══════════════════════════════════════════════════════════════════════
# ── SIDEBAR ───────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def render_sidebar():
    """Render sidebar: branding, navigation, data source, parameters."""
    sb = st.sidebar

    # ── Branding ──
    sb.markdown("""
    <div style="padding:18px 18px 10px 18px;">
      <div style="font-family:'Inter',sans-serif; font-size:15px;
                  font-weight:700; color:#0f1923; letter-spacing:-0.3px;">
        📈 Demand Intelligence
      </div>
      <div style="font-size:10.5px; color:#5a6475; margin-top:2px; letter-spacing:0.03em;">
        Supply Chain Forecasting Platform
      </div>
    </div>
    <div style="height:1px; background:#e2e6ec; margin:0 0 8px 0;"></div>
    """, unsafe_allow_html=True)

    # ── Navigation ──
    sb.markdown('<div style="padding:6px 18px 4px 18px;"><span style="font-size:10px;color:#5a6475;text-transform:uppercase;letter-spacing:0.1em;font-weight:600;">Navigation</span></div>',
                unsafe_allow_html=True)

    NAV_OPTIONS = [
        "📊  Overview",
        "🔬  Model Lab",
        "📉  Residual Analysis",
        "🧩  Decomposition",
        "🎛️  Parameter Explorer",
        "🏆  Model Rankings",
        "⚠️  Diagnostics",
        "🎯  Challenge Mode",
        "🤖  ML Studio",
    ]
    selected_page = sb.radio("nav", NAV_OPTIONS, label_visibility="collapsed")

    sb.markdown('<div style="height:1px;background:#e2e6ec;margin:8px 0;"></div>', unsafe_allow_html=True)

    # ── Data Source ──
    sb.markdown('<div style="padding:0 18px 4px 18px;"><span style="font-size:10px;color:#5a6475;text-transform:uppercase;letter-spacing:0.1em;font-weight:600;">Data Source</span></div>',
                unsafe_allow_html=True)

    data_mode = sb.radio("data_source", ["🔧 Synthetic Generator", "📂 Upload Your Data"],
                          label_visibility="collapsed")

    uploaded_df     = None
    demand_col      = None
    upload_success  = False

    if "Upload" in data_mode:
        uploaded_file = sb.file_uploader(
            "Upload CSV or Excel", type=["csv", "xlsx", "xls"],
            label_visibility="collapsed"
        )
        if uploaded_file:
            df = load_user_data(uploaded_file)
            if df is not None and len(df) > 0:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    sb.markdown('<div style="padding:0 0 4px 0;font-size:11px;color:#5a6475;">Select demand column:</div>',
                                unsafe_allow_html=True)
                    demand_col   = sb.selectbox("Demand column", numeric_cols, label_visibility="collapsed")
                    uploaded_df  = df
                    upload_success = True
                    sb.markdown(f'<div class="kb-box">✓ Loaded <b>{len(df)}</b> rows · <b>{demand_col}</b> selected</div>',
                                unsafe_allow_html=True)
                else:
                    sb.error("No numeric columns found in file.")
            else:
                sb.error("Could not read file.")

    sb.markdown('<div style="height:1px;background:#e2e6ec;margin:8px 0;"></div>', unsafe_allow_html=True)

    # ── Demand Profile (only shown for synthetic) ──
    p = {}
    if "Synthetic" in data_mode:
        sb.markdown('<div style="padding:0 18px 4px 18px;"><span style="font-size:10px;color:#5a6475;text-transform:uppercase;letter-spacing:0.1em;font-weight:600;">Demand Profile</span></div>',
                    unsafe_allow_html=True)
        p["n_periods"]   = sb.slider("Total periods",            30,   120,   60,   5)
        p["base_level"]  = sb.slider("Base level (units)",       100, 5000, 1000,  50)
        p["trend_slope"] = sb.slider("Trend slope (units/period)", -10.0, 20.0, 3.0, 0.5)
        p["season_amp"]  = sb.slider("Seasonality amplitude",    0,    500,  150,  25)
        p["season_len"]  = sb.slider("Season length (periods)",  4,    24,   12,   1)
        p["noise_std"]   = sb.slider("Noise std dev",            0,    300,  60,   10)
        p["n_outliers"]  = sb.slider("Outlier injections",       0,    8,    2,    1)
        p["outlier_mag"] = sb.slider("Outlier magnitude (×σ)",   1.0,  6.0,  3.0,  0.5)
        p["seed"]        = sb.number_input("Random seed", 0, 9999, 42)
        sb.markdown('<div style="height:1px;background:#e2e6ec;margin:8px 0;"></div>', unsafe_allow_html=True)

    # ── Train / Test ──
    sb.markdown('<div style="padding:0 18px 4px 18px;"><span style="font-size:10px;color:#5a6475;text-transform:uppercase;letter-spacing:0.1em;font-weight:600;">Train / Test Split</span></div>',
                unsafe_allow_html=True)
    p["train_pct"]    = sb.slider("Training set (%)", 50, 90, 75, 5)
    p["extra_periods"] = sb.slider("Forecast horizon (periods)", 4, 30, 12, 2)

    sb.markdown('<div style="height:1px;background:#e2e6ec;margin:8px 0;"></div>', unsafe_allow_html=True)

    # ── Model Parameters ──
    sb.markdown('<div style="padding:0 18px 4px 18px;"><span style="font-size:10px;color:#5a6475;text-transform:uppercase;letter-spacing:0.1em;font-weight:600;">Model Parameters</span></div>',
                unsafe_allow_html=True)
    p["ma_n"]    = sb.slider("MA — Window n",         2,    20,  3,    1)
    p["ses_a"]   = sb.slider("SES — Alpha α",         0.01, 0.99, 0.30, 0.01)
    p["holt_a"]  = sb.slider("Holt — Alpha α",        0.01, 0.99, 0.30, 0.01)
    p["holt_b"]  = sb.slider("Holt — Beta β",         0.01, 0.99, 0.20, 0.01)
    p["damp_a"]  = sb.slider("Damped — Alpha α",      0.01, 0.99, 0.30, 0.01)
    p["damp_b"]  = sb.slider("Damped — Beta β",       0.01, 0.99, 0.20, 0.01)
    p["damp_ph"] = sb.slider("Damped — Phi φ",        0.50, 1.00, 0.90, 0.01)

    sb.markdown('<div style="height:1px;background:#e2e6ec;margin:8px 0;"></div>', unsafe_allow_html=True)

    # ── Visibility ──
    sb.markdown('<div style="padding:0 18px 4px 18px;"><span style="font-size:10px;color:#5a6475;text-transform:uppercase;letter-spacing:0.1em;font-weight:600;">Show Models</span></div>',
                unsafe_allow_html=True)
    p["show_naive"]  = sb.checkbox("Naïve",          True)
    p["show_ma"]     = sb.checkbox("Moving Average", True)
    p["show_ses"]    = sb.checkbox("SES",            True)
    p["show_holt"]   = sb.checkbox("Holt's Linear",  True)
    p["show_damped"] = sb.checkbox("Damped Trend",   True)

    # ── Reference notes ──
    sb.markdown('<div style="height:1px;background:#e2e6ec;margin:8px 0;"></div>', unsafe_allow_html=True)
    sb.markdown("""
    <div style="padding:0 18px 12px 18px;">
    <div class="kb-box"><b>α (alpha)</b> — smoothing intensity. High α = reactive, tracks noise. Low α = stable, slow to adapt.</div>
    <div class="kb-box"><b>β (beta)</b> — trend smoothing. Lower values give smoother trend estimates.</div>
    <div class="kb-box"><b>φ (phi)</b> — damping factor. φ &lt; 1 prevents trend from extrapolating infinitely.</div>
    <div class="warn-box"><b>MAPE Warning</b> — biased toward under-forecasting. Use MAE% for decisions.</div>
    </div>
    """, unsafe_allow_html=True)

    p["data_mode"]     = data_mode
    p["uploaded_df"]   = uploaded_df
    p["demand_col"]    = demand_col
    p["upload_success"] = upload_success
    p["selected_page"]  = selected_page
    return p


# ══════════════════════════════════════════════════════════════════════
# ── PAGES ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def page_header(icon, title, subtitle=""):
    st.markdown(f"""
    <div class="page-header">
      <div>
        <h2>{icon} {title}</h2>
        {"" if not subtitle else f'<div class="subtitle">{subtitle}</div>'}
      </div>
    </div>
    """, unsafe_allow_html=True)


def page_overview(d_full, train_end, extra, kpis_all, p):
    page_header("📊", "Demand Overview",
                "Real-time summary of demand characteristics and model performance")

    # ── Demand stats row ──
    mean_d = d_full.mean()
    std_d  = d_full.std()
    cov    = std_d / mean_d * 100
    best_model = min(kpis_all, key=lambda m: kpis_all[m]["mae_pct"]
                     if not np.isnan(kpis_all[m]["mae_pct"]) else np.inf)
    best_mae   = kpis_all[best_model]["mae_pct"]
    risk, risk_color = forecast_risk_level(best_mae)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Mean Demand",   f"{mean_d:.0f} u")
    c2.metric("Std Deviation", f"{std_d:.0f} u")
    c3.metric("CoV",           f"{cov:.1f}%",
              "High variability" if cov > 30 else "Stable")
    c4.metric("Best Model",    best_model, f"MAE {best_mae:.1f}%")
    c5.metric("Forecast Risk", risk, "vs mean demand")

    st.markdown("")

    # ── Main chart ──
    forecasts = {}
    if p["show_naive"]:  forecasts["Naïve"]          = (C["naive"],  p["f_naive"])
    if p["show_ma"]:     forecasts["Moving Average"] = (C["ma"],     p["f_ma"])
    if p["show_ses"]:    forecasts["SES"]            = (C["ses"],    p["f_ses"])
    if p["show_holt"]:   forecasts["Holt's Linear"]  = (C["holt"],   p["f_holt"])
    if p["show_damped"]: forecasts["Damped Trend"]   = (C["damped"], p["f_damp"])

    fig = make_model_chart(list(range(len(d_full))), d_full, train_end, forecasts, extra)
    st.plotly_chart(fig, use_container_width=True)

    # ── KPI summary cards (test set) ──
    st.markdown('<div class="section-label">Test Set KPIs — All Models</div>', unsafe_allow_html=True)
    model_colors = {
        "Naïve": C["naive"], "Moving Average": C["ma"],
        "SES": C["ses"], "Holt's Linear": C["holt"], "Damped Trend": C["damped"]
    }
    for model, kpis in kpis_all.items():
        risk_l, _ = forecast_risk_level(kpis["mae_pct"])
        badge_cls = {"Low": "badge-green", "Medium": "badge-amber", "High": "badge-red"}.get(risk_l, "badge-blue")
        col_color = model_colors.get(model, C["dim"])
        st.markdown(
            f'<span style="font-size:12px;font-weight:600;color:{col_color};">{model}</span> '
            f'<span class="badge {badge_cls}">{risk_l} risk</span> '
            f'<span class="stat-pill" style="margin-left:4px;">MAE <span>{kpis["mae_pct"]:.1f}%</span></span>'
            f'<span class="stat-pill">Bias <span>{kpis["bias_pct"]:+.1f}%</span></span>'
            f'<span class="stat-pill">FVA <span>{kpis["fva"]:+.1f}%</span></span>',
            unsafe_allow_html=True
        )
    st.markdown("")


def page_model_lab(d_full, train_end, extra, kpis_all, p):
    page_header("🔬", "Model Lab",
                "Full KPI breakdown per model — train/test split evaluation")

    forecasts = {}
    if p["show_naive"]:  forecasts["Naïve"]          = (C["naive"],  p["f_naive"])
    if p["show_ma"]:     forecasts["Moving Average"] = (C["ma"],     p["f_ma"])
    if p["show_ses"]:    forecasts["SES"]            = (C["ses"],    p["f_ses"])
    if p["show_holt"]:   forecasts["Holt's Linear"]  = (C["holt"],   p["f_holt"])
    if p["show_damped"]: forecasts["Damped Trend"]   = (C["damped"], p["f_damp"])

    fig = make_model_chart(list(range(len(d_full))), d_full, train_end, forecasts, extra)
    st.plotly_chart(fig, use_container_width=True)

    # ── KPI rows ──
    st.markdown('<div class="section-label">Detailed KPIs (evaluated on test set only)</div>',
                unsafe_allow_html=True)
    model_colors = {
        "Naïve": C["naive"], "Moving Average": C["ma"],
        "SES": C["ses"], "Holt's Linear": C["holt"], "Damped Trend": C["damped"]
    }
    for model, kpis in kpis_all.items():
        col_color = model_colors.get(model, C["dim"])
        st.markdown(f'<b style="color:{col_color};font-size:12px;">{model}</b>', unsafe_allow_html=True)
        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
        c1.metric("Bias",         f"{kpis['bias_abs']:+.1f}",   f"{kpis['bias_pct']:+.1f}%")
        c2.metric("MAE",          f"{kpis['mae']:.1f}",          f"{kpis['mae_pct']:.1f}%")
        c3.metric("RMSE",         f"{kpis['rmse']:.1f}",         f"{kpis['rmse_pct']:.1f}%")
        c4.metric("MAPE",         f"{kpis['mape']:.1f}%",        "ref only")
        c5.metric("FVA vs Naïve", f"{kpis['fva']:+.1f}%",        "positive = better")
        c6.metric("Bias Dir",     "Over" if kpis['bias_abs'] > 0 else "Under", "forecast")
        c7.metric("N Periods",    str(kpis['n']),                 "evaluated")
        st.markdown("")


def page_residuals(d_full, train_end, kpis_all, p):
    page_header("📉", "Residual Analysis",
                "Error patterns, bias detection, and distribution shape per model")

    model_map = {
        "Naïve":          (C["naive"],  p["f_naive"],  p["show_naive"]),
        "Moving Average": (C["ma"],     p["f_ma"],     p["show_ma"]),
        "SES":            (C["ses"],    p["f_ses"],    p["show_ses"]),
        "Holt's Linear":  (C["holt"],   p["f_holt"],   p["show_holt"]),
        "Damped Trend":   (C["damped"], p["f_damp"],   p["show_damped"]),
    }

    tabs = st.tabs([m for m, (_, _, show) in model_map.items() if show])
    visible = [(m, c, f) for m, (c, f, show) in model_map.items() if show]

    for tab, (model, color, f_arr) in zip(tabs, visible):
        with tab:
            fig = make_residuals_chart(d_full, f_arr, model, color)
            st.plotly_chart(fig, use_container_width=True)
            kpis = kpis_all[model]
            c1, c2, c3 = st.columns(3)
            c1.metric("Bias (abs)",  f"{kpis['bias_abs']:+.1f}")
            c2.metric("MAE",         f"{kpis['mae']:.1f}")
            c3.metric("RMSE",        f"{kpis['rmse']:.1f}")


def page_decomposition(d_full, p):
    page_header("🧩", "Demand Decomposition",
                "Separate trend, seasonality, and residual noise components")

    if "Synthetic" in p["data_mode"]:
        trend_comp, season_comp, residual = decompose_demand(
            d_full, p["trend_slope"], p["base_level"], p["season_amp"], p["season_len"]
        )
    else:
        # For user data: simple linear detrend
        t = np.arange(len(d_full))
        slope, intercept = np.polyfit(t, d_full, 1)
        trend_comp  = intercept + slope * t
        season_comp = np.zeros(len(d_full))
        residual    = d_full - trend_comp

    fig = make_decomposition_chart(d_full, trend_comp, season_comp, residual)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Trend Magnitude", f"{trend_comp[-1] - trend_comp[0]:+.0f} u", "total change")
    c2.metric("Season Amplitude", f"±{np.abs(season_comp).max():.0f} u" if season_comp.any() else "None")
    c3.metric("Residual Std",    f"{residual.std():.1f} u")


def page_parameters(d_full, train_end, extra, p):
    page_header("🎛️", "Parameter Explorer",
                "Understand model mechanics, sensitivity analysis, and auto-optimisation")

    d_train = d_full[:train_end]

    tab1, tab2, tab3, tab4 = st.tabs([
        "📐 Model Formulas",
        "🔭 Model Memory",
        "🗺️ Sensitivity Map",
        "🤖 Auto-Optimiser",
    ])

    # ── Tab 1: Formulas ──
    with tab1:
        st.markdown('<div class="section-label">Mathematical Formulas</div>', unsafe_allow_html=True)

        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown('<div class="formula-card"><div class="formula-title">Naïve Forecast</div>',
                        unsafe_allow_html=True)
            st.latex(r"F_{t+1} = D_t")
            st.markdown('<div style="font-size:11px;color:#5a6475;margin-top:6px;">Last observed demand carried forward. Simplest possible baseline.</div></div>',
                        unsafe_allow_html=True)

            st.markdown('<div class="formula-card"><div class="formula-title">Moving Average (n periods)</div>',
                        unsafe_allow_html=True)
            st.latex(r"F_{t+1} = \frac{1}{n}\sum_{i=0}^{n-1} D_{t-i}")
            st.markdown('<div style="font-size:11px;color:#5a6475;margin-top:6px;">Equal weight to last n observations. Smooths noise but lags trends.</div></div>',
                        unsafe_allow_html=True)

            st.markdown('<div class="formula-card"><div class="formula-title">Simple Exponential Smoothing (SES)</div>',
                        unsafe_allow_html=True)
            st.latex(r"L_t = \alpha \cdot D_t + (1-\alpha) \cdot L_{t-1}")
            st.latex(r"F_{t+m} = L_t \quad \forall m \geq 1")
            st.markdown('<div style="font-size:11px;color:#5a6475;margin-top:6px;">Exponentially decaying weights on past observations. No trend component — flat forecast.</div></div>',
                        unsafe_allow_html=True)

        with col_r:
            st.markdown('<div class="formula-card"><div class="formula-title">Holt\'s Linear Trend</div>',
                        unsafe_allow_html=True)
            st.latex(r"L_t = \alpha \cdot D_t + (1-\alpha)(L_{t-1} + T_{t-1})")
            st.latex(r"T_t = \beta(L_t - L_{t-1}) + (1-\beta) T_{t-1}")
            st.latex(r"F_{t+m} = L_t + m \cdot T_t")
            st.markdown('<div style="font-size:11px;color:#5a6475;margin-top:6px;">Two smoothing equations: level + trend. Forecast grows linearly into the horizon.</div></div>',
                        unsafe_allow_html=True)

            st.markdown('<div class="formula-card"><div class="formula-title">Damped Trend (Gardner & McKenzie)</div>',
                        unsafe_allow_html=True)
            st.latex(r"L_t = \alpha \cdot D_t + (1-\alpha)(L_{t-1} + \varphi T_{t-1})")
            st.latex(r"T_t = \beta(L_t - L_{t-1}) + (1-\beta)\varphi T_{t-1}")
            st.latex(r"F_{t+m} = L_t + \left(\sum_{i=1}^{m}\varphi^i\right) T_t")
            st.markdown('<div style="font-size:11px;color:#5a6475;margin-top:6px;">φ &lt; 1 causes trend to decay — prevents runaway long-horizon projections. Most robust model in this suite.</div></div>',
                        unsafe_allow_html=True)

        # FVA formula
        st.markdown('<div class="section-label" style="margin-top:16px;">Accuracy Metrics</div>',
                    unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="formula-card"><div class="formula-title">Mean Absolute Error (MAE)</div>', unsafe_allow_html=True)
            st.latex(r"\text{MAE} = \frac{1}{n}\sum_{t=1}^{n}|D_t - F_t|")
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="formula-card"><div class="formula-title">Root Mean Squared Error</div>', unsafe_allow_html=True)
            st.latex(r"\text{RMSE} = \sqrt{\frac{1}{n}\sum_{t=1}^{n}(D_t - F_t)^2}")
            st.markdown("</div>", unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="formula-card"><div class="formula-title">Forecast Value Added</div>', unsafe_allow_html=True)
            st.latex(r"\text{FVA} = \frac{\text{MAE}_{\text{Naïve}} - \text{MAE}_{\text{model}}}{\text{MAE}_{\text{Naïve}}} \times 100\%")
            st.markdown("</div>", unsafe_allow_html=True)

    # ── Tab 2: Model Memory ──
    with tab2:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**SES — weight decay by α**")
            fig_w = make_weight_chart(p["ses_a"])
            st.plotly_chart(fig_w, use_container_width=True)
            st.markdown(f'<div class="kb-box">α = <b>{p["ses_a"]}</b> — each period ago receives weight (1-α)^t × α. Adjust the SES α slider to see how memory changes.</div>',
                        unsafe_allow_html=True)
        with col_b:
            st.markdown("**Damped vs Undamped — long horizon**")
            fig_phi = make_phi_chart(d_full, p["holt_a"], p["holt_b"], train_end)
            st.plotly_chart(fig_phi, use_container_width=True)
            st.markdown('<div class="warn-box">Without damping (φ=1), the trend extrapolates indefinitely — dangerous for long-horizon planning. φ=0.85 shown.</div>',
                        unsafe_allow_html=True)

    # ── Tab 3: Sensitivity ──
    with tab3:
        st.markdown("**Holt's MAE sensitivity map — α × β grid**")
        fig_heat = make_sensitivity_heatmap(d_train)
        st.plotly_chart(fig_heat, use_container_width=True)
        st.markdown('<div class="info-box">The ⭐ marks the optimal (α, β) pair that minimises MAE on the training set. Use this as a guide when manually tuning.</div>',
                    unsafe_allow_html=True)

    # ── Tab 4: Auto-Optimiser ──
    with tab4:
        st.markdown("Grid search to minimise MAE on the training set. Use results to guide manual slider tuning.")
        col_a, col_b, col_c = st.columns(3)
        if col_a.button("Optimise SES"):
            with st.spinner("Searching α space…"):
                res = optimise_ses(d_train)
            st.success(f"✅ SES: α = **{res['alpha']}** → Train MAE = {res['mae']:.1f}")
        if col_b.button("Optimise Holt's"):
            with st.spinner("Searching α × β grid…"):
                res = optimise_holts(d_train)
            st.success(f"✅ Holt's: α = **{res['alpha']}**, β = **{res['beta']}** → Train MAE = {res['mae']:.1f}")
        if col_c.button("Optimise Damped"):
            with st.spinner("Searching α × β × φ grid — may take a moment…"):
                res = optimise_damped(d_train)
            st.success(f"✅ Damped: α = **{res['alpha']}**, β = **{res['beta']}**, φ = **{res['phi']}** → Train MAE = {res['mae']:.1f}")


def page_rankings(kpis_all):
    page_header("🏆", "Model Rankings",
                "Comparative performance across all models — ranked by accuracy")

    c1, c2 = st.columns(2)
    with c1:
        fig_ladder = make_ladder_chart(kpis_all)
        st.plotly_chart(fig_ladder, use_container_width=True)
    with c2:
        fig_fva = make_fva_chart(kpis_all)
        st.plotly_chart(fig_fva, use_container_width=True)

    st.markdown('<div class="section-label">Full KPI Comparison Table</div>', unsafe_allow_html=True)
    rows = []
    for model, kpis in kpis_all.items():
        risk_l, _ = forecast_risk_level(kpis["mae_pct"])
        rows.append({
            "Model":      model,
            "Bias":       f"{kpis['bias_abs']:+.1f}",
            "Bias %":     f"{kpis['bias_pct']:+.1f}%",
            "MAE":        f"{kpis['mae']:.1f}",
            "MAE %":      f"{kpis['mae_pct']:.1f}%",
            "RMSE":       f"{kpis['rmse']:.1f}",
            "RMSE %":     f"{kpis['rmse_pct']:.1f}%",
            "MAPE":       f"{kpis['mape']:.1f}%",
            "FVA":        f"{kpis['fva']:+.1f}%",
            "Risk Level": risk_l,
        })
    df_table = pd.DataFrame(rows).set_index("Model")
    st.dataframe(df_table, use_container_width=True)


def page_diagnostics(kpis_all, p, d_full):
    page_header("⚠️", "Diagnostics",
                "Automated analysis — model suitability, bias flags, supply chain risk indicators")

    flags = generate_diagnostics(kpis_all, p)
    for icon, model, message in flags:
        if icon == "✅":
            st.success(f"**{model}** — {message}")
        elif icon == "🔴":
            st.error(f"**{model}** — {message}")
        elif icon == "⚠️":
            st.warning(f"**{model}** — {message}")
        else:
            st.info(f"**{model}** — {message}")

    st.markdown('<div class="section-label" style="margin-top:16px;">Demand Characteristics</div>',
                unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    cov = d_full.std() / d_full.mean() * 100
    c1.metric("Mean Demand",     f"{d_full.mean():.0f} u")
    c2.metric("Std Deviation",   f"{d_full.std():.0f} u")
    c3.metric("CoV",             f"{cov:.1f}%",
              "High" if cov > 30 else "Normal")
    if "Synthetic" in p.get("data_mode", ""):
        c4.metric("Trend",       f"{p['trend_slope']:+.1f} u/period",
                  "Detected" if abs(p["trend_slope"]) > 1 else "None")
    else:
        slope, _ = np.polyfit(np.arange(len(d_full)), d_full, 1)
        c4.metric("Est. Trend",  f"{slope:+.2f} u/period")

    st.markdown('<div class="section-label" style="margin-top:16px;">Supply Chain Implications</div>',
                unsafe_allow_html=True)
    best_model = min(kpis_all, key=lambda m: kpis_all[m]["mae_pct"]
                     if not np.isnan(kpis_all[m]["mae_pct"]) else np.inf)
    best_mae   = kpis_all[best_model]["mae_pct"]
    risk, _    = forecast_risk_level(best_mae)

    st.markdown(f"""
    <div class="kb-box">
    <b>Best Performing Model:</b> {best_model} (MAE% = {best_mae:.1f}%)<br>
    <b>Forecast Risk Level:</b> {risk}<br>
    <b>Implication:</b>
    {'At &lt;10% error, safety stock requirements are relatively low. Good forecast confidence.' if risk == 'Low' else
     'At 10–25% error, moderate safety stock buffer required. Consider model improvement or upstream demand signal sharing.' if risk == 'Medium' else
     'At &gt;25% error, elevated safety stock or high stockout risk. Investigate demand structure — ML models may outperform here.'}
    </div>
    """, unsafe_allow_html=True)

    cov_val = d_full.std() / d_full.mean()
    st.markdown(f"""
    <div class="{"warn-box" if cov_val > 0.3 else "kb-box"}">
    <b>Demand Variability (CoV = {cov_val*100:.1f}%):</b>
    {"High variability — intermittent or lumpy demand patterns may require specialised models (Croston, TSB) rather than exponential smoothing." if cov_val > 0.3 else
     "Moderate to low variability — standard exponential smoothing methods are appropriate."}
    </div>
    """, unsafe_allow_html=True)


def page_challenge(p):
    page_header("🎯", "Challenge Mode",
                "Mystery demand scenarios — diagnose the pattern and pick the best model")

    challenge_idx = st.selectbox(
        "Select scenario",
        options=list(range(len(CHALLENGES))),
        format_func=lambda i: f"Scenario {i+1}: {CHALLENGES[i]['name']}",
        label_visibility="visible",
    )
    ch = CHALLENGES[challenge_idx]

    ch_demand = generate_demand(
        n_periods=60,
        base_level=ch["params"]["base_level"],
        trend_slope=ch["params"]["trend_slope"],
        seasonality_amp=ch["params"]["seasonality_amp"],
        season_length=12,
        noise_std=ch["params"]["noise_std"],
        n_outliers=2, outlier_magnitude=3.0,
        seed=challenge_idx * 7 + 13,
    )
    ch_train_end = 45

    fig_ch = go.Figure()
    fig_ch.add_trace(go.Scatter(
        x=list(range(60)), y=ch_demand, mode="lines",
        name="Mystery Demand",
        line=dict(color=C["demand"], width=2),
    ))
    fig_ch.add_vline(x=ch_train_end, line_dash="dash", line_color=C["teal"],
                     line_width=1, opacity=0.5)
    fig_ch.update_layout(**PLOTLY_LAYOUT,
                          title="Mystery Demand — What demand pattern do you see?",
                          height=260, xaxis_title="Period", yaxis_title="Units")
    st.plotly_chart(fig_ch, use_container_width=True)

    c1, c2 = st.columns([2, 1])
    with c1:
        if st.checkbox("Show hint 💡"):
            st.info(f"**Hint:** {ch['hint']}")
        user_model = st.radio(
            "Which model is best for this demand?",
            ["Naïve", "Moving Average", "SES", "Holt's Linear", "Damped Trend"],
            horizontal=True,
        )
    with c2:
        st.markdown('<div class="kb-box"><b>What to look for:</b><br>• Flat line → SES or MA<br>• Rising/falling → Holt\'s<br>• Wavering up = Damped<br>• Very noisy flat → MA with large n</div>',
                    unsafe_allow_html=True)

    if st.button("🏁  Score My Answer"):
        ch_test = ch_demand[ch_train_end:]
        results = {}
        for name, f_func, kwargs in [
            ("Naïve",         naive_forecast,    {"extra_periods": 15}),
            ("Moving Average",moving_average,    {"n": 3, "extra_periods": 15}),
            ("SES",           lambda d, **kw: simple_exp_smoothing(d, alpha=0.3, **kw)[0],
                                               {"extra_periods": 15}),
            ("Holt's Linear", lambda d, **kw: holts_linear(d, alpha=0.3, beta=0.2, **kw)[0],
                                               {"extra_periods": 15}),
            ("Damped Trend",  lambda d, **kw: damped_trend(d, alpha=0.3, beta=0.2, phi=0.9, **kw)[0],
                                               {"extra_periods": 15}),
        ]:
            f_arr    = f_func(ch_demand[:ch_train_end], **kwargs)
            test_len = min(len(ch_test), len(f_arr))
            errors   = np.abs(np.array(f_arr[:test_len]) - np.array(ch_test[:test_len]))
            valid    = ~np.isnan(errors)
            results[name] = errors[valid].mean() if valid.sum() > 0 else np.nan

        best_m   = min(results, key=lambda x: results[x] if not np.isnan(results[x]) else np.inf)
        user_mae = results.get(user_model, np.nan)
        best_mae = results.get(best_m, np.nan)

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if user_model == best_m:
                st.success(f"🏆 **Perfect!** {user_model} is the best model (MAE = {user_mae:.1f})")
            elif user_mae <= best_mae * 1.10:
                st.success(f"✅ **Very close!** {user_model} (MAE={user_mae:.1f}) within 10% of best: {best_m} (MAE={best_mae:.1f})")
            else:
                st.error(f"❌ {user_model} (MAE={user_mae:.1f}) — best was **{best_m}** (MAE={best_mae:.1f}). {ch['hint']}")

        with col2:
            st.markdown("**All model scores:**")
            for name, mae in sorted(results.items(), key=lambda x: x[1] if not np.isnan(x[1]) else np.inf):
                star = " ⭐" if name == best_m   else ""
                you  = " ← you" if name == user_model else ""
                st.markdown(f"- **{name}**: MAE = {mae:.1f}{star}{you}")

        st.info(f"**Expert answer:** {ch['best_model']} — {ch['hint']}")


def page_ml_studio():
    page_header("🤖", "ML Studio",
                "Machine learning forecasting — coming next")

    st.markdown("""
    <div style="background:#ffffff;border:1px solid #e2e6ec;border-radius:10px;padding:28px 32px;margin-top:12px;">
      <div style="font-size:28px;margin-bottom:12px;">🚧</div>
      <div style="font-size:16px;font-weight:600;color:#0f1923;margin-bottom:8px;">ML Forecasting Module — In Development</div>
      <div style="font-size:13px;color:#5a6475;line-height:1.7;max-width:600px;">
        The statistical engine above handles level, trend, and seasonality well.
        ML models will extend this with feature-engineered inputs and non-linear pattern capture.
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<div class="section-label">Planned Models</div>', unsafe_allow_html=True)

    planned = [
        ("XGBoost", "Gradient-boosted trees with lag features, rolling statistics, calendar variables",
         "badge-amber", "High priority"),
        ("LightGBM", "Faster gradient boosting — better for large SKU counts and long histories",
         "badge-amber", "High priority"),
        ("Prophet", "Facebook's decomposable model — strong on seasonality and holidays",
         "badge-blue",  "Planned"),
        ("LSTM / GRU", "Recurrent neural networks for sequence modelling — complex demand patterns",
         "badge-purple","Planned"),
        ("Ensemble", "Weighted blend of statistical + ML outputs — robust production baseline",
         "badge-teal",  "Planned"),
    ]

    for name, desc, badge_cls, status in planned:
        st.markdown(f"""
        <div style="background:#ffffff;border:1px solid #e2e6ec;border-radius:7px;
                    padding:12px 16px;margin-bottom:8px;display:flex;align-items:flex-start;gap:14px;">
          <div style="flex:1;">
            <span style="font-size:13px;font-weight:600;color:#0f1923;">{name}</span>
            <span class="badge {badge_cls}" style="margin-left:8px;">{status}</span>
            <div style="font-size:12px;color:#5a6475;margin-top:3px;">{desc}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-label" style="margin-top:16px;">What ML Adds</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <b>Feature engineering pipeline:</b> lag features (D_{t-1}, D_{t-7}, D_{t-28}),
    rolling mean/std windows, calendar effects (day-of-week, month, holiday flags),
    external regressors (promotions, price, weather).
    </div>
    <div class="kb-box">
    <b>When ML outperforms statistical models:</b> non-linear demand patterns,
    external causal variables, large intermittent demand, portfolio-level cross-SKU learning.
    </div>
    <div class="warn-box">
    <b>When statistical models win:</b> short history (&lt;2 years), low data volume,
    interpretability requirements, rapid deployment with no training infrastructure.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# ── MAIN ──────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def main():
    # Sidebar
    p = render_sidebar()

    # ── Synthetic defaults — always present so downstream code never KeyErrors ──
    SYNTHETIC_DEFAULTS = {
        "n_periods":   60,
        "base_level":  1000.0,
        "trend_slope": 0.0,
        "season_amp":  0,
        "season_len":  12,
        "noise_std":   60,
        "n_outliers":  0,
        "outlier_mag": 3.0,
        "seed":        42,
    }
    for k, v in SYNTHETIC_DEFAULTS.items():
        p.setdefault(k, v)

    # ── Resolve demand array ──
    if "Upload" in p["data_mode"]:
        if not p["upload_success"]:
            # File not yet uploaded — show a friendly prompt, use fallback synthetic data
            st.info(
                "📂 **Upload mode selected.** Use the sidebar to upload a CSV or Excel file "
                "with a numeric demand column. Showing synthetic demo data until a file is loaded."
            )
            d_full = generate_demand(
                n_periods=p["n_periods"],
                base_level=p["base_level"],
                trend_slope=p["trend_slope"],
                seasonality_amp=p["season_amp"],
                season_length=p["season_len"],
                noise_std=p["noise_std"],
                n_outliers=p["n_outliers"],
                outlier_magnitude=p["outlier_mag"],
                seed=int(p["seed"]),
            )
        else:
            df     = p["uploaded_df"]
            col    = p["demand_col"]
            d_full = df[col].dropna().values.astype(float)
            d_full = np.clip(d_full, 0, None)

            if len(d_full) < 10:
                st.error(
                    f"❌ Column **{col}** has only {len(d_full)} non-null values. "
                    "Need at least 10 data points for forecasting."
                )
                return

            # Overwrite synthetic keys with real-data equivalents for diagnostics
            p["n_periods"]   = len(d_full)
            p["base_level"]  = float(d_full.mean())
            p["trend_slope"] = 0.0
            p["season_amp"]  = 0
            p["season_len"]  = 12

            st.success(
                f"✅ **{col}** loaded — {len(d_full):,} periods · "
                f"Mean = {d_full.mean():.0f} · "
                f"Min = {d_full.min():.0f} · "
                f"Max = {d_full.max():.0f}"
            )
    else:
        d_full = generate_demand(
            n_periods=p["n_periods"],
            base_level=p["base_level"],
            trend_slope=p["trend_slope"],
            seasonality_amp=p["season_amp"],
            season_length=p["season_len"],
            noise_std=p["noise_std"],
            n_outliers=p["n_outliers"],
            outlier_magnitude=p["outlier_mag"],
            seed=int(p["seed"]),
        )

    n          = len(d_full)
    train_end  = int(n * p["train_pct"] / 100)
    extra      = p["extra_periods"]
    d_train    = d_full[:train_end]

    # ── Compute forecasts ──
    f_naive              = naive_forecast(d_full, extra_periods=extra)
    f_ma                 = moving_average(d_full, n=p["ma_n"], extra_periods=extra)
    f_ses, _             = simple_exp_smoothing(d_full, alpha=p["ses_a"], extra_periods=extra)
    f_holt, _, _         = holts_linear(d_full, alpha=p["holt_a"], beta=p["holt_b"], extra_periods=extra)
    f_damp, _, _         = damped_trend(d_full, alpha=p["damp_a"], beta=p["damp_b"],
                                         phi=p["damp_ph"], extra_periods=extra)

    # Attach to p for page access
    p.update({"f_naive": f_naive, "f_ma": f_ma, "f_ses": f_ses,
               "f_holt": f_holt, "f_damp": f_damp})

    # ── Compute KPIs (test set only) ──
    test_actual = d_full[train_end:]
    def test_kpi(f_arr):
        n_test = len(test_actual)
        return compute_kpis(
            test_actual,
            f_arr[train_end:train_end + n_test],
            f_naive[train_end:train_end + n_test],
        )

    kpis_all = {
        "Naïve":          test_kpi(f_naive),
        "Moving Average": test_kpi(f_ma),
        "SES":            test_kpi(f_ses),
        "Holt's Linear":  test_kpi(f_holt),
        "Damped Trend":   test_kpi(f_damp),
    }

    # ── Route to selected page ──
    page = p["selected_page"]

    if   "Overview"   in page: page_overview(d_full, train_end, extra, kpis_all, p)
    elif "Model Lab"  in page: page_model_lab(d_full, train_end, extra, kpis_all, p)
    elif "Residual"   in page: page_residuals(d_full, train_end, kpis_all, p)
    elif "Decompos"   in page: page_decomposition(d_full, p)
    elif "Parameter"  in page: page_parameters(d_full, train_end, extra, p)
    elif "Ranking"    in page: page_rankings(kpis_all)
    elif "Diagnostic" in page: page_diagnostics(kpis_all, p, d_full)
    elif "Challenge"  in page: page_challenge(p)
    elif "ML Studio"  in page: page_ml_studio()


# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
