"""
Supply Chain DS — Integrated Pipeline
======================================
End-to-end pipeline combining:
  1. Demand forecasting with external drivers (XGBoost)
  2. EUDR supplier deforestation risk scoring
  3. Risk-adjusted safety stock calculation

Author: Hanzila Bin Younus
Context: EM CDE — University of Salzburg
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from scipy.optimize import minimize


# ─────────────────────────────────────────────
# 1. DEMAND FORECASTING MODULE
# ─────────────────────────────────────────────

def generate_fmcg_demand(n_weeks=260, seed=42):
    """
    Generate realistic FMCG demand with trend, seasonality,
    temperature correlation, and event spikes.
    Simulates an energy drink product (Red Bull-like context).
    """
    np.random.seed(seed)
    dates = pd.date_range(start='2019-01-01', periods=n_weeks, freq='W')

    trend = 1000 + 1.5 * np.arange(n_weeks)
    seasonality = 150 * np.sin(2 * np.pi * np.arange(n_weeks) / 52)
    temperature = 15 + 12 * np.sin(2 * np.pi * np.arange(n_weeks) / 52)
    temp_effect = 8 * temperature
    noise = np.random.normal(0, 60, n_weeks)

    event_weeks = [20, 45, 84, 135, 190, 240]
    event_boost = np.zeros(n_weeks)
    for w in event_weeks:
        if w < n_weeks:
            event_boost[w] = np.random.uniform(200, 500)

    demand = trend + seasonality + temp_effect + event_boost + noise
    demand = np.clip(demand, 0, None)

    return pd.DataFrame({
        'demand': demand,
        'temperature': temperature,
        'has_event': (event_boost > 0).astype(int)
    }, index=dates)


def create_ml_features(df, lags=[1, 2, 4, 8, 52]):
    """Create supervised ML features from time series."""
    df = df.copy()
    for lag in lags:
        df[f'demand_lag_{lag}'] = df['demand'].shift(lag)
    df['rolling_mean_4']  = df['demand'].shift(1).rolling(4).mean()
    df['rolling_mean_12'] = df['demand'].shift(1).rolling(12).mean()
    df['week_of_year']    = df.index.isocalendar().week.astype(int)
    df['month']           = df.index.month
    df['temp_lag_1']      = df['temperature'].shift(1)
    return df.dropna()


def train_demand_model(df, forecast_horizon=12):
    """Train XGBoost demand forecasting model."""
    df_f = create_ml_features(df)
    feature_cols = [c for c in df_f.columns if c != 'demand']

    X = df_f[feature_cols]
    y = df_f['demand']

    X_train = X.iloc[:-forecast_horizon]
    X_test  = X.iloc[-forecast_horizon:]
    y_train = y.iloc[:-forecast_horizon]
    y_test  = y.iloc[-forecast_horizon:]

    model = XGBRegressor(
        n_estimators=300, learning_rate=0.05,
        max_depth=4, random_state=42, verbosity=0
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)

    return model, predictions, y_test.values, mae, feature_cols


# ─────────────────────────────────────────────
# 2. EUDR SUPPLIER RISK MODULE
# ─────────────────────────────────────────────

def get_supplier_portfolio():
    """Return simulated supplier portfolio with EUDR risk scores."""
    return pd.DataFrame([
        {'supplier_id': 'COL-001', 'commodity': 'coffee',
         'volume_tons': 45,  'eudr_risk': 0.12, 'country': 'Colombia'},
        {'supplier_id': 'COL-002', 'commodity': 'coffee',
         'volume_tons': 120, 'eudr_risk': 0.18, 'country': 'Colombia'},
        {'supplier_id': 'ETH-001', 'commodity': 'coffee',
         'volume_tons': 95,  'eudr_risk': 0.09, 'country': 'Ethiopia'},
        {'supplier_id': 'ETH-002', 'commodity': 'coffee',
         'volume_tons': 210, 'eudr_risk': 0.11, 'country': 'Ethiopia'},
        {'supplier_id': 'BRA-001', 'commodity': 'sugar',
         'volume_tons': 500, 'eudr_risk': 0.48, 'country': 'Brazil'},
    ])


def portfolio_weighted_risk(suppliers, commodity):
    """Volume-weighted EUDR risk for a commodity."""
    sub = suppliers[suppliers['commodity'] == commodity]
    if len(sub) == 0:
        return 0.0
    return float(np.average(sub['eudr_risk'], weights=sub['volume_tons']))


def risk_to_disruption_prob(risk_score):
    """Convert EUDR risk score to annual supply disruption probability."""
    return float(np.clip(
        0.02 + 0.68 * (1 / (1 + np.exp(-8 * (risk_score - 0.35)))),
        0.02, 0.70
    ))


# ─────────────────────────────────────────────
# 3. SAFETY STOCK MODULE
# ─────────────────────────────────────────────

def risk_adjusted_safety_stock(mean_demand, std_demand, lead_time,
                                service_level_z, disruption_prob,
                                disruption_duration=4):
    """
    Safety stock = demand variability buffer + EUDR disruption buffer.

    Parameters
    ----------
    mean_demand : float — weekly units
    std_demand : float — weekly std dev
    lead_time : int — weeks
    service_level_z : float — 1.64 = 95%, 2.05 = 98%
    disruption_prob : float — annual probability
    disruption_duration : int — expected disruption length (weeks)
    """
    ss_demand = service_level_z * std_demand * np.sqrt(lead_time)
    weekly_prob = 1 - (1 - disruption_prob) ** (1 / 52)
    ss_disruption = weekly_prob * mean_demand * disruption_duration

    return {
        'ss_demand': round(ss_demand, 1),
        'ss_disruption': round(ss_disruption, 1),
        'total': round(ss_demand + ss_disruption, 1)
    }


# ─────────────────────────────────────────────
# 4. MAIN PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(verbose=True):
    """Run the full integrated pipeline."""

    if verbose:
        print('=' * 55)
        print(' Supply Chain DS — Integrated Pipeline')
        print('=' * 55)

    # Step 1: Demand forecasting
    if verbose:
        print('\n[1/3] Demand Forecasting (XGBoost + external drivers)')
    df = generate_fmcg_demand()
    model, predictions, actuals, mae, features = train_demand_model(df)

    if verbose:
        mape = float(np.mean(np.abs(actuals - predictions) / actuals) * 100)
        print(f'      MAE:  {mae:.1f} units/week')
        print(f'      MAPE: {mape:.1f}%')
        print(f'      Features used: {len(features)}')

    # Step 2: EUDR risk scoring
    if verbose:
        print('\n[2/3] EUDR Supplier Risk Scoring')
    suppliers = get_supplier_portfolio()

    coffee_risk = portfolio_weighted_risk(suppliers, 'coffee')
    sugar_risk  = portfolio_weighted_risk(suppliers, 'sugar')
    coffee_disruption = risk_to_disruption_prob(coffee_risk)
    sugar_disruption  = risk_to_disruption_prob(sugar_risk)

    if verbose:
        print(f'      Coffee portfolio risk:    {coffee_risk:.3f} '
              f'→ {coffee_disruption*100:.1f}% disruption prob')
        print(f'      Sugar portfolio risk:     {sugar_risk:.3f} '
              f'→ {sugar_disruption*100:.1f}% disruption prob')

    at_risk = suppliers[suppliers['eudr_risk'] > 0.4]
    if verbose and len(at_risk) > 0:
        print(f'      High-risk suppliers: {list(at_risk.supplier_id)}')

    # Step 3: Risk-adjusted safety stock
    if verbose:
        print('\n[3/3] Risk-Adjusted Safety Stock')

    coffee_ss = risk_adjusted_safety_stock(
        mean_demand=180, std_demand=25, lead_time=8,
        service_level_z=1.64, disruption_prob=coffee_disruption
    )
    sugar_ss = risk_adjusted_safety_stock(
        mean_demand=500, std_demand=60, lead_time=4,
        service_level_z=1.64, disruption_prob=sugar_disruption
    )

    if verbose:
        print(f'      Coffee SS: {coffee_ss["total"]:.0f} tons '
              f'(disruption buffer: {coffee_ss["ss_disruption"]:.0f})')
        print(f'      Sugar SS:  {sugar_ss["total"]:.0f} tons '
              f'(disruption buffer: {sugar_ss["ss_disruption"]:.0f})')

    # Summary
    results = {
        'demand_mae': mae,
        'coffee_eudr_risk': coffee_risk,
        'sugar_eudr_risk': sugar_risk,
        'coffee_safety_stock': coffee_ss,
        'sugar_safety_stock': sugar_ss,
        'high_risk_suppliers': list(at_risk['supplier_id']),
        'forecast_predictions': predictions,
        'forecast_actuals': actuals
    }

    if verbose:
        print('\n' + '=' * 55)
        print(' Pipeline complete.')
        print('=' * 55)

    return results


if __name__ == '__main__':
    results = run_pipeline(verbose=True)
