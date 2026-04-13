# Part IV — Integrated Tool

End-to-end pipeline combining demand forecasting, EUDR risk scoring,
and safety stock optimisation into a single runnable system.

## Files

| File | Purpose |
|------|---------|
| `pipeline.py` | Core pipeline — importable module + CLI runner |
| `dashboard.py` | Streamlit interactive dashboard |
| `case_study.md` | Business case write-up for FMCG context |

## Quick Start

### Run the pipeline (CLI)
```bash
cd 04_integrated_tool
python pipeline.py
```

### Launch the dashboard
```bash
pip install streamlit
streamlit run dashboard.py
```

The dashboard opens at `http://localhost:8501` with:
- Demand forecast tab (XGBoost + external drivers)
- EUDR supplier risk tab (portfolio scoring + compliance)
- Safety stock tab (risk-adjusted recommendations)
- Sidebar sliders to simulate supplier risk scenarios

## Architecture

```
pipeline.py
├── generate_fmcg_demand()          → synthetic FMCG demand dataset
├── create_ml_features()            → lag features, rolling stats, calendar
├── train_demand_model()            → XGBoost training + evaluation
├── get_supplier_portfolio()        → simulated supplier GPS + risk data
├── portfolio_weighted_risk()       → volume-weighted commodity risk
├── risk_to_disruption_prob()       → risk score → annual disruption %
├── risk_adjusted_safety_stock()    → standard SS + EUDR buffer
└── run_pipeline()                  → orchestrates all steps
```

## Replacing Simulation with Real Data

| Component | Simulation | Production replacement |
|-----------|------------|----------------------|
| Demand data | `generate_fmcg_demand()` | ERP / SAP export |
| Supplier GPS | hardcoded DataFrame | Procurement system API |
| Deforestation risk | `simulate_deforestation_check()` | GEE Hansen pipeline (notebook 01) |
| Weather data | sine wave | Open-Meteo API (notebook 04) |
