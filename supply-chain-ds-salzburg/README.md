# Supply Chain Data Science — Salzburg

> A progressive portfolio built alongside EM CDE studies at the University of Salzburg,
> covering demand forecasting, ML models, and geospatial EUDR compliance risk scoring
> for FMCG supply chains.

---

## About

This repository documents a structured learning and building journey at the intersection of:

- **Supply chain demand forecasting** — statistical and ML models applied to FMCG contexts
- **Geospatial AI** — satellite-based supplier risk scoring using Google Earth Engine
- **EUDR compliance** — automated deforestation risk assessment for ingredient procurement

Previously co-developed [GeoGemma](https://github.com/yourusername/geogemma) —
winner of **Best AI Use Case** at the Google & ADB Asia-Pacific Solution Challenge 2024 —
a geospatial LLM built on Google Earth Engine. This repository applies those geospatial
methods to supply chain compliance and demand forecasting.

---

## Repository Structure

```
supply-chain-ds-salzburg/
│
├── 01_statistical_forecasting/     # Part I: Classical forecasting models
│   ├── 01_moving_average.ipynb
│   ├── 02_exponential_smoothing.ipynb
│   ├── 03_double_exponential_smoothing.ipynb
│   ├── 04_triple_exponential_smoothing.ipynb
│   ├── 05_outlier_handling.ipynb
│   ├── 06_forecast_kpis.ipynb
│   └── README.md
│
├── 02_machine_learning/            # Part II: ML models for demand forecasting
│   ├── 01_decision_trees.ipynb
│   ├── 02_random_forests.ipynb
│   ├── 03_xgboost_forecasting.ipynb
│   ├── 04_external_demand_drivers.ipynb
│   ├── 05_neural_networks.ipynb
│   └── README.md
│
├── 03_eudr_supplier_risk/          # Part III: Geospatial EUDR compliance tool
│   ├── 01_gee_forest_data_pipeline.ipynb
│   ├── 02_supplier_deforestation_scoring.ipynb
│   ├── 03_risk_score_to_supply_volatility.ipynb
│   └── README.md
│
├── 04_integrated_tool/             # Part IV: End-to-end pipeline + dashboard
│   ├── pipeline.py
│   ├── dashboard.py
│   ├── case_study.md
│   └── README.md
│
├── data/
│   ├── raw/                        # Original downloaded datasets
│   ├── processed/                  # Cleaned, model-ready data
│   └── data_sources.md             # Dataset origins and licenses
│
├── requirements.txt
├── .gitignore
└── README.md                       # This file
```

---

## Roadmap

| Month | Focus | Status |
|-------|-------|--------|
| April 2025 | Statistical forecasting models (Part I) | 🔄 In progress |
| May 2025 | ML models — trees, forests, XGBoost (Part II) | ⏳ Upcoming |
| June 2025 | External demand drivers + features | ⏳ Upcoming |
| July 2025 | EUDR geospatial supplier risk pipeline (Part III) | ⏳ Upcoming |
| August 2025 | Integrated tool + dashboard + case study (Part IV) | ⏳ Upcoming |

---

## Stack

```
Python 3.10+      pandas / numpy / matplotlib / scipy
scikit-learn      Decision trees, random forests, clustering
XGBoost           Gradient boosting for demand forecasting
Google Earth Engine  Satellite imagery & deforestation data
Streamlit         Interactive dashboard
Jupyter           All notebooks
```

---

## Key References

- Vandeput, N. (2021). *Data Science for Supply Chain Forecasting* (2nd ed.). De Gruyter.
- EU Regulation (EU) 2023/1115 — EU Deforestation Regulation (EUDR)
- Google Earth Engine — [earthengine.google.com](https://earthengine.google.com)
- Global Forest Watch — [globalforestwatch.org](https://www.globalforestwatch.org)

---

## Contact

**Hanzila Bin Younus**
EM CDE Student — University of Salzburg
[LinkedIn](https://linkedin.com/in/hanzila-bin-younus-geogemma) · [GeoGemma](https://github.com/yourusername/geogemma)
