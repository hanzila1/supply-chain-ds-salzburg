# Part III — EUDR Supplier Risk

Satellite-based deforestation risk scoring for ingredient suppliers,
connected to supply chain planning. Built on Google Earth Engine methodology
from the GeoGemma project (Google DeepMind-funded, APAC Solution Challenge 2024).

## Notebooks

| # | Notebook | Concepts |
|---|----------|----------|
| 01 | `01_gee_forest_data_pipeline.ipynb` | GEE setup, Hansen dataset, point-level deforestation check |
| 02 | `02_supplier_deforestation_scoring.ipynb` | Portfolio scoring, risk aggregation, compliance dashboard |
| 03 | `03_risk_score_to_supply_volatility.ipynb` | Risk → disruption probability → safety stock → working capital |

## The Core Idea

```
GPS coordinates (from procurement)
        ↓
Google Earth Engine — Hansen Forest Loss Dataset
        ↓
Deforestation risk score per supplier plot
        ↓
Volume-weighted portfolio risk per commodity
        ↓
Supply disruption probability
        ↓
Adjusted safety stock recommendation
```

## EUDR Context

The EU Deforestation Regulation (Regulation EU 2023/1115) requires companies
placing commodities on the EU market to prove they are deforestation-free
after 31 December 2020.

**Enforcement deadline:** 30 December 2026 (large companies)

**Commodities in scope:** coffee, cocoa, palm oil, soy, rubber, wood, cattle

**For FMCG companies like Red Bull:** coffee and sugar are the primary exposure points.

## GEE Setup

```bash
pip install earthengine-api geemap
earthengine authenticate
```

Sign up: https://signup.earthengine.google.com (free for research)

All notebooks run in simulation mode if GEE is not configured.
