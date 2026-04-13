# Data Sources

All datasets used in this project are publicly available and free to use.

---

## Statistical Forecasting & ML (Parts I & II)

### Simulated FMCG Demand Data
- **Source:** Generated synthetically using `numpy` (see notebooks)
- **Description:** Simulated beverage demand with trend, seasonality, and noise
- **License:** N/A — self-generated

### M5 Forecasting Competition Dataset
- **Source:** [Kaggle M5 Competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy)
- **Description:** Walmart sales data — 42,840 time series across product categories
- **License:** CC BY 4.0 (for academic/research use)
- **Download:** Requires Kaggle account

### Rossmann Store Sales
- **Source:** [Kaggle Rossmann](https://www.kaggle.com/competitions/rossmann-store-sales)
- **Description:** Historical sales data for 1,115 stores
- **License:** CC BY 4.0

---

## External Demand Drivers (Part II — Chapter 04)

### Open-Meteo Weather API
- **Source:** [open-meteo.com](https://open-meteo.com)
- **Description:** Free weather API — temperature, precipitation, wind
- **License:** Free for non-commercial use, CC BY 4.0 for data
- **API:** No key required for basic use

### Google Trends
- **Source:** [trends.google.com](https://trends.google.com) via `pytrends`
- **Description:** Search interest over time — useful as demand proxy
- **License:** Public data

---

## EUDR Supplier Risk Pipeline (Part III)

### Global Forest Watch — Hansen Dataset
- **Source:** [Global Forest Watch](https://www.globalforestwatch.org)
- **GEE Asset:** `UMD/hansen/global_forest_change_2023_v1_11`
- **Description:** Annual tree cover loss at 30m resolution, 2000–2023
- **License:** CC BY 4.0

### Google Earth Engine
- **Source:** [earthengine.google.com](https://earthengine.google.com)
- **Description:** Cloud platform for planetary-scale geospatial analysis
- **Access:** Free for research — requires GEE account registration
- **Sign up:** [signup.earthengine.google.com](https://signup.earthengine.google.com)

### EUDR Country Risk Classifications
- **Source:** [European Commission](https://environment.ec.europa.eu/topics/forests/deforestation/regulation-deforestation-free-products_en)
- **Description:** Official country benchmarking by deforestation risk level
- **License:** Public domain (EU Open Data)

### Simulated Supplier Coordinates
- **Source:** Self-generated (see `03_eudr_supplier_risk/01_gee_forest_data_pipeline.ipynb`)
- **Description:** Synthetic GPS coordinates for coffee/sugar growing regions
- **Note:** Replace with real supplier data for production use

---

## Notes

- Never commit raw data files to this repository (see `.gitignore`)
- All notebooks include download/generation instructions for their required data
- For any API requiring credentials, use `.env` file (never commit credentials)
