# Case Study: EUDR-Aware Supply Chain Forecasting for a Global FMCG Company

**Context:** This case study applies the methodology developed in this repository
to a hypothetical global energy drink company with operations in Salzburg, Austria.

---

## Business Problem

A global FMCG company sources coffee extract and sugar from suppliers across
Colombia, Ethiopia, and Brazil. With the EU Deforestation Regulation (EUDR)
enforcement deadline approaching (December 2026), the company faces two challenges:

1. **Compliance risk:** Some suppliers may not be EUDR-compliant, risking EU market ban
2. **Supply risk:** Non-compliant suppliers may face disruption, affecting production

The company's DS team and sustainability procurement team are working in silos —
the DS team optimises forecasting and inventory without considering EUDR risk,
and the sustainability team is managing compliance without connecting it to
supply chain planning.

---

## Solution: Integrated EUDR-Aware Supply Chain Pipeline

### Step 1 — Demand Forecasting with External Drivers

XGBoost model trained on 5 years of weekly demand data with features:
- Lag demand (1, 2, 4, 8, 52 weeks)
- Temperature data (Open-Meteo API — free)
- Sporting/cultural events calendar
- Calendar features (week of year, month, seasonality)

**Result:** 12-week forecast with ~8% MAPE, significantly outperforming
Holt-Winters baseline due to external driver incorporation.

### Step 2 — EUDR Supplier Deforestation Risk Scoring

Google Earth Engine pipeline querying the Hansen Global Forest Change dataset
at each supplier GPS coordinate. For each plot:
- Checks for tree cover loss after 31 December 2020 (EUDR cutoff)
- Assigns a risk score combining: loss probability, tree cover baseline, country risk
- Aggregates to volume-weighted portfolio risk per commodity

**Result:**
| Commodity | Portfolio Risk | Disruption Prob |
|-----------|---------------|-----------------|
| Coffee | 0.13 (LOW) | 3.2% /year |
| Sugar | 0.48 (HIGH) | 31.5% /year |

Key finding: the Brazilian sugar supplier (BRA-001) is the highest-risk supplier
in the portfolio, with a deforestation risk score of 0.48.

### Step 3 — Risk-Adjusted Safety Stock

Standard safety stock (for demand variability) + EUDR disruption buffer:

```
SS_total = SS_demand + SS_disruption
SS_demand    = z × σ × √(lead_time)
SS_disruption = P(weekly disruption) × mean_demand × disruption_duration
```

**Results at 95% service level:**

| Ingredient | Standard SS | EUDR Buffer | Total SS | Buffer % |
|------------|-------------|-------------|----------|----------|
| Coffee | 116 tons | 8 tons | 124 tons | 6.5% |
| Sugar | 196 tons | 122 tons | 318 tons | 38.4% |

**Key insight:** The sugar ingredient requires 38% more safety stock than
standard calculations suggest — purely due to EUDR supply disruption risk.
This represents significant working capital that could be released by
improving supplier compliance.

---

## Business Impact

### Scenario Analysis

If the company switches BRA-001 to a certified low-risk sugar supplier (risk: 0.08):

| Metric | Current | After Switch | Saving |
|--------|---------|-------------|--------|
| Sugar safety stock | 318 tons | 198 tons | 120 tons |
| Working capital (€8/kg) | €2.54M | €1.58M | **€960K /year** |
| Disruption probability | 31.5% | 5.1% | ↓ 84% |

### Strategic Value

1. **Compliance is financeable** — connecting EUDR compliance to working capital
   savings creates a business case Finance can support

2. **DS and sustainability teams have shared KPIs** — this pipeline creates
   a common language between two teams that previously worked in isolation

3. **Proactive vs reactive** — most companies will respond to EUDR disruptions
   after they happen; this pipeline enables pre-emptive supplier diversification

---

## Methodology Notes

- **Satellite data:** Hansen Global Forest Change (UMD, 30m resolution, annual)
- **EUDR cutoff:** 31 December 2020 (as per Regulation EU 2023/1115)
- **Risk-to-disruption mapping:** Sigmoid function calibrated to industry disruption data
- **Safety stock formula:** Extension of standard statistical safety stock incorporating supply-side risk

---

## Limitations & Future Work

- GPS coordinates must come from actual procurement data (not simulated)
- Risk-to-disruption probability mapping should be calibrated on historical disruption events
- Pipeline should run quarterly as new Hansen data releases annually
- Natural language query interface (GeoGemma methodology) would allow
  sustainability managers to query supplier risk without coding

---

## Technical Stack

```
Python 3.10 · XGBoost · Google Earth Engine API
pandas · numpy · scipy · Streamlit
Open-Meteo API (weather) · Hansen GFC Dataset
```

---

*Hanzila Bin Younus — EM CDE, University of Salzburg*
*Building on GeoGemma — Best AI Use Case, Google & ADB APAC Solution Challenge 2024*
