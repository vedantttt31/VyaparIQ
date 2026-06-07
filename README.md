---
title: VyaparIQ
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: docker
pinned: false
short_description: Wheat price intelligence for MP mandi traders
---

# 🌾 VyaparIQ — Wheat Price Intelligence for MP Mandis

> Real-time wheat price predictions and buying signals for agricultural commodity traders in Madhya Pradesh, India.

**Live Demo:** [huggingface.co/spaces/Vedantttt31/VyaparIQ](https://huggingface.co/spaces/Vedantttt31/VyaparIQ)
**GitHub:** [github.com/vedantttt31/VyaparIQ](https://github.com/vedantttt31/VyaparIQ)

---

## What is VyaparIQ?

Wheat traders in Indian mandis make decisions worth lakhs of rupees every day — using gut feel and phone calls. No accessible, affordable, data-driven tool exists for them.

VyaparIQ is built to change that.

It pulls daily wheat auction data from government mandis across Madhya Pradesh, predicts tomorrow's modal price, and fires a buying signal when prices are temporarily depressed — validated to recover 67.3% of the time within 3 days.

---

## Features

**Layer 1 — Price Prediction**
- Predicts next day modal price for any district-variety-grade combination
- 96.4% accuracy (MAPE 3.6%), MAE Rs.86/quintal
- 71% of predictions within Rs.100 of actual market price
- MSP floor monitoring — alerts when predicted price is below government minimum

**Layer 2 — Buying Signal**
- Fires when price drops Rs.60+ below 3-day rolling average
- Active only in high-recovery months: Nov, Dec, Jan, Mar, May
- 67.3% of signals led to price recovery within 3 days
- Average gain of Rs.34/quintal on triggered signals

**Live Ticker**
- Real-time price changes across all 8 MP districts
- Updates daily from agmarknet.gov.in

---

## Data

- **Source:** agmarknet.gov.in — Government of India agricultural market data
- **Coverage:** 8 MP districts — Betul, Bhopal, Dewas, Harda, Hoshangabad, Sehore, Ujjain, Vidisha
- **Range:** November 2025 — present (updated daily)
- **Records:** 10,000+ wheat auction records
- **Weather:** Open-Meteo API for daily max temperature

---

## Model

| Detail | Value |
|---|---|
| Algorithm | Linear Regression |
| Target | Next day modal price (T+1) |
| MAE | Rs.86.28 / quintal |
| RMSE | Rs.133.44 / quintal |
| Accuracy | 96.4% (MAPE 3.6%) |
| Train period | Nov 2025 - Mar 2026 |
| Test period | Apr 2026 - May 2026 |
| Scope | Regular wheat, Nov-May season |

**Why Linear Regression over XGBoost?**

XGBoost and Random Forest were tested but both underperformed Linear Regression (MAE Rs.112 and Rs.87.6 respectively vs Rs.86.3). With one season of data spread across 45 markets and 10 varieties, per-group training data is sparse. XGBoost overfit on seasonal patterns that don't generalize. Linear Regression captured the dominant signal — yesterday's price predicts tomorrow's price — cleanly and reliably.

---

## Key Findings from EDA

- **Price leads arrivals by one month** — prices bottom in March before peak arrivals hit in April. The market prices in the harvest before trucks arrive (price discovery).
- **MSP functions as a genuine floor** — average prices breached Rs.2425 MSP in March and April during harvest surge.
- **Sharbati is a separate market** — trades Rs.600+ above regular wheat with different seasonal dynamics. Excluded from v1 due to insufficient data.
- **Daily arrivals weakly predict price** — correlation of -0.07. Season position dominates over daily supply fluctuations.

---

## Architecture

agmarknet.gov.in API → update_pipeline.py → data_clean.csv
                                                     |
Open-Meteo API ─────────────────────────────────────►|
                                                     ▼
                                              model_lr.pkl (retrained daily)
                                                     |
                                                     ▼
                                           app.py (Streamlit dashboard)
                                                     |
                                                     ▼
                                      Hugging Face Spaces (Docker)

**Automation:** GitHub Actions runs update_pipeline.py every morning at 6am IST.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data collection | Python requests — agmarknet POST API (reverse engineered) |
| Weather | Open-Meteo API |
| Data processing | Pandas, NumPy |
| ML model | Scikit-learn Linear Regression |
| Dashboard | Streamlit |
| Deployment | Docker, Hugging Face Spaces |
| Automation | GitHub Actions (cron schedule) |
| Version control | Git |

---

## Limitations

- **Season scope:** Model trained on Nov 2025 - May 2026 only. June-October months have no training data.
- **One season of data:** SARIMA and LSTM models require multiple seasons — revisit when 2+ years accumulate.
- **Sharbati excluded:** Insufficient data for v1. Planned for v2.
- **Weather proxy:** Single MP temperature used for all districts.
- **MAE vs trader benchmark:** Experienced traders predict within Rs.20-30/quintal using local network. VyaparIQ at Rs.86 MAE is most useful for less experienced traders or as a systematic cross-district signal.

---

## Running Locally

git clone https://github.com/vedantttt31/VyaparIQ.git
cd VyaparIQ
pip install -r requirements.txt
streamlit run app.py

Daily update:
python update_pipeline.py

---

## Roadmap

- [ ] Layer 3 — Grain quality classifier via phone camera
- [ ] Sharbati-specific model (when 2+ seasons available)
- [ ] FastAPI /predict endpoint
- [ ] District-level weather integration
- [ ] WhatsApp alert integration for buy signals
- [ ] SARIMA/Prophet models with multi-season data

---

## Project Background

Built by a 3rd year engineering student whose family trades wheat in MP mandis. Domain knowledge from the ground up — every modeling decision was validated against real trading reality, not just metrics.

---

*Data source: agmarknet.gov.in (Government of India) · Weather: Open-Meteo · Built with Python, Scikit-learn, Streamlit, Docker, GitHub Actions*
