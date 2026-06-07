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

---

## What is VyaparIQ?

Wheat traders in Indian mandis make decisions worth lakhs of rupees every day — using gut feel and phone calls. No accessible, affordable, data-driven tool exists for them.

VyaparIQ is built to change that.

It pulls daily wheat auction data from government mandis across Madhya Pradesh, predicts tomorrow's modal price, and fires a buying signal when prices are temporarily depressed — validated to recover 67.3% of the time within 3 days.

---

## Features

**Layer 1 — Price Prediction**
- Predicts next day modal price for any district-variety-grade combination
- 96.4% accuracy (MAPE 3.6%), MAE ₹86/quintal
- 71% of predictions within ₹100 of actual market price
- MSP floor monitoring — alerts when predicted price is below government minimum

**Layer 2 — Buying Signal**
- Fires when price drops ₹60+ below 3-day rolling average
- Active only in high-recovery months: Nov, Dec, Jan, Mar, May
- 67.3% of signals led to price recovery within 3 days
- Average gain of ₹34/quintal on triggered signals

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
| MAE | ₹86.28 / quintal |
| RMSE | ₹133.44 / quintal |
| Accuracy | 96.4% (MAPE 3.6%) |
| Train period | Nov 2025 — Mar 2026 |
| Test period | Apr 2026 — May 2026 |
| Scope | Regular wheat, Nov–May season |

**Why Linear Regression over XGBoost?**

XGBoost and Random Forest were tested but both underperformed Linear Regression (MAE ₹112 and ₹87.6 respectively vs ₹86.3). With one season of data spread across 45 markets and 10 varieties, per-group training data is sparse. XGBoost overfit on seasonal patterns that don't generalize. Linear Regression captured the dominant signal — yesterday's price predicts tomorrow's price — cleanly and reliably.

---

## Key Findings from EDA

- **Price leads arrivals by one month** — prices bottom in March before peak arrivals hit in April. The market prices in the harvest before trucks arrive (price discovery).
- **MSP functions as a genuine floor** — average prices breached ₹2425 MSP in March and April during harvest surge.
- **Sharbati is a separate market** — trades ₹600+ above regular wheat with different seasonal dynamics. Modeled separately (insufficient data for v1).
- **Daily arrivals weakly predict price** — correlation of -0.07. Season position dominates over daily supply fluctuations.

---

## Architecture
