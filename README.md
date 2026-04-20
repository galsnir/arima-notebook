# ARIMA Time-Series Forecasting from Scratch

A Jupyter notebook that builds an **ARIMA** (AutoRegressive Integrated Moving Average) model from scratch, validates it on simulated data, and then applies both the custom and `statsmodels` implementations to forecast Amazon (AMZN) stock prices.

Authors: **Gal Snir** (313588279) and **Efraim Yosofov** (318765716). Home Assignment 3.

---

## Table of contents

- [Overview](#overview)
- [What's in the notebook](#whats-in-the-notebook)
- [Algorithm](#algorithm)
- [Datasets](#datasets)
- [Methodology](#methodology)
- [Custom vs. statsmodels ARIMA](#custom-vs-statsmodels-arima)
- [Project structure](#project-structure)
- [Requirements](#requirements)
- [How to run](#how-to-run)
- [Notes](#notes)

---

## Overview

[ARIMA](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average) is one of the workhorses of classical time-series forecasting. It models a (possibly non-stationary) series with three components:

- **AR(p)** — \(p\) lagged observations of the series itself.
- **I(d)** — \(d\) successive differencings to make the series stationary.
- **MA(q)** — \(q\) lagged forecast errors.

This notebook implements the model from scratch as a `ManualARIMA` class, then tests it against a known-truth simulated process and a real-world financial series.

## What's in the notebook

| Part | Description |
| --- | --- |
| **1 — Design the algorithm** | Explanation of the AR, I, and MA components, parameter triplet \((p, d, q)\), stationarity, and how the manual implementation is structured. |
| **2 — Implement the algorithm** | The `ManualARIMA` class: `fit`, `predict`, residual handling, AR / MA coefficient estimation, drift / trend handling, and a prediction cap to avoid runaway forecasts. |
| **3 — Validate on simulated data** | A controlled "unit test" on an ARIMA(1, 1, 1) process with known coefficients (\(\phi_1 = 0.7\), \(\theta_1 = 0.3\)), confirming the implementation captures the underlying dynamics. |
| **4 — Real-world experiment (AMZN)** | EDA, stationarity testing (ADF), ACF / PACF inspection, grid search over \((p, d, q)\), residual diagnostics, forecasting, and quantitative comparison vs. `statsmodels.tsa.arima.model.ARIMA`. |
| **5 — Report** | A written summary of differences between the manual and library implementations and overall conclusions. |

## Algorithm

The `ManualARIMA(p, d, q)` class follows the standard ARIMA recipe:

1. **Differencing.** Apply \(d\) successive differences to the input series until it is (approximately) stationary.
2. **AR estimation.** Fit AR coefficients \(\phi_1, \dots, \phi_p\) on the differenced series.
3. **MA estimation.** Estimate MA coefficients \(\theta_1, \dots, \theta_q\) from the residuals of the AR fit.
4. **Forecasting.** Iteratively roll the model forward, using AR + MA contributions and the in-sample residual cache.
5. **Inverse differencing.** Cumulatively sum predictions back to the original scale, optionally adding a learned drift / trend term.
6. **Realism guard.** A `max_value` cap (twice the maximum observed value during fitting) prevents the model from producing absurd forecasts (e.g. negative prices or runaway extrapolations).

Key attributes tracked on the class:

- `ar_coef`, `ma_coef` — fitted coefficients.
- `history`, `errors` — caches used during rolling forecasts.
- `original_series`, `residuals` — book-keeping for diagnostics.
- `trend_slope`, `drift` — captured trend information.
- `max_value` — prediction cap.

## Datasets

### 1. Simulated ARIMA(1, 1, 1)

- 110 points (100 train / 10 test).
- AR coefficient \(\phi_1 = 0.7\), MA coefficient \(\theta_1 = 0.3\), Gaussian noise \(\mathcal{N}(0, 1)\).
- Generated as an ARMA(1, 1) process integrated with a cumulative sum, so the first difference returns a stationary series.

This serves as a **unit test**: with known parameters, we can verify that the manual implementation recovers them and produces sensible residuals.

### 2. Amazon (AMZN) stock prices

- Source: `yfinance` (or the bundled CSV, see [How to run](#how-to-run)).
- Period: 1997-05-15 → 2025-03-10 (the entire history of AMZN as a public company).
- Frequency: business-day (`asfreq('B')`), missing values filled with `ffill`.

The bundled `amzn_stock_prices.csv` lets you run the notebook end-to-end **without** an internet connection.

## Methodology

The real-world experiment follows standard time-series best practices:

1. **Loading & cleaning.** Forward-fill missing days, ensure a continuous business-day index.
2. **EDA.**
   - Summary statistics, rolling mean / std plots.
   - Stationarity tests: **Augmented Dickey-Fuller (ADF)** on raw and differenced series.
   - **ACF / PACF** plots to suggest reasonable starting values for \(p\) and \(q\).
   - Seasonal decomposition (`statsmodels.tsa.seasonal.seasonal_decompose`).
   - Histogram, lag plots, outlier inspection (e.g. stock splits), yearly returns aggregation.
3. **Train/test split.** Chronological split — never random — to respect causality.
4. **Model selection.** Grid search over a small range of \((p, d, q)\) using AIC / BIC and out-of-sample error.
5. **Diagnostics.** Residual normality / autocorrelation checks (Ljung-Box).
6. **Forecasting.** Multi-step-ahead predictions plotted against the held-out test set.
7. **Evaluation metrics.**
   - **MSE** — mean squared error.
   - **MAE** — mean absolute error.
   - **Directional accuracy** — fraction of steps where the sign of the predicted change matches the actual change (especially relevant for trading-style use cases).

## Custom vs. statsmodels ARIMA

Both implementations are evaluated side-by-side. Key differences highlighted in the notebook:

| Aspect | `ManualARIMA` (ours) | `statsmodels.tsa.arima.model.ARIMA` |
| --- | --- | --- |
| Coefficient estimation | Direct least-squares style updates on differenced series and residuals | Maximum likelihood via Kalman filter |
| Differencing | Manual `np.diff` followed by inverse `cumsum` | Built into the state-space representation |
| Trend / drift | Learned slope reapplied during inverse differencing | Configurable via `trend` parameter (`'n'`, `'c'`, `'t'`, `'ct'`) |
| Forecast safety | Hard `max_value` cap | None by default |
| Performance | Pure NumPy; intentionally readable | Highly optimized, with confidence intervals, model selection helpers, and ARIMAX support |

The notebook concludes with a discussion of when each implementation is preferable and what was learned about AMZN's series specifically (strong trend, increasing volatility, weak seasonality, evidence of structural breaks around stock splits).

## Project structure

```
arima-notebook/
├── ARIMA.ipynb              # The main notebook
├── amzn_stock_prices.csv    # Historical AMZN closing prices
├── README.md                # This file
├── requirements.txt         # Python dependencies
└── .gitignore
```

## Requirements

- Python 3.9 or newer
- `numpy`
- `pandas`
- `matplotlib`
- `scipy`
- `scikit-learn`
- `statsmodels`
- `yfinance` (only required if you want to refresh the data live)
- `jupyter`

Install everything with:

```bash
pip install -r requirements.txt
```

## How to run

```bash
git clone https://github.com/galsnir/arima-notebook.git
cd arima-notebook
pip install -r requirements.txt
jupyter notebook ARIMA.ipynb
```

Then run the cells top-to-bottom. By default the notebook will look for `amzn_stock_prices.csv` in the working directory, so the included file lets it run **offline**. If you'd rather pull a fresh series from Yahoo Finance, the relevant cell uses `yfinance.download("AMZN", ...)` — just uncomment / re-run it.

## Notes

- The custom implementation prioritizes clarity over raw performance and is **not** intended as a drop-in replacement for `statsmodels.ARIMA`.
- Stock-price forecasting is notoriously hard. The point of this notebook is methodological — comparing implementations, illustrating the ARIMA workflow, and producing principled diagnostics — not to make trading decisions.
- All randomness uses fixed seeds for reproducibility.
